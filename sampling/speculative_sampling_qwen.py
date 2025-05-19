import torch
from tqdm import tqdm
import torch
from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder

# 添加一个新的KVCacheModel扩展类，用于处理词表大小不匹配的问题
class AlignedKVCacheModel(KVCacheModel):
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0, target_vocab_size : int = None) -> None:
        super().__init__(model, temperature, top_k, top_p)
        self.target_vocab_size = target_vocab_size
        self.original_vocab_size = None  # 将在第一次前向传播时确定
        
    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug=False) -> torch.Tensor:
        # 不调用父类方法，完全重新实现
        input_ids = input_ids.to(self._device)
        
        if self._past_key_values is None:
            # 首次前向传播
            outputs = self._model(input_ids)
            logits = outputs.logits
            
            # 确保词表大小一致
            if self.target_vocab_size is not None:
                if self.original_vocab_size is None:
                    self.original_vocab_size = logits.shape[-1]
                
                if self.original_vocab_size != self.target_vocab_size:
                    logits = logits[..., :self.target_vocab_size]
                    # 可选: 归一化
                    logits = logits / logits.sum(dim=-1, keepdim=True)
            
            self._prob_history = logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], 
                                                        self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # 增量前向传播
            # 计算缓存长度
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
            
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            last_input_id = last_input_id.to(self._device)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
            # 处理词表大小
            if self.target_vocab_size is not None:
                if self.original_vocab_size is None:
                    self.original_vocab_size = not_cached_q.shape[-1]
                
                if self.original_vocab_size != self.target_vocab_size:
                    not_cached_q = not_cached_q[..., :self.target_vocab_size]
                    # 可选: 归一化
                    not_cached_q = not_cached_q / not_cached_q.sum(dim=-1, keepdim=True)
            
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], 
                                                self._temperature, self._top_k, self._top_p)
            
            # 现在两者维度应该一致，可以安全连接
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q

@torch.no_grad()
def speculative_sampling_qwen(
        prefix: torch.Tensor,
        approx_model: torch.nn.Module,
        target_model: torch.nn.Module,
        max_len: int,
        gamma: int = 4,
        temperature: float = 1,
        D: int = 0,
        top_p: float = 0,
        verbose: bool = False,
        random_seed: int = None
) -> torch.Tensor:
    """
    Speculative Decoding for Qwen 系列（词表可能不等长）。
    只在循环外建立一次 KV-cache；循环内完全增量推理。
    """
    # ---------- 基础准备 ----------
    seq_len   = prefix.shape[1]
    total_len = seq_len + max_len
    assert prefix.shape[0] == 1, "目前仅支持 batch = 1"

    device = next(target_model.parameters()).device
    prefix = prefix.to(device)

    v_small = getattr(approx_model.config, "vocab_size", approx_model.lm_head.out_features)
    v_large = getattr(target_model.config, "vocab_size", target_model.lm_head.out_features)
    v_min   = min(v_small, v_large)

    approx_cache = AlignedKVCacheModel(approx_model, temperature, top_k, top_p, v_min)
    target_cache = AlignedKVCacheModel(target_model, temperature, top_k, top_p, v_min)

    # ---------- 首次 Prefill ----------
    approx_cache.reset_cache()
    target_cache.reset_cache()
    approx_cache.generate(prefix, 0)        # 仅建立 KV
    target_cache.generate(prefix, 0)
    
    _ = approx_cache._forward_with_kvcache(prefix)
    _ = target_cache._forward_with_kvcache(prefix)

    accepted_cnt = resample_cnt = target_cnt = 0

    while prefix.shape[1] < total_len:
        prefix_len = prefix.shape[1]

        # === 1) 小模型生成 γ 个草稿 ===
        draft  = prefix.clone()
        q_last = approx_cache._prob_history[:, -1, :]          # prefix 最后 1 token 的分布
        for _ in range(gamma):
            if random_seed is not None:
                torch.manual_seed(random_seed)
            next_tok = sample(q_last).to(device)
            draft    = torch.cat((draft, next_tok), dim=1)
            q_last   = approx_cache._forward_with_kvcache(draft)   # 更新 KV & 拿到下一步 logits

        # draft 现在是 prefix + g₀…g_{γ-1}

        # === 2) 大模型一次并行算 γ 步 ===
        _ = target_cache._forward_with_kvcache(draft)              # 更新 KV，得到 p₀…p_{γ-1}

        # === 3) 接受 / 拒绝判定 ===
        n = prefix_len - 1                                         # 初始有效前缀末尾下标
        for i in range(gamma):
            j = draft[:, prefix_len + i]                           # g_i
            p = target_cache._prob_history[:, prefix_len + i, j]
            q = approx_cache._prob_history[:, prefix_len + i, j]

            if random_seed is not None:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device=device)

            if r > p / q:                                          # —— Reject
                n = prefix_len + i - 1
                break
            accepted_cnt += 1
            if verbose:
                print(f"[accept] {Decoder().decode(j)}")

        # === 4) 回滚到 n（最后一个被接受的 token） ===
        approx_cache.rollback(n + 1)
        target_cache.rollback(n + 1)
        prefix = draft[:, : n + 1]                                 # 保留已接受子串

        # === 5) 采样修正 / 或继续 ===
        if n < prefix_len + gamma - 1:                             # 出现拒绝
            diff_logit = max_fn(
                target_cache._prob_history[:, n, :] -
                approx_cache._prob_history[:, n, :]
            )
            t = sample(diff_logit).to(device)
            resample_cnt += 1
        else:                                                      # 全部通过
            t = sample(target_cache._prob_history[:, -1, :]).to(device)
            target_cnt += 1

        # === 6) 把新 token 写入前缀并推进 KV ===
        prefix = torch.cat((prefix, t), dim=1)
        _ = approx_cache._forward_with_kvcache(prefix)             # 仅 1 token 前向
        _ = target_cache._forward_with_kvcache(prefix)

    if verbose:
        print(f"gen {prefix.shape[1] - seq_len} tok | "
              f"accept={accepted_cnt}  resample={resample_cnt}  target={target_cnt}")
    return prefix



@torch.no_grad()
def speculative_sampling_v2_qwen(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling, adapted for Qwen models with different vocabulary sizes.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    # 获取模型所在设备并确保一致性
    target_device = next(target_model.parameters()).device
    # 将tensor移到大模型所在的设备
    prefix = prefix.to(target_device)
    
    # 获取Qwen模型的词表大小
    small_vocab_size = approx_model.config.vocab_size if hasattr(approx_model, 'config') and hasattr(approx_model.config, 'vocab_size') else approx_model.lm_head.out_features
    large_vocab_size = target_model.config.vocab_size if hasattr(target_model, 'config') and hasattr(target_model.config, 'vocab_size') else target_model.lm_head.out_features
    
    # 使用较小的词表大小作为目标
    min_vocab_size = min(small_vocab_size, large_vocab_size)

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                
                # 对齐词表大小
                if q.shape[-1] > min_vocab_size:
                    q = q[..., :min_vocab_size]
                    # 重新归一化
                    q = q / q.sum(dim=-1, keepdim=True)
                
                next_tok = sample(norm_logits(q[:, -1, :],
                                  temperature, top_k, top_p))
                # 确保next_tok在正确的设备上
                next_tok = next_tok.to(target_device)
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            
            # 对齐词表大小
            if p.shape[-1] > min_vocab_size:
                p = p[..., :min_vocab_size]
                # 重新归一化
                p = p / p.sum(dim=-1, keepdim=True)
            
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = target_device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=target_device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    # 确保t在正确的设备上
                    t = t.to(target_device)
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
                # 确保t在正确的设备上
                t = t.to(target_device)
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix 