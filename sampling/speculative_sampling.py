import torch
from tqdm import tqdm
import torch
from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
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
    approx_device = next(approx_model.parameters()).device
    target_device = next(target_model.parameters()).device
    
    # 将tensor移到大模型所在的设备
    prefix = prefix.to(target_device)
    # 多了一个使用kvcache加载model以提高采样速度
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    device = target_device
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # 每次循环前重置缓存，避免尺寸不匹配问题
        approx_model_cache.reset_cache()
        target_model_cache.reset_cache()
        
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        # prefix 已经token化了，不需要传token，只需要传tensor就可以了
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            # 拒接采样接受，跟deepmind的版本的逻辑是一样的，表述不同
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]).to(device))}\033[0m")

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        # 确保t在正确的设备上
        t = t.to(device)
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"target_accept_rate: {accepted_count / (target_sample_count + resample_count)}")
    return prefix


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
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

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):                              # gamma 是间隔检证长度（生成gamma个token验证一次）
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits                      # 输入x，输出logits （小模型的推测在这一步已经完成）
                next_tok = sample(norm_logits(q[:, -1, :],      # 采样token分数，经过norm_logits后，转变为概率分布-> 采样分布=Tokenid
                                  temperature, top_k, top_p))   # q[:, -1, :] 取的是当前序列最后一个 token 的 logits
                # 确保next_tok在正确的设备上
                next_tok = next_tok.to(target_device)
                x = torch.cat((x, next_tok), dim=1)             # 将当前序列和下一个token拼接起来，总共取完gamma个token   
            
            # normalize the logits
            for i in range(q.shape[1]):                         # 这里将输出q的所有 token都进行归一化，以便后续对比TargerM的概率分布
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
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

