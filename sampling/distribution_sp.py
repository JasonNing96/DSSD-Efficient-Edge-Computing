# distribution_sp.py
"""
Distributed (simulated) Speculative Sampling.

Draft  (Node-A):  小模型负责 γ 个猜测
Verify (Node-B):  大模型决定接受/拒绝，并在需要时重采

当前版本＝纯本机仿真：
  • 两个"节点"只是两个 Python 对象，通过 SimpleChannel 传消息；
  • 如需跨进程，把 SimpleChannel 换成 multiprocessing.Queue 即可；
  • 如需跨主机，把 send/recv 换成 Torch-RPC / ZeroMQ / gRPC 均可，
    DraftMsg / VerifyMsg 的字段已经冻结，可直接序列化。
"""
from __future__ import annotations
import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch    
# from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer, AutoModelForCausalLM

# 依赖同目录 / 已安装的原文件
from kvcache_model import KVCacheModel
from utils import sample, norm_logits, max_fn


# ---------- 消息定义 ----------
@dataclass      # 草稿消息
class DraftMsg:
    """Node-A → Node-B"""
    draft_tokens: List[int]               # γ 个猜测 token
    draft_probs: torch.Tensor             # [γ, vocab]，已 softmax
    prefix_len: int                       # Draft 起点 (inclusive)


@dataclass      # 验证消息
class VerifyMsg:
    """Node-B → Node-A"""
    accepted_len: int                     # Draft 中被接受的前缀长度 n∈[0,γ]
    need_resample: bool                   # True ⇒ new_token 由 (p-q)+ 重采
    new_token: int                        # 本轮最终加入序列的 token
    p_of_new: float                       # 大模型对 new_token 的采样概率


# ---------- 通信层（可替换） ----------
class SimpleChannel:    # 队列
    """极简同步『信道』——同进程对象直接调用，不做序列化"""
    def __init__(self):
        self._buf: List = []

    # A → B
    def send(self, obj):
        self._buf.append(obj)

    # B ← A
    def recv(self):
        assert self._buf, "channel empty"
        return self._buf.pop(0)


# ---------- Draft 节点 ----------
class DraftWorker:
    """
    小模型 + KV-Cache
    只暴露两个 API：
        1) generate(gamma)  → DraftMsg
        2) commit(DraftMsg, VerifyMsg)  用 verify 结果更新自身状态
    """
    def __init__(
        self,
        model_small: torch.nn.Module,
        tokenizer: AutoTokenizer,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.model = KVCacheModel(model_small, temperature, top_k, top_p)
        self.device = next(model_small.parameters()).device
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        # 以 <BOS> 开始，保持与 VerifyWorker 同步
        self.prefix = torch.tensor(
            [[tokenizer.bos_token_id]], dtype=torch.long, device=self.device
        )

    # ---- A->B ----
    @torch.no_grad()
    def generate(self, gamma: int) -> DraftMsg:
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []

        for _ in range(gamma):
            # big-prob for next-token under current prefix
            q = self.model._forward_with_kvcache(self.prefix, use_debug=False)  # [1,vocab]
            tok = sample(q)                                                     # [1,1]
            draft_tokens.append(int(tok.item()))
            draft_probs.append(q.squeeze(0).cpu())  # 传 CPU tensor，仿真场景无需压缩
            self.prefix = torch.cat([self.prefix, tok], dim=1)

        probs_tensor = torch.stack(draft_probs, dim=0)  # [γ,vocab]
        msg = DraftMsg(draft_tokens, probs_tensor, prefix_len=self.prefix.shape[1] - gamma)
        return msg

    # ---- B->A ----
    @torch.no_grad()
    def commit(self, dmsg: DraftMsg, vmsg: VerifyMsg) -> None:
        """根据 Verify 结果裁剪 KV 缓存并追加 new_token"""
        # 1) 回滚小模型到 prefix_len + accepted_len
        keep_len = dmsg.prefix_len + vmsg.accepted_len
        self.model.rollback(keep_len)
        self.prefix = self.prefix[:, :keep_len]

        # 2) 把被接受的 draft token 追加到序列
        if vmsg.accepted_len > 0:
            accepted_ids = torch.tensor(
                [dmsg.draft_tokens[: vmsg.accepted_len]],
                dtype=torch.long,
                device=self.device,
            )
            self.prefix = torch.cat([self.prefix, accepted_ids], dim=1)

        # 3) 追加本轮最终 token（可能是重采，也可能是 target-sample）
        new_tok = torch.tensor([[vmsg.new_token]], dtype=torch.long, device=self.device)
        self.prefix = torch.cat([self.prefix, new_tok], dim=1)

        # 4) 将 new_tok 加入 KV-Cache：再 forward 一步即可
        _ = self.model._forward_with_kvcache(self.prefix, use_debug=False)


# ---------- Verify 节点 ----------
class VerifyWorker:
    """大模型 + KV-Cache，持有『最终正确前缀』"""
    def __init__(
        self,
        model_big: torch.nn.Module,
        tokenizer: AutoTokenizer,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.model = KVCacheModel(model_big, temperature, top_k, top_p)
        self.device = next(model_big.parameters()).device
        self.prefix = torch.tensor(
            [[tokenizer.bos_token_id]], dtype=torch.long, device=self.device
        )
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    # ---- 核心：对 DraftMsg 做一次验证，返回 VerifyMsg ----
    @torch.no_grad()
    def verify(self, dmsg: DraftMsg) -> VerifyMsg:
        gamma = len(dmsg.draft_tokens)
        # ---------- 先跑一遍 big-model，拿到 draft 段 + 1 的 logits ----------
        seq = torch.cat(
            [
                self.prefix,
                torch.tensor([dmsg.draft_tokens], dtype=torch.long, device=self.device),
            ],
            dim=1,
        )
        _ = self.model.generate(seq, 1)  # γ 个 draft + 1 extra，Google 原实现

        prefix_len = dmsg.prefix_len
        accepted = 0

        # ---------- 逐 token 计算 p/q ----------
        for i, tok in enumerate(dmsg.draft_tokens):
            p_prob = self.model._prob_history[0, prefix_len + i, tok].item()
            q_prob = dmsg.draft_probs[i, tok].item()
            if torch.rand(1, device=self.device, generator=self.rng).item() > p_prob / q_prob:
                break  # 拒绝
            accepted += 1

        # ---------- 拒绝分支 ----------
        if accepted < gamma:
            need_rs = True
            pos_n = prefix_len + accepted  # 第一个被拒绝的位置
            p_dist = self.model._prob_history[:, pos_n, :].clone()
            q_dist = dmsg.draft_probs[accepted].to(self.device).unsqueeze(0)
            new_token_dist = max_fn(p_dist - q_dist)
            # new_tok = int(sample(new_token_dist.unsqueeze(0)).item())
            new_tok = int(sample(new_token_dist).item())
            p_of_new = p_dist[0, new_tok].item()
            # p_of_new = p_dist[new_tok].item()

            # 回滚到已接受末尾
            keep_len = pos_n  # 不保留被拒绝 token 本身
            self.model.rollback(keep_len)
            self.prefix = self.prefix[:, :keep_len]

        # ---------- 全部接受分支 ----------
        else:
            need_rs = False
            # _prob_history[-1] 对应"extra"那一步的分布
            next_dist = self.model._prob_history[0, -1, :]
            new_tok = int(sample(next_dist.unsqueeze(0)).item())
            p_of_new = next_dist[new_tok].item()

            # Google 实现回滚到 n+2（保留 extra logits，抛弃 extra token id）
            self.model.rollback(prefix_len + gamma + 1)
            self.prefix = self.prefix  # 长度未变（extra token 未写入 prefix）

        # ---------- 更新 prefix（加入已接受 + new_tok） ----------
        # 在 verify 中，同步问题是一个主要问题，要考虑 draft 和 vertify 之间的同步
        if accepted > 0:
            accepted_ids = torch.tensor(
                [dmsg.draft_tokens[:accepted]],
                dtype=torch.long,
                device=self.device,
            )
            self.prefix = torch.cat([self.prefix, accepted_ids], dim=1)

        self.prefix = torch.cat(
            [self.prefix, torch.tensor([[new_tok]], dtype=torch.long, device=self.device)],
            dim=1,
        )
        # 触发一次 forward，把 new_tok 写进 KV-Cache
        _ = self.model._forward_with_kvcache(self.prefix, use_debug=False)

        return VerifyMsg(accepted, need_rs, new_tok, p_of_new)


# ---------- 主循环 ----------
@torch.no_grad()
def distributed_speculative_sampling(
    prompt_ids: torch.Tensor,
    small_model: torch.nn.Module,
    big_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 128,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    return:
        generated_ids  (torch.Tensor)  = prompt + new_tokens
        stats          (dict)
    """
    device = next(big_model.parameters()).device

    # ---- 两端实例 & 虚拟"信道" ----
    channel_A2B = SimpleChannel()
    channel_B2A = SimpleChannel()

    draft_worker  = DraftWorker(small_model, tokenizer, temperature, top_k, top_p, seed=random_seed)
    verify_worker = VerifyWorker(big_model, tokenizer, temperature, top_k, top_p)

    # 让两端都带上 prompt
    draft_worker.prefix = prompt_ids.to(device)
    verify_worker.prefix = prompt_ids.to(device)

    # ---- 统计量 ----
    total_draft_tokens = 0
    total_accepted = 0
    start_time = time.time()

    # ---- 生成循环 ----
    while draft_worker.prefix.shape[1] - prompt_ids.shape[1] < max_new_tokens:
        # A → B
        dmsg = draft_worker.generate(gamma)
        total_draft_tokens += gamma
        channel_A2B.send(dmsg)

        # B
        dmsg_recv: DraftMsg = channel_A2B.recv()
        vmsg = verify_worker.verify(dmsg_recv)
        total_accepted += vmsg.accepted_len
        channel_B2A.send(vmsg)

        # A
        vmsg_recv: VerifyMsg = channel_B2A.recv()
        draft_worker.commit(dmsg, vmsg_recv)

        if verbose:
            print(
                f"[step] accept {vmsg.accepted_len}/{gamma} | "
                f"{'resample' if vmsg.need_resample else 'target'} "
                f"→ {tokenizer.decode([vmsg.new_token]).strip()}"
            )

        # 触发结束符
        if vmsg.new_token == tokenizer.eos_token_id:
            break

    elapsed = time.time() - start_time
    new_tokens = draft_worker.prefix.shape[1] - prompt_ids.shape[1]

    stats = dict(
        accept_rate=total_accepted / total_draft_tokens,
        tokens_generated=new_tokens,
        throughput=new_tokens / elapsed if elapsed > 0 else float("inf"),
    )

    return draft_worker.prefix, stats


# ---------- Benchmark ----------
def benchmark(
    model_small_name: str,
    model_big_name: str,
    prompt: str,
    runs: int = 5,
    gamma: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(model_big_name)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids

    small = AutoModelForCausalLM.from_pretrained(model_small_name).half().cuda()
    big = AutoModelForCausalLM.from_pretrained(model_big_name).half().cuda()

    print(f"Benchmarking distributed-SP:  γ={gamma}, runs={runs}")
    agg = {"accept_rate": 0.0, "tokens_generated": 0, "throughput": 0.0}

    for r in range(runs):
        _, st = distributed_speculative_sampling(
            prompt_ids,
            small,
            big,
            tokenizer,
            max_new_tokens,
            gamma,
            temperature,
            top_k,
            top_p,
        )
        for k in agg:
            agg[k] += st[k]
        print(f"run{r+1}: {st}")

    for k in agg:
        agg[k] /= runs
    print(f"\n===> Average over {runs} runs\n{agg}")


# ---------- CLI ----------
def _cli(): 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    parser = argparse.ArgumentParser(description="Distributed Speculative Sampling (single-box simulation)")
    parser.add_argument("--small", required=True, default= '/home/ningjiahong/LLM/bloomz-560m/AI-ModelScope/bloom-560m/')
    parser.add_argument("--big", required=True, default = '/home/ningjiahong/LLM/AI-ModelScope/bloomz-7b1/')
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--benchmark", action="store_true", default= False, help="run 5-round benchmark")
    parser.add_argument("--verbose", action="store_true", default = False)
    parser.add_argument("--profiling", default= False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.big)
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    small = AutoModelForCausalLM.from_pretrained(args.small).half().cuda()
    big = AutoModelForCausalLM.from_pretrained(args.big).half().cuda()

    if args.benchmark:
        benchmark(
            args.small,
            args.big,
            args.prompt,
            runs=5,
            gamma=args.gamma,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        out, st = distributed_speculative_sampling(
            prompt_ids,
            small,
            big,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=args.verbose,
        )
        print(tokenizer.decode(out[0]))
        print(st)


if __name__ == "__main__":
    _cli()
