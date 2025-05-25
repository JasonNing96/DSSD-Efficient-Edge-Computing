import time
import argparse
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative import autoregressive_sampling, speculative_sampling_with_acceptance_rate

# ------------------------------------------------------------
# Simulation of transmission time based on channel conditions
# ------------------------------------------------------------
def transmission_simulator(token_count: int, rtt: float, bandwidth: float) -> float:
    """
    One-way transmission delay: RTT/2 + serialization delay
    bandwidth: tokens per second
    """
    serialize = token_count / bandwidth
    return rtt / 2 + serialize

# ------------------------------------------------------------
# Sampling helper (reuse original sample signature)
# ------------------------------------------------------------
def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    logits = logits / temperature
    # top-k, top-p filtering if needed
    if top_k > 0 or top_p > 0.0:
        from speculative import top_k_top_p_filter
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ------------------------------------------------------------
# Draft on SLM (small model)
# ------------------------------------------------------------
def draft_step(slm: torch.nn.Module,
               prefix: torch.Tensor,
               gamma: int,
               temperature: float,
               top_k: int,
               top_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run small model for gamma steps, return extended sequence and q_logits for each step
    """
    x = prefix.to(slm.device)
    # Collect q_probs for all proposals
    q_probs = []
    for _ in range(gamma):
        logits = slm(x).logits           # (1, seq, vocab)
        logits_last = logits[:, -1, :]
        idx = sample(logits_last, temperature, top_k, top_p)
        # store probability vector
        q_probs.append(F.softmax(logits_last, dim=-1).detach().cpu())
        x = torch.cat((x, idx), dim=1)
    # Stack to shape (gamma, vocab)
    q_probs = torch.stack(q_probs, dim=0)
    return x, q_probs

# ------------------------------------------------------------
# Verify on LLM (large model) for one speculative batch
# ------------------------------------------------------------
def verify_step(llm: torch.nn.Module,
                x_draft: torch.Tensor,
                q_probs: torch.Tensor,
                gamma: int,
                temperature: float,
                top_k: int,
                top_p: float) -> Tuple[int, torch.Tensor]:
    """
    Given draft sequence and small-model probs, run large model to accept/reject and diff-sample.
    Returns rollback index n and corrected token t.
    """
    x_l = x_draft.to(llm.device)
    p_logits = llm(x_l).logits             # (1, seq+gamma, vocab)
    p_logits = p_logits[:, -gamma-1:-1, :].detach().cpu()  # only positions of proposals
    p_probs = F.softmax(p_logits / temperature, dim=-1)

    # Decide acceptance
    prefix_len = x_draft.shape[1] - gamma
    n = prefix_len - 1
    for i in range(gamma):
        # proposal at position i
        p_i = p_probs[i]                  # (vocab,)
        q_i = q_probs[i]                  # (vocab,)
        # token id drafted
        tok_id = x_draft[0, prefix_len + i].cpu().item()
        ratio = p_i[tok_id] / q_i[tok_id]
        if torch.rand(1).item() > ratio:
            # reject
            n = prefix_len + i - 1
            # diff-sampling at this position
            diff = (p_i - q_i).clamp(min=0)
            diff = diff / diff.sum()
            t = torch.multinomial(diff, num_samples=1)
            return n, t
        # else accept and continue
    # all accepted
    n = prefix_len + gamma - 1
    # sample from p_probs last
    t = torch.multinomial(p_probs[-1], num_samples=1)
    return n, t

# ------------------------------------------------------------
# Distributed speculative decoding only
# ------------------------------------------------------------
def distributed_speculative(slm: torch.nn.Module,
                             llm: torch.nn.Module,
                             input_ids: torch.Tensor,
                             args) -> Dict[str, float]:
    prefix = input_ids.clone()
    total_generated = 0
    total_accept = 0
    t_s_list, t_l_list, t_c_list = [], [], []
    while total_generated < args.max_tokens:
        # Draft
        # t0 = time.time()
        x_draft, q_probs = draft_step(
            slm, prefix, args.gamma,
            args.temperature, args.top_k, args.top_p)
        # t_s = time.time() - t0
        # Uplink
        t_c1 = transmission_simulator(args.gamma, args.rtt, args.bandwidth)
        time.sleep(t_c1)
        # Verify
        t1 = time.time()
        n, t_corr = verify_step(
            llm, x_draft, q_probs, args.gamma,
            args.temperature, args.top_k, args.top_p)
        # t_l = time.time() - t1
        # Downlink
        time.sleep(args.rtt/2)
        # Update
        prefix = torch.cat((x_draft[:, :n+1].to(prefix.device),
                             t_corr.to(prefix.device)), dim=1)
        total_generated += (n + 1 - (prefix.shape[1] - args.gamma))
        total_accept += max(0, n - (prefix.shape[1] - args.gamma - 1))
        # t_s_list.append(t_s)
        # t_l_list.append(t_l)
        t_c_list.append(t_c1)

    # Metrics
    # T_s = sum(t_s_list)/len(t_s_list)
    # T_l = (sum(t_l_list)/len(t_l_list))/args.gamma
    T_c = sum(t_c_list)/len(t_c_list)
    # wall = sum(t_s_list)+sum(t_l_list)+sum(t_c_list)+len(t_c_list)*(args.rtt/2)
    # throughput = total_generated / wall
    # alpha = total_accept / (args.gamma * len(t_s_list))
    # b = T_c / T_s
    # c = T_s / T_l
    return {
        'throughput_dist': throughput,
        'alpha': alpha,
        'b': b,
        'c': c
    }

# ------------------------------------------------------------
# Baseline autoregressive sampling (pure LLM)
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="Alan Turing theorized that computers would one day become ") 
    parser.add_argument('--model_s', type=str, default="./LLM/opt-125m")
    parser.add_argument('--model_l', type=str, default="./LLM/opt-1.3b")
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--rtt', type=float, default=0.02)
    parser.add_argument('--bandwidth', type=float, default=1e6)
    parser.add_argument('--max_tokens', type=int, default=60)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--random_seed', type=int, default=321)
    args = parser.parse_args()

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(args.model_l)
    slm = AutoModelForCausalLM.from_pretrained(args.model_s).eval().to('cuda:0')
    llm = AutoModelForCausalLM.from_pretrained(args.model_l).eval().to('cuda:1')

    # Prepare input
    input_ids = tokenizer(args.input, return_tensors='pt').input_ids.to('cuda:0')

    # # Distributed speculative
    # dist_stats = distributed_speculative(slm, llm, input_ids, args)
    # print("Distributed Speculative:")
    # for k,v in dist_stats.items(): print(f"  {k}: {v:.4f}")

    # Baseline autoregressive
    print("Baseline autoregressive:")
    t0 = time.time()
    _prefix, t_ar, _, tp_ar = autoregressive_sampling(
        input_ids.to('cuda:1'), llm,
        args.max_tokens,
        args.temperature, args.top_k, args.top_p)
    print(f"  throughput_base: {tp_ar:.4f}")

if __name__ == '__main__':
    main()
