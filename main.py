import argparse
from speculative import autoregressive_sampling, sample
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Dict
import time
import torch.nn.functional as F
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--input', type=str, default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="./LLM/opt-1.3b") 
    parser.add_argument('--max_len', type=int, default=80) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    # parser.add_argument('--benchmark', type=bool, default=False)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--rtt', type=float, default=0.02)
    parser.add_argument('--bandwidth', type=float, default=1000)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)

    return parser.parse_args()

def transmission_simulator(token_count: int, rtt: float, bandwidth: float) -> float:
    """
    One-way transmission delay: RTT/2 + serialization delay
    bandwidth: tokens per second
    """
    serialize = token_count / bandwidth
    return rtt / 2 + serialize

def draft_step(slm: torch.nn.Module,
               prefix: torch.Tensor,
               gamma: int,
               temperature: float,
               top_k: int,
               top_p: float,
               device: torch.device):
    """
    在 device 上用小模型连采 γ 步，返回拼好的序列 x_draft
    以及最后一次小模型前向的完整 logits q_full_logits (1, T+γ, V)
    """
    x = prefix.to(device)
    with torch.no_grad():
        for _ in range(gamma):
            logits = slm(x).logits                   # (1, seq, V)
            next_tok = sample(logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat((x, next_tok), dim=1)
        # “最后一次” logits 已经是对整个 x 的前向结果
        q_full_logits = logits                    # (1, seq+γ, V)
    return x, q_full_logits



def verify_step(llm: torch.nn.Module,
                x_draft: torch.Tensor,
                q_logits: torch.Tensor,   # 来自 draft_step，shape (1, prefix_len+γ-1, V)
                gamma: int,
                temperature: float,
                device: torch.device):
    """
    只验证前 gamma-1 个 proposal，用 q_logits 里已有的时序，
    最后一个 proposal 直接从 p_full 上采样。
    """
    x_l = x_draft.to(device)
    with torch.no_grad():
        p_full = llm(x_l).logits                 # -> (1, prefix+γ, V)

    # 归一化
    p = F.softmax(p_full   / temperature, dim=-1)      # (1, prefix+γ, V)
    q = F.softmax(q_logits / temperature, dim=-1)      # (1, prefix+γ-1, V)

    prefix_len = x_draft.size(1) - gamma
    n = prefix_len - 1

    # 只验证前 gamma-1 步
    for i in range(gamma - 1):
        t = prefix_len + i
        tok_id = int(x_draft[0, t].item())
        p_val = p[0, t, tok_id].detach().cpu()
        q_val = q[0, t, tok_id].detach().cpu()
        if torch.rand(1).item() > (p_val / q_val):
            # 在第 i 步被拒绝
            n = t - 1
            diff = (p[0, t].detach().cpu() - q[0, t].detach().cpu()).clamp(min=0)
            diff /= diff.sum()
            t_corr = torch.multinomial(diff, 1).unsqueeze(0)  # shape (1,1)
            return n, t_corr

    # 前 gamma-1 步都通过了，把最后一步当 accept：
    # 回滚到 prefix_len + (γ-1) - 1 = prefix_len+γ-2
    n = prefix_len + gamma - 2
    # 在真正的最后时刻 t_last = prefix_len+γ-1 上采样
    t_last = prefix_len + gamma - 1
    diff = p[0, t_last]  # 直接用大模型分布
    t_corr = torch.multinomial(diff, 1).unsqueeze(0)    # (1,1)
    return n, t_corr

def normal_generate( small_model, large_model,tokenizer, input_ids, device):
    print("Baseline autoregressive:")
    _prefix, t_ar, _, tp_ar = autoregressive_sampling(
        input_ids.to(device), large_model,
        args.max_len,
        args.temperature, args.top_k, args.top_p)
    
    print('text: ', tokenizer.decode(_prefix[0], skip_special_tokens=True))
    print(f"  throughput_base: {tp_ar:.4f}")
    
    
def generate_with_sp(draft_model: torch.nn.Module,
                     target_model: torch.nn.Module,
                     input_ids: torch.Tensor,
                     tokenizer: AutoTokenizer,
                     device_1: torch.device,
                     device_2: torch.device):
    '''
    分布式 Speculative Decoding 核心逻辑，只做投机采样。
    '''
    args = parse_arguments()

    # 1) 准备
    max_total_len = args.max_len + input_ids.shape[1]
    rtt       = args.rtt
    bandwidth = args.bandwidth

    torch.manual_seed(args.seed)
    # 局部维护 prefix，而不再直接修改 input_ids
    prefix = input_ids.to(device_1)

    # 2) 进度条循环
    with tqdm(total=max_total_len, desc="speculative sampling") as pbar:
        # 初始化进度条为已有的 prefix 长度
        pbar.update(prefix.shape[1])

        initial_len       = prefix.shape[1]
        total_proposals   = 0     # 累计尝试的 proposal 数
        accepted_proposals= 0     # 累计被 accept 的 proposal 数
        total_comm_time   = 0.0   # 累计模拟的通信时间
        dsp_start         = time.time()
        
        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]

            # 2.1) UAV 端：草稿 gamma 步
            x_draft, q_probs = draft_step(
                draft_model, prefix,
                args.gamma, args.temperature,
                args.top_k, args.top_p,
                device=device_1
            )

            # 模拟上行延迟，并累加
            delta = x_draft.shape[1] - prefix.shape[1]
            t_up  = transmission_simulator(delta, rtt, bandwidth)
            total_comm_time += t_up
            time.sleep(t_up)

            # 累计 proposal 数
            total_proposals += delta

            # 2.2) BS 端：验证 + 回滚 + 差分采样
            n, t_corr = verify_step(
                target_model, x_draft, q_probs,
                args.gamma, args.temperature,
                device=device_2
            )

            # 2.3) UAV 端：截断 + 校正 token 拼接
            # 注意：x_draft[:, :n+1] 已包含初始 prefix 和所有 accept 的 proposal
            prefix = torch.cat([
                x_draft[:, : n+1].to(device_1),
                t_corr.to(device_1)
            ], dim=1)

            accepted = (n - (old_len - 1))
            accepted_proposals += accepted
            
            # 2.4) 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            
    dsp_time = time.time() - dsp_start
    total_tokens = prefix.shape[1] - initial_len
    dsp_throughput = total_tokens / dsp_time
    acceptance_rate = accepted_proposals / total_proposals if total_proposals > 0 else 0.0

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== Distributed SP Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput : \033[91m{dsp_throughput}\033[0m", "DSP wall time", dsp_time, "Generated tokens", total_tokens, "Acceptance rate", acceptance_rate )
    print(f"Acceptance rate  : \033[91m{acceptance_rate:.3f}\033[0m  ({accepted_proposals}/{total_proposals})")    
            
        
        
    
    

if __name__ == "__main__":
    args = parse_arguments()
    device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name).to(device_1)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name).to(device_2)
    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    
    # normal_generate()
    generate_with_sp(draft_model, target_model, input_ids, tokenizer, device_1, device_2)
    normal_generate(draft_model, target_model, tokenizer, input_ids, device_2)