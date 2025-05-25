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
    parser.add_argument('--max_len', type=int, default=60) 
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
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run small model for gamma steps, return extended sequence and q_logits for each step
    """
    x = prefix.to(device)
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
                q_probs: torch.Tensor,     # (γ, 1, V)
                gamma: int,
                temperature: float,
                top_k: int,
                top_p: float,
                device: torch.device) -> Tuple[int, torch.Tensor]:
    """
    Returns:
      n: rollback index (int)
      t: correction token of shape (1,1)
    """
    # 1) 大模型前向 + 取 proposal 部分
    x_l = x_draft.to(device)
    p_full  = llm(x_l).logits                                      # (1, seq+γ, V)
    p_logits= p_full[:, -gamma-1:-1, :].detach().cpu().squeeze(0)  # (γ, V)
    p_probs = F.softmax(p_logits / temperature, dim=-1)           # (γ, V)

    # 2) q_probs 从 (γ,1,V) -> (γ,V)
    q_probs = q_probs.detach().cpu().squeeze(1)                   # (γ, V)

    # 3) accept/reject
    prefix_len = x_draft.shape[1] - gamma
    n = prefix_len - 1
    for i in range(gamma):
        tok_id = x_draft[0, prefix_len + i].cpu().item()
        p_i = p_probs[i]   # (V,)
        q_i = q_probs[i]   # (V,)
        ratio = p_i[tok_id] / q_i[tok_id]
        if torch.rand(1).item() > ratio:
            # 拒绝：回滚 + 差分采样
            n = prefix_len + i - 1
            diff = (p_i - q_i).clamp(min=0)
            diff = diff / diff.sum()
            t = torch.multinomial(diff, num_samples=1)  # -> shape (1,)
            t = t.unsqueeze(1)                          # -> shape (1,1)
            return n, t

    # 4) 全部接受：在最后一个 p_probs 上采样
    n = prefix_len + gamma - 1
    t = torch.multinomial(p_probs[-1], num_samples=1)  # (1,)
    t = t.unsqueeze(1)                                 # (1,1)
    return n, t


def normal_generate():
    args = parse_arguments()
    
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)

    small_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    large_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)

    input_ids = tokenizer.encode(args.input, return_tensors='pt')

    torch.manual_seed(args.seed)
    # output, sp_time, sp_len, sp_acceptance_rate, sp_throughput = speculative_sampling_with_acceptance_rate(input_ids, small_model, large_model, args.max_len, gamma = args.gamma)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # # print(f"speculative_sampling: {generated_text}")
    # print(f"speculative throughput: \033[91m{sp_throughput}\033[0m")


    torch.manual_seed(args.seed)
    output, ag_time, ag_len, ag_throughput = autoregressive_sampling(input_ids, large_model, args.max_len, top_k = 10, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"autoregressive_sampling: {generated_text}")
    print(f"autoregressive throughput: \033[91m{ag_throughput}\033[0m")
    # print(f"speculative throughput / autoregressive throughput: \033[94m{sp_throughput/ag_throughput}\033[0m")
    
    
def generate_with_sp(draft_model : torch.nn.Module, target_model : torch.nn.Module, input_ids : torch.Tensor, tokenizer : AutoTokenizer, device_1 : torch.device, device_2 : torch.device):
    '''
    Args: T, b, c, rtt, bandwidth
    '''
    args = parse_arguments()

    T = args.max_len + len(input_ids[0])
    b = 0
    c = 0
    rtt = args.rtt
    bandwidth = args.bandwidth
    accepted_count = 0
    rejected_count = 0

    torch.manual_seed(args.seed)
    t_draft_start = time.time()
    # time.sleep(transmission_simulator(T, rtt, bandwidth))
    
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while input_ids.shape[1] < T:
            x_draft, q_probs = draft_step(draft_model, input_ids, args.gamma, args.temperature, args.top_k, args.top_p, device=device_1)
            time.sleep(transmission_simulator(T, rtt, bandwidth))
            
            n, t_corr = verify_step(target_model, x_draft, q_probs, args.gamma, args.temperature, args.top_k, args.top_p, device=device_2)
            
            prefix = torch.cat((x_draft[:, :n+1].to(device_1),
                             t_corr.to(device_1)), dim=1)

            # print(f"\033[91m{tokenizer.decode(prefix[0], skip_special_tokens=True)}\033[0m")
            pbar.update(n)
    
    
    text = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print(text)
            
        
        
    
    

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