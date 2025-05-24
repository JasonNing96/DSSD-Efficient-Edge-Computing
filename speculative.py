from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--input', type=str, default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="./LLM/opt-1.3b") 
    parser.add_argument('--max_len', type=int, default=100) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    # parser.add_argument('--benchmark', type=bool, default=False)
    parser.add_argument('--gamma', type=int, default=4)
    
    args = parser.parse_args()
    return args

def top_k_top_p_filter(logits, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def sample(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """_summary_
    Args:
        logits (torch.Tensor): shape (batch, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p
    Returns:
        torch.Tensor: next token with shape as (batch, 1)
    """
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    idx_next = torch.multinomial(probs, num_samples=1)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next


def autoregressive_sampling(prefix : torch.Tensor, model : torch.nn.Module, max_len : int, temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(prefix)
    T = len(prefix) + max_len
    t1 = time.time()
    with tqdm(total=max_len, desc="autoregressive sampling") as pbar:
        while n < T:
            logits = model(prefix).logits[::, -1, :]
            idx_next = sample(logits, temperature, top_k, top_p)
            prefix = torch.cat((prefix, idx_next), dim=1)
            n += 1
            pbar.update(1)
    t2 = time.time()
    print(f"autoregressive throughput: {T / (t2 - t1)} tokens/s", 'time_cost: ', t2 - t1, 'generated_length: ', len(T))
    return prefix, (t2 - t1), len(T)

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

def norm_logits(p : torch.Tensor):
    """
        normalize logits using softmax to probabilities along the last dimension.
    """
    return F.softmax(p, dim=-1)

def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, max_len : int , gamma : int = 4, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        N (int): the overall max generated tokens number.
        K (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    t1 = time.time()
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(q[:, -1, :], 
                                  temperature, top_k, top_p)
                x = torch.cat((x, next_tok), dim=1)

            q = norm_logits(q)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = norm_logits(target_model(x).logits)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            n = prefix_len + gamma - 1
            for i in range(gamma):
                r = torch.rand(1)
                j = x[:, prefix_len + i]
                # print(f"sum on {prefix_len + i - 1}: {torch.sum(p[:, prefix_len + i - 1, :])}, {torch.sum(q[:, prefix_len + i - 1, :])}")

                if r > (p[:, prefix_len + i - 1, j]) / (q[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break


            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]

            if n < prefix_len + gamma - 1:
                # reject someone, sample from the pos n
                # print(f"sum on {n}: {torch.sum(p[:, n, :])}, {torch.sum(q[:, n, :])}")
                t = sample(max_fn(p[:, n, :] - q[:, n, :]), 
                           temperature, top_k, top_p)
                # print(f"reject and sample {n}")
            else:
                # all draft model decoding accepted
                assert n == p.shape[1] - 1
                t = sample(p[:, -1, :], 
                           temperature, top_k, top_p)

            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)
    t2 = time.time()
    print(f"speculative sampling throughput: {max_len / (t2 - t1)} tokens/s")
    return prefix, t2 - t1


def speculative_sampling_with_acceptance_rate(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, max_len : int , gamma : int = 4, temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        N (int): the overall max generated tokens number.
        K (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    accepted_count = 0
    rejected_count = 0
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    t1 = time.time()
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits                      # 小模型单次生成token的概率分布
                next_tok = sample(q[:, -1, :], 
                                  temperature, top_k, top_p)    # 小模型单次生成token
                x = torch.cat((x, next_tok), dim=1)              # 将小模型生成的token加入到prefix中

            q = norm_logits(q)                                  # 小模型单次生成token的概率分布归一化
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = norm_logits(target_model(x).logits)             # 大模型单次生成token的概率分布(这部分没有使用cache)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            n = prefix_len + gamma - 1                          # 大模型生成token的结束位置
            for i in range(gamma):                              # 开始投机采样拒绝
                r = torch.rand(1)                                # 生成一个随机数
                j = x[:, prefix_len + i]                         # 获取大模型生成token的位置
                # print(f"sum on {prefix_len + i - 1}: {torch.sum(p[:, prefix_len + i - 1, :])}, {torch.sum(q[:, prefix_len + i - 1, :])}")

                if r > (p[:, prefix_len + i - 1, j]) / (q[:, prefix_len + i - 1, j]): # 如果随机数大于大模型生成token的概率分布除以小模型生成token的概率分布
                    n = prefix_len + i - 1                          # 更新大模型生成token的结束位置
                    rejected_count += 1                             # 拒绝计数
                    if verbose:
                        print(f"\033[91m{j.item()}\033[0m", end=' ')
                    break
                else:
                    accepted_count += 1
                    if verbose:
                        print(f"\033[97m{j.item()}\033[0m", end=' ')
                        
            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]

            if n < prefix_len + gamma - 1:                       # 回滚逻辑
                # reject someone, sample from the pos n
                # print(f"sum on {n}: {torch.sum(p[:, n, :])}, {torch.sum(q[:, n, :])}")
                t = sample(max_fn(p[:, n, :] - q[:, n, :]), temperature, top_k, top_p)
                # print(f"reject and sample {n}")
            else:
                # all draft model decoding accepted
                assert n == p.shape[1] - 1
                t = sample(p[:, -1, :], temperature, top_k, top_p)

            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)
    t2 = time.time()
    total_samples = accepted_count + rejected_count
    acceptance_rate = accepted_count / total_samples
    print('Acceptance rate: ', acceptance_rate, 'Accept: ', accepted_count, 'Reject: ', rejected_count)
    print(f"speculative sampling throughput: {T / (t2 - t1)} tokens/s", 'time_cost: ', t2 - t1, 'generated_length: ', len(T))
    return prefix, (t2 - t1), len(T), acceptance_rate

def generate(input_text, draft_model_name, target_model_name, max_len=20, verbose=False, seed=123, benchmark=False, gamma=4):

    # NOTE() draft_model_name and target_model_name should use the same tokenizer!
    tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

    small_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')


    torch.manual_seed(seed)
    output = speculative_sampling(input_ids, small_model, large_model, max_len, gamma = gamma)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"speculative_sampling: {generated_text}")


    torch.manual_seed(seed)
    output = autoregressive_sampling(input_ids, large_model, max_len, top_k = 10, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"autoregressive_sampling: {generated_text}")

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.draft_model_name, args.target_model_name, args.max_len, args.verbose, args.seed, args.benchmark, args.gamma)