from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
import time

# ==== DSSD patch BEGIN ====
def decompress_logits(compressed_data: bytes, vocab_size: int, k: int = 8) -> torch.Tensor:
    """
    将压缩的数据解压为稀疏的 logits
    """
    # 1. 解析压缩数据
    data = torch.frombuffer(compressed_data, dtype=torch.uint8)
    
    # 2. 分离索引和概率
    ids_bytes = data[:k*4]    # int32 索引，4字节 * k
    prob_bytes = data[k*4:]   # float16 概率，2字节 * k
    
    ids = ids_bytes.view(torch.int32).long()
    probs = prob_bytes.view(torch.float16)
    
    # 3. 重构稀疏 logits
    sparse_logits = torch.full((vocab_size,), float('-inf'))
    sparse_logits[ids] = torch.log(probs)  # 转回 log 概率
    
    return sparse_logits
def tx_delay_bytes(size_B: int, rtt: float, bw_Bps: float) -> float:
    """
    一次上传耗时 = RTT/2 + size / 带宽
    bw_Bps : Byte / second
    """
    return rtt / 2 + size_B / bw_Bps

def compress_logits(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    logits: (V,) 原始 logits → 返回压缩后的张量
    """
    top = torch.topk(logits, k=k)
    ids  = top.indices.to(torch.int32)        # int32 × k (4字节)
    prob = torch.softmax(top.values, dim=-1).to(torch.float16)  # float16 × k (2字节)
    
    # 转换为字节格式的张量并拼接
    ids_bytes = ids.view(torch.uint8)
    prob_bytes = prob.view(torch.uint8)
    
    # 返回压缩后的张量，而不是 bytes
    return torch.cat([ids_bytes, prob_bytes])

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()
# ==== DSSD patch END ====

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
    print(f"autoregressive throughput: {T / (t2 - t1)} tokens/s", 'time_cost: ', t2 - t1, 'generated_length: ', T)
    return prefix, (t2 - t1), T, T / (t2 - t1)

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

def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, max_len : int , gamma : int = 4, temperature : float = 1, top_k : int = 0, top_p : float = 0, device : str = 'cuda:0') -> torch.Tensor:
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
    # ===== 新增：统计 accept/reject 次数 =====
    accepted_count = 0
    rejected_count = 0
    t1 = time.time()
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):                       # 小模型生成gamma个token
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits               # 小模型单次生成token的概率分布
                next_tok = sample(q[:, -1, :], 
                                  temperature, top_k, top_p)
                x = torch.cat((x, next_tok), dim=1)      # 将小模型生成的token加入到prefix中(n,n+1,n+2,...,n+gamma-1)

            q = norm_logits(q)                          # 小模型单次生成token的概率分布归一化
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = norm_logits(target_model(x).logits)      # 大模型单次生成token的概率分布(n,n+gama)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            n = prefix_len + gamma - 1                   # 大模型生成token的结束位置
            for i in range(gamma):                       # 开始确认逻辑
                r = torch.rand(1).to(device)                        # 生成一个随机数
                j = x[:, prefix_len + i]                 # 获取大模型生成token的位置
                # print(f"sum on {prefix_len + i - 1}: {torch.sum(p[:, prefix_len + i - 1, :])}, {torch.sum(q[:, prefix_len + i - 1, :])}")
                if r > (p[:, prefix_len + i - 1, j]) / (q[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    rejected_count += 1
                    break
                else:
                    accepted_count += 1


            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]

            if n < prefix_len + gamma - 1:               # 回滚逻辑(有拒绝,被回滚的n)
                # reject someone, sample from the pos n
                # print(f"sum on {n}: {torch.sum(p[:, n, :])}, {torch.sum(q[:, n, :])}")
                t = sample(max_fn(p[:, n, :] - q[:, n, :]),     # 差分q(0:n+gama) -> q(n,n+gama)
                           temperature, top_k, top_p)           # X1, logist,q1(0,n+q1) 
                # print(f"reject and sample {n}")
            else:                                        # 所有小模型生成的token都被接受
                # all draft model decoding accepted
                assert n == p.shape[1] - 1
                t = sample(p[:, -1, :], 
                           temperature, top_k, top_p)

            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)
    t2 = time.time()
    print(f"speculative sampling throughput: {max_len / (t2 - t1)} tokens/s")
    # ===== 新增：打印 acceptance rate =====
    total_proposals = accepted_count + rejected_count
    accept_rate = accepted_count / total_proposals if total_proposals > 0 else 0.0
    print(f"Acceptance rate: {accept_rate:.3f} "
          f"({accepted_count}/{total_proposals})")
    print(f"speculative sampling throughput: {max_len / (t2 - t1):.2f} tokens/s")
    return prefix, t2 - t1


def speculative_sampling_with_acceptance_rate(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, max_len : int , gamma : int = 4, temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, device : str = 'cuda:0') -> torch.Tensor:
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
            x = prefix.to(device)
            prefix_len = prefix.shape[1]        
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = (approx_model(x).logits)                  # 小模型单次生成token的概率分布
                next_tok = sample(q[:, -1, :], 
                                  temperature, top_k, top_p).to(device)    # 小模型单次生成token
                x = torch.cat((x, next_tok), dim=1).to(device)              # 将小模型生成的token加入到prefix中

            q = norm_logits(q).to(device)                             # 小模型单次生成token的概率分布归一化
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = norm_logits(target_model(x).logits).to(device)             # 大模型单次生成token的概率分布(这部分没有使用cache)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            n = prefix_len + gamma - 1                          # 大模型生成token的结束位置
            for i in range(gamma):                              # 开始投机采样拒绝
                r = torch.rand(1).to(device)                                # 生成一个随机数
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
            prefix = x[:, :n + 1]               # 更新prefix,回滚

            if n < prefix_len + gamma - 1:                       # 回滚逻辑
                # reject someone, sample from the pos n
                # print(f"sum on {n}: {torch.sum(p[:, n, :])}, {torch.sum(q[:, n, :])}")
                t = sample(max_fn(p[:, n, :] - q[:, n, :]), temperature, top_k, top_p)              
                # print(f"reject and sample {n}")， 直接采样概率更大的那个p
            else:
                # all draft model decoding accepted
                assert n == p.shape[1] - 1
                t = sample(p[:, -1, :], temperature, top_k, top_p).to(device)

            prefix = torch.cat((prefix, t), dim=1).to(device)  
            pbar.update(n - pbar.n)
    t2 = time.time()
    total_samples = accepted_count + rejected_count
    acceptance_rate = accepted_count / total_samples
    print('Acceptance rate: ', acceptance_rate, 'Accept: ', accepted_count, 'Reject: ', rejected_count)
    print(f"speculative sampling throughput: {T / (t2 - t1)} tokens/s", 'time_cost: ', t2 - t1, 'generated_length: ', T)
    return prefix, (t2 - t1), T, acceptance_rate, T / (t2 - t1)

def generate(input_text, draft_model_name, target_model_name, max_len=20, verbose=False, seed=123, benchmark=False, gamma=4, device='cuda:0'):
    # NOTE() draft_model_name and target_model_name should use the same tokenizer!
    tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

    small_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(device)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)


    torch.manual_seed(seed)
    sp_text, _  = speculative_sampling(input_ids, small_model, large_model, max_len, gamma = gamma, device=device)
    generated_text = tokenizer.decode(sp_text[0], skip_special_tokens=True)
    # print(f"speculative_sampling: {generated_text}")
    
    # torch.manual_seed(seed)
    sp_text, sp_time, sp_len, sp_acceptance_rate, sp_throughput = speculative_sampling_with_acceptance_rate(input_ids, small_model, large_model, max_len, gamma = gamma, device=device)
    generated_text = tokenizer.decode(sp_text[0], skip_special_tokens=True)
    # print(f"speculative_sampling: {generated_text}")
    print(f"speculative throughput: \033[91m{sp_throughput}\033[0m")


    # torch.manual_seed(seed)
    # ag_text, ag_time, ag_len, ag_throughput = autoregressive_sampling(input_ids, large_model, max_len, top_k = 10, temperature=0.7, device=device)
    # generated_text = tokenizer.decode(ag_text[0], skip_special_tokens=True)
    # # print(f"autoregressive_sampling: {generated_text}")
    # print(f"autoregressive throughput: \033[91m{ag_throughput}\033[0m")

    # # torch.manual_seed(seed)
    # agsm_text, agsm_time, agsm_len, agsm_throughput = autoregressive_sampling(input_ids, small_model, max_len, top_k = 10, temperature=0.7, device=device)
    # generated_text = tokenizer.decode(agsm_text[0], skip_special_tokens=True)
    # # print(f"speculative_sampling: {generated_text}")
    # print(f"speculative throughput: \033[91m{agsm_throughput}\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--input', type=str, default="I have 10 apples. I find 3 gold coins in the bottom of a river. The river runs near a big city that has something to do with what I can spend the coins on. ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/llama160m/")
    parser.add_argument('--target_model_name', type=str, default="./LLM/Llama-2-7b/") 
    parser.add_argument('--max_len', type=int, default=128) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--benchmark', type=bool, default=False)
    parser.add_argument('--gamma', type=int, default=8)
    args = parser.parse_args()
    
    generate(args.input, args.draft_model_name, args.target_model_name, args.max_len, args.verbose, args.seed, args.benchmark, args.gamma, device='cuda:3')