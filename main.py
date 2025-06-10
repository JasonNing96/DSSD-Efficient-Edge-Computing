import argparse
from speculative import autoregressive_sampling, sample, tensor_nbytes, compress_logits, tx_delay_bytes
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Dict
import time
import torch.nn.functional as F
from tqdm import tqdm
import csv, time, os, json
from collections import OrderedDict

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
    parser.add_argument('--bandwidth', type=float, default=1000, help='Bandwidth in Mbps')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--csv_path', type=str, default="results.csv")
    parser.add_argument('--device_1', type=str, default="cuda:6")
    parser.add_argument('--device_2', type=str, default="cuda:1")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--use_dist_summary', action='store_true', help='upload compressed distribution instead of raw logits')
    parser.add_argument('--no_cache', action='store_true', help='disable Δ-prompt cache (ablation)')
    return parser.parse_args()

class Recorder:
    def __init__(self, csv_path="results.csv"):
        self.rows = []
        self.csv_path = csv_path

    def add_entry(self, **kw):
        # kw: model_s, model_l, gamma, rtt, bw, dsp_thr, base_thr, speedup,
        #     b, c, accept_rate, T_comm, T_slm, T_llm, prompt_len
        row = OrderedDict(kw)          # keep order
        self.rows.append(row)
        # append to disk incrementally
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header: w.writeheader()
            w.writerow(row)

    def summary(self):
        print(json.dumps(self.rows, indent=2))
        


def transmission_simulator(token_count: int, rtt: float, bandwidth: float, bits_per_token: int = 32) -> float:
    """
    One-way transmission delay: RTT/2 + serialization delay
    bandwidth: Mbps (Megabits per second)
    bits_per_token: bits per token (default is 32 bits)
    """
    # 计算总比特数
    total_bits = token_count * bits_per_token
    
    # 将 Mbps 转换为 bps，然后计算序列化延迟
    bandwidth_bps = bandwidth * 1e6  # Mbps → bps
    serialize = total_bits / bandwidth_bps
    return rtt / 2 + serialize

class UAVNode:
    """无人机节点 - 负责draft阶段的小模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        
    def draft_step(self, prefix, gamma):
        """
        UAV端执行draft步骤，生成gamma个候选token
        返回：
          x_draft: (1, prefix_len+γ) 
          q_step_logits_stack: (γ, V)
          dup_bytes: 传输字节数
        """
        x = prefix.to(self.device)
        q_stack = []
        
        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits  # (1, seq, V)
                q_stack.append(logits[0, -1].cpu())  # 存储最后一行 (V,)
                next_tok = sample(logits[:, -1, :],
                                self.args.temperature, 
                                self.args.top_k, 
                                self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)
        
        q_step_logits = torch.stack(q_stack, dim=0)  # -> (γ, V)
        
        # 计算传输字节数
        raw_bytes = tensor_nbytes(q_step_logits)
        if self.args.use_dist_summary:
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_step_logits)
            dup_bytes = comp_bytes
        else:
            dup_bytes = raw_bytes
            
        return x, q_step_logits, dup_bytes

class BSNode:
    """基站节点 - 负责verify阶段的大模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
    
    def verify_step(self, x_draft, q_steps, gamma):
        """
        BS端执行verify步骤，验证UAV生成的候选token
        返回：
          n: 接受的token位置
          t_corr: 校正token
          correct_num: 接受的token数量  
          reject_num: 拒绝的token数量
        """
        correct_num = reject_num = 0
        prefix_len = x_draft.size(1) - gamma
        
        # 1) 大模型一次前向
        p_all = self.model(x_draft.to(self.device)).logits.cpu()  # (1, prefix_len+γ, V)
        p_slice = p_all[0, prefix_len-1 : prefix_len+gamma-1, :]  # 取 γ 行 -> (γ, V)

        # 2) softmax归一化
        p_probs = F.softmax(p_slice / self.args.temperature, dim=-1)  # (γ, V)
        q_probs = F.softmax(q_steps / self.args.temperature, dim=-1)  # (γ, V)

        # 3) accept/reject判断
        for i in range(gamma):
            tok_id = int(x_draft[0, prefix_len+i].item())
            if torch.rand(1).item() > (p_probs[i, tok_id] / q_probs[i, tok_id]):
                # 首次拒绝 → 回滚到 prefix_len+i-1
                n = prefix_len + i - 1
                diff = (p_probs[i] - q_probs[i]).clamp(min=0)
                diff = diff / diff.sum()
                t_corr = torch.multinomial(diff, 1).unsqueeze(0)  # (1,1)
                reject_num += 1
                return n, t_corr, correct_num, reject_num
            else:
                correct_num += 1
        
        # 全部通过：n = 最后位置
        n = prefix_len + gamma - 1
        prob_last = F.softmax(p_all[0, n] / self.args.temperature, dim=-1)
        t_corr = torch.multinomial(prob_last, 1).unsqueeze(0)  # shape (1,1)
        return n, t_corr, correct_num, reject_num

def normal_generate(large_model, tokenizer, input_ids, device, args):
    print("Baseline autoregressive:")
    input_ids = input_ids.to(device)
    _prefix, t_ar, _, tp_ar = autoregressive_sampling(
        input_ids, large_model.to(device),
        args.max_len,
        args.temperature, args.top_k, args.top_p)
    
    print('text: ', tokenizer.decode(_prefix[0], skip_special_tokens=True))
    print(f"  throughput_base: {tp_ar:.4f}")
    return _prefix, t_ar, tp_ar

def generate_with_distributed_sp(uav_node, bs_node, input_ids, tokenizer, args):
    """
    分布式投机采样主循环，使用UAV和BS节点协作
    """
    input_ids = input_ids.to(uav_node.device)
    
    total_comm = total_slm = total_llm = 0.0
    rounds = correct_nums = reject_nums = 0
    total_dup_bytes = 0
    
    # 准备参数
    max_total_len = args.max_len + input_ids.shape[1]
    rtt = args.rtt
    bandwidth = args.bandwidth

    torch.manual_seed(args.seed)
    prefix = input_ids

    # 主循环
    with tqdm(total=max_total_len, desc="distributed speculative sampling") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        dsp_start = time.time()
        
        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1
            
            # UAV端：执行draft步骤
            t0 = time.time()
            x_draft, q_probs, dup_bytes = uav_node.draft_step(prefix, args.gamma)
            total_dup_bytes += dup_bytes
            
            # 模拟上行传输延迟
            bw_Bps = args.bandwidth * 1e6 / 8  # Mbps → B/s
            t_up = tx_delay_bytes(dup_bytes, rtt, bw_Bps)
            
            total_slm += time.time() - t0
            total_comm += t_up
            time.sleep(t_up)

            # BS端：执行verify步骤
            t1 = time.time()
            n, t_corr, correct_num, reject_num = bs_node.verify_step(
                x_draft, q_probs, args.gamma)
            correct_nums += correct_num
            reject_nums += reject_num
            total_llm += time.time() - t1
            
            # UAV端：接收结果并更新prefix
            prefix = torch.cat([
                x_draft[:, : n+1],
                t_corr.to(uav_node.device)
            ], dim=1)

            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            
    dsp_time = time.time() - dsp_start
    total_tokens = prefix.shape[1] - initial_len
    dsp_throughput = total_tokens / dsp_time
    acceptance_rate = correct_nums / (rounds * args.gamma)

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== Distributed SP Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{dsp_throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    return generated, dsp_throughput, dsp_time, acceptance_rate, total_comm, total_dup_bytes

args = parse_arguments()
recorder = Recorder(args.csv_path)

if __name__ == "__main__":
    args = parse_arguments()
    torch.cuda.empty_cache()  # 清理未使用的显存
    device_1 = torch.device(args.device_1) 
    device_2 = torch.device(args.device_2)
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)
    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    
    # 创建UAV和BS节点
    uav_node = UAVNode(draft_model, device_1, args)
    bs_node = BSNode(target_model, device_2, args)
    
    # 执行分布式投机采样
    generated, dsp_throughput, dsp_time, acceptance_rate, total_comm, total_dup_bytes = \
        generate_with_distributed_sp(uav_node, bs_node, input_ids, tokenizer, args)
    
    torch.cuda.empty_cache() 
    _prefix, t_ar_llm, tp_ar_llm = normal_generate(target_model, tokenizer, input_ids, device_2, args)
    
    torch.cuda.empty_cache() 
    _prefix, t_ar_slm, tp_ar_slm = normal_generate(draft_model, tokenizer, input_ids, device_1, args)
    
    # print(f"使用压缩: {args.use_dist_summary}, 原始大小: {raw_bytes}, 传输大小: {dup_bytes}")
    recorder.add_entry(
        model_s = args.draft_model_name.rstrip('/').split('/')[-1],
        model_l = args.target_model_name.rstrip('/').split('/')[-1],
        speedup = round(dsp_throughput/tp_ar_llm, 2),
        b       = round(total_comm/max(t_ar_slm, 1e-4), 1),
        c       = round(t_ar_slm/t_ar_llm, 1),
        accept_rate = round(acceptance_rate, 3),
        dsp_thr = round(dsp_throughput, 2),
        base_thr= round(tp_ar_llm, 2),
        slm_thr = round(tp_ar_slm, 2),
        gamma   = args.gamma,
        rtt_ms  = args.rtt*1e3,
        bw_Mbps = args.bandwidth,
        prompt_len = args.max_len,
        t_ar_slm = round(t_ar_slm, 2),
        t_ar_llm = round(t_ar_llm, 2),
        dup_B    = round(total_dup_bytes / 1024, 1),   # KB
    )