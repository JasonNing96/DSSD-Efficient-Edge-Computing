import torch
import argparse
import contexttimer
from colorama import Fore, Style
# from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer, AutoModelForCausalLM
import os

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from sampling.speculative_sampling_qwen import speculative_sampling_qwen, speculative_sampling_v2_qwen
from globals import Decoder

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default='/home/ningjiahong/LLM/AI-ModelScope/bloomz-560m/')
    parser.add_argument('--target_model_name', type=str, default='/home/ningjiahong/LLM/AI-ModelScope/bloomz-7b1/')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=42, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


# my local models
# 这里可以直接不用，自己加载Local model的位置
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama2-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama2-70b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=42, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 5
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")
    
    # print(f"accept_rate: {output[1]}, target_accept_rate: {output[2]}")
def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device_1 = 'cuda:0'
    torch_device_2 = 'cuda:1'
    # 加载approx_model:小模型， Target_model:大模型，不过分词表应该是同一个
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    print("finish loading models")
    
    # 检查是否为Qwen模型
    is_qwen_model = False
    if hasattr(small_model, 'config') and hasattr(small_model.config, 'model_type') and 'qwen' in small_model.config.model_type.lower():
        is_qwen_model = True
    elif hasattr(large_model, 'config') and hasattr(large_model.config, 'model_type') and 'qwen' in large_model.config.model_type.lower():
        is_qwen_model = True
    
    # 检查词表大小是否不同
    small_vocab_size = small_model.config.vocab_size if hasattr(small_model, 'config') and hasattr(small_model.config, 'vocab_size') else small_model.lm_head.out_features
    large_vocab_size = large_model.config.vocab_size if hasattr(large_model, 'config') and hasattr(large_model.config, 'vocab_size') else large_model.lm_head.out_features
    
    if small_vocab_size != large_vocab_size and verbose:
        print(f"检测到词表大小不同: 小模型={small_vocab_size}, 大模型={large_vocab_size}")
        if not is_qwen_model:
            print("注意: 模型不是Qwen系列，但词表大小不同。建议使用相同词表大小的模型。")
    
    # 不要使用固定的torch_device，因为模型使用了device_map="auto"
    # encode 输入文本，返回token id
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # 确定第一个模型参数所在的设备，并将输入张量移到该设备上
    first_param_device = next(small_model.parameters()).device
    input_ids = input_ids.to(first_param_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(random_seed)
    
    # 自回归采样, 大模型
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling, input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(random_seed)
    # 自回归采样, 小模型
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(random_seed)
    
    # 投机采样，deepmind的版本
    # if is_qwen_model and small_vocab_size != large_vocab_size:
    #     output = speculative_sampling_v2_qwen(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    # else:
    #     output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(random_seed)
    
    # 根据模型类型和词表大小选择合适的投机采样方法
    if is_qwen_model and small_vocab_size != large_vocab_size:
        output = speculative_sampling_qwen(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"Qwen适配版 google's speculative_sampling: {generated_text}")
        
        if use_benchmark:
            benchmark(speculative_sampling_qwen, "SP_Qwen", use_profiling, input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    else:
        output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"google's speculative_sampling: {generated_text}")
        
        if use_benchmark:
            benchmark(speculative_sampling, "SP", use_profiling,
                    input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args = parse_arguments()
    
    # gamma 是沟通次数，主要修改的就是这个参数； 
    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
