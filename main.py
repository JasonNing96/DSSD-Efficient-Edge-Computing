import argparse
from speculative import speculative_sampling, autoregressive_sampling
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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


def main():
    args = parse_arguments()
    
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)

    small_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    large_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)

    input_ids = tokenizer.encode(args.input, return_tensors='pt')


    torch.manual_seed(args.seed)
    output = speculative_sampling(input_ids, small_model, large_model, args.max_len, gamma = args.gamma)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"speculative_sampling: {generated_text}")


    torch.manual_seed(args.seed)
    output = autoregressive_sampling(input_ids, large_model, args.max_len, top_k = 10, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"autoregressive_sampling: {generated_text}")



if __name__ == "__main__":
    main()