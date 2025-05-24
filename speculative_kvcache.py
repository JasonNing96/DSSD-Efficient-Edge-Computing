import time
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    batch_size, vocab_size = logits.size()
    # Top-K filtering
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).expand_as(logits)
        logits = torch.where(logits < min_values, filter_value, logits)
    # Top-P (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits.view(-1).index_fill_(0, indices_to_remove, filter_value)
    return logits

@torch.no_grad()
def speculative_sampling(
    approx_model, target_model, tokenizer,
    input_ids, max_new_tokens=50, gamma=5,
    temperature=1.0, top_k=0, top_p=0.0,
    device='cuda', verbose=False
):
    """
    Optimized Speculative Sampling:
    - Uses KV caching and batched target_model calls for accepted tokens
    - Benchmarks tokens per second
    - Logs accept/reject counts when verbose
    """
    approx_model.eval(); target_model.eval()
    approx_model.to(device); target_model.to(device)

    # Initialize prefix and caches
    generated = input_ids.to(device)
    total_generated = 0
    iters = 0
    out_t = target_model(input_ids=generated, use_cache=True)
    prefix_past = out_t.past_key_values

    # Start timer
    start_time = time.time()
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    while total_generated < max_new_tokens:
        iters += 1
        # --- Draft with approx_model ---
        out_a_pref = approx_model(input_ids=generated, use_cache=True)
        approx_past = out_a_pref.past_key_values
        token = generated[:, -1:]

        draft_ids = []
        draft_logits = []
        for _ in range(gamma):
            out_a = approx_model(input_ids=token, past_key_values=approx_past, use_cache=True)
            logits_a = out_a.logits[:, -1, :]
            approx_past = out_a.past_key_values
            # Sample from approx distribution q
            scaled = logits_a / temperature
            filtered = top_k_top_p_filtering(scaled, top_k, top_p)
            next_id = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
            draft_ids.append(next_id)
            draft_logits.append(logits_a)
            token = next_id
        draft_ids = torch.cat(draft_ids, dim=1)           # (1, gamma)
        draft_logits = torch.stack(draft_logits, dim=1)    # (1, gamma, vocab)

        # --- Compute target logits for draft in one call ---
        out_t2 = target_model(input_ids=draft_ids, past_key_values=prefix_past, use_cache=True)
        p_logits = out_t2.logits                         # (1, gamma, vocab)
        target_past_full = out_t2.past_key_values        # prefix + gamma tokens

        # --- Accept/Reject decision ---
        seq = draft_ids[0]
        q_vals = draft_logits[0, torch.arange(gamma), seq]
        p_vals = p_logits[0, torch.arange(gamma), seq]
        log_ratio = p_vals - q_vals
        log_rand = torch.log(torch.rand(gamma, device=device))
        accept_mask = log_rand <= log_ratio

        # All accepted
        if accept_mask.all():
            generated = torch.cat([generated, draft_ids], dim=1)
            prefix_past = target_past_full
            total_generated += gamma
            if verbose:
                print(f"[Iter {iters}] prefix_len={generated.size(1)} accepted={gamma} rejected=0")
        else:
            # Find first rejection index
            first_reject = (~accept_mask).nonzero(as_tuple=False)[0].item()
            accepted_count = first_reject
            # Batch call for accepted tokens
            if accepted_count > 0:
                acc_ids = draft_ids[:, :accepted_count]
                out_acc = target_model(input_ids=acc_ids, past_key_values=prefix_past, use_cache=True)
                prefix_past = out_acc.past_key_values
                generated = torch.cat([generated, acc_ids], dim=1)
                total_generated += accepted_count

            # Residual sampling for rejected token
            j = first_reject
            p_prob = F.softmax(p_logits[0, j] / temperature, dim=-1)
            q_prob = F.softmax(draft_logits[0, j] / temperature, dim=-1)
            resid = p_prob - q_prob
            resid.clamp_(min=0.0)
            if resid.sum() <= 0:
                resid = p_prob
            else:
                resid = resid / resid.sum()
            next_id = torch.multinomial(resid, num_samples=1).unsqueeze(0)
            out_rej = target_model(input_ids=next_id, past_key_values=prefix_past, use_cache=True)
            prefix_past = out_rej.past_key_values
            generated = torch.cat([generated, next_id], dim=1)
            total_generated += 1

            if verbose:
                rejected_count = gamma - accepted_count
                print(f"[Iter {iters}] prefix_len={generated.size(1)} accepted={accepted_count} rejected={rejected_count}")

    # End timer
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    tps = max_new_tokens / elapsed
    return generated, tps


@torch.no_grad()
def autoregressive_sampling(model, tokenizer, input_ids, max_new_tokens=50,
                             temperature=1.0, top_k=0, top_p=0.0,
                             device='cuda'):
    model.eval(); model.to(device)
    generated = input_ids.to(device)
    past = None
    total_generated = 0

    start = time.time()
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    while total_generated < max_new_tokens:
        out = model(input_ids=generated[:, -1:], past_key_values=past, use_cache=True)
        logits, past = out.logits[:, -1, :], out.past_key_values
        scaled = logits / temperature
        filtered = top_k_top_p_filtering(scaled, top_k, top_p)
        next_id = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)
        total_generated += 1

    if device.startswith('cuda'):
        torch.cuda.synchronize()
    tps = max_new_tokens / (time.time() - start)
    return generated, tps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Speculative Sampling Demo")
    parser.add_argument("--model_small", type=str, default="./LLM/opt-125m", required=True)
    parser.add_argument("--model_large", type=str, default="./LLM/opt-1.3B", required=True)
    parser.add_argument("--prefix", type=str, default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_large)
    small = AutoModelForCausalLM.from_pretrained(
        args.model_small, torch_dtype=torch.float16, device_map='auto'
    )
    large = AutoModelForCausalLM.from_pretrained(
        args.model_large, torch_dtype=torch.float16, device_map='auto'
    )
    input_ids = tokenizer(args.prefix, return_tensors='pt').input_ids

    seq_sp, tps_sp = speculative_sampling(
        small, large, tokenizer,
        input_ids, max_new_tokens=args.max_new_tokens,
        gamma=args.gamma, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        device='cuda', verbose=args.verbose
    )
    print(f"Speculative Sampling ({tps_sp:.1f} tok/s):")
    print(tokenizer.decode(seq_sp[0], skip_special_tokens=True))

    seq_ar, tps_ar = autoregressive_sampling(
        large, tokenizer, input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        device='cuda'
    )
    print(f"Autoregressive Sampling ({tps_ar:.1f} tok/s):")
    print(tokenizer.decode(seq_ar[0], skip_special_tokens=True))
