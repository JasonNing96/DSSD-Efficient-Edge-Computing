import torch
from typing import Optional

from sampling.utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break
    
class KVCacheModel:
    """
    Optimized KV-cache wrapper for causal LM models.
    - Batch normalization of logits (vectorized norm_logits).
    - Fast multi-token sampling via HF generate(use_cache=True) with score capture.
    - Streamlined rollback trimming with list comprehensions.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._device = next(model.parameters()).device
        self._past_key_values = None
        self._prob_history = None

    @torch.no_grad()
    def _forward_with_kvcache(
        self,
        input_ids: torch.LongTensor,
        use_debug: bool = False
    ) -> torch.Tensor:
        """
        Single-step forward with KV-cache and batch norm.
        Populates self._prob_history and self._past_key_values for the prefix or new tokens.
        """
        input_ids = input_ids.to(self._device)
        # PREFILL or NEXT TOKEN
        if self._past_key_values is None:
            outputs = self._model(input_ids, use_cache=True)
            logits = outputs.logits  # (B, L, V)
            B, L, V = logits.size()
            flat = logits.view(-1, V)  # (B*L, V)
            flat = norm_logits(flat, self._temperature, self._top_k, self._top_p)
            self._prob_history = flat.view(B, L, V)
            self._past_key_values = outputs.past_key_values
        else:
            new_ids = input_ids[:, -1:].to(self._device)
            outputs = self._model(
                new_ids,
                past_key_values=self._past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]  # (B, V)
            probs = norm_logits(logits, self._temperature, self._top_k, self._top_p)
            self._prob_history = torch.cat([self._prob_history, probs.unsqueeze(1)], dim=1)
            self._past_key_values = outputs.past_key_values
            if use_debug:
                print("[KVCache] Debug Prob History Shape:", self._prob_history.shape)
            return probs
        if use_debug:
            print("[KVCache] Debug Prefill Prob History Shape:", self._prob_history.shape)
        return self._prob_history[:, -1, :]

    @torch.no_grad()
    def _generate_with_kvcache(
        self,
        prefix: torch.LongTensor,
        gamma: int,
        **generate_kwargs
    ) -> torch.LongTensor:
        """
        Fast multi-token sampling with score capture.
        Supports gamma == 0 for prefill-only use.
        """
        prefix = prefix.to(self._device)
        # handle prefill only (no new tokens)
        if gamma == 0:
            # reset and prefill prefix
            self.reset_cache()
            _ = self._forward_with_kvcache(prefix)
            return prefix
        # reset and prefill prefix
        self.reset_cache()
        _ = self._forward_with_kvcache(prefix)
        # generate with scores
        outputs = self._model.generate(
            prefix,
            max_new_tokens=gamma,
            do_sample=True,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
            **generate_kwargs
        )
        sequences = outputs.sequences  # (B, L+gamma)
        new_scores = outputs.scores     # list of gamma (B, V)
        new_probs = []
        for score in new_scores:
            flat = norm_logits(score, self._temperature, self._top_k, self._top_p)
            new_probs.append(flat.unsqueeze(1))
        new_probs_tensor = torch.cat(new_probs, dim=1)  # (B, gamma, V)
        self._prob_history = torch.cat([self._prob_history, new_probs_tensor], dim=1)
        if hasattr(outputs, 'past_key_values'):
            self._past_key_values = outputs.past_key_values
        return sequences

    def generate(self, input_ids: torch.LongTensor, gamma: int) -> torch.LongTensor:
        """
        Interface-compatible generate(): supports gamma==0 for scoring only, else full generate.
        """
        return self._generate_with_kvcache(input_ids, gamma)

    @torch.no_grad()
    def rollback(
        self,
        end_pos: int
    ) -> None:
        """
        Trim KV-cache and prob history back to sequence length `end_pos`.
        """
        if self._past_key_values is None:
            return
        past_trimmed = []
        for k, v in self._past_key_values:
            if isinstance(self._model, BloomForCausalLM):
                k_trim = k[:, :, :end_pos, :]
                v_trim = v[:, :end_pos, :]
            else:
                k_trim = k[..., :end_pos, :]
                v_trim = v[..., :end_pos, :]
            past_trimmed.append((k_trim, v_trim))
        self._past_key_values = past_trimmed
        if self._prob_history is not None and end_pos <= self._prob_history.shape[1]:
            self._prob_history = self._prob_history[:, :end_pos, :]

    def reset_cache(self) -> None:
        """Completely reset KV-cache and probability history."""
        self._past_key_values = None
        self._prob_history = None
