
# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

python main.py \
    --input "Alan Turing theorized that computers would one day become " \
    --draft_model_name "./LLM/opt-125m" \
    --target_model_name "./LLM/opt-1.3b" \
    --max_len 60 \
    --gamma 4 \
    --verbose True


python distribute_sp.py \
    --input "Alan Turing theorized that computers would one day become " \
    --model_s "./LLM/opt-125m" \
    --model_l "./LLM/opt-1.3b" \
    --gamma 4 \
    --rtt 0.02 \
    --bandwidth 1000.0 \
    --max_tokens 60 \
    --temperature 0.7 \
    --top_k 10 \
    --top_p 0.0

