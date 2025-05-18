CUDA_VISIBLE_DEVICES=4 python main.py \
    --input "How to make a good cup of coffee?" \
    --target_model_name /home/ningjiahong/LLM/AI-ModelScope/bloomz-7b1/ \
    --approx_model_name /home/ningjiahong/LLM/bloomz-560m/AI-ModelScope/bloom-560m/ \
    --max_tokens 128 \
    --gamma 7 \
    --verbose \
    --seed 42

CUDA_VISIBLE_DEVICES=2 python main.py \
    --input "I have 10 apples. I find 3 gold coins in the bottom of a river. The river runs near a big city that has something to do with what I can spend the coins on. I then lose 4 apples but gain a gold coin. Three birds run into my path and drop 6 apples each. I play an online game and win 6 gold coins but I have to share them equally with my 2 teammates. I buy apples for all the coins I have. The price of an apple is 0.5 coins. How many apples do I have? And where is the river? Use step-by-step reasoning to solve this problem." \
    --target_model_name /home/ningjiahong/LLM/Qwen2.5-14B-Instruct/ \
    --approx_model_name /home/ningjiahong/LLM/Qwen2.5-0.5B-Instruct/ \
    --max_tokens 240 \
    --gamma 8 \
    --seed 42