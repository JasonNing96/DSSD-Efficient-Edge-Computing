#!/usr/bin/env bash
set -euo pipefail

# 一些固定参数
INPUT="I have 10 apples. I find 3 gold coins in the bottom of a river. The river runs near a big city that has something to do with what I can spend the coins on."
DRAFT_MODEL="./LLM/opt-1.3b"
MAX_LEN=128
SEED=321
TEMPERATURE=1
TOP_K=10
TOP_P=0
CSV="./results/results_13.b_cpu_H800.csv"

# 如果之前有旧结果，就删掉
[ -f $CSV ] && rm $CSV

# 循环参数
TARGET_MODELS=("./LLM/opt-1.3b" "./LLM/opt-6.7b" "./LLM/opt-13b")
GAMMAS=(2 4 6 8)
RTTS=(0.0 0.02 0.05)
BWS=(10000000 50000000 100000000)

for target_model in "${TARGET_MODELS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for rtt in "${RTTS[@]}"; do
            for bw in "${BWS[@]}"; do
            echo "=== Running DSP: γ=$gamma, RTT=${rtt}s, BW=$(($bw/1000000))Mbps ==="
            python main.py \
                --input "$INPUT" \
                --draft_model_name "$DRAFT_MODEL" \
                --target_model_name "$target_model" \
                --max_len $MAX_LEN \
                --gamma $gamma \
                --seed $SEED \
                --temperature $TEMPERATURE \
                --top_k $TOP_K \
                --top_p $TOP_P \
                --bandwidth $bw \
                --rtt $rtt \
                --csv_path $CSV \
                --device_1 "cpu" \
                --device_2 "cuda:2"

            echo "  -> Appended results to $CSV"
            done
        done
    done
done

echo "All experiments done. Final results in $CSV"
