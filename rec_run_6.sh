while true
do
    TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
        --exp_name "inference_Qwen2-1.5B-Instruct_cot_summurize" \
        --model_name "Qwen/Qwen2-1.5B-Instruct" \
        --dataset_name "cais/mmlu" \
        --device_map "cuda:6"
done