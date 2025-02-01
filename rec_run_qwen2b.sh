while true
do
    TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
        --exp_name "inference_qwenvl2b_cot_summurize" \
        --model_name "Qwen/Qwen2-VL-2B-Instruct" \
        --dataset_name "cais/mmlu" \
        --device_map "cuda:0"
done