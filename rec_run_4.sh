while true
do
    CUDA_VISIBLE_DEVICES=4 TRANSFORMERS_CACHE=$PWD/.cache python3 run_deepseek.py \
        --exp_name "inference_deepseek-vl-7b-chat_cot_summurize" \
        --model_name "deepseek-ai/deepseek-vl-7b-chat" \
        --dataset_name "cais/mmlu" \
        --device_map "auto"
done