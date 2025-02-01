while true
do
    CUDA_VISIBLE_DEVICES=5 TRANSFORMERS_CACHE=$PWD/.cache python3 run_deepseek.py \
        --exp_name "inference_deepseek-llm-7b-chat_cot_summurize" \
        --model_name "deepseek-ai/deepseek-llm-7b-chat" \
        --dataset_name "cais/mmlu" \
        --device_map "auto"
done