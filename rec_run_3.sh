while true
do
    TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
        --exp_name "inference_Qwen2-VL-7B-Instruct_cot_summarize" \
        --model_name "Qwen/Qwen2-VL-7B-Instruct" \
        --dataset_name "Idavidrein/gpqa" \
        --device_map "auto" \
        --seed "0"
done