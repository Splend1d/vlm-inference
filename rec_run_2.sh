while true
do
    # CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
    #     --exp_name "inference_Qwen2-VL-7B-Instruct_cot_summarize" \
    #     --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    #     --dataset_name "Splend1dchan/gpqa_diamond_visual_noise" \
    #     --device_map "auto" \
    #     --seed "0"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
    #     --exp_name "inference_QVQ-72B-Preview_cot_summurize" \
    #     --model_name "Qwen/QVQ-72B-Preview" \
    #     --dataset_name "Splend1dchan/gpqa_diamond_visual_noise" \
    #     --device_map "auto" \
    #     --seed "0"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
	    --exp_name "inference_Qwen2-VL-72B-Instruct_cot_summurize" \
		--model_name "Qwen/Qwen2-VL-72B-Instruct" \
		--dataset_name "Splend1dchan/gpqa_diamond_visual_noise" \
		--device_map "auto" \
		--seed "0"
done
