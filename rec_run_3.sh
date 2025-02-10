while true
do
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
    #     --exp_name "inference_Qwen2-72B-Instruct_cot_summurize" \
    #     --model_name "Qwen/Qwen2-72B-Instruct" \
    #     --dataset_name "cais/mmlu" \
    #     --device_map "auto" \
    #     --seed "0"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
	#     --exp_name "inference_Qwen2.5-VL-72B-Instruct_cot_summurize" \
	# 	--model_name "Qwen/Qwen2.5-VL-72B-Instruct" \
	# 	--dataset_name "data/MMMU_local" \
	# 	--device_map "auto" \
	# 	--seed "0"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRANSFORMERS_CACHE=$PWD/.cache python3 run_qwen2b.py \
	--exp_name "inference_QWQ-32B-Preview_cot_summurize" \
		            --model_name "Qwen/QWQ-32B-Preview" \
			            --dataset_name "data/MMMU_descriptive_Qwen2.5-VL-72B-Instruct" \
				            --device_map "auto" \
					            --seed "0"
	break
done
