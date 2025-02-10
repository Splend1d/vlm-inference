import argparse
import os
import subprocess
import logging
import random 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
#from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from datasets import load_dataset, get_dataset_config_names
import torch
import json
import time

from inference_qwen_cot_summurize import get_answer_from_messages_qwen, get_answer_from_messages_deepseek
from data import parse_MMMU, parse_MMMU2, parse_MMLU, parse_gpqa
from json_utils import make_serializable

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

# Dataset-specific configurations
DATASET_CONFIG = {
    "MMMU/MMMU": {
        "split": "validation",
        "subsets": ["__ALL__"],
        "dataset_preprocess_fn": "parse_MMMU",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "data/MMMU_local": {
        "split": "validation",
        "subsets": ["__ALL__"],
        "dataset_preprocess_fn": "parse_MMMU2",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "cais/mmlu": {
        "split": "test",
        "subsets": ["all"],
        "dataset_preprocess_fn": "parse_MMLU",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "Idavidrein/gpqa": {
        "split": "train",
        "subsets": ["gpqa_diamond"],
        "dataset_preprocess_fn": "parse_gpqa",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "Splend1dchan/gpqa_diamond_visual_noise": {
        "split": "test",
        "subsets": ["random_noise"],
        "dataset_preprocess_fn": "parse_gpqa_diamond_visual_noise",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "Splend1dchan/MMMU_descriptive": {
        "split": "validation",
        "subsets": ["__ALL__"],
        "dataset_preprocess_fn": "parse_MMMU_descriptive",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "data/MMMU_descriptive_Qwen2.5-VL-72B-Instruct": {
        "split": "validation",
        "subsets": ["__ALL__"],
        "dataset_preprocess_fn": "parse_MMMU_descriptive",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    },
    "data/MMMU_descriptive": {
        "split": "validation",
        "subsets": ["__ALL__"],
        "dataset_preprocess_fn": "parse_MMMU_descriptive",  # Replace with actual function if needed
        "options_key": "options_for_logits_compare",
        "messages_key": "messages"
    }
}

MESSAGE_PREPROCESS_CONFIG = {
    "Qwen2": get_answer_from_messages_qwen,
    "QVQ": get_answer_from_messages_qwen,
    "deepseek": get_answer_from_messages_deepseek
}

PREPROCESSOR_OVERRIDE = {
    "Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-72B-Instruct": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/QWQ-32B-Preview": "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "deepseek-ai/deepseek-moe-16b-chat": "deepseek-ai/deepseek-vl2-small",
}

# Load model and processor
def load_model_and_processor(model_name, device_map):
    if "Qwen2.5-VL" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    elif "Qwen2-VL" in model_name or "QVQ" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    elif "deepseek" in model_name:
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(torch.bfloat16).to(device_map).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    if model_name in PREPROCESSOR_OVERRIDE:
        preprocessor_name = PREPROCESSOR_OVERRIDE[model_name]
    else:
        preprocessor_name = model_name
        
    if "deepseek" in preprocessor_name:
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(preprocessor_name)
    else:
        processor = AutoProcessor.from_pretrained(preprocessor_name)
    return model, processor

# Dynamically import preprocess functions
def import_dataset_preprocess_fn(module_name, func_name):
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)

# Main inference function
def run_inference_on_dataset(
    dataset_name, exp_name, model, processor
):
    if dataset_name not in DATASET_CONFIG:
        logging.error(f"Dataset {dataset_name} not configured!")
        return

    dataset_config = DATASET_CONFIG[dataset_name]
    dataset_preprocess_fn = import_dataset_preprocess_fn("data", dataset_config["dataset_preprocess_fn"])
    
    
    processor_type = processor.__class__.__name__
    if "Qwen2" in processor_type:
        message_dataset_preprocess_fn = MESSAGE_PREPROCESS_CONFIG["Qwen2"]
    elif "deepseek" in processor_type:
        message_dataset_preprocess_fn = MESSAGE_PREPROCESS_CONFIG["deepseek"]
    

    subsets = get_dataset_config_names(dataset_name)
    print(dataset_config["subsets"])
    if dataset_config["subsets"] == ["__ALL__"]:
        pass
    else:
        for subset in dataset_config["subsets"]:
            assert subset in subsets, f"Subset name not found: {subset}"
        subsets = dataset_config["subsets"]
    
    random.shuffle(subsets)
    for subset in subsets:
        logging.info(f"Processing subset: {subset}")

        result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{subset}"
        os.makedirs(result_dir, exist_ok=True)
        
        ds = load_dataset(dataset_name, subset, split=dataset_config["split"], cache_dir=None)
        ds = ds.map(dataset_preprocess_fn, num_proc=1)
        #print(dataset_preprocess_fn)
        
        if "mmlu" in dataset_name:
            cur_time = int(time.time())
            ds = ds.shuffle(seed=cur_time)
            print("shuffling dataset, seed=",cur_time)

        for data in ds:
            str_id = data["str_id"]
            label = data['label']
            logging.info(f"Running inference for ID: {str_id}")

            file_path = os.path.join(result_dir, f"{str_id}.json")
            if os.path.exists(file_path):
                logging.info(f"Results already exist for ID: {str_id}. Skipping.")
                continue

            subprocess.run(f"touch {file_path}".split())
            messages = eval(data[dataset_config["messages_key"]])

            # Replace string paths with PIL.Image instances
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image":
                        content["image"] = data[content["image"]]

            options = data.get(dataset_config["options_key"], [])
            answer, trace = message_dataset_preprocess_fn(
                messages, model, processor, options
            )

            with open(file_path, 'w') as json_file:
                json.dump(make_serializable({"trace": trace, "final_answer":answer, "label": label}), json_file, indent=4)

            logging.info(f"Completed ID: {str_id}")

# Unit test function
def unit_run(model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    answer, trace = inference_qwen_cot_summurize.get_answer_from_messages(
        messages, model, processor, ["A", "B", "C", "D"]
    )
    logging.info("Final Answer: " + answer)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a dataset using Qwen2VL model.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--device_map", type=str, required=True, help="Device map.")
    parser.add_argument("--seed", type=int, required=False, default = 0, help="Random Seed.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    random.seed(args.seed)
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_name, args.device_map)

    # Run inference
    run_inference_on_dataset(args.dataset_name, args.exp_name, model, processor)

    # Unit test
    # unit_run(model, processor)
