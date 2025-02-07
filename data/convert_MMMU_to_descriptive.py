from datasets import load_dataset, Dataset, DatasetDict
from datasets import get_dataset_config_names, load_dataset
import random
import torch
import os
import time
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_MAX_DIM = 1000

model_name = "Qwen/Qwen2-VL-72B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model = model.eval()

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")


# Load MMMU dataset (all available configs)
subjects = get_dataset_config_names("MMMU/MMMU")
mmmu_images = []

@torch.no_grad()
def get_descriptive(example):
    
    result_dir = f"data/MMMU_descriptive"
    example["str_id"] = example["id"]
    str_id = example["str_id"]
    file_path = os.path.join(result_dir, f"{str_id}.json")

    with open(file_path, 'r') as json_file:
        js = json.loads(json_file.read())
    #print(js)
    example['description'] = js["description"]
    
                
    return example
random.shuffle(subjects)
for subject in ["Music"]:# + subjects:
    #mmmu = load_dataset("MMMU/MMMU", subject, keep_in_memory=True)["validation"]  # Load all available configurations
    mmmu = load_dataset("HuggingFaceM4/MMMU", keep_in_memory=True)["validation"]  # Load all available configurations
    def is_music(example):
        return example["id"].startswith("validation_Music")
    mmmu = mmmu.filter(is_music)
    #mmmu = mmmu.shuffle(seed=int(time.time()))
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]
    mmmu = mmmu.map(get_descriptive, num_proc=1)
    #mmmu.push_to_hub("Splend1dchan/MMMU_descriptive")

    dataset_dict = DatasetDict({"validation": mmmu})
    # Save the modified dataset to Hugging Face DatasetDict with a random subset name
    # dataset_name = f"Splend1dchan/MMLU_visual_noise"
    print("saving",subject)
    #if subject not in done:
    print("***\n\n\n")
    print("uploading",subject)
    print("\n\n\n***")
    dataset_dict.push_to_hub("Splend1dchan/MMMU_descriptive", subject)
        


