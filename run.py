from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import inference_qwen_cot_summurize
from data import parse_MMMU
from datasets import load_dataset, get_dataset_config_names
import torch
# default: Load the model on the available device(s)
import logging
import os
from PIL import Image
import json
from utils import make_serializable

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

exp_name = "inference_qvq_cot_summurize"
dataset_name = "MMMU/MMMU"

model_name = "Qwen/QVQ-72B-Preview"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")
print(processor)
#choices_idx = processor.tokenizer.convert_tokens_to_ids(["A","B","C","D"])
#print(choices_idx)
#input("test")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

configs = get_dataset_config_names("MMMU/MMMU")
for config in configs:
    ds = load_dataset("MMMU/MMMU",config, split = "validation", keep_in_memory = True)
    print(ds)
    ds = ds.map(parse_MMMU, num_proc = 100)
    print(ds[0])
    print(ds[0]["image_1"])
    #print(ds[0]["messages"][0]["content"][0]["image"])
    #s()
    
    result_dir = f"results/{exp_name}/{dataset_name.split('/')[-1]}/{config}"
    os.makedirs(result_dir, exist_ok = True)
    
    for data in ds:
        #print(data)
        messages = eval(data["messages"])
        
        # replace string to PIL.Image
        for i in range(len(messages)):
            for j in range(len(messages[i]["content"])):
                if messages[i]["content"][j]["type"] == "image":
                    #print(data[messages[i]['content'][j]['image']])
                    messages[i]["content"][j]["image"] = data[messages[i]['content'][j]['image']]
        
        #print(isinstance(messages[0]["content"][0]["image"], Image.Image))
        #print(messages)
        #messages[0]["content"][0]["image"] is Image.Image)
        options = data["options_for_logits_compare"]
        print("options",options)
        str_id = data["id"]
        answer, trace = inference_qwen_cot_summurize.get_answer_from_messages(messages, model, processor, options)
        
        file_path = os.path.join(result_dir, f"{str_id}.json")
        with open(file_path, 'w') as json_file:
            json.dump(make_serializable(trace), json_file, indent=4)
        #logging.info("Final Answer " + answer)
        
def unit_run():
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

    # Preparation for inference
    answer, trace = inference_qwen_cot_summurize.get_answer_from_messages(messages, model, processor, ["A","B","C","D"])

    logging.info("Final Answer " + answer)
