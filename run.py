from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import inference_qwen_cot_summurize
import torch
# default: Load the model on the available device(s)
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)


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

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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
answer = inference_qwen_cot_summurize.get_answer_from_messages(messages, model, processor, ["A","B","C","D"])

logging.info("Final Answer",answer)
