from datasets import load_dataset, Dataset, DatasetDict
from datasets import get_dataset_config_names, load_dataset
import random
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen2-VL-72B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

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
    
    meta_prompt = "You are an agent that will describe an image throughly, given an image and a question to answer." \
    "After this, a blind person who is very good at reasoning will answer the question based on your description. " \
    "The descrption should be verbose enough to answer the question without the original image." \
    "For example, if the image is a diagram and the question asks about the details of the standard procedure, you should try to recreate the diagram with output text" \
    "But if the question is asking about the color of a component, you just have to describe colors of the component." \
    "Whenever a statement is not related to the visual clue, such as doing math after getting some numbers out from a table, you can leave it to the blind person don't have to output that" \
    "You can provide the answer if the question is very visual and there is no good way to describe it"
    
    meta_prompt = "Describe the image as detailed as possible, such that the following QUESTION can be answered solely based on your description without the actual image. It is ok if the description already contains the answer."
    meta_prompt_after = "Describe the image as detailed as possible, such that the QUESTION  above can be answered solely based on your description without the actual image. It is ok if the description already contains the answer."
    question = meta_prompt+ "\n\n***START of QUESTION***\n"+example['question']
    for n, option in enumerate(eval(example["options"])):
        question += f"{chr(ord('A')+n)}. {option}\n"
        
    question += "\n***END of QUESTION***\n\n"
    question += meta_prompt_after
    
    content = []
    for image_id in [1,2,3,4,5,6,7]:
        if example[f"image_{image_id}"] is not None:
            content.append({"type": "image", "image": example[f"image_{image_id}"]})
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    print(question)
    print(messages)
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    #print(messages,text)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(model.device)
    print(inputs["input_ids"].shape)
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    print(generated_ids)

    output_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)
    input("pause")
    
    result_dir = f"data/MMMU_descriptive"
    example["str_id"] = example["id"]
    str_id = example["str_id"]
    file_path = os.path.join(result_dir, f"{str_id}.json")
    with open(file_path, 'w') as json_file:
        json.dump(make_serializable({"question": example['question'], 
                                     "options": example['options'], 
                                     "id": example["id"],
                                     "description": output_text}), json_file, indent=4)
                
    return example

for subject in subjects:
    mmmu = load_dataset("MMMU/MMMU", subject)  # Load all available configurations

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
    mmmu = mmmu.map(get_descriptive)
        


