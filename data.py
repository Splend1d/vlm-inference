import json
import hashlib
import random


OPTIONS = list("ABCDEFGHIJKLMNOPQRS")

def hash_string(input_string):
    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the object with the string (encoded as bytes)
    sha256_hash.update(input_string.encode('utf-8'))
    
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()[:12]


def parse_MMMU(example):
    # keys: ['id', 'question', 'options', 'explanation', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'img_type', 'answer', 'topic_difficulty', 'question_type', 'subfield']
    
    # Example message:
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
    
    question = example['question']+ "\n\n"
    for n, option in enumerate(eval(example["options"])):
        question += f"{chr(ord('A')+n)}. {option}\n"
    
    content = []
    for image_id in [1,2,3,4,5,6,7]:
        if example[f"image_{image_id}"] is not None:
            content.append({"type": "image", "image": f"image_{image_id}"})
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    example["messages"] = json.dumps(messages)
    example["options_for_logits_compare"] = OPTIONS[:len(eval(example["options"]))]
    example["str_id"] = example["id"]
    example['label'] = example['answer']
    return example

def parse_MMLU(example):
    # keys: ['question', 'subject', 'choices', 'answer']
    
    # Example message:
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."},
    #         ],
    #     }
    # ]
    
    question = example['question']+ "\n\n"
    for n, option in enumerate(example["choices"]):
        question += f"{chr(ord('A')+n)}. {option}\n"
    
    content = []
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    example["messages"] = json.dumps(messages)
    example["options_for_logits_compare"] = OPTIONS[:len(example["choices"])]
    example["str_id"] = example["subject"] + hash_string(example['question'])
    example['label'] = chr(ord('A')+example['answer'])
    return example

def parse_gpqa(example):
    # keys: ['question', 'subject', 'choices', 'answer']
    
    # Example message:
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."},
    #         ],
    #     }
    # ]
    
    choice_indices = [1, 2, 3, 4]
    choice_order = random.sample(choice_indices, len(choice_indices))
    ans_idx = choice_order.index(4)
    
    ordered_choices = [
        example[f"Incorrect Answer {i}"].strip() if i != 4 else example["Correct Answer"].strip()
        for i in choice_order
    ]
    
    question = example['Question']+ "\n\n"
    for n, option in enumerate(ordered_choices):
        question += f"{chr(ord('A')+n)}. {option}\n"
    
    content = []
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    example["messages"] = json.dumps(messages)
    example["options_for_logits_compare"] = OPTIONS[:len(ordered_choices)]
    example["str_id"] = hash_string(example['Question'])
    example['label'] = chr(ord('A')+ ans_idx) 
    
    return example