import json

OPTIONS = list("ABCDEFGHIJKLMNOPQRS")

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
    return example