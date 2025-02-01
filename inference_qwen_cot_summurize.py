from qwen_vl_utils import process_vision_info
import logging
import torch

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

def get_inputs_from_messages_qwen(messages, processor):
    print(messages, processor)
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
    #print(text, inputs)
    return inputs

@torch.no_grad()
def get_answer_from_messages_qwen(messages, model, processor, choices):
    inputs = get_inputs_from_messages_qwen(messages, processor)
    inputs = inputs.to(model.device)
    #print(model.device)
    print(inputs["input_ids"].shape)
    #print(inputs["attention_mask"])
    #print(inputs["input_ids"][inputs["attention_mask"][0] == 1])
    #input()

    # Inference: Generation of the output
    # print(inputs["input_ids"])
    # print(inputs["attention_mask"])
    # print(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    print(generated_ids)

    output_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    #print(output_text)

    #follow_up_question = "Now, rewrite the final answer in the most concise way:"
    
    messages.extend(
        [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": output_text},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now, output the final answer and nothing else:"},
            ],
        }]
        
    )
    logging.info(str(messages))
    #print(messages)


    inputs = get_inputs_from_messages_qwen(messages, processor)
    inputs = inputs.to(model.device)
    
    if len(choices):
        last_logits = model(**inputs, return_dict = True).logits[:,-1]
        logging.info(last_logits.shape)
        #choices_idx = processor.tokenizer.tokenize(choices)
        #logging.info(choices_idx)
        
        
        choices_ids = torch.LongTensor(processor.tokenizer.convert_tokens_to_ids(choices)).to(last_logits.device)

        select = torch.argmax(last_logits[:,choices_ids]).item()
        final_answer = choices[select]
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": final_answer},
                ],
            }
            
        )
    else:
        # Use the reasoning as the answer
        messages.pop()
        final_answer = output_text
        
        
        # generated_ids = model.generate(**inputs, max_new_tokens=20)

        # final_answer = processor.batch_decode(
        #     generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]

    
    
    return final_answer, messages


from deepseek_vl.utils.io import load_pil_images

def get_inputs_from_messages_deepseek(messages, processor, prepend_messages=None):
    
    # conversation = [
    #     {
    #         "role": "User",
    #         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
    #                    "<image_placeholder>a dog wearing a santa hat, "
    #                    "<image_placeholder>a dog wearing a wizard outfit, and "
    #                    "<image_placeholder>what's the dog wearing?",
    #         "images": [
    #             "images/dog_a.png",
    #             "images/dog_b.png",
    #             "images/dog_c.png",
    #             "images/dog_d.png",
    #         ],
    #     },
    #     {"role": "Assistant", "content": ""}
    # ]
    
    print(messages)
    for i in range(len(messages)):
        new_content = ""
        new_images = []
        message = messages[i]
        for c in message['content']:
            if c['type'] == "text":
                new_content += c['text'].replace("<image>", "<image_placeholder>")
            elif c['type'] == "image":
                new_images.append(c['image'])
            else:
                raise ValueError
        messages[i]["content"] = new_content
        messages[i]["images"] = new_images
    
    
    if prepend_messages is not None:
        messages = prepend_messages + messages
    
    messages.append({"role": "assistant", "content": ""})
    #print(messages)
    #input("messages")
    pil_images = load_pil_images(messages)
    #print(processor)
    #print(pil_images)
    prepare_inputs = processor(
        conversations=messages,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    )
    
    
    messages.pop() # remove the null assistant for the processor
    
    print(processor.tokenizer.decode(prepare_inputs["input_ids"][0]))
    
    
    #print(text, inputs)
    return prepare_inputs, messages

@torch.no_grad()
def get_answer_from_messages_deepseek(messages, model, processor, choices):

    
    inputs, messages = get_inputs_from_messages_deepseek(messages, processor)
    inp = processor.tokenizer.decode(
        inputs["inputs_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(inp)
    input()
    if model.__class__.__name__  == "MultiModalityCausalLM":
        generation_root_model = model.language_model
    else:
        generation_root_model = model
    inputs = inputs.to(generation_root_model.device)
    
    #print(model.__class__.__name__)
    if model.__class__.__name__ == "DeepseekForCausalLM":
        inputs_embeds = None
        #print("no input embeds")
    elif model.__class__.__name__ == "LlamaForCausalLM":
        inputs_embeds = None
    else:
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        inputs_embeds = inputs_embeds.to(generation_root_model.device)
        
    
    #inputs["inputs_embeds"] = inputs_embeds
    #print(inputs)
    #inputs.pop("input_ids")
    #input("input_keys")
    #print(model.device)
    #print(inputs["input_ids"].shape)
    #print(inputs["attention_mask"])
    #print(inputs["input_ids"][inputs["attention_mask"][0] == 1])
    #input()

    # Inference: Generation of the output
    # print(inputs["input_ids"])
    # print(inputs["attention_mask"])
    # print(model.device)
    if inputs_embeds is None:
        generated_ids = generation_root_model.generate(    
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=1024,
            use_cache=True
        )
    else:
        generated_ids = generation_root_model.generate(    
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=1024,
            use_cache=True
        )
    print("generated_ids",generated_ids[:, :][0])

    
    if model.__class__.__name__ == "DeepseekForCausalLM":
        output_text = processor.tokenizer.decode(
            generated_ids[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        # deepseek-VL generate does not include inputs
        output_text = processor.tokenizer.decode(
            generated_ids[:, :][0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
    print("＊＊＊Output Text＊＊＊\n",output_text,"\n＊＊＊End of Output Text＊＊＊")

    #follow_up_question = "Now, rewrite the final answer in the most concise way:"
    
    
    new_messages = \
        [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": output_text},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now, output the final answer and nothing else:"},
            ],
        }]
        
    logging.info(str(new_messages))
    print("After step 1:",messages)


    inputs, messages = get_inputs_from_messages_deepseek(new_messages, processor, prepend_messages=messages)
    
    print("***After step 2:***\n",messages)
    inputs = inputs.to(model.device)
    
    if model.__class__.__name__ == "DeepseekForCausalLM":
        inputs_embeds = None
        #print("no input embeds")
    if model.__class__.__name__ == "LlamaForCausalLM":
        inputs_embeds = None
    else:
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
    
    if len(choices):
        if inputs_embeds is None:
            last_logits = generation_root_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict = True,
                use_cache=True
            ).logits[:,-1]
        else:
            last_logits = generation_root_model(    
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                return_dict = True,
                use_cache=True
            ).logits[:,-1]
            
        #last_logits = model(**inputs, return_dict = True).logits[:,-1]
        logging.info(last_logits.shape)
        #choices_idx = processor.tokenizer.tokenize(choices)
        #logging.info(choices_idx)
        
        choices_ids = torch.LongTensor(processor.tokenizer.convert_tokens_to_ids(choices)).to(last_logits.device)

        select = torch.argmax(last_logits[:,choices_ids]).item()
        final_answer = choices[select]
        
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": final_answer},
                ],
            }
            
        )
    else:
        # Use the reasoning as the answer
        messages.pop()
        final_answer = output_text

    
    
    return final_answer, messages

