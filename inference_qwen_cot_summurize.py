from qwen_vl_utils import process_vision_info
import logging
import torch

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(message)s'
)

def get_inputs_from_messages(messages, processor):
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
def get_answer_from_messages(messages, model, processor, choices):
    inputs = get_inputs_from_messages(messages, processor)
    inputs = inputs.to(model.device)
    #print(model.device)
    #print(inputs["input_ids"])
    #print(inputs["attention_mask"])
    #print(inputs["input_ids"][inputs["attention_mask"][0] == 1])
    #input()

    # Inference: Generation of the output
    #print(inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    #print(output_text)

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
                {"type": "text", "text": "Output the correct answer in letter:"},
            ],
        }]
        
    )
    logging.info(str(messages))
    #print(messages)


    inputs = get_inputs_from_messages(messages, processor)

    last_logits = model(**inputs, return_dict = True).logits[:,-1]
    logging.info(last_logits.shape)
    #choices_idx = processor.tokenizer.tokenize(choices)
    #logging.info(choices_idx)
    
    
    choices_ids = torch.LongTensor(processor.tokenizer.convert_tokens_to_ids(choices)).to(last_logits.device)

    select = torch.argmax(last_logits[:,choices_ids]).item()

    return choices[select]