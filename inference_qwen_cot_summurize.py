from qwen_vl_utils import process_vision_info


def get_inputs_from_messages(messages, processor):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(messages,text)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(text, inputs)
    return inputs

def get_answer_from_messages(messages, model, processor, choices):
    inputs = get_inputs_from_messages(messages, processor)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    print(inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    messages.append(
        {
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
        }
    )


    inputs = get_inputs_from_messages(messages, processor)

    last_logits = model(**inputs, return_logits = True)[-1]

    choices_ids = processer.tokenize(choices)

    select = torch.argmax(last_logits[choices_ids]).item()

    return choices[select]