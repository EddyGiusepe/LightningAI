from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from litserve.specs.openai import ChatMessage
import base64, torch
from typing import List
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_image_str):
    # Strip the prefix (e.g., 'data:image/jpeg;base64,')
    base64_data = base64_image_str.split(",")[1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image


class Phi:
    def __init__(self, device):
        model_id = "microsoft/Phi-3.5-vision-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=16,  # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        )
        self.device = device

    def apply_chat_template(self, messages: List[ChatMessage]):
        final_messages = []
        images = []
        for message in messages:
            msg = {}
            if message.role == "user":
                prompt = None
                placeholder = ""
                msg["role"] = "user"
                content_list = message.content
                for i, content in enumerate(content_list):
                    if content.type == "text":
                        prompt = content.text
                    elif content.type == "image_url":
                        url = content.image_url.url
                        image = decode_base64_image(url)
                        images.append(image)
                        msg["content"] = image
                        placeholder += f"<|image_{i}|>\n"
                msg["content"] = placeholder + prompt
            elif message.role == "assistant":
                content = message.content
                msg["role"] = "assistant"
                msg["content"] = content
            final_messages.append(msg)
        prompt = self.processor.tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
        return prompt, images

    def __call__(self, inputs):
        prompt, images = inputs
        if len(images)==0:
            images = None
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.2,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args,)
        return inputs, generate_ids

    def decode_tokens(self, outputs):
        inputs, generate_ids = outputs
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response