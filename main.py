from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs["load_in_4bit"] = True
kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, **kwargs
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cuda")
image_processor = vision_tower.image_processor

import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from transformers import TextStreamer
from fastapi import FastAPI, Header, Form, Request

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    """
    Load an image from a URL or a local file.

    Args:
    image_file (str): A URL or a local file path to the image.

    Returns:
    Image: An image object in RGB format.
    """

    # Check if the image file is a URL or a local file.
    if is_url(image_file):
        # Load image from a URL.
        return load_image_from_url(image_file)
    else:
        # Load image from a local file.
        return load_image_from_file(image_file)


def is_url(file_path):
    """
    Check if the given file path is a URL.

    Args:
    file_path (str): The file path to check.

    Returns:
    bool: True if the file path is a URL, False otherwise.
    """
    # Check for http or https in the beginning of the file path.
    return file_path.startswith("http") or file_path.startswith("https")


def load_image_from_url(url):
    """
    Load an image from a URL.

    Args:
    url (str): The URL of the image.

    Returns:
    Image: An image object in RGB format.
    """
    try:
        # Send a request to the URL.
        response = requests.get(url)

        # Raise an exception if the request was unsuccessful.
        response.raise_for_status()

        # Open the image from the response content.
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.RequestException as e:
        print(f"Error while fetching the image from URL: {e}")
        return None
    except IOError as e:
        print(f"Error while opening the image: {e}")
        return None


def load_image_from_file(file_path):
    """
    Load an image from a local file.

    Args:
    file_path (str): The local file path of the image.

    Returns:
    Image: An image object in RGB format.
    """
    try:
        # Open the image from the local file path.
        image = Image.open(file_path).convert("RGB")
        return image
    except IOError as e:
        print(f"Error while opening the image file: {e}")
        return None


def caption_image(image_file, prompt):

    image = load_image(image_file)

    if image is not None:

        disable_torch_init()

        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()

        print(conv)

        # fetching roles
        roles = conv.roles
        image_tensor = (
            image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )

        inp = f"{roles[0]}: {prompt}"
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + inp
        )

        print("inp", inp)

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        raw_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

        print("outputs:", outputs)
        conv.messages[-1][-1] = outputs

        output = outputs.rsplit("</s>", 1)[0]

        print(output)
        return image, output
    else:
        return None, "Error"


app = FastAPI()


@app.get("/")
async def index():
    return {"model": "working"}


@app.post("/generate")
async def generate_caption(
    request: Request, url: str = Form(...), prompt: str = Form(...)
):

    if url:
        image, output = caption_image(url, prompt)
        return {"result": output}
    else:
        return {"error": "Failed to process the image"}
