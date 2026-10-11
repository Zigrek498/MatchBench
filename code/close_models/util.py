import os
import random
random.seed(42)
from google.genai import types
import base64
import jsonlines
import json
from openai import OpenAI
from collections import defaultdict

TEST_DATA_PATH = "xxx"
TRAIN_DATA_PATH = "xxx"
IMAGE_DIR = r"xxx"
RESULT_DIR = r"xxx"
ICL_PATH = r"xxx"

ROLE_PROMPT = (
    "You are an expert in figural reasoning. Given a textual instruction "
    "and an image displaying one problem figure with four visual answer options (A, B, C, D), "
    "identify the correct option. Respond with only one letter (A/B/C/D) without explanation.\n"
)

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

train_data = read_jsonl(TRAIN_DATA_PATH)
train_by_cat = defaultdict(list)
for d in train_data:
    cat = d['image'].rsplit('-', 1)[0]
    train_by_cat[cat].append(d)
train_by_cat = dict(train_by_cat)

def encode_image_to_base64(path):
  with open(path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def init_json(file_path):
    with open(file_path, 'w') as f:
        json.dump({}, f, indent=2)

def construct_few_shot(entry, num_shot, model, mode="random"):
    global train_data
    cat = entry['image'].rsplit('-', 1)[0]
    icl_file = os.path.join(ICL_PATH, f'{model}_{num_shot}shot.json')
    if not os.path.exists(icl_file):
        init_json(icl_file)
    with open(icl_file, 'r') as f:
        icl_dict = json.load(f)

    img_path = os.path.join(IMAGE_DIR, entry['image'])
    base64_image = encode_image_to_base64(img_path)
    test_content = [
        {"type": "input_text", "text": "Input: Image: "},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
        {"type": "input_text", "text": f"\nInstruction: {entry['question']}\nOutput:"}
    ]
    
    fewshot_prompt = f"Given the following {num_shot} examples to learn the figural reasoning task:\n" if num_shot else ""
    content = [
        {"type": "input_text", "text": ROLE_PROMPT + fewshot_prompt},
    ]

    if mode == "random":
        if entry['image'] in icl_dict:
            sampled_data = icl_dict[entry['image']]
        else:
            sampled_data = random.sample(train_by_cat[cat], num_shot)
            icl_dict[entry['image']] = sampled_data
    else:
        sampled_data = []
    
    with open(icl_file, 'w') as f:
        json.dump(icl_dict, f, indent=2)

    if len(sampled_data) != num_shot:
        raise ValueError("len(sampled_data) != num_shot")
    else:
        fewshot_content = []
        for idx, example in enumerate(sampled_data):
            ex_img_path = os.path.join(IMAGE_DIR, example['image'])
            ex_base64_image = encode_image_to_base64(ex_img_path)
            fewshot_content.extend([
                {"type": "input_text", "text": f"Example{idx+1}:\nInput: Image: "},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{ex_base64_image}"},
                {"type": "input_text", "text": f"\nInstruction: {example['question']}\nOutput: {example['answer']}\n"}
            ])
    content.extend(fewshot_content)
    content.extend(test_content)
    return content

def encode_image_bytes(img_path):
    with open(img_path, "rb") as f:
        return f.read()

def construct_few_shot_gemini(entry, num_shot, mode="random"):
    global train_data
    
    model = "gemini"
    icl_file = os.path.join(ICL_PATH, f"{model}_{num_shot}shot.json")

    if not os.path.exists(icl_file):
        init_json(icl_file)

    with open(icl_file, "r") as f:
        icl_dict = json.load(f)

    img_path = os.path.join(IMAGE_DIR, entry["image"])
    test_image = encode_image_bytes(img_path)

    test_parts = [
        types.Part.from_text(text="Input: Image: "),
        types.Part.from_bytes(data=test_image, mime_type="image/jpeg"),
        types.Part.from_text(
            text=f"\nInstruction: {entry['question']}\nOutput:"
        ),
    ]

    fewshot_prompt = (
        f"Given the following {num_shot} examples to learn the figural reasoning task:\n"
        if num_shot
        else ""
    )

    content_parts = [
        types.Part.from_text(text=ROLE_PROMPT + fewshot_prompt),
    ]

    if mode == "random":
        if entry["image"] in icl_dict:
            sampled_data = icl_dict[entry["image"]]
        else:
            cat = entry['image'].rsplit('-', 1)[0]
            sampled_data = random.sample(train_by_cat[cat], num_shot)
            icl_dict[entry["image"]] = sampled_data
    else:
        sampled_data = []

    with open(icl_file, "w") as f:
        json.dump(icl_dict, f, indent=2)

    if len(sampled_data) != num_shot:
        raise ValueError("len(sampled_data) != num_shot")

    for idx, example in enumerate(sampled_data):
        ex_img_path = os.path.join(IMAGE_DIR, example["image"])
        ex_image = encode_image_bytes(ex_img_path)

        content_parts.extend([
            types.Part.from_text(text=f"Example{idx+1}:\nInput: Image: "),
            types.Part.from_bytes(data=ex_image, mime_type="image/jpeg"),
            types.Part.from_text(
                text=f"\nInstruction: {example['question']}\nOutput: {example['answer']}\n"
            ),
        ])

    content_parts.extend(test_parts)

    return [
        types.Content(
            role="user",
            parts=content_parts
        )
    ]