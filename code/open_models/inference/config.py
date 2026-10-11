import os

MODEL_DIR = "xxx"
BASE_DIR = "xxx"
DATA_PATH = os.path.join(BASE_DIR, "split_v1", "test.jsonl")
IMAGE_DIR = os.path.join(BASE_DIR, "img_v1")
RESULT_DIR = {
    "base": os.path.join(BASE_DIR, "results", "base"),
    "ft": os.path.join(BASE_DIR, "results", "ft"),
    "test": os.path.join(BASE_DIR, "results", "test")
}
ROLE_PROMPT = (
    "You are an expert in figural reasoning. Given a textual instruction "
    "and an image displaying one problem figure with four visual answer options (A, B, C, D), "
    "identify the correct option. Respond with only one letter (A/B/C/D) without explanation.\n"
)