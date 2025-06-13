from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import os
import faiss
import json

df = load_dataset('json', data_files='medquad_cleaned_small.jsonl', split='train')
corpus = df["answer"]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

def tokenize_function(examples):
    from models import gpt2_tokenizer
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
    return gpt2_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
    )

# --- VQA-RAD Open Ended Dataset ---
def load_vqarad_open_ended():
    with open("vqa-rad/vqa_rad_open_ended.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    image_names = [item["image_name"] for item in data if "image_name" in item]
    questions = [item["question"] for item in data if "question" in item]
    answers = [item["answer"] for item in data if "answer" in item]
    # Use image_name as image_path with the folder prefix
    image_paths = [os.path.join("vqa-rad", "VQA_RAD Image Folder", name) for name in image_names]
    return image_paths, questions, answers

image_paths, vqarad_questions, vqarad_answers = load_vqarad_open_ended()

# --- FAISS image index for VQA-RAD ---
faiss_index = faiss.read_index("faiss_image_vqarad.index")