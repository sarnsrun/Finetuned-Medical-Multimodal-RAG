import numpy as np
import faiss
import json
import os
import uuid
import torch
import csv
import gc
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import bert_score
from datasets import load_dataset

# Import specific classes for retraining
from transformers import (
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer, 
    GPT2LMHeadModel, 
    GPT2Tokenizer
)
from peft import LoraConfig, get_peft_model, TaskType

from data import df, vectorizer, vqarad_answers, tokenize_function, faiss_index
from models import (
    gpt2_model, gpt2_tokenizer, blip_model, blip_processor, embedder, device
)

# Load FAISS index
index_text = faiss.read_index("medquad_embedder_small.index")

def retrieve(query, top_k=3):
    answers = df["answer"]
    query_embed = embedder.encode(query)
    query_vec = np.array([query_embed]).astype(np.float32)
    _, I = index_text.search(query_vec, top_k)
    return [answers[i] for i in I[0]]

@torch.no_grad()
def retrieve_image(image: Image.Image, top_k=3):
    image = image.convert("RGB").resize((384, 384))
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    vision_outputs = blip_model.vision_model(**inputs)
    image_embed = vision_outputs.pooler_output.cpu().numpy()
    distances, indices = faiss_index.search(image_embed, top_k)
    return [vqarad_answers[i] for i in indices[0]]

def choose_expected_response(prompt, dataset):
    all_texts = [prompt] + list(dataset['question'])
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_index = similarities.argmax()
    return dataset['answer'][best_index]

def get_embedding(text):
    return embedder.encode(text.strip().lower())

def comparison_evaluator(response, expected):
    response, expected = response.strip().lower(), expected.strip().lower()
    lev_distance = Levenshtein.distance(response, expected)
    lev_similarity = max(0, 1 - lev_distance / max(len(response), len(expected)))
    emb1, emb2 = get_embedding(response), get_embedding(expected)
    cos_sim = cosine_similarity([emb1], [emb2])[0][0]
    P, R, F1 = bert_score.score([response], [expected], lang="en")
    return lev_similarity, cos_sim, P.mean().item(), R.mean().item(), F1.mean().item()

def rag(query, dataset, max_context_length=300):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)[:max_context_length]
    reference_answer = choose_expected_response(query, dataset)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(inputs['input_ids'], max_new_tokens=1000, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_answer.split("Answer:")[-1].strip()
    return answer, reference_answer

@torch.no_grad()
def rag_image(image: Image.Image, question: str, top_k=3, max_new_tokens=50):
    reference_answer = retrieve_image(image, top_k)
    context_text = " ".join(reference_answer)
    combined_text = question + " Context: " + context_text
    inputs = blip_processor(images=image, text=combined_text, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_answer = blip_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_answer, reference_answer

def save_feedback_gpt2(user_input: str, corrected_output: str):
    entry = {
        "question": user_input.strip(),
        "answer": corrected_output.strip()
    }
    with open('medquad_cleaned_small.jsonl', "a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")

def save_feedback_blip(user_input: str, corrected_output: str, image_file=None):
    image_folder = 'roco-dataset-master/data/test/radiology/images'
    os.makedirs(image_folder, exist_ok=True)
    image_path = None
    if image_file:
        ext = os.path.splitext(image_file.filename)[1] or '.png'
        filename = f"{uuid.uuid4().hex}{ext}"
        image_path = os.path.join(image_folder, filename)
        image_file.save(image_path)
    else:
        image_path = user_input.strip()
    write_header = not os.path.isfile('test.csv') or os.path.getsize('test.csv') == 0
    with open('test.csv', "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
        if write_header:
            writer.writeheader()
        writer.writerow({"image_path": image_path, "caption": corrected_output.strip()})

def retrain_gpt2():
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists('medquad_cleaned_small.jsonl'):
        raise Exception("No feedback data found (medquad_cleaned_small.jsonl missing). Please save some corrections first.")

    # Using 'gpt2' base prevents "nested adapter" warnings, but if you want to keep knowledge,
    # use your finetuned one. We suppress warnings by ignoring the existing config if possible.
    base_model_id = "sarnsrun/gpt2-medquad-finetuned"
    
    print(f"Loading base model: {base_model_id}")
    model = GPT2LMHeadModel.from_pretrained(base_model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Prepare New LoRA Adapter
    # We create a new configuration to retrain on the feedback
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Wrap model
    model = get_peft_model(model, lora_config)
    model.to(device)

    print("Loading feedback dataset...")
    df_retrain = load_dataset('json', data_files='medquad_cleaned_small.jsonl', split='train')
    tokenized_dataset = df_retrain.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-retrained_checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    output_dir = "gpt2-local-best"
    print(f"Saving retrained model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()