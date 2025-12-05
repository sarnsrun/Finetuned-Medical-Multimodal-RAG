import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    BlipProcessor, BlipForQuestionAnswering,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import os

os.environ["WANDB_DISABLED"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_model_path = "sarnsrun/gpt2-medquad-finetuned"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)

blip_model_path = "sarnsrun/blip-vqa-finetuned"
blip_model = BlipForQuestionAnswering.from_pretrained(blip_model_path).to(device)
blip_processor = BlipProcessor.from_pretrained(blip_model_path, use_fast=True)

embedder = SentenceTransformer('all-mpnet-base-v2')

def reload_gpt2_model():
    global gpt2_model, gpt2_tokenizer
    
    local_model_path = "gpt2-local-best"
    base_model_path = "sarnsrun/gpt2-medquad-finetuned"

    print("Reloading model...")

    if os.path.exists(local_model_path):
        print(f"Found locally retrained model at {local_model_path}. Loading...")
        
        # Load the base model first
        base_model = GPT2LMHeadModel.from_pretrained(base_model_path).to(device)
        
        # Load the adapter (LoRA)
        gpt2_model = PeftModel.from_pretrained(base_model, local_model_path).to(device)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
    else:
        print("No local model found. Reloading base model.")
        gpt2_model = GPT2LMHeadModel.from_pretrained(base_model_path).to(device)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)

    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.eval() # Ensure model is in eval mode
    print("Model reloaded successfully.")