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

gpt2_model_path = "gpt2-finetuned"

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = PeftModel.from_pretrained(base_model=base_model, PeftModelPath=gpt2_model_path)

blip_model_path = "blip-vqa-finetuned"
blip_model = BlipForQuestionAnswering.from_pretrained(blip_model_path).to(device)
blip_processor = BlipProcessor.from_pretrained(blip_model_path, use_fast=True)

embedder = SentenceTransformer('all-mpnet-base-v2')

def reload_gpt2_model():
    global gpt2_model, gpt2_tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = PeftModel.from_pretrained(base_model=base_model, PeftModelPath=gpt2_model_path)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token