import os
import torch
import clip
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import clip
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RocoClipCapDataset(Dataset):
    def __init__(self, csv_file, clip_preprocess, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        caption = str(row["caption"])
        keywords = str(row.get("keywords", "")).strip()
        if keywords:
            caption += f". Keywords: {keywords}"

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        image_tensor = self.clip_preprocess(image)

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# Define the projection module
class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_dim=512, gpt2_type="gpt2"):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.prefix_proj = nn.Linear(prefix_dim, self.gpt.transformer.wte.embedding_dim)

    def forward(self, tokens, mask, prefix_embed):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_embed = self.prefix_proj(prefix_embed).unsqueeze(1)
        embedding_cat = torch.cat((prefix_embed, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, attention_mask=None)
        return out

# Load CLIP and Tokenizer
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Dataset and Loader
train_dataset = RocoClipCapDataset("roco-dataset-master/test.csv", clip_preprocess, tokenizer)
val_dataset = RocoClipCapDataset("roco-dataset-master/val_split.csv", clip_preprocess, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model
model = ClipCaptionModel().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            prefix_embed = clip_model.encode_image(images).float()

        outputs = model(input_ids[:, :-1], batch["attention_mask"][:, :-1], prefix_embed)
        logits = outputs.logits[:, 1:].reshape(-1, logits.shape[-1])
        labels = input_ids[:, 1:].reshape(-1)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")

    # Validation loop (optional: add loss tracking like train)

# Save model
torch.save(model.state_dict(), "clipcap_roco.pth")
print("Training complete and model saved.")