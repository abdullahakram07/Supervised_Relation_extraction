import os
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LukeTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import LukeForEntityPairClassification, AdamW
import pytorch_lightning as pl
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")
import pickle
import requests, zipfile, io
import gdown
import numpy as np

app = FastAPI()
# Load the pre-trained model
model_path = "/content/checkpoint.ckpt"
def download_model(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
class LUKE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=len(label2id))

    def forward(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids,
                             entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        return outputs

    def common_step(self, batch, batch_idx):
        labels = batch['label']
        del batch['label']
        outputs = self(**batch)
        logits = outputs.logits

        criterion = torch.nn.CrossEntropyLoss() # multi-class classification
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/batch['input_ids'].shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader

class Item(BaseModel):
    text: str
    entity_spans: str


def load_model(model_path):
    loaded_model = LUKE.load_from_checkpoint(checkpoint_path=model_path)
    return loaded_model

def download_data():
    url = "https://www.dropbox.com/s/izi2x4sjohpzoot/relation_extraction_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


file_id = '1nL7EEHLbiuFHt7NX_Q-Zf1gS4jgc6cuq'
model_path = 'model_weights/checkpoint.ckpt'  # Change the output path as needed

if not os.path.exists('model_weights'):
    os.makedirs('model_weights')

download_model(file_id, model_path)
print(f"Model checkpoint downloaded and saved to: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download_data()
model = load_model(model_path)
df = pd.read_pickle("relation_extraction_dataset.pkl")
df.reset_index(drop=True, inplace=True)
id2label = dict()
for idx, label in enumerate(df.string_id.value_counts().index):
  id2label[idx] = label

@app.post("/predict")
def predict(item: Item):
    try:
        text = Item.text
        entity_spans = Item.entity_spans
        entity_spans = [tuple(x) for x in entity_spans]

        inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")

        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Ensure the model is also on the same device
        loaded_model.to(device)

        # Forward pass
        outputs = loaded_model.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        print("Sentence:", text)
        print("Predicted class idx:", id2label[predicted_class_idx])
        print("Confidence:", F.softmax(logits, -1).max().item())

        # You can return the prediction as JSON
        return {"prediction": predicted_class_idx}

    except Exception as e:
        # Handle exceptions gracefully
        raise HTTPException(status_code=500, detail=str(e))
