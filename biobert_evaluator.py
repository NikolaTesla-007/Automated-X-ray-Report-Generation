# -*- coding: utf-8 -*-
"""
biobert_evaluator.py
author: V Subrahmamya Raghu Ram Kihore Parupudi
"""

# from google.colab import drive
# drive.mount('/content/drive')

#pip install transformers

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

def load_dataset(path):
  '''load and return dataset'''
  data = pd.read_csv(path)
  return data

def get_real_and_predicted_reports_as_lists(data,cols=("target_text","improved_report")):
   '''
   convert the given cols to lists and return
   '''
   real_reports = data[cols[0]].tolist()   
   predicted_reports = data[cols[1]].tolist()
   return real_reports, predicted_reports

def tokenize_sentences(reports, pad=True, truncate=True):
   '''create tokens from reports'''
   return tokenizer(reports, padding=pad, truncation=truncate, return_tensors="pt")

def generate_embeddingd(real_tokens,predicted_tokens,real_dims=1,pred_dims=1):
   '''generate embeddings from sentence tokens'''
   with torch.no_grad():
    real_embeddings = model(**real_tokens).last_hidden_state.mean(dim=real_dims)
    predicted_embeddings = model(**predicted_tokens).last_hidden_state.mean(dim=pred_dims)
    return real_embeddings, predicted_embeddings

if __name__ == "__main__":
    # Load the BioBERT model for semantic similarity
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Load the CSV file
    data = load_dataset("/content/drive/Shared drives/DGM/codes/t5_pred_v2.csv")

    # Extract the real and prediction columns
    real_reports,predicted_reports = get_real_and_predicted_reports_as_lists(data,cols=("target_text","improved_report"))

    # Tokenize the input sentences
    tokenized_real = tokenize_sentences(real_reports)
    tokenized_predicted = tokenize_sentences(predicted_reports)

    # Generate embeddings
    real_embeddings,predicted_embeddings = generate_embeddingd(tokenized_real,tokenized_predicted,real_dims=1,pred_dims=1)

    # Calculate cosine similarity
    similarities = cosine_similarity(real_embeddings, predicted_embeddings)

    # Add similarity scores to the DataFrame
    data["improved_similarity"] = similarities.diagonal()

    # save the data
    data.to_csv("/content/drive/Shared drives/DGM/codes/bio.csv")