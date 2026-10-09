import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns
import time, ast
import re, json, random, math
from datetime import datetime, timedelta
from tqdm import tqdm
import os, pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from collections import defaultdict

SAVE_PATH = "checkpoints/pubmedbert_gemma_eval"
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv("data/healthbench_probe/gemma_eval_judge.csv")
df = df.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df["rubric_text"] = df["rubric_text"].astype(str)
df["binary_label"] = df["binary_label"].astype(int)
print("\nClass Distribution:")
print(df["binary_label"].value_counts(normalize=True))
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["rubric_text"].tolist(),
    df["binary_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["binary_label"]
    )

df2 = pd.read_csv("data/healthbench_probe/gemma_hard_judge.csv")
df2 = df2.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df2["rubric_text"] = df2["rubric_text"].astype(str)
df2["binary_label"] = df2["binary_label"].astype(int)

print("\nClass Distribution:")
print(df2["binary_label"].value_counts(normalize=True))

train_texts2, val_texts2, train_labels2, val_labels2 = train_test_split(
    df2["rubric_text"].tolist(),
    df2["binary_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df2["binary_label"]
)

def forward_func(input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    return outputs.logits

lig = LayerIntegratedGradients(
    forward_func,
    model.bert.embeddings
)

def interpret_rubric(text, target_class=None):

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    input_ids = encoding["input_ids"].to("cuda")
    attention_mask = encoding["attention_mask"].to("cuda")

    # Prediction
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    probs = torch.softmax(outputs.logits, dim=-1)

    pred_class = torch.argmax(probs, dim=-1).item()

    if target_class is None:
        target_class = pred_class

    # Baseline tokens = PAD
    baseline_input_ids = torch.zeros_like(input_ids).to("cuda")

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        return_convergence_delta=True
    )

    # Sum across embedding dimension
    attributions = attributions.sum(dim=-1).squeeze(0)

    # Normalize
    attributions = attributions / torch.norm(attributions)

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)
    )

    scores = attributions.detach().cpu().numpy()

    return {
        "tokens": tokens,
        "scores": scores,
        "prediction": pred_class,
        "probabilities": probs.detach().cpu().numpy()
    }
    
def show_top_tokens(result, top_k=10):

    token_scores = []

    for tok, score in zip(result["tokens"], result["scores"]):

        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        token_scores.append((tok, float(score)))

    token_scores = sorted(
        token_scores,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\nPrediction: {result['prediction']}\n")

    print("Top influential tokens:\n")

    for tok, score in token_scores[:top_k]:
        print(f"{tok:20s} {score:.4f}")
        
global_token_scores = defaultdict(list)
global_token_scores2 = defaultdict(list)

def aggregate_attributions(texts):
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"Processed {idx}")
        try:
            result = interpret_rubric(text)
            for tok, score in zip(result["tokens"], result["scores"]):
                if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                tok = tok.lower()
                global_token_scores[tok].append(abs(float(score)))
        except Exception as e:
            print(e)

    avg_scores = {}
    for tok, scores in global_token_scores.items():
        avg_scores[tok] = np.mean(scores)
    sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def aggregate_attributions2(texts):
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"Processed {idx}")
        try:
            result = interpret_rubric(text)
            for tok, score in zip(result["tokens"], result["scores"]):
                if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                tok = tok.lower()
                global_token_scores2[tok].append(abs(float(score)))
        except Exception as e:
            print(e)

    avg_scores = {}
    for tok, scores in global_token_scores2.items():
        avg_scores[tok] = np.mean(scores)
    sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

positive_scores = defaultdict(list)
negative_scores = defaultdict(list)
positive_scores2 = defaultdict(list)
negative_scores2 = defaultdict(list)

def aggregate_by_class(texts):
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"Processed {idx}")
        result = interpret_rubric(text)
        pred = result["prediction"]

        for tok, score in zip(result["tokens"], result["scores"]):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            tok = tok.lower()
            if pred == 1:
                positive_scores[tok].append(abs(float(score)))
            else:
                negative_scores[tok].append(abs(float(score)))
                
def aggregate_by_class2(texts):
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"Processed {idx}")
        result = interpret_rubric(text)
        pred = result["prediction"]

        for tok, score in zip(result["tokens"], result["scores"]):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            tok = tok.lower()
            if pred == 1:
                positive_scores2[tok].append(abs(float(score)))
            else:
                negative_scores2[tok].append(abs(float(score)))
                
# Obtain globally relevant tokens across all validation samples
top_tokens1 = aggregate_attributions(val_texts)
top_tokens2 = aggregate_attributions2(val_texts2)

# Obtain class-specific influential tokens
aggregate_by_class(val_texts)
aggregate_by_class2(val_texts2)