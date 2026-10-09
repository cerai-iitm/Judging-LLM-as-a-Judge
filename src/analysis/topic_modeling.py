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

from bertopic import BERTopic

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

topic_model = BERTopic(
    embedding_model=None,
    calculate_probabilities=True,
    verbose=True
)

topics1, probs1 = topic_model.fit_transform(df["rubric_text"].tolist(), embeddings1)
topic_model.get_topic_info().head(20)
topic_info_df1 = topic_model.get_topic_info()
topic_model.save("bertopic_model1")

topics2, probs2 = topic_model.fit_transform(df2["rubric_text"].tolist(), embeddings2)
topic_model.get_topic_info().head(20)
topic_info_df2 = topic_model.get_topic_info()
topic_model.save("bertopic_model2")

# Labeled Conditioned BERTopic
texts_0 = df[df["binary_label"] == 0]["rubric_text"].tolist()
texts_1 = df[df["binary_label"] == 1]["rubric_text"].tolist()

topic_model_0 = BERTopic(verbose=True)
topics_0, probs_0 = topic_model_0.fit_transform(texts_0)

topic_model_1 = BERTopic(verbose=True)
topics_1, probs_1 = topic_model_1.fit_transform(texts_1)

info_0 = topic_model_0.get_topic_info()
info_1 = topic_model_1.get_topic_info()

print("\nCLASS 0 TOPICS\n")
print(info_0.head(20))

print("\nCLASS 1 TOPICS\n")
print(info_1.head(20))


texts_0 = df2[df2["binary_label"] == 0]["rubric_text"].tolist()
texts_1 = df2[df2["binary_label"] == 1]["rubric_text"].tolist()

topic_model_0 = BERTopic(verbose=True)
topics_0, probs_0 = topic_model_0.fit_transform(texts_0)

topic_model_1 = BERTopic(verbose=True)
topics_1, probs_1 = topic_model_1.fit_transform(texts_1)

info_0 = topic_model_0.get_topic_info()
info_1 = topic_model_1.get_topic_info()

print("\nCLASS 0 TOPICS\n")
print(info_0.head(20))

print("\nCLASS 1 TOPICS\n")
print(info_1.head(20))