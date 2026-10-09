import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

eval_df = pd.read_csv("data/healthbench_probe/gemma_eval_judge.csv")
hard_df = pd.read_csv("data/healthbench_probe/gemma_hard_judge.csv")
os.makedirs("results/probe/gemma", exist_ok=True)

def clean_df(df):
    df = df.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
    df["rubric_text"] = df["rubric_text"].astype(str)
    df["binary_label"] = df["binary_label"].astype(int)
    return df

eval_df = clean_df(eval_df)
hard_df = clean_df(hard_df)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ClinicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds)
    }

def create_dataset(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    return ClinicalDataset(encodings, labels)

def run_experiment(train_df, test_df, experiment_name):
    print("\n" + "="*80)
    print(experiment_name)
    print("="*80)

    train_texts = train_df["rubric_text"].tolist()
    train_labels = train_df["binary_label"].tolist()
    test_texts = test_df["rubric_text"].tolist()
    test_labels = test_df["binary_label"].tolist()

    train_dataset = create_dataset(train_texts, train_labels)
    test_dataset = create_dataset(test_texts, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"checkpoints/pubmedbert_gemma_{experiment_name}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.01,
        logging_steps=20,
        fp16=True,
        report_to="none",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    print("\nValidation Results:")
    print(results)
    print("\nClassification Report:\n")
    report = classification_report(labels, preds)
    print(report)
    print("\nConfusion Matrix:\n")
    cm = confusion_matrix(labels, preds)
    print(cm)

    with open(f"results/probe/gemma/1_{experiment_name}.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write(experiment_name + "\n")
        f.write("="*80 + "\n\n")
        f.write("METRICS\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write("CLASSIFICATION REPORT\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("CONFUSION MATRIX\n\n")
        f.write(str(cm))
    return results

def make_balanced(df):
    df_majority = df[df["binary_label"] == 0]
    df_minority = df[df["binary_label"] == 1]
    minority_size = len(df_minority)
    sampled_majority = df_majority.sample(n=minority_size, random_state=SEED)
    balanced_df = pd.concat(
        [sampled_majority, df_minority]
    ).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)
    return balanced_df

balanced_eval_df = make_balanced(eval_df)
balanced_hard_df = make_balanced(hard_df)

all_results = {}

all_results["eval_to_hard_weighted"] = run_experiment(
    train_df=eval_df,
    test_df=hard_df,
    experiment_name="eval_to_hard_weighted"
)

all_results["hard_to_eval_weighted"] = run_experiment(
    train_df=hard_df,
    test_df=eval_df,
    experiment_name="hard_to_eval_weighted"
)

all_results["eval_to_hard_balanced"] = run_experiment(
    train_df=balanced_eval_df,
    test_df=hard_df,
    experiment_name="eval_to_hard_balanced"
)

all_results["hard_to_eval_balanced"] = run_experiment(
    train_df=balanced_hard_df,
    test_df=eval_df,
    experiment_name="hard_to_eval_balanced"
)

summary_df = pd.DataFrame(all_results).T
print(summary_df)