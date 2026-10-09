import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# The model can be swapped here everytime for training
df = pd.read_csv("data/healthbench_probe/gemma_eval_judge.csv")
df = df.dropna(subset=["rubric_text", "binary_label"]).reset_index(drop=True)
df["rubric_text"] = df["rubric_text"].astype(str)
df["binary_label"] = df["binary_label"].astype(int)

texts = df["rubric_text"].tolist()
labels = df["binary_label"].tolist()

print("\nClass Distribution:")
print(df["binary_label"].value_counts(normalize=True))

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print("\n" + "="*60)
    print(f"FOLD {fold+1}")
    print("="*60)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("\nClass Weights:")
    print(class_weights)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=256
    )
    train_dataset = ClinicalDataset(train_encodings, train_labels)
    val_dataset = ClinicalDataset(val_encodings, val_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(model.device)
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=f"checkpoints/cv_fold_{fold+1}",
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    results = trainer.evaluate()
    print("\nValidation Results:")
    print(results)

    predictions_output = trainer.predict(val_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels_fold = predictions_output.label_ids

    print("\nClassification Report:\n")
    print(classification_report(labels_fold, preds))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(labels_fold, preds))

    all_fold_metrics.append({
        "fold": fold + 1,
        "accuracy": results["eval_accuracy"],
        "f1": results["eval_f1"],
        "precision": results["eval_precision"],
        "recall": results["eval_recall"],
        "balanced_accuracy": results["eval_balanced_accuracy"]
    })

results_df = pd.DataFrame(all_fold_metrics)
print("\n" + "="*60)
print("CROSS VALIDATION RESULTS")
print("="*60)
print(results_df)
print("\nAVERAGE METRICS:\n")
print(results_df.mean(numeric_only=True))

os.makedirs("results/probe", exist_ok=True)
output_file = "results/probe/cross_validation_results_eval.txt"
with open(output_file, "w") as f:
    f.write("="*70 + "\n")
    f.write("5-FOLD CROSS VALIDATION RESULTS\n")
    f.write("="*70 + "\n\n")

    for fold_result in all_fold_metrics:
        f.write(f"Fold {fold_result['fold']}\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy           : {fold_result['accuracy']:.4f}\n")
        f.write(f"F1 Score           : {fold_result['f1']:.4f}\n")
        f.write(f"Precision          : {fold_result['precision']:.4f}\n")
        f.write(f"Recall             : {fold_result['recall']:.4f}\n")
        f.write(f"Balanced Accuracy  : {fold_result['balanced_accuracy']:.4f}\n")

        f.write("\n")

    results_df = pd.DataFrame(all_fold_metrics)
    mean_metrics = results_df.mean(numeric_only=True)
    std_metrics = results_df.std(numeric_only=True)

    f.write("="*70 + "\n")
    f.write("AVERAGE METRICS ACROSS FOLDS\n")
    f.write("="*70 + "\n\n")

    f.write(
        f"Accuracy           : "
        f"{mean_metrics['accuracy']:.4f} "
        f"+/- {std_metrics['accuracy']:.4f}\n"
    )

    f.write(
        f"F1 Score           : "
        f"{mean_metrics['f1']:.4f} "
        f"+/- {std_metrics['f1']:.4f}\n"
    )

    f.write(
        f"Precision          : "
        f"{mean_metrics['precision']:.4f} "
        f"+/- {std_metrics['precision']:.4f}\n"
    )

    f.write(
        f"Recall             : "
        f"{mean_metrics['recall']:.4f} "
        f"+/- {std_metrics['recall']:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy  : "
        f"{mean_metrics['balanced_accuracy']:.4f} "
        f"+/- {std_metrics['balanced_accuracy']:.4f}\n"
    )

print(f"\nCross-validation results saved to: {output_file}")