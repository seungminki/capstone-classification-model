import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from torch.optim import AdamW

# from transformers import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import os

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    accuracy_score,
    classification_report,
)

from model.bert import MultiLabelDataset, BertForMultiLabelClassification
from settings import ST_MODEL_NAME, BATCHES_FILE_PATH, SAVE_MODEL_DIR


tokenizer = BertTokenizer.from_pretrained(ST_MODEL_NAME)

num_epoch = 3


def train():

    df = pd.read_csv(BATCHES_FILE_PATH)

    # X: 자연어 텍스트, y: 멀티 라벨 확률값
    X = df["text"].tolist()
    y = df.drop(columns=["board_id", "post_id", "text"]).values.astype(np.float32)
    labels = df.drop(columns=["board_id", "post_id", "text"]).columns.tolist()
    label_names = df.drop(columns=["board_id", "post_id", "text"]).columns.tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42  # 전체의 10%가 val
    )

    train_dataset = MultiLabelDataset(X_train, y_train)
    val_dataset = MultiLabelDataset(X_val, y_val)
    test_dataset = MultiLabelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForMultiLabelClassification.from_pretrained(
        ST_MODEL_NAME, num_labels=len(labels)
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    # optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Training
    model.train()
    for epoch in range(num_epoch):  # epoch
        total_loss = 0
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 👇 tqdm progress bar 업데이트
            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss=loss.item())

        print(
            f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(train_loader):.4f}"
        )
        micro_f1, hamming, subset_acc = evaluate(
            model, val_loader, device, label_names=label_names
        )
        print(f"Validation Micro-F1: {micro_f1:.4f}")
        print(f"Validation Hamming Loss: {hamming:.4f}")
        print(f"Validation Subset Acc: {subset_acc:.4f}")

    micro_f1, hamming, subset_acc = evaluate(
        model, test_loader, device, label_names=label_names
    )
    print(f"Final Test Micro-F1: {micro_f1:.4f}")
    print(f"Final Test Hamming Loss: {hamming:.4f}")
    print(f"Final Test Subset Acc: {subset_acc:.4f}")

    # Saved Model
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    print(f"Model and tokenizer saved to {SAVE_MODEL_DIR}")


def evaluate(model, dataloader, device, label_names, threshold=0.5):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs["logits"].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    y_score = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    y_true_binary = (y_true >= 0.5).astype(int)

    micro_f1 = f1_score(y_true_binary, y_pred, average="micro", zero_division=0)
    # macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    hamming = hamming_loss(y_true_binary, y_pred)
    subset_acc = accuracy_score(y_true_binary, y_pred)

    print(
        classification_report(
            y_true_binary, y_pred, target_names=label_names, zero_division=0
        )
    )

    # macro_auc = roc_auc_score(y_true, y_score, average="macro")  # 라벨별 AUC 평균
    micro_auc = roc_auc_score(
        y_true_binary, y_score, average="micro"
    )  # 전체 샘플 단위로 평균

    print(f"📊 AUC  Micro: {micro_auc:.4f}")

    return micro_f1, hamming, subset_acc
