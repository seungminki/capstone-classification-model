import torch
import pandas as pd
from tqdm import tqdm
import json

from transformers import BertTokenizer
from model.bert import BertForMultiLabelClassification

from settings import TRAINED_MODEL_LOCAL_PATH


class MultiLabelPredictor:
    def __init__(self, model, tokenizer, label_names, threshold=0.5, device=None):
        self.model = model.to(device or self._get_device())
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.threshold = threshold
        self.device = device or self._get_device()

        self.model.eval()

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, dataloader, return_probs=False):
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = outputs["logits"].cpu().numpy()  # sigmoid 후 확률값

                pred_labels = (probs >= self.threshold).astype(int)

                all_probs.extend(probs)
                all_preds.extend(pred_labels)

        pred_df = pd.DataFrame(all_preds, columns=self.label_names)
        if return_probs:
            prob_df = pd.DataFrame(
                all_probs, columns=[f"{l}_prob" for l in self.label_names]
            )
            return pd.concat([pred_df, prob_df], axis=1)

        return pred_df

    def predict_from_texts(
        self, texts, return_probs=False, batch_size=16, max_length=128
    ):
        self.model.eval()
        all_preds = []
        all_probs = []

        # Tokenization
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Batch-wise prediction
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            input_batch = input_ids[i : i + batch_size]
            mask_batch = attention_mask[i : i + batch_size]

            with torch.no_grad():
                outputs = self.model(input_ids=input_batch, attention_mask=mask_batch)
                probs = outputs["logits"].cpu().numpy()
                pred_labels = (probs >= self.threshold).astype(int)

                all_probs.extend(probs)
                all_preds.extend(pred_labels)

        pred_df = pd.DataFrame(all_preds, columns=self.label_names)
        if return_probs:
            prob_df = pd.DataFrame(
                all_probs, columns=[f"{l}_prob" for l in self.label_names]
            )
            return pd.concat([pred_df, prob_df], axis=1)

        return pred_df

    def save_predictions(self, dataloader, path="predictions.csv", return_probs=True):
        df = self.predict(dataloader, return_probs=return_probs)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ Predictions saved to {path}")


if __name__ == "__main__":

    with open("model/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    model = BertForMultiLabelClassification.from_pretrained(TRAINED_MODEL_LOCAL_PATH)
    tokenizer = BertTokenizer.from_pretrained(TRAINED_MODEL_LOCAL_PATH)
    model.eval()

    predictor = MultiLabelPredictor(
        model=model,
        tokenizer=tokenizer,
        label_names=labels,  # 예: ["연애", "취업", "감정", ...]
        threshold=0.5,
    )

    texts = [
        "시험기간 너무 힘들어요",
        "자취방 추천 부탁해요",
        "동아리 사람들 너무 좋아",
    ]  # 여러 건

    texts = [
        "두정 왜택시안잡힘 1교시 전공이라 과제물 개큰거 들고가야해서 택시타려는데 미친 왜 안잡히냐"  # 단건
    ]

    predictor = MultiLabelPredictor(model, tokenizer, label_names=labels, threshold=0.5)
    df = predictor.predict_from_texts(texts, return_probs=True)
    df.iloc[:, :20]
