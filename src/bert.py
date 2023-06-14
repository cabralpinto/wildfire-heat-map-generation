import logging
import shutil
import sys
import warnings
from functools import partialmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("always")
logging.disable(logging.ERROR)
tqdm.__init__ = partialmethod(tqdm.__init__, leave=False)
Path("metrics").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

# "fire/related", "neuralmind/bert-[base|large]-portuguese-cased"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
label, tokenizer, model = sys.argv[1:4]
name = f"{label}-{model.replace('/', '-').lower()}"
data = pd.read_excel("data/news.xlsx")
tokenizer = AutoTokenizer.from_pretrained(tokenizer)
inputs = tokenizer(
    data["text"].tolist(), padding="max_length", truncation=True, max_length=100
)
labels = data[label].unique()
fire = labels.argmax()
print(fire)
exit()
outputs = np.where(data[label].to_numpy()[:, None] == labels)[1]
dataset = Dataset.from_dict(inputs | {"label": outputs})
resume = Path(f"metrics/{name}.csv").exists()
metrics = pd.read_csv(f"metrics/{name}.csv") if resume else pd.DataFrame()
splits = StratifiedShuffleSplit(30, test_size=0.1, random_state=0).split(data, outputs)
for index, (train, test) in enumerate(tqdm(list(splits), leave=True)):
    if index < metrics.shape[0]:
        continue
    classifier = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=np.unique(outputs).size
    ).to(device)
    arguments = TrainingArguments(
        "checkpoint",
        num_train_epochs=3 + 3 * ("large" in model),
        optim="adamw_torch",
        **{
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        }
        if "large" in model
        else {},
    )
    trainer = Trainer(classifier, arguments, train_dataset=dataset.select(train))
    trainer.train()
    predictions = trainer.predict(dataset.select(test)).predictions.argmax(1) == fire
    metrics.loc[index, ["accuracy", "precision", "recall", "f1-score"]] = (
        accuracy_score(outputs[test] == fire, predictions),
        *precision_recall_fscore_support(
            outputs[test] == fire,
            predictions,
            average="binary",
            pos_label=True,
        ),
    )[:4]
    metrics.loc[index] *= 100
    if (
        metrics.shape[0] == 1
        or metrics.at[index, "f1-score"] > metrics["f1-score"][:index].max()
    ):
        trainer.save_model(f"models/{name}")
    metrics.to_csv(f"metrics/{name}.csv", index=False)
    shutil.rmtree("checkpoint", ignore_errors=True)
    shutil.rmtree("mlruns", ignore_errors=True)
print(metrics.mean().combine(metrics.sem(), "{0:.1f} ± {1:.1f}".format).to_string())

# remove warnings and verbose output
