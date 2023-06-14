import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm

Path("metrics").mkdir(parents=True, exist_ok=True)

label = sys.argv[1]  # "fire" or "related"
data = pd.read_excel("data/news.xlsx")
name = f"{label}-svm"
resume = Path(f"metrics/{name}.csv").exists()
inputs = data["text"]
outputs = data[label]
fire = outputs.max()
metrics = pd.read_csv(f"metrics/{name}.csv") if resume else pd.DataFrame()
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), SVC())
hyperparameters = {
    "svc__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "svc__gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "svc__kernel": ["linear", "rbf"],
}
grid = GridSearchCV(pipeline, hyperparameters, scoring="f1_macro", verbose=1)
pipeline: Pipeline = grid.fit(inputs, outputs).best_estimator_
splits = StratifiedShuffleSplit(30, test_size=0.1, random_state=0).split(data, outputs)
for index, (train, test) in enumerate(tqdm(list(splits))):
    if index < metrics.shape[0]:
        continue
    predictions = pipeline.fit(inputs[train], outputs[train]).predict(inputs[test])
    metrics.loc[index, ["accuracy", "precision", "recall", "f1-score"]] = (
        accuracy_score(outputs[test] == fire, predictions == fire),
        *precision_recall_fscore_support(
            outputs[test] == fire,
            predictions == fire,
            average="binary",
            pos_label=True,
        ),
    )[:4]
    metrics.loc[index] *= 100
    metrics.to_csv(f"metrics/{name}.csv", index=False)
print(metrics.mean().combine(metrics.sem(), "{0:.1f} ± {1:.1f}".format).to_string())
