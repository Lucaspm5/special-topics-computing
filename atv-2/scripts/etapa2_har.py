#!/usr/bin/env python3
"""
scripts/etapa2_har.py
Etapa 2 — selecionar mean/std, mapear atividades, escalar (StandardScaler fit no train),
salvar train_processed.csv, test_processed.csv e arrays .npy em data/processed/.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

# --- localizar dataset (robusto) ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
base = project_root / "data" / "raw" / "UCI HAR Dataset"
if not base.exists():
    found = list((project_root / "data" / "raw").rglob("features.txt"))
    if found:
        base = found[0].parent
    else:
        print("ERRO: não encontrei o UCI HAR Dataset em data/raw. Extraia-o em data/raw/")
        sys.exit(1)

# --- carregar feature names (assegurar unicidade, como na etapa1) ---
features = pd.read_csv(base / "features.txt", sep=r"\s+", header=None, names=["idx", "feature"])
feature_names = features["feature"].tolist()
seen = {}
unique_feature_names = []
for f in feature_names:
    if f not in seen:
        seen[f] = 0
        unique_feature_names.append(f)
    else:
        seen[f] += 1
        unique_feature_names.append(f"{f}_{seen[f]}")
feature_names = unique_feature_names

# --- selecionar features mean() ou std(), excluindo meanFreq ---
selected_mask = [("mean()" in f or "std()" in f) and ("meanFreq" not in f) for f in feature_names]
selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]

print(f"Features totais: {len(feature_names)}  → selecionadas (mean/std): {len(selected_features)}")

# --- função para ler split ---
def read_split(split):
    X = pd.read_csv(base / split / f"X_{split}.txt", sep=r"\s+", header=None, names=feature_names)
    y = pd.read_csv(base / split / f"y_{split}.txt", header=None, names=["activity_id"])
    subj = pd.read_csv(base / split / f"subject_{split}.txt", header=None, names=["subject"])
    return X, y, subj

X_train, y_train, subj_train = read_split("train")
X_test, y_test, subj_test = read_split("test")

# --- subset de features escolhidas ---
X_train_sel = X_train[selected_features].copy()
X_test_sel = X_test[selected_features].copy()

# --- escalar: fit no train, transformar train/test ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# --- mapear ids para nomes de atividade ---
activity_map_df = pd.read_csv(base / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_map_df["id"], activity_map_df["activity"]))
y_train["activity"] = y_train["activity_id"].map(activity_map)
y_test["activity"] = y_test["activity_id"].map(activity_map)

# --- salvar em data/processed ---
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# montar DataFrames finais (subject, activity_id, activity, features...)
train_df = pd.DataFrame(X_train_scaled, columns=selected_features)
train_df.insert(0, "subject", subj_train["subject"].values)
train_df.insert(1, "activity_id", y_train["activity_id"].values)
train_df.insert(2, "activity", y_train["activity"].values)
train_df.to_csv(processed_dir / "train_processed.csv", index=False)

test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
test_df.insert(0, "subject", subj_test["subject"].values)
test_df.insert(1, "activity_id", y_test["activity_id"].values)
test_df.insert(2, "activity", y_test["activity"].values)
test_df.to_csv(processed_dir / "test_processed.csv", index=False)

# salvar arrays numpy pra modelos rápidos
np.save(processed_dir / "X_train.npy", X_train_scaled)
np.save(processed_dir / "X_test.npy", X_test_scaled)
np.save(processed_dir / "y_train.npy", y_train["activity_id"].values)
np.save(processed_dir / "y_test.npy", y_test["activity_id"].values)

# salvar lista de features selecionadas
with open(processed_dir / "selected_features.txt", "w") as f:
    for feat in selected_features:
        f.write(feat + "\n")

print("Processamento concluído.")
print("Arquivos salvos em:", processed_dir)
print("train_processed.csv shape:", train_df.shape)
print("test_processed.csv shape:", test_df.shape)
