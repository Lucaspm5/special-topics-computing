#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# caminho base dos dados
base = Path("../data/raw/UCI HAR Dataset")

# carregar nomes das features
features = pd.read_csv(base / "features.txt", sep=r"\s+", header=None, names=["idx", "feature"])
feature_names = features["feature"].tolist()

# garantir nomes únicos (corrige erro de duplicatas)
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

# atividades
activity_map = pd.read_csv(
    base / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
).set_index("id")["activity"].to_dict()

# treino
X_train = pd.read_csv(
    base / "train" / "X_train.txt",
    sep=r"\s+",
    header=None,
    names=feature_names
)
y_train = pd.read_csv(base / "train" / "y_train.txt", header=None, names=["activity_id"])
subject_train = pd.read_csv(base / "train" / "subject_train.txt", header=None, names=["subject"])

# teste
X_test = pd.read_csv(
    base / "test" / "X_test.txt",
    sep=r"\s+",
    header=None,
    names=feature_names
)
y_test = pd.read_csv(base / "test" / "y_test.txt", header=None, names=["activity_id"])
subject_test = pd.read_csv(base / "test" / "subject_test.txt", header=None, names=["subject"])

# concatenar
X = pd.concat([X_train, X_test], ignore_index=True)
y = pd.concat([y_train, y_test], ignore_index=True)
subjects = pd.concat([subject_train, subject_test], ignore_index=True)

# saída
print("=== Informações do dataset ===")
print("Número total de amostras:", X.shape[0])
print("Número de atributos (features):", X.shape[1])
print("Número de classes:", y["activity_id"].nunique())
print()

print("=== Classes ===")
for k, v in activity_map.items():
    print(f"{k}: {v}")
