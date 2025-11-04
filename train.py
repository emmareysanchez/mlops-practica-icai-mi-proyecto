import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow

# Carga
df = pd.read_csv("data/processed/iris_full.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Experimento MLflow
mlflow.set_experiment("Iris-MLflow-LogReg")

with mlflow.start_run():
    # Hiperparámetros
    params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "multinomial",
        "random_state": 42,
    }

    # Modelo: StandardScaler + LogisticRegression (multiclase)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver=params["solver"],
                max_iter=params["max_iter"],
                multi_class=params["multi_class"],
                random_state=params["random_state"],
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # Log a MLflow (sin guardar pkl)
    for k, v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1m)

    print(f"accuracy={acc:.4f}  f1_macro={f1m:.4f}")
