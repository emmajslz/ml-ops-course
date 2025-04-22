from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class PCOSFlow(FlowSpec):

    data_path = Parameter("data-path", help="Path to PCOS training data CSV")

    @step
    def start(self):
        print("Flow started")
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('metaflow-experiment')
        self.next(self.load_data)

    @step
    def load_data(self):
        self.pcos = pd.read_csv(self.data_path)
        self.pcos = self.pcos.set_index("ID")
        self.next(self.train_models)

    @step
    def train_models(self):
        df = self.pcos
        X = df.drop(columns="PCOS")
        y = df["PCOS"]

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, None]}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_acc = 0
        best_model = None
        i = 0

        for params in ParameterGrid(param_grid):
            clf = Pipeline([
                ("preprocess", preprocessor),
                ("clf", RandomForestClassifier(**params, random_state=42))
            ])
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"Params: {params}, Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = params

            with mlflow.start_run(run_name=f"Metaflow - Combo {i}"):
                mlflow.log_params(params)
                mlflow.log_metric("test_accuracy", acc)
            i += 1

        self.model = best_model
        self.best_acc = best_acc
        self.best_params = best_params
        self.next(self.register_model)

    @step
    def register_model(self):
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model",
                                     registered_model_name="PCOS_Model")
        print("Model registered with MLflow.")
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed.")

if __name__ == '__main__':
    PCOSFlow()