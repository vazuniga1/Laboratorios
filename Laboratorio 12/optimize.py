import os
import pickle
import optuna
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model

# Se cargan los datos
df = pd.read_csv('water_potability.csv')
X, y = df.drop("Potability", axis=1), df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar el directorio para los artefactos
artifacts_dir = "mlruns/artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

def optimize_model(trial):
      params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": trial.suggest_loguniform('eta', 0.01, 0.1),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            }

      # Se crea  el modelo
      model = xgb.XGBClassifier(**params)

      name_run = f"XGBoost con lr {params['eta']}"
      # Registrar los resultados en MLflow
      with mlflow.start_run(run_name = name_run):
        # se entrena el modelo
        model.fit(X_train, y_train)

        # Se predice y calcula el f1
        y_pred = model.predict(X_test)
        valid_f1 = f1_score(y_test, y_pred, average="weighted")
        
        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", valid_f1)

      return valid_f1

def main():
    mlflow.set_experiment("XGBoost Potability Experiment")

    # Se crea un estudio de Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(optimize_model, n_trials=300)

    # Se devuelve el mejor modelo
    best_model = study.best_trial
    best_params = best_model.params
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Se guarda el mejor modelo
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Se guarda gráficos de Optuna
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f"{plots_dir}/optimization_history.png")
    plt.close()

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f"{plots_dir}/param_importance.png")
    plt.close()

    # Se loggean gráficos en MLflow
    mlflow.log_artifact(f"{plots_dir}/optimization_history.png")
    mlflow.log_artifact(f"{plots_dir}/param_importance.png")

    # Se guarda la importancia de características
    xgb.plot_importance(best_model, importance_type="weight")
    plt.savefig(f"{artifacts_dir}/feature_importance.png")
    plt.close()

    # Se registran las versiones de las librerías
    mlflow.log_artifacts(artifacts_dir)
    mlflow.set_tag("mlflow.version", mlflow.__version__)
    mlflow.set_tag("optuna.version", optuna.__version__)
    mlflow.set_tag("xgboost.version", xgb.__version__)

    return best_model


if __name__ == "__main__":
    best_model = main()