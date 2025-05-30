import yaml
import mlflow

from download_data import download_and_prepare_data
from train import train_pipeline
from evaluate import evaluate_model

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with mlflow.start_run():

        def log_params_flat(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    log_params_flat(v, prefix + k + ".")
                else:
                    mlflow.log_param(prefix + k, v)

        log_params_flat(config)

        images_dir, train_df, val_df, test_df = download_and_prepare_data(config)

        model, _ = train_pipeline(config, images_dir, train_df, val_df)

        evaluate_model(model, images_dir, test_df, config)

        mlflow.pytorch.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()