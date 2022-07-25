MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
mlflow models serve -m "models:/pytorch_simplenn_mnist/3" --env-manager=local --enable-mlserver --port 6000