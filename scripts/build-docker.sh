MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    mlflow models build-docker \
        -m "models:/pytorch_simplenn_mnist/1" \
        -n my-docker-image \
        --enable-mlserver