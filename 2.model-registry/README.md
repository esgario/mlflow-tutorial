# MLFlow Model Registry

O model registry introduz alguns conceitos que facilitam a gestão do ciclo de vida de um modelo de machine learning com o MLFlow. Basicamente os modelos logados recebem um nome, uma versão e uma tag que indica o estágio atual do modelo por exemplo, `Staging`, `Production` ou `Archived`.

Além disso, o model registry fornece uma forma de mantermos o tracking da linhagem do modelo, ou seja, mantém um registro de qual experimento gerou determinado modelo.

## Passo a passo

A utilização do Model Registry requer que o armazenamento seja feito em um banco de dados. Para isso, utilizaremos o SQLite para a execução deste tutorial, portanto, verifique se o SQLite está instalado no seu sistema.

### 1. Treinando um modelo e logando os resultados no SQLite.

Para treinar o modelo utilizaremos o mesmo script do **módulo 1**. Com a diferença de que agora nós exportaremos uma variável de ambiente com a URI do nosso banco de dados.

```bash
$ export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
$ python ../1.tracking/train.py 
```

Para visualizar o resultado devemos iniciar a interface do MLFlow, neste caso também devemos passar a URI do nosso banco.

```bash
$ mlflow ui --backend-store-uri sqlite:///mlflow.db --serve-artifacts
```

obs: use `mlflow ui` para ambiente de desenvolvimento e `mlflow serve` para ambiente de produção.


### 2. Registrando modelo

Após o modelo ter sido logado, nós podemos criar um registro do modelo no Model Registry. Esse registro pode ser feito programaticamente ou via interface de usuário.

Acesse [este link](https://mlflow.org/docs/latest/model-registry.html#ui-workflow) para ver um exemplo de como registrar um modelo via interface de usuário.


### 3. Carregando modelo

Para carregar o modelo precisamos do nome e da versão do modelo registrado. Com esses dados em mãos basta chamar o método `load_model` do Model Registry. Por exemplo:

```python
import torch
import mlflow.pytorch

model_name = "pytorch_simplenn_mnist"
model_version = 1

model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
```

Rode o script `predict.py` para testar o carregamento do modelo.

```bash
$ export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
$ python predict.py
```

Existem diferentes formas de se passar a URI do modelo, por exemplo, você pode passar uma URI do S3 ou então o caminho completo da pasta do modelo. Para mais exemplos de URI veja [este link](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.load_model).

### 4. Servindo modelo

Por fim podemos servir diretamente um modelo do Model Registry da seguinte forma:

```bash
$ export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
$ mlflow models serve -m "models:/pytorch_simplenn_mnist/1" --env-manager=virtualenv --enable-mlserver
```