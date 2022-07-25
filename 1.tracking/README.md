# MLFlow Tracking

O MLFlow tracking é um componente que permite ao usuário criar e gerenciar experimentos. Fornece uma API e uma interface de usuário que nos permite salvar e visualizar métricas, parâmetros, modelos e artefatos.

Os resultados dos experimentos realizados com MLFlow são armazenados localmente ou em um servidor remoto. Por padrão, os resultados são armazenados localmente em arquivos dentro do diretório `mlruns`.

Para mais informações, consulte a [documentação](https://mlflow.org/docs/latest/tracking.html).

## Passo a passo

Nosso código é composto por três arquivos:

* dataset.py: Contém a função que carrega os dataloaders que servirão para treinar e testar o modelo.
* model.py: Contém a função que cria o modelo.
* train.py: Script principal com a função de treino do modelo. É dentro deste script que logamos as métricas e os parâmetros com o MLFlow tracking.

Dentro do train.py você encontrará o seguinte trecho de código:

```python
with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_acc, test_loss = test(test_dataloader, model, loss_fn)

        mlflow.log_metric("test_acc", test_acc, step=t)
        mlflow.log_metric("test_loss", test_loss, step=t)

        if test_acc > best_acc:
            best_acc = test_acc
            mlflow.log_metric("best_acc", best_acc, step=t)
            mlflow.pytorch.log_model(model, "model")
```

Neste trecho de código você pode observar que os parâmetros são logados com o método `log_param`, as métricas são logadas com o método `log_metric` e o modelo é salvo com o método `log_model`. Além destes, temos outros métodos muito úteis que podem ser utilizados por exemplo, `log_artifact` para salvar arquivos no diretório `artifacts` e `log_image` para salvar imagens. Para mais informações, consulte a [documentação](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html).

### 1. Treinando um modelo

Para treinarmos um modelo novo basta rodar o script train.py:

```bash
$ python train.py
```

obs: Você deve executar o script de dentro do diretório `1.tracking`.

### 2. Visualizando os resultados

Após o treino, o script irá criar uma pasta chamada `mlruns` no diretório atual. Dentro desta pasta serão armazenados os arquivos de log do experimento. O MLFlow disponibiliza uma interface de usuário que nos permite visualizar os resultados do experimento. Para isso, basta abrir o terminal e digitar:

```bash
$ mlflow ui
```

Agora em um navegador podemos acessar a interface de usuário do MLFlow no endereço [http://127.0.0.1:5000](http://127.0.0.1:5000).