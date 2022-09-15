# Treinamento e validação de modelos preditivos com AM

## Download do projeto

Clone o [repositório](https://github.com/rjribeiro/machine_learning)
e mude para pasta kfold

```git clone https://github.com/rjribeiro/machine_learning.git && cd kfold```


## Configuração do ambiente


### Opção 1: [Docker](https://docs.docker.com/get-started/)
Execute ```bash run_jupyter.sh```

Esse comando irá buildar e executar uma imagem contendo docker contendo o jupyter. 
O diretório do projeto será todo mapeado para dentro de um container. 

    
### Opção 2: [Conda](https://docs.conda.io/en/latest/)
Execute ```conda env create -f conda_environment.yml python=3.10```
