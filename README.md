Projeto: Classificador de Cenas Urbanas vs. Naturais

```
Bruno N. de Paula Silva – G793030
Heitor Fernandez Soares – G7853G7
Guilherme Silva Pereira da Rocha - F352463
Arthur Araújo Pereira – G87EDA1
```


Projeto: https://github.com/Bruno70i/franca-np2

Disciplina: Processamento de Imagens e Visão Computacional - Prof. França Instituição: UNIP (Universidade Paulista)

Este repositório contém o código-fonte e a documentação do trabalho "Classificador de Cenas Urbanas vs. Naturais", que utiliza Redes Neurais Convolucionais (CNNs) e Transfer Learning para classificar imagens.
O objetivo técnico foi construir um modelo de alta precisão, e o objetivo analítico foi documentar o uso e as limitações de I.As generativas no processo de desenvolvimento.


# Sumário

1. Tecnologias Utilizadas
2. Estrutura do Projeto
3. Instalação e Configuração do Ambiente
4. Configuração do Dataset
5. Como Executar
6. Resultados Obtidos
 
# 1. Tecnologias Utilizadas

- Python 3.11.4
- Visual Studio Code: Ambiente de desenvolvimento.
- TensorFlow / Keras: Framework principal para a construção e treinamento do modelo.
- MobileNetV2: Arquitetura de CNN pré-treinada utilizada para o Transfer Learning.
- Scikit-learn: Utilizado para o cálculo das métricas de avaliação (Matriz de Confusão, Relatório de Classificação, Curva ROC).
- Matplotlib & Seaborn: Utilizados para a plotagem e visualização dos resultados.
- NumPy: Para manipulação de arrays.

 
# 2. Estrutura do Projeto
A estrutura de pastas utilizada foi projetada para ser lida automaticamente pelo ImageDataGenerator do Keras.
 
O modelo precisa de três conjuntos de dados para aprender corretamente, “train, validation e test”:
Para o modelo aprender corretamente, ele utiliza três conjuntos de dados com funções distintas:

1.	data/train/ (O Livro de Estudo): É o conjunto principal e maior, usado exclusivamente para o treinamento. É aqui que o modelo analisa os padrões, texturas e formas para aprender o que define "natural" e "urbana".

2.	data/validation/ (O Simulado): É um conjunto menor que o modelo não usa para aprender. Ele é usado ao final de cada ciclo (Época) como uma "mini-prova" para monitorar a saúde do modelo e detectar overfitting (memorização).

3.	data/test/ (A Prova Final): É um conjunto totalmente separado, usado apenas uma vez no final de todo o processo. Ele fornece a "nota" final e imparcial do modelo, usada para gerar a Matriz de Confusão e a acurácia de 99%.
Além disso, as sub-pastas .../natural/ e .../urbana/ são cruciais para a rotulagem automática. O Keras não sabe o que é uma imagem; ele aprende o rótulo pelo nome da pasta. Ele automaticamente define natural/ como Classe 0 e urbana/ como Classe 1 (por ordem alfabética), criando os pares (imagem, rótulo) necessários para o aprendizado.
Em resumo, a estrutura de pastas serve para organizar o processo (treino, simulado, prova) e automatizar a rotulagem (natural, urbana). 


# 3. Instalação e Configuração do Ambiente
Este projeto utiliza um ambiente virtual (.venv) para gerenciar as dependências.

Crie e ative o ambiente virtual:

# Criar o ambiente
``` 
python -m venv .venv
```

# Ativar no Windows (PowerShell)
```
.venv\Scripts\Activate.ps1 
```


Crie o arquivo requirements.txt: Copie e cole o conteúdo abaixo em um arquivo requirements.txt na raiz do projeto.

```
Tensorflow
Matplotlib
Seaborn
scikit-learn
numpy
```

Instale as dependências:

``` 
pip install -r requirements.txt
```


 
# 4. Configuração do Dataset
O modelo foi treinado com o dataset "Intel Image Classification", disponível no Kaggle.
Link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
O dataset do Kaggle é dividido em 6 classes (buildings, forest, glacier, mountain, sea, street). Para este trabalho, elas devem ser agrupadas manualmente nas pastas natural e urbana dentro da estrutura data/.

Observação Importante: Se você clonar direto do github: https://github.com/Bruno70i/franca-np2 
não precisará fazer o passo abaixo, pois as imagens já estarão separadas pelas pastas corretamente.

Instruções Detalhadas:
1.	Baixe e Descompacte: Baixe o .zip do Kaggle e descompacte-o. Você verá as pastas seg_train, seg_test, a seg_pred você pode excluir.

2.	Crie a Estrutura: Crie as pastas data/train, data/test e data/validation. Dentro de cada uma, crie as subpastas natural e urbana.

3.	Popule as Pastas de Teste (data/test):
- Copie as imagens de seg_test/buildings e seg_test/street para data/test/urbana/.
-	Copie as imagens de seg_test/forest, seg_test/glacier, seg_test/mountain e seg_test/sea para data/test/natural/.

4.	Popule as Pastas de Treino (data/train):
-	Faça o mesmo com as pastas seg_train: copie as imagens de buildings e street para data/train/urbana/.
-	Copie as imagens de forest, glacier, mountain e sea para data/train/natural/.

5.	Crie o Conjunto de Validação:
-	Vá até a pasta data/train/urbana.
-	Selecione e mova (Recortar/Colar) cerca de 20% das imagens para data/validation/urbana/.
-	Faça o mesmo para data/train/natural/, movendo 20% das imagens para data/validation/natural/.

 
# 5. Como Executar
Após configurar o ambiente (Passo 3) e o dataset (Passo 4), a execução é feita por um único comando.

Certifique-se de que seu ambiente virtual (.venv) está ativado.


# Execute o script principal
``` 
python main.py
```


O script irá:
1.	Imprimir o status de carregamento dos dados.
2.	Baixar os pesos do MobileNetV2 na primeira execução.
3.	Treinar o modelo por 10 épocas, mostrando o progresso.
4.	Ao final, imprimir o Relatório de Classificação no terminal.
5.	Abrir 3 janelas de gráficos Matplotlib com os resultados da avaliação.
