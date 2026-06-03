# Demonstração do App
https://youtu.be/I0AXbnFv1pY

# Apresentação do Projeto
https://youtu.be/mZ6LNzksBaE

# 🌸 IrisScan - Classificação de Espécies de Flores com Machine Learning

## Sobre o Projeto

O **IrisScan** é um projeto de **Machine Learning e Visão Computacional**, desenvolvido como parte da disciplina **Projeto Aplicado II** do curso de **Ciência de Dados**.

O objetivo do projeto é classificar automaticamente espécies de flores do gênero **Iris**, utilizando tanto:

- **Dados tabulares** (medidas físicas das flores)
- **Imagens reais** das flores utilizando **Deep Learning**

A proposta combina conceitos de **Aprendizado de Máquina, Redes Neurais Convolucionais (CNNs)** e **Visão Computacional**, demonstrando como a Inteligência Artificial pode ser aplicada à identificação de espécies vegetais.

---

## Objetivo do Projeto

O principal objetivo do projeto é desenvolver modelos capazes de identificar corretamente a espécie de uma flor Iris com base em suas características físicas ou em uma imagem enviada pelo usuário.

As espécies classificadas são:

- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

---

## Conjunto de Dados

### 1. Dataset Iris (Dados Tabulares)

O projeto utiliza o clássico **dataset Iris**, amplamente utilizado em estudos de Machine Learning.

O conjunto de dados contém:

- **150 amostras**
- **50 flores de cada espécie**
- **4 características numéricas:**
  - Comprimento da sépala
  - Largura da sépala
  - Comprimento da pétala
  - Largura da pétala

Esses dados foram utilizados para treinar modelos tradicionais de classificação supervisionada.

---

### 2. Dataset de Imagens Reais

Como o dataset original Iris não possui imagens, foi criada uma base própria contendo **50 imagens reais de cada espécie**, coletadas de fontes públicas.

### Fontes das imagens

- http://www.signa.org  
- https://www.wildflower.org  
- https://plants.ces.ncsu.edu  
- https://www.inaturalist.org

---

## Modelos Utilizados

Foram implementados e comparados diferentes algoritmos de classificação supervisionada.

### Machine Learning Tradicional

- Regressão Logística
- K-Nearest Neighbors (**KNN**)
- Árvore de Decisão

### Deep Learning

- Redes Neurais Convolucionais (**CNNs**)
- Transfer Learning:
  - **MobileNetV2**
  - **ResNet50**

---

## Análise Exploratória dos Dados (EDA)

Antes da modelagem, foi realizada uma análise exploratória para compreender os padrões do conjunto de dados.

O projeto inclui:

- Verificação de valores ausentes  
- Estatísticas descritivas  
- Análise de correlação  
- Visualizações gráficas dos dados

### Visualizações utilizadas

- Pairplot
- Boxplot
- Gráfico de Violino
- Heatmap de Correlação

---

## Tecnologias Utilizadas

### Linguagem de Programação

- Python

### Bibliotecas e Frameworks

- Scikit-learn
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

---

## Resultados Obtidos

### Modelos com Dados Tabulares

| Modelo | Acurácia |
|---------|-----------|
| Regressão Logística | 100% |
| KNN | 100% |
| Árvore de Decisão | 100% |

O dataset Iris é altamente estruturado e possui classes bem separadas, o que favorece altos níveis de precisão.

### Modelos com Imagens

Também foram utilizadas técnicas de **Data Augmentation**, como:

- Rotação
- Zoom
- Espelhamento horizontal (*horizontal flip*)

Essas técnicas ajudaram a melhorar a generalização do modelo.

---

## Aplicação IrisScan

Foi desenvolvido um aplicativo funcional utilizando **Streamlit**, permitindo que o usuário:

1. Faça upload de uma imagem da flor  
2. Execute a classificação  
3. Receba a previsão da espécie automaticamente

---

## Aprendizados do Projeto

Durante o desenvolvimento do IrisScan, foi possível aplicar conhecimentos em:

- Análise Exploratória de Dados (EDA)
- Machine Learning Supervisionado
- Avaliação de modelos e métricas de desempenho
- Matriz de confusão
- Visão Computacional
- Redes Neurais Convolucionais (CNNs)
- Transfer Learning
- Desenvolvimento de aplicações com Streamlit

---

## Integrantes do Grupo

Projeto desenvolvido por:

- **Amarilis Oliveira dos Reis**  
- **Nicole Xavier do Nascimento**  
- **Lourenço Netto Ribeiro Correa**

---

## Conclusão

O projeto **IrisScan** demonstrou que técnicas de **Machine Learning** e **Deep Learning** podem ser utilizadas com alta precisão na classificação de espécies vegetais.

Além dos resultados técnicos obtidos, o projeto reforça o potencial da Inteligência Artificial para aplicações em áreas como **botânica, agricultura e preservação ambiental**, conectando tecnologia e ciência de forma prática.

---

⭐ Se você gostou do projeto, considere deixar uma estrela no repositório!
