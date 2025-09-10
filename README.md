# Aprenda Machine Learning

## Sumário
- [Aprenda Machine Learning](#aprenda-machine-learning)
  - [Sumário](#sumário)
  - [Introducão](#introducão)
  - [Análise de Dados](#análise-de-dados)
    - [Visualizacão](#visualizacão)
      - [Informacões gerais](#informacões-gerais)
      - [Gráfico em Barra](#gráfico-em-barra)
  - [Tratamento dos Dados](#tratamento-dos-dados)
    - [Tipos de Dados](#tipos-de-dados)
      - [Numéricos e Categóricos](#numéricos-e-categóricos)
      - [X e Y](#x-e-y)
    - [Codificacão](#codificacão)
    - [Padronização](#padronização)
    - [Normalização](#normalização)
  - [Salvamento de variáveis](#salvamento-de-variáveis)
  - [Modelos de Classificacão](#modelos-de-classificacão)
    - [Navie Bayes](#navie-bayes)
    - [Random Forest Classifier](#random-forest-classifier)
    - [K-Nearest Neighborn Classifier (KNN)](#k-nearest-neighborn-classifier-knn)
  - [Modelos de Regressão](#modelos-de-regressão)
    - [K-Nearest Neighborn Regressor (KNN Regressor)](#k-nearest-neighborn-regressor-knn-regressor)
    - [SGD Regressor](#sgd-regressor)
  - [Ensambles](#ensambles)
    - [AutoGluon](#autogluon)
  - [Redes Neurais](#redes-neurais)
    - [Aprendizagem da rede neural](#aprendizagem-da-rede-neural)
      - [Feedforward](#feedforward)
      - [Activation Function](#activation-function)
      - [Loss function](#loss-function)
      - [Backpropagation](#backpropagation)
      - [Iteration e Epochs](#iteration-e-epochs)
    - [Implementacão do algoritmo](#implementacão-do-algoritmo)
  - [Usabilidade](#usabilidade)
    - [Clone o repositório](#clone-o-repositório)
    - [Instale as Dependências](#instale-as-dependências)
      - [Usando o Pip](#usando-o-pip)
      - [Usando o Anaconda](#usando-o-anaconda)
  - [Contatos](#contatos)
  - [Referências](#referências)


## Introducão

Esse é um repositório introdutório ao conceito de Machine Learning e alguns algoritmos usados nessa área. Partindo do zero até o entendimento da análise e aplicacão dos modelos.

Aqui será tratado tópicos relacionados a visualizacão dos dados, tipos de dados, pré-processamento funcionamento da aprendizagem e etc. Trazendo de maneira teórica e prática cada tópico.

Os códigos devem ser feitos -em sua maioria- utilizando arquivos ```.ipynb```, usando o [Jupyter Notebook](https://jupyter.org/) ou o [Google Colab](https://colab.research.google.com/)

## Análise de Dados

A etapa inicial do processo de Machine Learning (ML) vem da análise de dados, nesse processo toda a base será avaliada para saber se as informacões contidas em si podem ser utilizadas nos modelos. Caso haja irregularidades, será nesse processo em que elas serão corrigidas.

### Visualizacão

Antes de tudo, precisamos ver quais são os dados que possuimos, para isso será utilizada a biblioteca [Pandas](https://pandas.pydata.org/) e também a ferramenta de geracão de gráficos [Plotly](https://plotly.com/python/), em especial o seu pacote `express`.

#### Informacões gerais

No pandas, podemos ler arquivos que contenham tabelas como `.csv`, `.parquet`, ``.db`` e etc. Com o seguinte comando usando o exemplo de um arquivo csv:

```python
import pandas as pd

data = pd.read_csv("caminho_do_arquivo")
```

Assim, o arquivo será guardado em uma variável do python. Essa tabela pode ser vista digitando novamente a variável.

[![display_data](resources/assets/data_display.png)](/src/pre_processing/data_visualization.ipynb)

Ainda com o pandas, podemos ver algumas informacões gerais sobre a tabela usando a funcão `.info()` e `.describe()`

[![info_and_describe](/resources/assets/info_and_describe.png)](/src/pre_processing/data_visualization.ipynb)

#### Gráfico em Barra

Para a producão do gráfico em barra, a funcão utilizada será a `bar()`. Com o seguitne escopo:

```python
import plotly.express as px

fig = px.bar(
    data_frame=tabela,
    x=coluna_da_tabela,
    y=outra_coluna,
    title=titulo_do_gráfico
)

fig.show()
```

No nosso caso temos esse exemplo:

[![bar](resources/assets/bar_graph.png)](/src/pre_processing/data_visualization.ipynb)

## Tratamento dos Dados

Nessa parte será tratado quais dados serão utilizados partindo de suas respectivas funcões. Entender cada detalhe disso será de extrema importancia para evitar incoerências no processo de aprendizagem.

### Tipos de Dados

#### Numéricos e Categóricos

Os dados serão reconhecidos de duas formas: numéricos e categóricos. Os numéricos são todos aqueles que apresentam valores inteiros ou reais (Integer, Float, Double...) que representam quantidades. Já os categóricos são aqueles que são definicões/categorias/nomes.

Vale ressaltar que nem sempre que um valor for um número signfique que pode ser tratado como numérico, talvez seja um caso de um numérico categórico! Exemplo: se um banco de dados representar o gênero de uma pessoa por 0: masculino e 1: feminino, isso será um caso de numérico categórico. Por isso o processo de separacão de dados deve ser analisado com cautela, pois cada erro desse resulta em uma falha gigante nos calculos de predicão.

#### X e Y

Outra denominacão bastante utilizada no ambiente de ML são os dados X e Y. Os dados X representam aqueles que serão as características da predicão, ou seja, será nesses dados que o modelo irá identificar os padrões comparando com os resultados. Já o Y é justamente o resultado, será o alvo da predicão. Os dados X serão o caminho e o Y é o destino.

Veja nessas imagens:

[![x](resources/assets/data_x.png)](src/pre_processing/pre_proessing.ipynb)

[![y](resources/assets/data_y.png)](src/pre_processing/pre_proessing.ipynb)

Os valores X são as caracteristicas do vegetal, quanto o Y representa o preco de cada produto. Nosso intuito será descobrir qual seria o valor ideal de um vegetal partindo de suas caracteristicas.

### Codificacão
Em geral os modelos não aceitam valores do tipo string, pois isso atraplha ou impossibilita os calculos matemáticos. Para resolver isso, esses dados devem ser transformados em valores numéricos, um dos recursos que nos auxiliam nessa atividade é o [LableEncoder](docs/LableEncoder.md), que retorna um ID para cada item da coluna.

Encontramos o LableEncoder na biblioteca do scikit-learn:

```python
from sklearn.preprocessing import LableEncoder

col_a_encoder = LableEncoder()

col_a_encoded = col_a_encoder.fit_transform(col_a)
```
Assim, podemos sair disso:

![uncode](resources/assets/veg_uncoded.png)

Para isso:

![code](resources/assets/veg_encoded.png)

Para poder reverter o processo de codificação (decodificação), o LableEncoder disponibiliza a função `inverse_transform()`.

```python
from sklearn.preprocessing import LableEncoder

encoder = LableEncoder()

col_decoded = encoder.inverse_transform(encoded_data)
```

### Padronização
A padronizacão se baseia em colocar os dados numérico em intervalos menores, para evitar que um modelo ache que determinada coluna X tenha uma importância maior que a Y apenas porque os seus valores estão em uma escala maior do que a outra, sendo que isso pode ser um fator irrelevante. Nesses casos que irá entrar a padronizacão, ela transformará os dados de tal forma que o seu desvio padrão seja 1 e a sua média 0

```math
z=\frac{x-\mu}{\sigma}
```

O scikit-learn também disponibiliza uma ferramenta para fazer isso, chamada StandardScaler

```python
from sklearn.preprocessing import StandardScaler

col_a_scaler = StandardScaler()

col_a_scaled = col_a_scaler.fit_transform(col_a)
```

Saindo disso:

![data_y](resources/assets/data_y.png)

Para isso:

![y_vector](resources/assets/y_scaled.png)

### Normalização
Um algoritmo bastante utilizado para a normalizção -que visa deixar os valores entre 0 e 1- é o OneHotEncoder, na qual transforma os dados em vetores canônicos para representar sua presença ou ausência naquela célula.

```python
from sklearn.preprocessing import OneHotEncoder

col_a_encoder = OneHotEncoder(sparse_output=False)

col_a_encoded = col_a_encoder.fit_transform(col_a)

pd.DataFrame(data=col_a_encoded, columns=col_a_encoder.get_feature_names_out())
```

Agora, será adicionada N colunas que representam as N diferentes features de cada coluna. Esse algoritmo é extremamente essêncial principalmente para modelos que se baseam em distâncias, tais como KNN, regressores e redes neurais

![normalization](resources/assets/normalization.png)

## Salvamento de variáveis
É possível salvar as variáveis do Python em um tipo especial de arquivo, esse tipo é o `.pkl`, para isso podemos utilizar a biblioteca `pickle` para poder fazer essa guarda de informação. As variáveis salvas, em nosso contexto, podem ser os dados pré processados e também os modelos treinados para poder ser utiizados em outros códigos.

Para fazer isso, basta seguir o passo a passo:

```python
import pickle as pkl

with open("file/path", "wb") as file:
    pkl.dump([your, variables], file)
```

O arquivo é escrito usando o método `wb`, permitindo a escrita de dados em binário. Pode ser armazenado mais de uma variável, basta colocar em uma lista assim como está no exemplo.

Para coletar essas variáveis, basta seguir o seguinte passo:

```python
import pickle as pkl

with open("file/path", "rb") as file:
    your, variables = pkl.load(file)
```

Dessa vez o método é o `rb`, que permite a leitura em binário. Os dados a serem coletados devem ser escritos na mesma sequência em que foram armazenados.

Nos nossos casos foi usado da seguinte forma:

```python
KNN_BASED_DATA_PATH = join("..", "..", "resources", "database", "knn_classifier_data.pkl")

with open(KNN_BASED_DATA_PATH, "wb") as file:
    pkl.dump([x_train, x_test, y_train, y_test], file)
```

```python
DATA_PATH=join("..", "..", "..", "resources", "database", "knn_classifier_data.pkl")

with open(DATA_PATH, "rb") as file:
    x_train, x_test, y_train, y_test= pkl.load(file)
```

O pickle é bastante interessante para exportar modelos treinados para outros arquivos/projetos. Assim, permitindo compartilhar e baixar modelos locais pré treinados.

## Modelos de Classificacão

Em geral, os modelos de classificacão servem para predizer valores categóricos com base nos dados que estão em X. Como exemplo: suponhamos que temos uma base de dados com dados de currículos para uma vaga de desenvolvimento de software, com base nesses dados e uma base de treinamento, poderíamos predizer qual área de desenvolvimento de software cada pessoa poderia ocupar. Por outro lado, os modelos de regressão servem para predizer valores numéricos reais, o que seriam o ideal para a nossa base de dados de vegetais, já que os resultados são os precos de cada vegetal com base em suas características.

### Navie Bayes
Naive Bayes é um classificador probabilístico simples, isso é, um modelo de classificação que calcula as diferentes probabilidades para cada diferente classe dado aos termos de entrada, baseado no "Teorema de Bayes":

```math
P(y∣X)=\frac{P(X∣y)⋅P(y)}{P(X)}
```
Onde, $P(y\; |\; X)$ é probabilidade da classe $y$ baseada nas variáveis de entrada $X$, $P(X\; | \;y)$ são as probabilidades anteriores das variáveis recebidas $X$ dada a classe $y$, $P(y)$ é a probabilidade anterior da classe $y$ e $P(X)$ é a probabilidade marginal dos termos.

O termo "Naive" em Naive Bayes vem da própria palavra em inglês que significa "Ingenuo". Isso acontece devido ao modelo assumir cada termo de entrada como independente dado a classe, Logo temos:

```math
	P(X) = P(X_1 ... X_n) ∴ P(y\;|\;X_1 ... X_n) = \frac{P(y) * P(X_1\;|\; y)...*P(X_n\;|\;y)}{P(X_1)...*P(X_n)}
```
Desse modo o teorema de Bayes torna-se:
```math
	P(y\;|\;X_1 ... X_n) = \frac{P(y) * ∏^n_{i=1} P(X_i\;|\;y)}{P(X_1)... P(X_n)}
```
Dado que o denominador $P(X_1 ... X_N)$ seja uma contante temos:
```math
P(y\;|\; X_1 ... X_n) = P(y) * ∏^n_{i=1} P(X_i\;|\;y)
```
Assim para funcionar nosso modelo calculamos os valores posteriores para cada classe y e escolhemos a com maior probabilidade.
```math
	ŷ = arg\;max_yP(y) * ∏^n_{i=1} P(X_i|y)
```
Isso torna-se nosso classificador.

Os diferentes Naive Bayes se diferem principalmente pelas suposições que fazem sobre a distribuição de $P(X_i\; |\; y)$.

Para valores continuos nos assumimos a "Distribuição Gaussiana": 
```math
P(x_i\;|\;y) = \frac{1}{\sqrt{2\piσ^2_y}}\;exp\; (-\frac{(x_i-μ_y)^2}{2σ^2_y})
```
Onde:
$μ_y$ é a media dos valores do termo $x_i$ para a classe $y$;
$σ^2_y$ é o desvio padrão do termo $x_i$ para a classe $y$;

Assim temos o que chamamos de Gaussian Naive Bayes, ou Naive Bayes Gaussiano.

### Random Forest Classifier
É um modelo de classificação que se baseia em conjuntos de árvores de decisões, criadas a partir de diferentes subconjuntos aleatórios de dados ou recursos.
- Decision Tree(Árvore de Decisão):
    A árvore de decisão é um modelo simples de interpretar, formada por raiz, que representa um atributo ou característica, ramos, representando uma regra de decisão, e nós, o resultado, é semelhante a um fluxograma que simula um pensamento humano auxiliando na tomada de decisões.
    A medida de seleção de atributos particiona os dados da melhor forma possível, também sendo regras de divisão que nos ajuda a determinar pontos de interrupção em um nó, fornecendo uma classificação para cada atributo. O atributo com a melhor pontuação será escolhido como um atributo de divisão.
As medidas de seleção são:
#### Ganho de Informação(Gain):
 O ganho de informações baseia-se na Entropia, é a impureza em um grupo de exemplos, sendo o ganho de informação a diminuição da entropia, calculando a diferença da entropia média depois e a entropia antes da divisão do conjunto de dados.
 
$$ \text{Info(D)} = - \sum_{i=1}^{m} p_i \log_2(p_i) $$

- Sendo Pi a probabilidade de uma tupla arbitraria em D, pertencente a Ci
- Porém, tende a favorecer atributos com maiores números de valores

 $$ \text{Info}_a(D) = \sum_{j=1}^{V} \frac {|D_j|}{|D|} \cdot \text{Info}(D_j) $$

 $$ \text{Gain(A)} = \text{Info(D)} - \text{Info}_A(D) $$

- O maior Gain(A) é escolhido como atributo de divisão no nó
#### Relação de ganho: Utiliza-se o Split Info para a normalização da tendência do ganho de informações:

$$ \text{SplintInfo}_A(D) = - \sum_{j=1}^{V} \frac {|D_j|}{|D|} \cdot \log{2}^{\frac{|D_j|}{|D|}} $$

#### Taxa de ganho(GainRatio):

$$ \text{GainRatio(A)} = \frac{Gain(A)}{SplintInfo_A(D)} $$

- O atributo com maior GainRatio é escolhido como raiz da árvore
#### Índice de Gini :

$$ \text{Gini(D)} = 1 - \sum_{i=1}^{m} Pi^2 $$

- Pi é a probabilidade de D pertencer a Ci
- Considera uma divisão binária em cada atributo


$$  \text{Gini}_A(D) = \frac{|D1|}{|D|} \text{Gini}(D_1) + \frac{|D2|}{|D|} \text{Gini}(D_2) $$


$$ \bigtriangleup{Gini(A)} = \text{Gini(D)} - \text{Gini}_A(D) $$


- O atributo com o menor índice de Gini é escolhido como raiz da árvore



### K-Nearest Neighborn Classifier (KNN)
TODO

## Modelos de Regressão
De forma geral, modelos de regressão retornam valores numéricos reais e trabalham em sua maioria com todas as fazes de pré processamentos. 

Como exemplo, vamos supor que tenhamos uma base de produtos de um comércio e queremos saber o preço ideal para os novos produtos que entram em estoque. Como o resultado é um número real e não uma categoria, os modelos de regressão são os ideais para esses casos. O que se encaixa bem na base de vegetais.

### K-Nearest Neighborn Regressor (KNN Regressor)
- O princípio entre os métodos de Nearest Neighbors(Vizinho mais Próximo) é encontrar um, já pré definido, número de treinamento de amostras mais próximas entre a distância de dois pontos, e prever o rótulo deles. O número de amostras pode ser  uma constante K definida pelo usuário, ou baseada na densidade de pontos do local. A distância pode, no geral, ser qualquer medida métrica, sendo o padrão Euclidiano a escolha mais comum.
- Os métodos baseados em Neighbors são conhecidos como machine learning não generalizantes métodos, desde que eles simplesmente lembram todo desse treinamento de dados.
- O modelo em regressão pode ser utilizado em casos que os rótulos dos dados são contínuos, e não discretos. O rótulo atribuído ao ponto de consulta é calculado baseado no principal dos rótulos de seu vizinho mais próximo.
- A regressão básica Nearest Neighbors usa pesos uniformes, isso é, em cada local de vizinhança contribui uniformemente para a classificação de ponto de consulta. Em algumas circunstâncias, isso pode ser vantajoso pesar os pontos de modo que os próximos pontos contribuem mais para a regressão do que os pontos mais distantes
#### Weight
    (weight= “uniform”) # indica peso igual para todos os pontos
    (weight = “distance”)  # indica peso proporcional ao inverso da distância entre os pontos de consulta



### SGD Regressor
O **SDGRegressor** é um modelo de _Machine Learning_ baseado em **Stochastic Gradient Descent (SGD)**, em vez do **Batch Gradient Descent**. Dessa forma, o modelo apresenta maior eficiência e é de mais simples de implementação em diferentes aplicações.

A **descida de gradiente** é um algoritmo de otimização iterativa usado para minimizar funções de custo, que representam o quão distantes estão as predições feitas pelo modelo em relação aos valores reais. O principal objetivo é ajustar os parâmetros do modelo de forma que o erro seja minimizado.

Na descida de gradiente tradicional, o gradiente é calculado com base em todo o conjunto de dados, o que pode ser computacionalmente custoso para grandes _datasets_.

Por outro lado, a descida de gradiente estocástica realiza atualizações de parâmetros a cada nova amostra processada. Isso permite que o modelo se adapte mais rapidamente aos dados, sendo especialmente vantajoso em grandes bases. Dessa forma, reduz-se a função de custo e a margem de erro final.

## Ensambles

### AutoGluon
TODO

## Redes Neurais
Redes Neurais ou Redes Neurais Artificiais é um tipo de modelo que tenta simular o funcionamento de neurônios biológicos. Na qual, funciona com base em um grafo onde os nós são os neurônios. Cada neurônio faz o papel de uma combinacão linear:

```math
f(x)=x_1*w_1+x_2*w_2+...x_n*w_n
```

Onde Xi é um, Wi é um peso associado ao neurônio. Um conjunto de nós que fazem, em paralelo, essa combinacão linear é chamada de camada. Quando cada nó de uma camada se conecta a todos os nós da camada seguinte, chama-se camada densa.

Essa formula da combinacão linear está para as redes neurais assim como o 0 e 1 estão para a computacão. Porém, esse conceito é limitado a linearidade da funcão, fazendo com que não haja distincão entre uma rede neural e um modelo de regressão logístico/linear. Com isso, é acrescentado uma funcão de ativacão nos neurônios e um bias. O bias pode ser interpretado como uma margem, quanto maior o bias e os pesos, mais relevante é a análise daquela camada:

```math
f(x)=(x_1*w_1+x_2*w_2+...x_n*w_n)+b
```

A funcão de ativacão dos nós serve para evitar ruídos e padronizar os formatos dos dados, além de evitar uma gama de dados nulos. As funcões mais conhecidas são: ReLu, Leaky ReLu, Sigmoid, Tanh, Softmax e Linear. Terá um tópico exclusivo para tratar de funcões de ativacão.

Para a programacão da rede neural será feita com as bibliotecas [TensorFlow](https://www.tensorflow.org/) e [Keras](https://keras.io/).
### Aprendizagem da rede neural
#### Feedforward
#### Activation Function
#### Loss function
#### Backpropagation
#### Iteration e Epochs
### Implementacão do algoritmo
## Usabilidade

Para ter esse repositóprio e manipular os arquivos localmentem, basta copiar e colar os seguintes comandos no terminal:

### Clone o repositório

```bash
git clone https://github.com/Rafaelszc/Aprenda-Machine-Learning.git
```

### Instale as Dependências

#### Usando o Pip

```bash
cd Aprenda-Machine-Learning/

pip install -r requirements.txt
```

Selecione o env do jupyter que foi instalado as requisições

#### Usando o Anaconda

```bash
cd Aprenda-Machine-Learning/

conda env create -f environment.yml

conda activate learn_machine_learning_env
```

Selecione o env `learn_machine_learning_env` como a env que irá rodar no jupyter

## Contatos
<div class="contact-images" align=center>
    <a href="https://github.com/Rafaelszc"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white%22" style="border-radius: 10px; height: 35px; padding-right: 2px;"></a>
    <a href="mailto:rafaelbjj84@gmail.com"><img src="https://img.shields.io/badge/GMAIL-100000?style=for-the-badge&logo=gmail&logoColor=red" style="border-radius: 10px; height: 35px"></a>
    <a href="https://www.linkedin.com/in/rafael-souza-5461762b8"><img src="https://img.shields.io/badge/LINKEDIN-100000?style=for-the-badge&logo=linkedin&logoColor=blue" style="border-radius: 10px; height: 35px; padding-left: 2px;"></a>
</div>

## Referências

SCIKIT-LEARN. scikit-learn: machine learning in Python — scikit-learn 0.20.3 documentation. Disponível em: <https://scikit-learn.org/stable/index.html>. 

PLOTLY. Plotly Python Graphing Library. Disponível em: <https://plotly.com/python/>.

JAIN, D. Data Preprocessing in Data Mining. Disponível em: <https://www.geeksforgeeks.org/data-preprocessing-in-data-mining/>. 

SCIKIT-LEARN. sklearn.preprocessing.LabelEncoder — scikit-learn 0.22.1 documentation. Disponível em: <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>.

SCIKIT-LEARN. 1.6. Nearest Neighbors — scikit-learn 1.7.2 documentation. Disponível em: <https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression>.


‌SCIKIT-LEARN. sklearn.preprocessing.OneHotEncoder — scikit-learn 0.22 documentation. Disponível em: <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>.

‌NAIR, A. Standardization vs Normalization | Towards Data Science. Disponível em: <https://towardsdatascience.com/standardization-vs-normalization-dc81f23085e3/>.

‌GEEKSFORGEEKS. Understanding Python Pickling with example. Disponível em: <https://www.geeksforgeeks.org/understanding-python-pickling-example/>.

‌DATACAMP. Random Forest Classification with Scikit-Learn. Disponível em: <https://www.datacamp.com/tutorial/random-forests-classifier-python>.
