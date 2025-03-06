# Aprenda Machine Learning

## Sumário
- [Aprenda Machine Learning](#aprenda-machine-learning)
    - [Sumário](#sumário)
    - [Introducao](#introducão)
    - [Análise de Dados](#análise-de-dados)
        - [Visualizacão](#visualizacão)
            - [Informacões gerais](#informacões-gerais)
            - [Gráfico em Barra](#gráfico-em-barra)
    - [Tratamento dos Dados](#tratamento-dos-dados)
        - [Tipos de Dados](#tipos-de-dados)
            - [Numéricos e Categóricos](#numéricos-e-categóricos)
            - [X e Y](#x-e-y)
        - [Codificacão](#codificacão)
        - [Normalizacão](#normalizacão)

    - [Usabilidade](#usabilidade)
        - [Clone o Repositório](#clone-esse-repositório)
        - [Instale as Dependências](#instale-as-dependências)
    - [Contatos](#contatos)


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
TODO

### Normalizacão
TODO

## Usabilidade

Para ter esse repositóprio e manipular os arquivos localmentem, basta copiar e colar os seguintes comandos no terminal:

### Clone o repositório

```bash
git clone https://github.com/Rafaelszc/Aprenda-Machine-Learning.git
```

### Instale as Dependências

```bash
cd Aprenda-Machine-Learning/

pip install -r requirements.txt
```

## Contatos
<div class="contact-images" align=center>
    <a href="https://github.com/Rafaelszc"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white%22" style="border-radius: 10px; height: 35px; padding-right: 2px;"></a>
    <a href="mailto:rafaelbjj84@gmail.com"><img src="https://img.shields.io/badge/GMAIL-100000?style=for-the-badge&logo=gmail&logoColor=red" style="border-radius: 10px; height: 35px"></a>
    <a href="https://www.linkedin.com/in/rafael-souza-5461762b8"><img src="https://img.shields.io/badge/LINKEDIN-100000?style=for-the-badge&logo=linkedin&logoColor=blue" style="border-radius: 10px; height: 35px; padding-left: 2px;"></a>
</div>