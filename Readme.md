# 📊 Análise de Empresas com Machine Learning (K-Means + Apriori)

## 📌 Descrição do Projeto
Este projeto tem como objetivo analisar o comportamento de empresas a partir de seus dados cadastrais, utilizando técnicas de **Aprendizado de Máquina Não Supervisionado**.

A proposta central foi:
- Identificar padrões ocultos nos dados
- Segmentar empresas com base em características similares
- Detectar possíveis fatores associados ao risco empresarial

---

## 🎯 Objetivos

- Realizar análise exploratória de dados empresariais
- Criar variáveis relevantes (ex: longevidade)
- Aplicar **clusterização (K-Means)** para segmentação
- Utilizar **Apriori** para extração de regras e explicação dos padrões
- Validar os resultados com variáveis externas (Simples Nacional)

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
  - KMeans
  - StandardScaler
- Mlxtend
  - Apriori
  - Association Rules
- Matplotlib / Seaborn

---

---

## 🔄 Etapas do Projeto

### 1. Pré-processamento
- Seleção de variáveis relevantes
- Tratamento de dados inconsistentes
- Conversão de tipos (datas, categóricos)
- Criação de novas variáveis:
  - **Longevidade (anos)**
  - **Faixas de longevidade**
- Transformação de variáveis categóricas (One-Hot Encoding para Apriori)

---

### 2. Padronização
Aplicação do **StandardScaler** para normalizar variáveis numéricas:
- ANOS
- VL_CAPITAL_SOCIAL
- Indicadores derivados

---

### 3. Clusterização (K-Means)
- Definição do número de clusters com:
  - Método do cotovelo (Elbow)
  - Silhouette Score
- Segmentação das empresas em perfis distintos

---

### 4. Regras de Associação (Apriori)
- Transformação da base em formato transacional
- Extração de **itemsets frequentes**
- Geração de regras com métricas:
  - Support
  - Confidence
  - Lift

---

### 5. Validação do Modelo
A variável **Simples Nacional** NÃO foi utilizada na clusterização.

Ela foi usada posteriormente para:
- Validar os padrões encontrados
- Interpretar o comportamento dos clusters

---

## 📊 Principais Resultados

### 🔹 Cluster A – Jovens Pouco Capitalizadas
- Empresas mais novas
- Maioria saudável
- Forte presença no Simples Nacional

---

### 🔹 Cluster B – Maduras Capitalizadas
- Empresas consolidadas
- Perfil estável
- Parte relevante ainda no Simples

---

### 🔹 Cluster C – Maduras Pouco Capitalizadas
- Empresas estáveis
- Baixa adesão ao Simples
- Indício de crescimento ou mudança tributária

---

### 🔹 Cluster D – Jovens Capitalizadas em Risco
- Alta taxa de exclusão do Simples (~78%)
- Forte associação com risco empresarial

---

## 🔍 Principais Insights

- Empresas fora do Simples Nacional apresentam maior associação com risco
- Permanência no Simples está ligada à estabilidade operacional
- Empresas maduras fora do Simples podem indicar crescimento, não necessariamente risco
- O modelo identificou padrões relevantes **sem utilizar variável alvo**

---

## 📈 Métricas Utilizadas (Apriori)

- **Support**: Frequência da regra no dataset
- **Confidence**: Probabilidade do consequente dado o antecedente
- **Lift**: Força da associação (acima de 1 indica relação relevante)

---

## 🚀 Possíveis Aplicações

- Análise de risco empresarial
- Segmentação de clientes (B2B)
- Apoio à decisão em crédito
- Inteligência de mercado

---

## 🎓 Contexto Acadêmico

Projeto desenvolvido na disciplina de:
**Aprendizado de Máquina Não Supervisionado**  
Faculdade Senac-DF

---

## 📌 Autor

Leonardo Zell

---

## 📬 Contato

[LinkedIn](#) *[(adicione seu link aqui)](https://www.linkedin.com/in/leonardozell/)*

