# Segmentação Inteligente de Empresas com Machine Learning Não Supervisionado

## 📌 Visão Geral
Este projeto tem como objetivo segmentar **mais de 1 milhão de empresas** a partir de dados cadastrais e financeiros, utilizando técnicas de **Machine Learning não supervisionado** e **Data Mining**.

A proposta foi identificar **perfis empresariais semelhantes**, validar a qualidade dos agrupamentos e, posteriormente, **explicar semanticamente os clusters gerados**.

O pipeline foi construído com foco em:
- descoberta de padrões
- segmentação de perfis
- interpretação de clusters
- geração de insights de negócio

---

## 🎯 Objetivos
- Agrupar empresas com características semelhantes
- Descobrir o número ideal de clusters
- Interpretar o perfil de cada grupo
- Identificar padrões frequentes dentro dos clusters
- Apoiar análises de risco, crescimento e perfil financeiro

---

## 🧠 Técnicas de IA & ML Utilizadas
### Machine Learning Não Supervisionado
- **K-Means Clustering**
- **PCA (Principal Component Analysis)**
- **Silhouette Score**
- **Elbow Method**

### Data Mining / Explainable Analytics
- **Apriori Algorithm**
- **Association Rules**
- **Cluster Explainability**

### Engenharia de Dados e Features
- Tratamento de encoding UTF-8
- Limpeza de dados textuais com Regex
- Tratamento de nulos
- Padronização numérica com `StandardScaler`
- Transformação logarítmica
- Binning / discretização com `qcut` e `cut`
- Encoding categórico (One-Hot, Frequency e Ordinal)

---

## 🛠️ Stack Tecnológica
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Mlxtend**
- **Matplotlib**
- **Jupyter Notebook / VS Code**

---

## 📊 Fluxo do Projeto
```text
Coleta e limpeza dos dados
        ↓
Feature Engineering
        ↓
Padronização
        ↓
K-Means
        ↓
Elbow + Silhouette
        ↓
PCA + Radar / Spider
        ↓
Cruzamento com ST_CADASTRO
        ↓
Apriori para explicar clusters
```

---

## 🧩 Variáveis Utilizadas
### Clusterização
- `ANOS`
- `VLR_FATURAMENTO_MAX`
- `VL_CAPITAL_SOCIAL_LOG`

### Explicação dos Clusters
- `ST_CADASTRO`
- Faixas de anos
- Faixas de faturamento
- Faixas de capital
- Cluster gerado

---

## 📈 Principais Resultados
O modelo identificou **3 perfis principais de empresas**:

### 🔵 Cluster 0 — Empresas Consolidadas Tradicionais
- empresas antigas
- capital social alto
- faturamento médio
- perfil estável e estruturado

### 🟢 Cluster 1 — Empresas Maduras de Baixo Capital
- empresas antigas
- capital social baixo
- faturamento médio
- maior incidência de baixadas

### 🟠 Cluster 2 — Empresas de Alto Faturamento
- alto faturamento
- cluster altamente puro
- forte previsibilidade estatística

---

## 🔍 Exemplo de Insight Gerado
Uma das regras mais fortes encontradas pelo Apriori foi:

> **FAT_ALTO → CLUSTER_2**

Isso mostrou que **99,9% das empresas com faturamento alto pertencem ao Cluster 2**, validando a separação financeira feita pelo K-Means.

---

## 💼 Aplicações de Negócio
Este projeto pode ser aplicado em:
- segmentação comercial
- análise de crédito
- prevenção de risco
- prospecção B2B
- políticas de relacionamento
- inteligência de mercado
- análise de churn empresarial

---

## 🚀 Próximos Passos
- Testar outros algoritmos de clusterização (DBSCAN, GMM, Hierárquico)
- Criar dashboard interativo em Power BI / Streamlit
- Adicionar score de risco por cluster
- Incorporar variáveis setoriais e geográficas
- Deploy do pipeline para uso analítico contínuo

---

## 👨‍💻 Autor
Projeto desenvolvido por **Leonardo Zell** como estudo aplicado de **Machine Learning, Analytics e Segmentação Empresarial**.

