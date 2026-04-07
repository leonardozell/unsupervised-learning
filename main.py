# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("basedf2.csv", sep=";")



# %%
df.head()


# %%

#df.DescSetor.value_counts()

# %%

#Substitui , por . na Coluna ANOS
df["ANOS"] = (
    df["ANOS"]
    .str.replace(",", ".", regex=False)
)

df["ANOS"] = pd.to_numeric(df["ANOS"], errors="coerce")

# %%

# BOXPLOT DA COLUNA VL CAPITAL SOCIAL
def plot_boxplots(df, columns, outline=True):
    
    
    num_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not num_columns:
        raise ValueError("Nenhuma das colunas fornecidas é numérica.")

    # Criar os boxplots
    plt.figure(figsize=(4, len(num_columns) * 4))  # Ajuste de tamanho baseado no número de colunas

    for i, col in enumerate(num_columns, 1):
        plt.subplot(len(num_columns), 1, i)

        # Criar o boxplot vertical
        sns.boxplot(data=df, y=col, showfliers=outline, orient="v")

        # Calcular os quartis e a média
        quartile_25 = df[col].quantile(0.25)
        quartile_50 = df[col].median()  # ou quantile(0.50)
        quartile_75 = df[col].quantile(0.75)

        # Calcular os limites para os outliers
        iqr = quartile_75 - quartile_25
        lower_bound = quartile_25 - 1.5 * iqr
        upper_bound = quartile_75 + 1.5 * iqr

        # Exibir os limites dos outliers
        plt.text(0, lower_bound, f'Início Outliers Inf: {lower_bound:,.2f}', horizontalalignment='center', color='purple', weight='bold')
        plt.text(0, upper_bound, f'Início Outliers Sup: {upper_bound:,.2f}', horizontalalignment='center', color='purple', weight='bold')


        # Plotar os valores dos quartis e da média
        plt.text(0, quartile_25, f'Q1: {quartile_25:,.2f}', horizontalalignment='center', color='blue', weight='bold')
        plt.text(0, quartile_50, f'Q2 (Mediana): {quartile_50:,.2f}', horizontalalignment='center', color='blue', weight='bold')
        plt.text(0, quartile_75, f'Q3: {quartile_75:,.2f}', horizontalalignment='center', color='blue', weight='bold')

        # Título e rótulo do eixo Y
        plt.title(f'Boxplot de {col}')
        plt.ylabel(col)

    plt.tight_layout()
    plt.show()


columns = ['VL_CAPITAL_SOCIAL']
plot_boxplots(df, columns, outline=True)


# %%

#Aplicando Logaritimo no Capital Social 
df['VL_CAPITAL_SOCIAL_LOG'] = np.log1p(df['VL_CAPITAL_SOCIAL'])

columns = ['VL_CAPITAL_SOCIAL_LOG']
plot_boxplots(df, columns, outline=True)




# %%
#Excluir registros vazios
df = df.dropna(subset=["TP_ESTAB", "MOT_SIT_CADASTRAL"])


# %%
#Arrumar nomes da coluna RA
df["RA"] = df["RA"].replace("São Sebastinão", "São Sebastião")


# %%
df.info()


# %%

#Selecionando as colunas que utilizaremos
colunas = [
    "ANOS",
    "VLR_FATURAMENTO_MAX",
    "VL_CAPITAL_SOCIAL_LOG",
]

df_modelo = df[colunas].copy()




# %%
df_modelo = df_modelo.dropna()


# %%
df_modelo.head()

# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_modelo_padronizada = scaler.fit_transform(df_modelo[colunas])

df_modelo_padronizada = pd.DataFrame(df_modelo_padronizada, columns=[coluna + '_PADRONIZADA' for coluna in colunas])
df_modelo_padronizada.head()


# %%
#Verificar padronização. Esperado: ~1
df_modelo_padronizada[[
    "ANOS_PADRONIZADA",
    "VLR_FATURAMENTO_MAX_PADRONIZADA",
    "VL_CAPITAL_SOCIAL_LOG_PADRONIZADA"
]].std()

# %%
df_modelo.head()


# %%

# COTOVELO KMEANS

from sklearn.cluster import KMeans

inertia = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_modelo_padronizada)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia")
plt.title("Método do Cotovelo")
plt.show()



# %%

#SILHOUETTE SCORE


from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_modelo_padronizada)
    
    score = silhouette_score(
        df_modelo_padronizada,
        labels,
        sample_size=5000,  
        random_state=42
    )
    
    silhouette_scores.append(score)

plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette")
plt.show()
# %%

#PLOTAR OS DOIS GRAFICOS EM UM SÓ
fig, ax1 = plt.subplots()

# eixo 1 → cotovelo (inertia)
ax1.plot(K_range, inertia, marker='o', color='blue')
ax1.set_xlabel("Número de clusters (k)")
ax1.set_ylabel("Inertia")

# eixo 2 → silhouette
ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, marker='s', color='red')
ax2.set_ylabel("Silhouette Score")

plt.title("Elbow vs Silhouette")
plt.show()


# %%
#DEFININDO O K

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_modelo_padronizada)

df_modelo["cluster"] = clusters


# %%

# Aplicando PCA e plotando grafico
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_modelo_padronizada)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_modelo["cluster"],
    cmap="viridis",
    s=10
)

plt.title("Clusters PCA")
plt.show()
# %%

colunas_perfil = [
    "ANOS",
    "VLR_FATURAMENTO_MAX",
    "VL_CAPITAL_SOCIAL_LOG",
]

df_perfil = df_modelo.groupby("cluster")[colunas_perfil].mean()

# %%

#Normalização dos dados
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_perfil_scaled = pd.DataFrame(
    scaler.fit_transform(df_perfil),
    columns=df_perfil.columns,
    index=df_perfil.index
)


# %%

# Gráfico SPIDER RADAR
labels = df_perfil_scaled.columns
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for i in df_perfil_scaled.index:

    values = df_perfil_scaled.loc[i].tolist()
    values += values[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title(f"Cluster {i}")
    plt.show()

# %%
df_modelo["ST_CADASTRO"] = df["ST_CADASTRO"]


# %%
# APRIORI PARA EXPLICAR OS CLUSTERS

from mlxtend.frequent_patterns import apriori, association_rules

# 1) Base para regras
df_apriori = df_modelo.copy()

# 2) Criar faixas (bins) das variáveis numéricas
df_apriori["ANOS_FAIXA"] = pd.qcut(
    df_apriori["ANOS"],
    q=3,
    labels=["ANOS_BAIXO", "ANOS_MEDIO", "ANOS_ALTO"]
)

df_apriori["FAT_FAIXA"] = pd.cut(
    df_apriori["VLR_FATURAMENTO_MAX"],
    bins=[0, 100000, 1000000, df_apriori["VLR_FATURAMENTO_MAX"].max()],
    labels=["FAT_BAIXO", "FAT_MEDIO", "FAT_ALTO"],
    include_lowest=True
)

df_apriori["CAPITAL_FAIXA"] = pd.qcut(
    df_apriori["VL_CAPITAL_SOCIAL_LOG"],
    q=3,
    labels=["CAPITAL_BAIXO", "CAPITAL_MEDIO", "CAPITAL_ALTO"]
)

# 3) Transformar cluster em categoria textual
df_apriori["CLUSTER_CAT"] = "CLUSTER_" + df_apriori["cluster"].astype(str)

# 4) Selecionar colunas categóricas para cesta
colunas_apriori = [
    "ANOS_FAIXA",
    "FAT_FAIXA",
    "CAPITAL_FAIXA",
    "ST_CADASTRO",
    "CLUSTER_CAT"
]

df_transacoes = df_apriori[colunas_apriori].astype(str)

# 5) One-hot para formato transacional
df_basket = pd.get_dummies(df_transacoes)

# 6) Rodar Apriori
freq_items = apriori(
    df_basket,
    min_support=0.05,
    use_colnames=True
)

# 7) Regras
rules = association_rules(
    freq_items,
    metric="lift",
    min_threshold=1.2
)

# 8) Filtrar regras que expliquem cluster
rules_cluster = rules[
    rules["consequents"].astype(str).str.contains("CLUSTER")
]

# melhores regras
rules_cluster = rules_cluster.sort_values(
    by="lift",
    ascending=False
)

rules_cluster.head(20)

# %%

df_modelo.head()
# %%
df_modelo.groupby("cluster")["ST_CADASTRO"].value_counts(normalize=True)
# %%
# reorganiza para formato de tabela
df_plot = df_modelo.groupby("cluster")["ST_CADASTRO"] \
    .value_counts(normalize=True) \
    .unstack()

# plot
df_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6)
)

plt.ylabel("Proporção")
plt.xlabel("Cluster")
plt.title("Distribuição de ST_CADASTRO por Cluster")

plt.legend(title="Status", bbox_to_anchor=(1.05, 1))
plt.show()
# %%
sns.heatmap(df_plot, annot=True, fmt=".2f")
plt.title("Heatmap - ST_CADASTRO por Cluster")
plt.show()
