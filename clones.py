# %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df
# %%
features = ["Distância Ombro a ombro",	"Tamanho do crânio"	,"Tamanho dos pés"	,"Status "]
target = ["General Jedi encarregado"]

X = df[features]
y = df[target]

X = X.replace({
    "Tipo 4": 4,
    "Tipo 2": 2,
    "Tipo 3": 3,
    "Tipo 1": 1,
    "Tipo 5": 5,
    "Defeituoso" : 0,
    "Apto" : 1
})

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)
# %%
