# %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%

y = df["Fruta"]
carateristicas = [	"Arredondada",	"Suculenta",	"Vermelha",	"Doce"]
x = df[carateristicas]

# %%

arvore.fit(x,y)

# %%

arvore.predict([[0,0,0,0]])
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(arvore,
               feature_names=carateristicas,
               class_names=arvore.classes_,
               filled=True)

# %%

pobra = arvore.predict_proba([[0,0,0,0]])[0]
pd.Series(pobra, index=arvore.classes_)