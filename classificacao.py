# %%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df["aprovado"] = (df["nota"] >5).astype(int)
df
# %%

plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.title("cerveja vs aprovacao")
plt.xlabel("cervejas")
plt.ylabel("aprovado")

# %%

from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

reg= linear_model.LogisticRegression(penalty=None,
                                     fit_intercept=True)

reg.fit(df[["cerveja"]], df["aprovado"])

regpredict = reg.predict(df[["cerveja"]].drop_duplicates())
reg_prob = reg.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

arvore_fuul = tree.DecisionTreeClassifier(random_state=42)
arvore_fuul.fit(df[["cerveja"]], df["aprovado"])
arvore_fuul_predict = arvore_fuul.predict(df[["cerveja"]].drop_duplicates())
arvore_fuul_proba = arvore_fuul.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[["cerveja"]], df["aprovado"])
arvore_d2_predict = arvore_d2.predict(df[["cerveja"]].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(df[["cerveja"]], df["aprovado"])
nb_predict = nb.predict(df[["cerveja"]].drop_duplicates())
nb_proba = nb.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.title("cerveja vs aprovacao")
plt.xlabel("cervejas")
plt.ylabel("aprovado")
plt.plot(df["cerveja"].drop_duplicates(), regpredict, color="tomato")
plt.plot(df["cerveja"].drop_duplicates(), reg_prob, color="red")

plt.plot(df["cerveja"].drop_duplicates(), nb_predict, color="green")
plt.plot(df["cerveja"].drop_duplicates(), nb_proba, color="magenta")

plt.plot(df["cerveja"].drop_duplicates(), arvore_d2_predict, color="blue")
plt.plot(df["cerveja"].drop_duplicates(), arvore_d2_proba, color="black")

plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", colors="black")

plt.legend(["obervacao", 
            "reg predict", 
            "red proba",
            "nb_predict",
            "nb_proba", 
            "arvore d2 predict",
            "arvore d2 proba"
            ])

# %%
