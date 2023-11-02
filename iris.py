import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def read_data(path):
  return pd.read_csv(path)

# Obtiene los datos separados entre entradas y objetivo
df = read_data('Iris.csv')
input = df.drop(columns=["Species"])
target = df["Species"].values
print(input, target)

# Crea el modelo
knn = KNeighborsClassifier(n_neighbors=1)
# Hace el entrenamiento para cada set de indices
cv_scores = cross_val_score(knn, input, target, cv=10)

# Prueba el modelo
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))