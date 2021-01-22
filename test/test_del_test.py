import eat
import pandas as pd

matrix = pd.read_excel("C:/Users/meryg/Desktop/EATBoost/test/bbdd.xlsx", header=True)
deep_eat = eat.deep_EAT_for_EATBoost.deepEATBoost(matrix, [0,1], [4])
deep_eat.fit_deep_EAT()
print(deep_eat._predict())