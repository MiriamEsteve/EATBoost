import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def plotCV(result):
    # Generar los colores
    colors = []
    for i in range(10):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # Grafica para j
    for j in result['J'].unique():

        plt.title("MSE for J = ", j)
        plt.xlabel("Number of iterations (M)")
        plt.ylabel("MSE")

        # Funcion para cada v
        i = 0
        for v in result['v']:
            cv_j_v = result.loc[(result['v'] == v) & (result['J'] == j)]
            plt.errorbar(cv_j_v['M'], cv_j_v['MSE'], color=colors[i], label=str(v), capsize=3)
            i += 1

        plt.legend(loc='upper right')
        plt.savefig("graficas/graficaJ"+str(j)+".png")
        plt.show()
