import pandas as pd 
import numpy as np 
from pyECLAT import ECLAT

base_mercado1 = pd.read_csv("mercado1.csv", header = None)

eclat = ECLAT(data = base_mercado1)
eclat.df_bin #Representação binaria 
eclat.uniq_ #Nome das colunas
indices, suporte = eclat.fit(min_support=0.3, min_combination=1, max_combination=3)

