import pandas as pd
import numpy as np


base_census = pd.read_csv("census.csv")

colunas = base_census.columns[:-1]

X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values


from sklearn.preprocessing import LabelEncoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])


from sklearn.preprocessing import MinMaxScaler #NORMALIZAÇÃO
scaler = MinMaxScaler()
X_census_scaler = scaler.fit_transform(X_census)



# ----- LOW VARIANCE: Enontrar quais dos 14 atributos, são os mais importântes
X_census.shape[1]

for i in range(X_census.shape[1]):
    print(X_census_scaler[:, i].var())

from sklearn.feature_selection import VarianceThreshold
selecao = VarianceThreshold(threshold=0.05)
X_census_variancia = selecao.fit_transform(X_census_scaler)
X_census_variancia.shape

#selecao.variances_ =  for i in range(X_census.shape[1]):
#                      print(X_census_scaler[:, i].var())

indices = np.where(selecao.variances_ > 0.05)
colunas[indices]

base_census_variancia = base_census.drop(columns = ["age", "workclass", "final.weight", 
                                                    "education.num", "race", "capital.gain", 
                                                    "capital.loos", "hour.per.week", "native.country"], axis = 1)

X_census_variancia = base_census_variancia.iloc[:, 0:5].values
y_census_variancia = base_census_variancia.iloc[:, 5].values

X_census_variancia[:, 0] = label_encoder_education.fit_transform(X_census_variancia[:,0])
X_census_variancia[:, 1] = label_encoder_marital.fit_transform(X_census_variancia[:,1])
X_census_variancia[:, 2] = label_encoder_occupation.fit_transform(X_census_variancia[:,2])
X_census_variancia[:, 3] = label_encoder_relationship.fit_transform(X_census_variancia[:,3])
X_census_variancia[:, 4] = label_encoder_sex.fit_transform(X_census_variancia[:,4])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,2,3,4])],remainder='passthrough')
X_census_variancia = onehotencorder.fit_transform(X_census_variancia).toarray()
X_census_variancia


#NORMALIZAÇÃO
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_census_variancia = scaler.fit_transform(X_census_variancia)
X_census_variancia


from sklearn.model_selection import train_test_split
X_census_treinamento_var, X_census_teste_var, y_census_treinamento_var, y_census_teste_var = train_test_split(X_census_variancia, y_census_variancia, test_size = 0.15, random_state = 0)
X_census_treinamento_var.shape, X_census_teste_var.shape

from sklearn.ensemble import RandomForestClassifier
random_forest_var = RandomForestClassifier(criterion = 'entropy', min_samples_leaf =  1, min_samples_split = 5, n_estimators = 200)
random_forest_var.fit(X_census_treinamento_var, y_census_treinamento_var)


from sklearn.metrics import accuracy_score, classification_report
previsoes = random_forest_var.predict(X_census_teste_var)
accuracy_score(y_census_teste_var, previsoes)



#------------EXTRA TREE: Extra trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

X_census_scaler.shape

selecao = ExtraTreesClassifier()
selecao.fit(X_census_scaler, y_census)

importancias = selecao.feature_importances_ #Importância dos atributos em porcentagem

indices = []
for i in range(len(importancias)):
    if importancias[i] >= 0.029:
        indices.append(i)

indices
colunas[indices]

X_census_extra = X_census[:, indices]


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7])],remainder='passthrough')
X_census_extra = onehotencorder.fit_transform(X_census_extra).toarray()
X_census_extra.shape


from sklearn.model_selection import train_test_split
X_census_treinamento_extra, X_census_teste_extra, y_census_treinamento_extra, y_census_teste_extra = train_test_split(X_census_extra, y_census, test_size = 0.15, random_state = 0)
X_census_treinamento_extra.shape, X_census_teste_extra.shape


from sklearn.ensemble import RandomForestClassifier
random_forest_extra = RandomForestClassifier(criterion = 'entropy', min_samples_leaf =  1, min_samples_split = 5, n_estimators = 200)
random_forest_extra.fit(X_census_treinamento_extra, y_census_treinamento_extra)


from sklearn.metrics import accuracy_score, classification_report
previsoes = random_forest_extra.predict(X_census_teste_extra)
accuracy_score(y_census_teste_extra, previsoes)
#0.8435
