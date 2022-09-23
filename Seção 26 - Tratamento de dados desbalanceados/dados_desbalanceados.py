import pandas as pd
import numpy as np
import seaborn as sns


# PREPARAÇÃO DOS DADOS

base_census = pd.read_csv("census.csv")
np.unique(base_census["income"], return_counts=True) #Dados desbalanceados. A quantidade de um está muito maior que outro.

sns.countplot(x= base_census["income"])

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


#---- SUBAMOSTRAGEM COM TOMEK LINKS
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy = "all")
X_under, y_under = tl.fit_resample(X_census, y_census)

X_under.shape, y_under.shape
np.unique(y_census, return_counts=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
X_census = onehotencorder.fit_transform(X_under).toarray()
X_census


from sklearn.model_selection import train_test_split
X_census_treinamento_under, X_census_teste_under, y_census_treinamento_under, y_census_teste_under = train_test_split(X_under, y_under, test_size=0.15, random_state=0)
X_census_treinamento_under.shape, X_census_teste_under.shape


# 84.70% com os dados originais
from sklearn.ensemble import RandomForestClassifier
random_forest_census = RandomForestClassifier(criterion = 'entropy', min_samples_leaf =  1, min_samples_split = 5, n_estimators = 100)
random_forest_census.fit(X_census_treinamento_under, y_census_treinamento_under)


from sklearn.metrics import accuracy_score, classification_report
previsoes = random_forest_census.predict(X_census_teste_under)
accuracy_score(y_census_teste_under, previsoes)
#86.04% com o TOMEK LINK

print(classification_report(y_census_teste_under, previsoes))



#------ SOBREAMOSTRAGEM COM SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy = "minority")
X_over, y_over = smote.fit_resample(X_census, y_census)

X_over.shape

np.unique(y_census, return_counts=True)
np.unique(y_over, return_counts=True)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
X_census = onehotencorder.fit_transform(X_over).toarray()
X_census

 
from sklearn.model_selection import train_test_split
X_census_treinamento_over, X_census_teste_over, y_census_treinamento_over, y_census_teste_over = train_test_split(X_over, y_over, test_size=0.15, random_state=0)
X_census_treinamento_over.shape, X_census_teste_over.shape


from sklearn.ensemble import RandomForestClassifier
random_forest_census = RandomForestClassifier(criterion = 'entropy', min_samples_leaf =  1, min_samples_split = 5, n_estimators = 200)
random_forest_census.fit(X_census_treinamento_over, y_census_treinamento_over)


from sklearn.metrics import accuracy_score, classification_report
previsoes = random_forest_census.predict(X_census_teste_over)
accuracy_score(y_census_teste_over, previsoes)
#90% com SMOTE

print(classification_report(y_census_teste_over, previsoes))