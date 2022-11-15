from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score

#entrada
#nombrearchivoentrenar nombrearchivovalidacion criterio depth

datos = pd.read_csv("training_data_small.csv")
cols = ["buenas_papas", "carne_fresca", "combos_familiares", "con_hongos", "con_huevo", "con_pepinillos", "de_pescado", "de_pollo", "jugosas", "juguetes", "malteadas", "mas_salsa", "opciones_quesos", "opciones_vegetarianas", "rapidez"]
clases = datos.pop('class')
# print(clases)
#sacamos a otro dataframe las class y esto iria en y

validation_data = pd.read_csv("validation_data.csv")
clasesValidaciones = validation_data.pop("class")
validation_data = validation_data.replace({"Si": 1, "No": 0})
validation_data = validation_data.iloc[:100]
xVal = validation_data[cols]
y = clasesValidaciones

datos = datos.replace({"Si": 1, "No": 0})
x = datos[cols]
y = clases
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
clf = clf.fit(x,y)

y_pred = clf.predict(xVal)
print("Accuracy:", accuracy_score(y, y_pred))
