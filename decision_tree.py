from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score

#entrada
#nombrearchivoentrenar nombrearchivovalidacion criterio depth

cols = ["buenas_papas", "carne_fresca", "combos_familiares", "con_hongos", "con_huevo", "con_pepinillos", "de_pescado", "de_pollo", "jugosas", "juguetes", "malteadas", "mas_salsa", "opciones_quesos", "opciones_vegetarianas", "rapidez"]


#validation data
validation_data = pd.read_csv("validation_data.csv")
clasesValidaciones = validation_data.pop("class")
validation_data = validation_data.replace({"Si": 1, "No": 0})
xVal = validation_data[cols]


#training data
datos = pd.read_csv("training_data_small.csv")
clases = datos.pop('class')
datos = datos.replace({"Si": 1, "No": 0})
x = datos[cols]
y = clases

#decision tree creation
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
#training
clf = clf.fit(x,y)
#test with validation data
y_pred = clf.predict(xVal)
print("Accuracy:", accuracy_score(y, y_pred))
