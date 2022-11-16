import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report
import pickle

nombrearchivoentrenar = sys.argv[1]
nombrearchivovalidacion = sys.argv[2]
cantidad_n = sys.argv[3]
depth = sys.argv[4]

# cols = ["buenas_papas", "carne_fresca", "combos_familiares", "con_hongos", "con_huevo", "con_pepinillos", "de_pescado", "de_pollo", "jugosas", "juguetes", "malteadas", "mas_salsa", "opciones_quesos", "opciones_vegetarianas", "rapidez"]


training_data = pd.read_csv(nombrearchivoentrenar)
clases = training_data.pop("class")
training_data = training_data.replace({"Si": 1, "No": 0})

cols = training_data.columns.tolist()


validation_data = pd.read_csv(nombrearchivovalidacion)
validation_data = validation_data.replace({"Si": 1, "No": 0})
xVal = validation_data[cols]
x= training_data[cols]
y = clases


rfc = RandomForestClassifier(n_estimators=int(cantidad_n), max_depth=int(depth))
rfc.fit(x,y)

nombre = "RandomForest-"+ cantidad_n + "-" + depth + ".pkl"
print(nombre)
filename = nombre
with open(filename, 'wb') as f:
    pickle.dump(rfc, f)