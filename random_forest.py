import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report

nombrearchivoentrenar = sys.argv[1]
nombrearchivovalidacion = sys.argv[2]
cantidad_n = sys.argv[3]
depth = sys.argv[4]

# cols = ["buenas_papas", "carne_fresca", "combos_familiares", "con_hongos", "con_huevo", "con_pepinillos", "de_pescado", "de_pollo", "jugosas", "juguetes", "malteadas", "mas_salsa", "opciones_quesos", "opciones_vegetarianas", "rapidez"]


training_data = pd.read_csv(nombrearchivoentrenar)
clases = training_data.pop("class")
training_data = training_data.replace({"Si": 1, "No": 0})
print()

cols = training_data.columns.tolist()


validation_data = pd.read_csv(nombrearchivovalidacion)
validation_data = validation_data.replace({"Si": 1, "No": 0})

x= training_data[cols]
y = clases


rfc = RandomForestClassifier(n_estimators=int(cantidad_n), max_depth=int(depth))
rfc.fit(x,y)

y_pred = rfc.predict(x)

print(classification_report(y_pred, y, digits = 4))

print()

# print(type(clases))