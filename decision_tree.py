from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from openpyxl import Workbook  
from openpyxl.chart import BarChart, Reference  
import sys


#entrada
#nombrearchivoentrenar nombrearchivovalidacion criterio depth

#receive the first, second, third and fourth argument in command line and save them in variables
nombrearchivoentrenar = sys.argv[1]
nombrearchivovalidacion = sys.argv[2]
criterio = sys.argv[3]
try:
    depth = sys.argv[4]
except IndexError:
    depth = None


cols = ["buenas_papas", "carne_fresca", "combos_familiares", "con_hongos", "con_huevo", "con_pepinillos", "de_pescado", "de_pollo", "jugosas", "juguetes", "malteadas", "mas_salsa", "opciones_quesos", "opciones_vegetarianas", "rapidez"]


#validation data
validation_data = pd.read_csv("validation_data.csv", header=None, names=cols)
clasesValidaciones = validation_data.pop("class")
validation_data = validation_data.replace({"Si": 1, "No": 0})
#validation_data = validation_data.iloc[:100] #esto
xVal = validation_data



#training data
datos = pd.read_csv("training_data_small.csv")
clases = datos.pop('class')
datos = datos.replace({"Si": 1, "No": 0})
x = datos[cols]
y = clases

# print(x.shape)
print("mjmjmjm",y.shape)


#decision tree creation
clf = tree.DecisionTreeClassifier(criterion=criterio, max_depth=None)
#training
clf = clf.fit(x,y)
# #test with validation data
y_pred = clf.predict(xVal)
print(y_pred, "que esta pasando")


report = classification_report(y, y_pred, digits = 4)
print(report)
