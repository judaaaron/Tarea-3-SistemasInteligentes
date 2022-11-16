from sklearn import tree
import pandas as pd
import sys
import pickle

#receive the first, second, third and fourth argument in command line and save them in variables
nombrearchivoentrenar = sys.argv[1]
nombrearchivovalidacion = sys.argv[2]
criterio = sys.argv[3]
try:
    depth = sys.argv[4]
except IndexError:
    depth = None


#validation data
validation_data = pd.read_csv(nombrearchivovalidacion)
clasesValidaciones = validation_data.pop("class")
validation_data = validation_data.replace({"Si": 1, "No": 0})
colsVal = validation_data.columns.tolist()
xVal = validation_data[colsVal]

#training data
training_data = pd.read_csv(nombrearchivoentrenar)
clases = training_data.pop('class')
training_data = training_data.replace({"Si": 1, "No": 0})
cols = training_data.columns.tolist()
x = training_data[cols]
y = clases


clf = tree.DecisionTreeClassifier(criterion=criterio, max_depth=None)
clf = clf.fit(x,y)
nombre = "DecisionTree-"+ criterio + "-" + depth + ".pkl"
print(nombre)
filename = nombre
with open(filename, 'wb') as f:
    pickle.dump(clf, f)