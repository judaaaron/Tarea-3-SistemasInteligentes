from sklearn import tree
import pandas as pd
import sys
import pickle

#receive the first, second, third and fourth argument in command line and save them in variables
trainFile = sys.argv[1]
criterio = sys.argv[2]
try:
    depth = sys.argv[3]
except IndexError:
    depth = None

preName = trainFile.split(".")[1]
preName = preName.split("_")[2]

training_data = pd.read_csv(trainFile)
clases = training_data.pop('class')
training_data = training_data.replace({"Si": 1, "No": 0})
cols = training_data.columns.tolist()
x = training_data[cols]
y = clases

if depth is not None:
    depth = int(depth)
clf = tree.DecisionTreeClassifier(criterion=criterio, max_depth=depth)
clf = clf.fit(x,y)
nombre = "./Modelos/DecisionTree/DecisionTree-" + preName + "-"+ criterio + "-" + str(depth) + ".pkl"
filename = nombre
with open(filename, 'wb') as f:
    pickle.dump(clf, f)


print("Modelo guardado en:", nombre, "con criterio =", criterio, "y max_depth =", str(depth))