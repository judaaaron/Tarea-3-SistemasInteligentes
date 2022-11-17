import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report
import pickle


trainFile = sys.argv[1]
n = sys.argv[2]
depth = sys.argv[3]

#for each archivosTraining, for each nPosibles, for each depth, create a model and save it in a file

#print(trainFile, n, depth)
preName = trainFile.split(".")[1]
preName = preName.split("_")[2]

training_data = pd.read_csv(trainFile)
clases = training_data.pop('class')
training_data = training_data.replace({"Si": 1, "No": 0})
cols = training_data.columns.tolist()
x = training_data[cols]
y = clases

rfc = RandomForestClassifier(n_estimators=int(n), max_depth=int(depth))
rfc.fit(x,y)
nombre = "./Modelos/RandomForest/RandomForest-" + preName + "-" + str(n) + "-" + str(depth) + ".pkl"
with open(nombre, 'wb') as f:
    pickle.dump(rfc, f)

print("Modelo guardado en:", nombre, "con n_estimators = ", n, "y max_depth = ", depth)
            


