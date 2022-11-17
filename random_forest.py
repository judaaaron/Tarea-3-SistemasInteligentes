import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle
import time

trainFile = sys.argv[1]
n = sys.argv[2]
depth = sys.argv[3]

#for each archivosTraining, for each nPosibles, for each depth, create a model and save it in a file

#print(trainFile, n, depth)
preName = trainFile.split(".")[1]
preName = preName.split("_")[2]

start = time.time()
training_data = pd.read_csv(trainFile)
clases = training_data.pop('class')
training_data = training_data.replace({"Si": 1, "No": 0})
cols = training_data.columns.tolist()
x = training_data[cols]
y = clases

rfc = RandomForestClassifier(n_estimators=int(n), max_depth=int(depth))
rfc.fit(x,y)

end = time.time()
print("Tiempo total de entrenamiento Random Forest: ", end-start) 

y_pred = rfc.predict(x)
val = accuracy_score(y_true=y, y_pred=y_pred)

datos = {"trainingTime": end-start, "trainingAcc": val, "modelo": rfc}
print("Datos de entrenamiento: ",datos)

nombre = "./Modelos/RandomForest/RandomForest-" + preName + "-" + str(n) + "-" + str(depth) + ".pkl"
with open(nombre, 'wb') as f:
    pickle.dump(datos, f)

print("Modelo guardado en:", nombre)
