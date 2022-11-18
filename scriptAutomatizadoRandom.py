import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pickle
import time

trainingfiles = ["training_data_small.csv", "training_data_medium.csv", "training_data_large.csv", "training_data_very_large.csv"]
ntotales = [5,10,50,100]
depths = [2,4,6]

for trainFile in trainingfiles:
    for n in ntotales:
        for depth in depths:

            preName = trainFile.split(".")[0]
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

            datos = {"trainingTime": end-start, "trainingAcc": val, "modelo": rfc, "n_estimators": n, "depth": depth, "tipo": "RandomForest", "size": preName}
            print("Datos de entrenamiento: ",datos)

            nombre = "./Modelos/RandomForest/RandomForest-" + preName + "-" + str(n) + "-" + str(depth) + ".pkl"
            with open(nombre, 'wb') as f:
                pickle.dump(datos, f)

            print("Modelo guardado en:", nombre)