from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
import pickle
import time

#receive the first, second, third and fourth argument in command line and save them in variables
trainingfiles = ["training_data_small.csv", "training_data_medium.csv", "training_data_large.csv", "training_data_very_large.csv"]
criterios = ["gini", "entropy"]
depths = [2,4,8,None]

for trainFile in trainingfiles:
    for criterio in criterios:
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

            if depth is not None:
                depth = int(depth)
            clf = DecisionTreeClassifier(criterion=criterio, max_depth=depth)
            clf = clf.fit(x,y)

            end = time.time()
            print("Tiempo total de entrenamiento Decision Tree: ", end-start)  

            y_pred = clf.predict(x)
            val = accuracy_score(y_true=y, y_pred=y_pred)

            datos = {"trainingTime": end-start, "trainingAcc": val, "modelo": clf, "criterio": criterio, "depth": depth, "tipo": "DecisionTree", "size": preName}
            print("Datos de entrenamiento: ",datos)

            nombre = "./Modelos/DecisionTree/DecisionTree-" + preName + "-"+ criterio + "-" + str(depth) + ".pkl"
            filename = nombre
            with open(filename, 'wb') as f:
                pickle.dump(datos, f)


            print("Modelo guardado en:", nombre)