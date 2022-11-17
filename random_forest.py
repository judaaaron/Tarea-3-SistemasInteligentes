import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report
import pickle


archivosTraining = ["training_data_small.csv", "training_data_medium.csv", "training_data_large.csv", "training_data_very_large.csv"]
nPosibles = [5, 10, 50, 100]
depths = [2, 4, 6]

#for each archivosTraining, for each nPosibles, for each depth, create a model and save it in a file

for trainFile in archivosTraining:
    for n in nPosibles:
        for depth in depths:
            i = 1
            preName = trainFile.split(".")[0]
            preName = preName.split("_")[2]
            
            training_data = pd.read_csv(trainFile)
            clases = training_data.pop('class')
            training_data = training_data.replace({"Si": 1, "No": 0})
            cols = training_data.columns.tolist()
            x = training_data[cols]
            y = clases
            
            rfc = RandomForestClassifier(n_estimators=n, max_depth=depth)
            rfc.fit(x,y)
            nombre = "./modelsRFC/RandomForest-" + preName + "-" + str(n) + "-" + str(depth) + ".pkl"
            filename = nombre
            with open(filename, 'wb') as f:
                pickle.dump(rfc, f)
            


