import sys
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score, classification_report, average_precision_score
import pickle
import matplotlib.pyplot as plt

import time
import csv
import os

#print(trainFile, n, depth)

modelFile = sys.argv[1]
validationFile = sys.argv[2]


#validation data
validation_data = pd.read_csv(validationFile)
clases = validation_data.pop('class')
clasesNoRepetidas = clases.drop_duplicates()
validation_data = validation_data.replace({"Si": 1, "No": 0})
cols = validation_data.columns.tolist()

x = validation_data[cols]
y = clases

#load model
f = open(modelFile, 'rb')
f.seek(0)
pickle_model = pickle.load(f)    

modelo = pickle_model["modelo"]
start = time.time()
y_pred = modelo.predict(x)
end = time.time()
print ("Tiempo total de predicción: ",end-start)


con = confusion_matrix(y, y_pred, labels=clasesNoRepetidas)
report = classification_report(y, y_pred, labels=clasesNoRepetidas, digits=4, output_dict=True, zero_division=0)

trainAcc = '{:.4f}'.format(pickle_model["trainingAcc"])
valAcc = '{:.4f}'.format(report["accuracy"])
avgF1 = '{:.4f}'.format(report["macro avg"]["f1-score"])
avgPrecision = '{:.4f}'.format(report["macro avg"]["precision"])
avgRecall = '{:.4f}'.format(report["macro avg"]["recall"])


con_disp = ConfusionMatrixDisplay(confusion_matrix= con, display_labels = clasesNoRepetidas)
feat_importances = pd.Series(modelo.feature_importances_, index=cols)
feat_importances.nlargest(15).plot(kind='barh')
plt.title("Importancia de las características")
# con_disp.plot()
plt.show()

isFile1 = os.path.exists("./Resultados/resultadosRandomForest.csv")
isFile2 = os.path.exists("./Resultados/resultadosDecisionTree.csv")
headerDT = ["Train Dataset", "Criterion", "Depth", "Train Acc.", "Val Acc.","Val. Avg Rec", "Val Avg. Prec", "Val Avg. F1", "Time Train", "Time Val."]
headerRF = ["Train Dataset", "N Trees", "Depth", "Train Acc.", "Val Acc.","Val. Avg Rec", "Val Avg. Prec", "Val Avg. F1", "Time Train", "Time Val."]

if(pickle_model["tipo"] == "RandomForest"):
    if(isFile1):
        with open('./Resultados/resultadosRandomForest.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if(pickle_model["size"] == "very"):
                pickle_model["size"] = "Very-Large"
            row = [pickle_model["size"], pickle_model["n_estimators"], pickle_model["depth"],trainAcc, valAcc, avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]) ,'{:.6f}'.format(end-start)]
            writer.writerow(row)
            print("Datos escritos en ./Resultados/resultadosRandomForest.csv satisfactoriamente")
    else:
        with open('./Resultados/resultadosRandomForest.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headerRF)
            if(pickle_model["size"] == "very"):
                pickle_model["size"] = "Very-Large"
            row = [pickle_model["size"], pickle_model["n_estimators"], pickle_model["depth"],trainAcc, valAcc, avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]) ,'{:.6f}'.format(end-start)]
            writer.writerow(row)
            print("Archivo creado y datos escritos en ./Resultados/resultadosRandomForest.csv satisfactoriamente")

elif(pickle_model["tipo"] == "DecisionTree"):
    if(isFile2):
        with open('./Resultados/resultadosDecisionTree.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if(pickle_model["size"] == "very"):
                pickle_model["size"] = "Very-Large"
            row = [pickle_model["size"], pickle_model["criterio"], pickle_model["depth"],trainAcc, valAcc, avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]) ,'{:.6f}'.format(end-start)]
            writer.writerow(row)
            print("Datos escritos en resultadosDecisionTree.csv satisfactoriamente")
    else:
        with open('./Resultados/resultadosDecisionTree.csv', 'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headerDT)
            if(pickle_model["size"] == "very"):
                pickle_model["size"] = "Very-Large"
            row = [pickle_model["size"], pickle_model["criterio"], pickle_model["depth"],trainAcc, valAcc, avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]) ,'{:.6f}'.format(end-start)]
            writer.writerow(row)
            print("Archivo creado y datos escritos en ./Resultados/resultadosDecisionTree.csv satisfactoriamente")