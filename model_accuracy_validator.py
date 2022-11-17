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
recall = recall_score(y, y_pred, average='macro')
print("Recall: ", recall)
f1 = f1_score(y, y_pred, average='macro')
print("F1: ", f1)


#acg precision sale de classification report

con_disp = ConfusionMatrixDisplay(confusion_matrix= con, display_labels = clasesNoRepetidas)
con_disp.plot()
plt.show()

isFile1 = os.path.exists("resultadosRandomForest.csv")
isFile2 = os.path.exists("resultadosDecisionTree.csv")
headerDT = ["Train Dataset", "Criterion", "Depth", "Train Acc.", "Val Acc.", "Val Avg. Prec", "Val Avg. F1", "Time Train", "Time Val."]
headerRF = ["Train Dataset", "N Trees", "Depth", "Train Acc.", "Val Acc.", "Val Avg. Prec", "Val Avg. F1", "Time Train", "Time Val."]

# if(modelFile.__contains__("RandomForest")):
#     if(isFile1):
#         with open('resultadosRandomForest.csv', 'a') as file:
#             writer = csv.writer(file)
#             writer.writerow([recall, f1, pickle_model["trainingTime"], pickle_model["trainingAcc"]])
#     else:
#         # si no existe el archivo, lo creo y agrego una linea
#         with open('resultadosRandomForest.csv', 'w') as file:
#             writer = csv.writer(file)
#             writer.writerow(headerRF)
#             writer.writerow([recall, f1, pickle_model["trainingTime"], pickle_model["trainingAcc"]])
#             print("Datos escritos en resultadosRandomForest.csv satisfactoriamente")
# else:
#     if(isFile2):
#         print("Existe el archivo")
#         with open('resultadosDecisionTree.csv', 'a') as file:
#             writer = csv.writer(file)
#             writer.writerow([modelFile])
#     else:
#         with open('resultadosDecisionTree.csv', 'w' ) as file:
#             writer = csv.writer(file)
#             writer.writerow(headerDT)
#             writer.writerow([modelFile])
    