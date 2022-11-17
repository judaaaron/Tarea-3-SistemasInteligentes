import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import time

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
start = time.time()
y_pred = pickle_model.predict(x)
end = time.time()
print ("Tiempo total de predicción: ",end-start)


con = confusion_matrix(y, y_pred)
recall = recall_score(y, y_pred, average='macro')
print("Recall: ", recall)
f1 = f1_score(y, y_pred, average='macro')
print("F1: ", f1)


con_disp = ConfusionMatrixDisplay(confusion_matrix= con, display_labels = clasesNoRepetidas)
con_disp.plot()
plt.show()

