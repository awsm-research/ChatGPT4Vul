import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import re

df = pd.read_csv("../../data/svp/test.csv")

trues = df["target"].tolist()
funcs = df["func_before"].tolist()

all_preds = []
count = 0
model_name = "gpt-3.5-turbo"
for i in range(len(trues)):
    try:
        with open(f"../response/svp_files/{model_name}/{i}.pkl", "rb") as f:
            data = pickle.load(f)
            pred = data.choices[0].message.content
            all_preds.append(int(pred))
    except:
        all_preds.append(0)
        count += 1
    
acc = accuracy_score(y_true=trues, y_pred=all_preds)
f1 = f1_score(y_true=trues, y_pred=all_preds)
pre = precision_score(y_true=trues, y_pred=all_preds)
recall = recall_score(y_true=trues, y_pred=all_preds)

print("Accuracy: ", acc)
print("F1: ", f1)
print("Precision: ", pre)
print("Recall: ", recall)
print(count)
