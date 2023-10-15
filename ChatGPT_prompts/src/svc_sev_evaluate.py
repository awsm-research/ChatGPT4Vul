import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pickle
import re

df = pd.read_csv("../../data/svc_severity/test.csv")
trues = df["CWE ID"].tolist()
sev_trues = df["Score"].tolist()

svc_preds = []
sev_preds = []
count = 0
filter_sev_true = []
model_name = "gpt-3.5-turbo"
for i in range(len(trues)):
    with open(f"../response/svc_sev_files/{model_name}/{i}.pkl", "rb") as f:
        data = pickle.load(f)
        pred = data.choices[0].message.content
    
    cwe_pred = re.findall('CWE-[0-9]+', pred)
    sev_pred = re.findall('[0-9]+.[0-9+]', pred)
    sev_pred = [p for p in sev_pred if "." in p]
    
    if cwe_pred == []:
        cwe_pred.append("CWE-NA")
    if sev_pred == []:
        count += 1
    else:
        # only append true and pred if ChatGPT returns a prediction
        sev_preds.append(float(sev_pred[0]))
        filter_sev_true.append(sev_trues[i])
    svc_preds.append(cwe_pred[0])
    

acc = accuracy_score(y_true=trues, y_pred=svc_preds)
mse = mean_squared_error(filter_sev_true, sev_preds)
mae = mean_absolute_error(filter_sev_true, sev_preds)

print("Accuracy: ", acc)
print("MSE: ", mse)
print("MAE: ", mae)

print(f"no sev pred count: {count}")