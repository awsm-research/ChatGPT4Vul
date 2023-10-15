import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

df = pd.read_csv("../../data/svp/gpt35_prediction.csv")
trues = df["target"].tolist()
flaw_lines = df["flaw_line"].tolist()

correct_pred = []
for i in range(len(flaw_lines)):
    try:
        with open(f"../response/svp_line_files/gpt-3.5-turbo/{i}.pkl", "rb") as f:
            data = pickle.load(f)
            pred = data.choices[0].message.content
        vul_lines = flaw_lines[i]
        vul_lines = vul_lines.split("/~/")
        vul_lines = [v.strip() for v in vul_lines]
        for vl in vul_lines:
            if vl in pred:
                correct_pred.append(1)
            else:
                correct_pred.append(0)
    except:
        continue
print(sum(correct_pred)/len(correct_pred)) 