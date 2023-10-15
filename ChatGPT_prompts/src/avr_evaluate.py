import pandas as pd
import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("../../data/avr/test.csv")
trues = df["target"].tolist()
correctness = []
bleu = []
meteor = []
model_name = "gpt-3.5-turbo"
for i in range(len(trues)):
    with open(f"../response/avr_files/{model_name}/{i}.pkl", "rb") as f:
        data = pickle.load(f)
        pred = data.choices[0].message.content
    pred = pred.split("\n")
    pred = " ".join(pred)
    
    if pred == trues[i]:
        correctness.append(1)
    else:
        correctness.append(0)

    # Compute BLEU score
    bleu_score = sentence_bleu([trues[i].split(" ")], pred.split(" "))
    bleu.append(bleu_score)
    
    me_score = meteor_score.meteor_score([trues[i].split(" ")], pred.split(" "))
    meteor.append(me_score)
        
print("%PP: ", round(sum(correctness)/len(correctness), 2))
print("BLEU: ", round(sum(bleu)/len(bleu), 4))
print("METEOR: ", round(sum(meteor)/len(meteor), 4))