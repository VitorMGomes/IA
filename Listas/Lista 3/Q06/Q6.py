import numpy as np
import pandas as pd

cm = np.array([
    [10, 4, 2, 1],  #A
    [1, 15, 2, 0],  #B
    [2, 3, 20, 5],  #C
    [4, 1, 2, 50]   #D
])

classes = ["A","B","C","D"]
metrics = {}

total = cm.sum()

for i, label in enumerate(classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = total - (TP + FN + FP)
    
    
    precisao = TP / (TP + FP) if (TP+FP) > 0 else 0
    recall   = TP / (TP + FN) if (TP+FN) > 0 else 0
    f1       = 2 * precisao * recall / (precisao + recall) if (precisao+recall) > 0 else 0
    tvp      = recall
    tfn      = FN / (TP + FN) if (TP+FN) > 0 else 0
    tfp      = FP / (FP + TN) if (FP+TN) > 0 else 0
    tvn      = TN / (FP + TN) if (FP+TN) > 0 else 0
    
    metrics[label] = [precisao, recall, f1, tvp, tfn, tfp, tvn]

df_metrics = pd.DataFrame(metrics, 
                          index=["Precisão","Recall","F1Score","TVP","TFN","TFP","TVN"]).T

#? df_percent = (df_metrics * 100).round(2)

print(df_metrics)