import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
import tifffile

# measure F1 score of GT and pred
def F1score(label, pred):
    return (2*np.sum(label[pred==1])/(2*np.sum(label[pred==1]) + np.sum(label[pred!=1]) + np.sum(pred[label!=1])))

# get files in folder
folder = f"dat_tissue/test_output"
cwd = os.getcwd()
Fullfilenames = os.listdir(cwd + "/" + folder)
fileList = []
for file in Fullfilenames:
    if file != ".DS_Store":
        fileList.append(file)

# make database of F1 scores
_df = []
for filename in fileList:
    label = sm.io.imread(f"dat_tissue/test_masks" + "/" + filename.replace("pred_img", "mask")).astype(int)//255

    pred = sm.io.imread(folder + "/" + filename).astype(int)[:,:,1]
    pred[pred<128] = 0
    pred[pred>=128] = 1


    _df.append(
        {
            "Filename": filename,
            "F1 score": F1score(label, pred),
        }
    )

dfConfusion = pd.DataFrame(_df)
dfConfusion.to_pickle(f"databases/dfConfusionTissue.pkl")

print(np.mean(dfConfusion["F1 score"]))