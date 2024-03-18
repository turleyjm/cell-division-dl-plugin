import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
import tifffile

# find matching division events between the ground truth and prediction using coords within a range of space-time
def sortConfusion(df, t, x, y, frameNum):

    a = df[df["T"] == t - frameNum]
    b = df[df["T"] == t]
    c = df[df["T"] == t + frameNum]

    df = pd.concat([a, b, c])

    xMax = x + 13
    xMin = x - 13
    yMax = y + 13
    yMin = y - 13
    if xMax > 511:
        xMax = 511
    if yMax > 511:
        yMax = 511
    if xMin < 0:
        xMin = 0
    if yMin < 0:
        yMin = 0

    dfxmin = df[df["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] <= xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    df = dfymin[dfymin["Y"] <= yMax]

    return df

# videos used in test dataset
fileType = "Test"
filenames = [
    "Unwound18h19",
    "Unwound18h20",
    "WoundL18h14",
    "WoundS18h17",
    "WoundXL18h05_",
]


# confusion matrix databases
_dfConfusion = []
for filename in filenames:
    # load the GT and predictions - from plugin made databases
    dfDL = pd.read_pickle(f"dat_division/dat_test/{filename}/dfDivisionDL{filename}.pkl")
    dfDivisions = pd.read_pickle(
        f"dat_division/dat_test/{filename}/dfDivisionGT{filename}.pkl"
    )

    for i in range(len(dfDivisions)):
        label = dfDivisions["Label"].iloc[i]
        ti = int(dfDivisions["T"].iloc[i])
        xi = int(dfDivisions["X"].iloc[i])
        yi = int(dfDivisions["Y"].iloc[i])
        ori = dfDivisions["Orientation"].iloc[i]

#         find matching GT and predictions 
        dfCon = sortConfusion(dfDL, ti, xi, yi, 1)
       
        if len(dfCon) > 0:
#             If match found then mark as true positive 
            label_DL = dfCon["Label"].iloc[0]
            t = dfCon["T"].iloc[0]
            x = dfCon["X"].iloc[0]
            y = dfCon["Y"].iloc[0]
            _dfConfusion.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Label DL": label_DL,
                    "T": int(t),
                    "X": x,
                    "Y": y,
                    "Orientation": ori,
                    "Orientation DL": dfCon["Orientation"].iloc[0] % 180,
                }
            )
        elif len(dfCon) == 0:
            # if false then count as false negative
            _dfConfusion.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Label DL": "falseNeg",
                    "T": int(ti),
                    "X": xi,
                    "Y": yi,
                    "Orientation": ori,
                }
            )

    for i in range(len(dfDL)):
        label = dfDL["Label"].iloc[i]
        ti = int(dfDL["T"].iloc[i])
        xi = int(dfDL["X"].iloc[i])
        yi = int(dfDL["Y"].iloc[i])
        
        # check cell division events from model are in ground truth
        dfCon = sortConfusion(dfDivisions, ti, xi, yi, 1)
        if len(dfCon) == 0:
            # if not then count as false positive
            _dfConfusion.append(
                {
                    "Filename": filename,
                    "Label": "falsePos",
                    "Label DL": label_DL,
                    "T": int(ti),
                    "X": xi,
                    "Y": yi,
                }
            )

# print F1 score of model 
dfConfusion = pd.DataFrame(_dfConfusion)
dfConfusion = dfConfusion[dfConfusion["T"] > 0]
dfConfusion = dfConfusion[dfConfusion["T"] < 90]
dfConfusion.to_pickle(f"databases/dfConfusionDivisions{fileType}.pkl")
falseNeg = len(dfConfusion[dfConfusion["Label DL"] == "falseNeg"])
falsePos = len(dfConfusion[dfConfusion["Label"] == "falsePos"])
print("U-NetCellDivision10")
print(
    f"True Pos: {len(dfConfusion) - falsePos - falseNeg}",
    f"False Pos: {falsePos}",
    f"False Neg: {falseNeg}",
)
truePos = len(dfConfusion) - falsePos - falseNeg
print(f"Dice score : {2*truePos/(2*truePos+falsePos+falseNeg)}")
print("excluding boundary division")
dfConfusion = dfConfusion[dfConfusion["X"] > 20]
dfConfusion = dfConfusion[dfConfusion["X"] < 492]
dfConfusion = dfConfusion[dfConfusion["Y"] > 20]
dfConfusion = dfConfusion[dfConfusion["Y"] < 492]
falseNeg = len(dfConfusion[dfConfusion["Label DL"] == "falseNeg"])
falsePos = len(dfConfusion[dfConfusion["Label"] == "falsePos"])
print(
    f"True Pos: {len(dfConfusion) - falsePos - falseNeg}",
    f"False Pos: {falsePos}",
    f"False Neg: {falseNeg}",
)
truePos = len(dfConfusion) - falsePos - falseNeg
print(f"Dice score : {2*truePos/(2*truePos+falsePos+falseNeg)}")

# error in the DL orientation measurements 
dfConfusion = pd.read_pickle(f"databases/dfConfusionDivisions{fileType}.pkl")
dfConfusion = dfConfusion[dfConfusion["X"] > 20]
dfConfusion = dfConfusion[dfConfusion["X"] < 492]
dfConfusion = dfConfusion[dfConfusion["Y"] > 20]
dfConfusion = dfConfusion[dfConfusion["Y"] < 492]
df = dfConfusion[dfConfusion["Label DL"] != "falseNeg"]
df = df[df["Label"] != "falsePos"]

ori_err = []
for i in range(len(df)):
    # find the abs differences in angle between 
    ori = df["Orientation"].iloc[i] * np.pi / 180
    ori_mask = df["Orientation DL"].iloc[i] * np.pi / 180

    error = np.dot(
        np.array([np.cos(2 * ori), np.sin(2 * ori)]),
        np.array([np.cos(2 * ori_mask), np.sin(2 * ori_mask)]),
    )
    dtheta = (np.arccos(error) / 2) * 180 / np.pi
    ori_err.append(dtheta)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(ori_err, density=False, bins=18)
ax.set_xlim([0, 90])
ax.set(xlabel="Orientation error", ylabel="Frequency")
ax.axvline(np.median(ori_err), c="k", label="mean")
fig.savefig(
    f"results/orientationError test data.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


# show divisions on focused 2D video
if False:

    for filename in filenames:
        # load database and video
        dfDivisions = pd.read_pickle(f"dat_division/dat_test/{filename}/dfDivision{filename}.pkl")
        vidFocus = sm.io.imread(f"dat_division/dat_test/{filename}/focus{filename}.tif").astype(float)
        
        [T, X, Y, rgb] = vidFocus.shape
        highlightDivisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

        
        for i in range(len(dfDivisions)):
            # for each cell division mark a blue disk around the site
            t, x, y = (
                dfDivisions["T"].iloc[i],
                dfDivisions["X"].iloc[i],
                dfDivisions["Y"].iloc[i],
            )
            x = int(x)
            y = int(y)
            t0 = int(t)
            ori = dfDivisions["Orientation"].iloc[i] * np.pi / 180

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)
            
            # draw line in direction of oriantation of division
            rr2, cc2, val = sm.draw.line_aa(
                int(551 - (y + 16 * np.sin(ori) + 20)),
                int(x + 16 * np.cos(ori) + 20),
                int(551 - (y - 16 * np.sin(ori) + 20)),
                int(x - 16 * np.cos(ori) + 20),
            )

            times = range(t0, t0 + 2)

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            i = 1
            for t in timeVid:
                highlightDivisions[t][rr0, cc0, 2] = 250 / i
                highlightDivisions[t][rr1, cc1, 2] = 0
                highlightDivisions[t][rr2, cc2, 2] = 250
                i += 1

        highlightDivisions = highlightDivisions[:, 20:532, 20:532]

        # save file
        highlightDivisions = np.asarray(highlightDivisions, "uint8")
        tifffile.imwrite(
            f"dat_division/dat_test/{filename}/divisions{filename}.tif",
            highlightDivisions,
        )

# show false positives and negatives U-NetCellDivision10
if False:
    dfConfusion = pd.read_pickle(f"databases/dfConfusionDivisions{fileType}.pkl")

    for filename in filenames:

        dfConfusionf = dfConfusion[dfConfusion["Filename"] == filename]
        vidFocus = sm.io.imread(f"dat_division/dat_test/{filename}/focus{filename}.tif").astype(float)
        [T, X, Y, rgb] = vidFocus.shape

        highlightDivisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

        dfFalsePos = dfConfusionf[dfConfusionf["Label"] == "falsePos"]

        for i in range(len(dfFalsePos)):
            t, x, y = (
                dfFalsePos["T"].iloc[i],
                dfFalsePos["X"].iloc[i],
                dfFalsePos["Y"].iloc[i],
            )
            x = int(x)
            y = int(y)
            t0 = int(t)

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

            times = range(t0, t0 + 2)

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            for t in timeVid:
                highlightDivisions[t][rr0, cc0, 2] = 250
                highlightDivisions[t][rr1, cc1, 2] = 0

        highlightDivisions = highlightDivisions[:, 20:532, 20:532]

        highlightDivisions = np.asarray(highlightDivisions, "uint8")
        tifffile.imwrite(
            f"dat_division/dat_test/{filename}/falsePositives{filename}.tif",
            highlightDivisions,
        )

        vidFocus = sm.io.imread(f"dat_division/dat_test/{filename}/focus{filename}.tif").astype(float)

        highlightDivisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

        dfFalseNeg = dfConfusionf[dfConfusionf["Label DL"] == "falseNeg"]

        for i in range(len(dfFalseNeg)):
            t, x, y = (
                dfFalseNeg["T"].iloc[i],
                dfFalseNeg["X"].iloc[i],
                dfFalseNeg["Y"].iloc[i],
            )
            x = int(x)
            y = int(y)
            t0 = int(t)

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

            times = range(t0, t0 + 2)

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            for t in timeVid:
                highlightDivisions[t][rr0, cc0, 2] = 250
                highlightDivisions[t][rr1, cc1, 2] = 0

        highlightDivisions = highlightDivisions[:, 20:532, 20:532]

        highlightDivisions = np.asarray(highlightDivisions, "uint8")
        tifffile.imwrite(
            f"dat_division/dat_test/{filename}/falseNegatives{filename}.tif",
            highlightDivisions,
        )
