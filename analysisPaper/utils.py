import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import skimage as sm
import skimage.io
import skimage.measure

plt.rcParams.update({"font.size": 20})




def centroid(polygon):
    """takes polygon and finds the certre of mass in (x,y)"""

    polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
    pts = list(polygon.exterior.coords)
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sx = 0
    sy = 0
    a = polygon.area
    for i in range(len(pts) - 1):
        sx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        sy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    cx = sx / (6 * a)
    cy = sy / (6 * a)
    return (cx, cy)

# ---- fileType functions ----

def findStartTime(filename):
    if "Wound" in filename:
        dfwoundDetails = pd.read_excel(f"dat/woundDetails.xlsx")
        t0 = dfwoundDetails["Start Time"][dfwoundDetails["Filename"] == filename].iloc[
            0
        ]
    else:
        t0 = 0

    return t0


def getFilesType(fileType):
    if fileType == "All":
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            filenames.append(filename)

        if ".DS_Store" in filenames:
            filenames.remove(".DS_Store")
        if "woundDetails.xls" in filenames:
            filenames.remove("woundDetails.xls")
        if "woundDetails.xlsx" in filenames:
            filenames.remove("woundDetails.xlsx")
        if "dat_pred" in filenames:
            filenames.remove("dat_pred")

    else:
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            if fileType in filename:
                filenames.append(filename)

    filenames.sort()

    return filenames, fileType


def getFileTitle(fileType):
    if fileType == "WoundL18h":
        fileTitle = "large wound"
    elif fileType == "WoundS18h":
        fileTitle = "small wound"
    elif fileType == "Unwound18h":
        fileTitle = "unwounded"


    return fileTitle


def getBoldTitle(fileTitle):
    if len(str(fileTitle).split(" ")) == 1:
        boldTitle = r"$\bf{" + fileTitle + "}$"
    elif len(str(fileTitle).split(" ")) == 2:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + "}$"
        )
    elif len(str(fileTitle).split(" ")) == 3:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[2]
            + "}$"
        )
    elif len(str(fileTitle).split(" ")) == 4:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[2]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[3]
            + "}$"
        )
    elif len(str(fileTitle).split(" ")) == 5:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[2]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[3]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[4]
            + "}$"
        )

    return boldTitle



def getColorLineMarker(fileType):
    
    if "Unwound18h":
        return "tab:blue"
    if "WoundS18h":
        return "tab:orange"
    if "WoundL18h":
        return "tab:green"


# ---------------


def ThreeD(a):
    lst = [[[] for col in range(a)] for col in range(a)]
    return lst


def sortTime(df, t):
    tMin = t[0]
    tMax = t[1]

    dftmin = df[df["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


def sortRadius(dfVelocity, t, r):
    rMin = r[0]
    rMax = r[1]
    tMin = t[0]
    tMax = t[1]

    dfrmin = dfVelocity[dfVelocity["R"] >= rMin]
    dfr = dfrmin[dfrmin["R"] < rMax]
    dftmin = dfr[dfr["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


def sortGrid(dfVelocity, x, y):
    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]

    dfxmin = dfVelocity[dfVelocity["X"] > xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] > yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def sortVolume(dfShape, x, y, t):
    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]
    tMin = t[0]
    tMax = t[1]

    dfxmin = dfShape[dfShape["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    dfy = dfymin[dfymin["Y"] < yMax]

    dftmin = dfy[dfy["T"] >= tMin]
    df = dftmin[dftmin["T"] < tMax]

    return df


def sortSection(dfVelocity, r, theta):
    rMin = r[0]
    rMax = r[1]
    thetaMin = theta[0]
    thetaMax = theta[1]

    dfxmin = dfVelocity[dfVelocity["R"] > rMin]
    dfx = dfxmin[dfxmin["R"] < rMax]

    dfymin = dfx[dfx["Theta"] > thetaMin]
    df = dfymin[dfymin["Theta"] < thetaMax]

    return df


def sortBand(dfRadial, band, pixelWidth):
    if band == 1:
        df = dfRadial[dfRadial["Wound Edge Distance"] < pixelWidth]
    else:
        df2 = dfRadial[dfRadial["Wound Edge Distance"] < band * pixelWidth]
        df = df2[df2["Wound Edge Distance"] >= (band - 1) * pixelWidth]

    return df


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def rotation_matrix(theta):
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R