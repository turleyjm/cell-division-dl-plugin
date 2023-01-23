"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import os
import shutil

import napari
import numpy as np
import pandas as pd
import skimage as sm
import tifffile
import timm
import torch
import torch.nn as nn
from fastai.vision.models.unet import DynamicUnet
from magicgui import magic_factory
from napari.layers import Image
from scipy import ndimage
from skimage.feature import blob_log

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def sortDL(df, t, x, y):

    a = df[df["T"] == t - 2]
    b = df[df["T"] == t - 1]
    c = df[df["T"] == t + 1]
    d = df[df["T"] == t + 2]

    df = pd.concat([a, b, c, d])

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
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def intensity(vid, ti, xi, yi):

    [T, X, Y] = vid.shape

    vidBoundary = np.zeros([T, 552, 552])

    for x in range(X):
        for y in range(Y):
            vidBoundary[:, 20 + x, 20 + y] = vid[:, x, y]

    rr, cc = sm.draw.disk([yi + 20, xi + 20], 9)
    div = vidBoundary[ti][rr, cc]
    div = div[div > 0]

    mu = np.mean(div)

    return mu


def maskOrientation(mask):
    S = np.zeros([2, 2])
    X, Y = mask.shape
    x = np.zeros([X, Y])
    y = np.zeros([X, Y])
    x += np.arange(X)
    y += (Y - 1 - np.arange(Y)).reshape(Y, 1)
    A = np.sum(mask)
    Cx = np.sum(x * mask) / A
    Cy = np.sum(y * mask) / A
    xx = (x - Cx) ** 2
    yy = (y - Cy) ** 2
    xy = (x - Cx) * (y - Cy)
    S[0, 0] = -np.sum(yy * mask) / A**2
    S[1, 0] = S[0, 1] = np.sum(xy * mask) / A**2
    S[1, 1] = -np.sum(xx * mask) / A**2
    TrS = S[0, 0] + S[1, 1]
    inertia = np.zeros(shape=(2, 2))
    inertia[0, 0] = 1
    inertia[1, 1] = 1
    q = S - TrS * inertia / 2
    theta = np.arctan2(q[0, 1], q[0, 0]) / 2

    return theta * 180 / np.pi


def blurFocusStack(vid, focusRange):

    vid = vid.astype("uint16")
    (T, Z, Y, X) = vid.shape
    variance = np.zeros([T, Z, Y, X])
    varianceMax = np.zeros([T, Y, X])
    surface = np.zeros([T, Y, X])
    focus = np.zeros([T, Y, X])
    focusDown = np.zeros([T, Y, X])
    focusUp = np.zeros([T, Y, X])

    for t in range(T):
        for z in range(Z):
            winMean = ndimage.uniform_filter(
                vid[t, z], (focusRange, focusRange)
            )
            winSqrMean = ndimage.uniform_filter(
                vid[t, z] ** 2, (focusRange, focusRange)
            )
            variance[t, z] = winSqrMean - winMean**2

    varianceMax = np.max(variance, axis=1)

    for z in range(Z):
        surface[variance[:, z] == varianceMax] = z

    for z in range(Z):
        focus[surface == z] = vid[:, z][surface == z]

    for z in range(Z - 1):
        focusUp[surface == z] = vid[:, z + 1][surface == z]
    focusUp[surface == z + 1] = vid[:, z + 1][surface == z + 1]

    focusDown[surface == 0] = vid[:, 0][surface == 0]
    for z in range(1, Z):
        focusDown[surface == z] = vid[:, z - 1][surface == z]

    focusUp = focusUp.astype("uint8")
    focus = focus.astype("uint8")
    focusDown = focusDown.astype("uint8")

    focusUp, focus, focusDown = normaliseBlur(focusUp, focus, focusDown, 60)

    return focusUp, focus, focusDown


def normaliseBlur(focusUp, focus, focusDown, mu0):
    focus = focus.astype("float")
    (T, X, Y) = focus.shape

    for t in range(T):
        mu = focus[t, 50:450, 50:450][focus[t, 50:450, 50:450] > 0]

        mu = np.quantile(mu, 0.5)

        ratio = mu0 / mu

        focus[t] = focus[t] * ratio
        focus[t][focus[t] > 255] = 255

        focusUp[t] = focusUp[t] * ratio
        focusUp[t][focusUp[t] > 255] = 255

        focusDown[t] = focusDown[t] * ratio
        focusDown[t][focusDown[t] > 255] = 255

    return (
        focusUp.astype("uint8"),
        focus.astype("uint8"),
        focusDown.astype("uint8"),
    )

    # ---------- Find Divisions ----------


@magic_factory(
    Dropdown={
        "choices": [
            "Division heatmap",
            "Division database",
            "Division & orientaton database",
        ]
    }
)
def cellDivision(
    Image_layer: "napari.layers.Image", Dropdown="Division heatmap"
) -> Image:
    filename = Image_layer.name
    resnet = timm.create_model("resnet34", pretrained=True)
    resnet.conv1 = nn.Conv2d(
        10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    m = resnet
    m = nn.Sequential(*list(m.children())[:-2])
    model = DynamicUnet(m, 1, (512, 512), norm_type=None).to(DEVICE)
    load_checkpoint(
        torch.load(
            "models/UNetCellDivision10.pth.tar",
            map_location=torch.device("cpu"),
        ),
        model,
    )

    DLinput = Image_layer.data
    if len(DLinput.shape) != 4:
        return "ERROR stack must be 4D"
    T, X, Y, rgb = DLinput.shape
    if rgb < 2:
        return """ERROR stack must had two or more colour
         channels (only first 2 will be used)"""

    mask = np.zeros([T, X, Y])
    DLinput = np.array(DLinput)

    for t in range(T - 4):
        vid = np.zeros([10, X, Y])
        for j in range(5):
            vid[2 * j] = DLinput[int(t + j), :, :, 1]
            vid[2 * j + 1] = DLinput[int(t + j), :, :, 0]

        vid = torch.tensor([vid / 256]).float()
        mask[t + 1] = torch.sigmoid(model(vid)).detach().numpy()

    if Dropdown == "Division heatmap":
        return Image(
            mask,
            name="divisions",
            colormap="blue",
            blending="additive",
            contrast_limits=[0, 0.35],
        )

    img = np.zeros([552, 552])
    img[20:532, 20:532] = mask[0]
    blobs = blob_log(
        img, min_sigma=10, max_sigma=25, num_sigma=25, threshold=30
    )
    blobs_logs = np.concatenate((blobs, np.zeros([len(blobs), 1])), axis=1)

    for t in range(1, T):
        img = np.zeros([552, 552])
        img[20:532, 20:532] = mask[t]
        blobs = blob_log(
            img, min_sigma=10, max_sigma=25, num_sigma=25, threshold=30
        )
        blobs_log = np.concatenate(
            (blobs, np.zeros([len(blobs), 1]) + t), axis=1
        )
        blobs_logs = np.concatenate((blobs_logs, blobs_log))

    label = 0
    _df = []
    for blob in blobs_logs:
        y, x, r, t = blob
        mu = intensity(mask, int(t), int(x - 20), int(y - 20))

        _df.append(
            {
                "Label": label,
                "T": int(t + 1),
                "X": int(x - 20),
                "Y": 532 - int(y),  # map coords without boundary
                "Intensity": mu,
            }
        )
        label += 1

    df = pd.DataFrame(_df)
    createFolder("temp_folder")
    df.to_pickle(f"temp_folder/_dfDivisions{filename}.pkl")
    dfRemove = pd.read_pickle(f"temp_folder/_dfDivisions{filename}.pkl")

    for i in range(len(df)):
        ti, xi, yi = df["T"].iloc[i], df["X"].iloc[i], df["Y"].iloc[i]
        labeli = df["Label"].iloc[i]
        dfmulti = sortDL(df, ti, xi, yi)
        dfmulti = dfmulti.drop_duplicates(subset=["T", "X", "Y"])

        if len(dfmulti) > 0:
            mui = df["Intensity"].iloc[i]
            for j in range(len(dfmulti)):
                labelj = dfmulti["Label"].iloc[j]
                muj = dfmulti["Intensity"].iloc[j]

                if mui < muj:
                    indexNames = dfRemove[dfRemove["Label"] == labeli].index
                    dfRemove.drop(indexNames, inplace=True)
                else:
                    indexNames = dfRemove[dfRemove["Label"] == labelj].index
                    dfRemove.drop(indexNames, inplace=True)

    dfDivisions = dfRemove.drop_duplicates(subset=["T", "X", "Y"])

    if Dropdown == "Division database":
        shutil.rmtree("temp_folder")
        dfDivisions.to_pickle(f"dfDivision{filename}.pkl")
        return Image(
            mask,
            name="divisions",
            colormap="blue",
            blending="additive",
            contrast_limits=[0, 0.35],
        )

    createFolder("orientationImages")

    for k in range(len(dfDivisions)):
        label = int(dfDivisions["Label"].iloc[k])
        t = int(dfDivisions["T"].iloc[k])
        x = int(dfDivisions["X"].iloc[k])
        y = int(512 - dfDivisions["Y"].iloc[k])

        xMax = int(x + 30)
        xMin = int(x - 30)
        yMax = int(y + 30)
        yMin = int(y - 30)
        if xMax > 512:
            xMaxCrop = 60 - (xMax - 512)
            xMax = 512
        else:
            xMaxCrop = 60
        if xMin < 0:
            xMinCrop = -xMin
            xMin = 0
        else:
            xMinCrop = 0
        if yMax > 512:
            yMaxCrop = 60 - (yMax - 512)
            yMax = 512
        else:
            yMaxCrop = 60
        if yMin < 0:
            yMinCrop = -yMin
            yMin = 0
        else:
            yMinCrop = 0

        vid = np.zeros([10, 120, 120])
        for i in range(5):
            image = np.zeros([60, 60])

            image[yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = DLinput[
                t - 1 + i, yMin:yMax, xMin:xMax, 1
            ]

            image = np.asarray(image, "uint8")
            tifffile.imwrite("temp_folder/images.tif", image)

            division = Image.open("temp_folder/images.tif")

            division = division.resize((120, 120))
            vid[2 * i] = division

            image = np.zeros([60, 60])

            image[yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = DLinput[
                t - 1 + i, yMin:yMax, xMin:xMax, 0
            ]

            image = np.asarray(image, "uint8")
            tifffile.imwrite("temp_folder/images.tif", image)

            division = Image.open("temp_folder/images.tif")

            division = division.resize((120, 120))
            vid[2 * i + 1] = division

        vid = np.asarray(vid, "uint8")
        tifffile.imwrite(f"orientationImages/division{label}.tif", vid)

    resnet = timm.create_model("resnet34", pretrained=True)
    resnet.conv1 = nn.Conv2d(
        10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    m = resnet
    m = nn.Sequential(*list(m.children())[:-2])
    model = DynamicUnet(m, 1, (120, 120), norm_type=None).to(DEVICE)
    load_checkpoint(
        torch.load(
            "models/UNetOrientation.pth.tar", map_location=torch.device("cpu")
        ),
        model,
    )

    _df = []
    for k in range(len(dfDivisions)):
        label = int(dfDivisions["Label"].iloc[k])
        vid = sm.io.imread(f"orientationImages/division{label}.tif").astype(
            int
        )[0]
        vid = torch.tensor([vid / 256]).float()
        mask = torch.sigmoid(model(vid)).detach().numpy()
        ori_mask = maskOrientation(mask)

        _df.append(
            {
                "Label": label,
                "T": dfDivisions["T"].iloc[k],
                "X": dfDivisions["X"].iloc[k],
                "Y": dfDivisions["Y"].iloc[k],
                "Orientation": ori_mask,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"dfDivision{filename}.pkl")

    shutil.rmtree("temp_folder")
    shutil.rmtree("orientationImages")
    dfDivisions.to_pickle(f"dfDivOri{filename}.pkl")
    return Image(
        mask,
        name="divisions",
        colormap="blue",
        blending="additive",
        contrast_limits=[0, 0.35],
    )
