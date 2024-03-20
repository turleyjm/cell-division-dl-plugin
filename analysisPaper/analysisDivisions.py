
from os.path import exists
from math import floor, log10, factorial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
import utils as util

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 16})

# -------------------

# weighted average and standard deviation 
def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)

def OLSfit(x, y, dy=None):
    """Find the best fitting parameters of a linear fit to the data through the
    method of ordinary least squares estimation. (i.e. find m and b for
    y = m*x + b)

    Args:
        x: Numpy array of independent variable data
        y: Numpy array of dependent variable data. Must have same size as x.
        dy: Numpy array of dependent variable standard deviations. Must be same
            size as y.

    Returns: A list with four floating point values. [m, dm, b, db]
    """
    if dy is None:
        # if no error bars, weight every point the same
        dy = np.ones(x.size)
    denom = np.sum(1 / dy**2) * np.sum((x / dy) ** 2) - (np.sum(x / dy**2)) ** 2
    m = (
        np.sum(1 / dy**2) * np.sum(x * y / dy**2)
        - np.sum(x / dy**2) * np.sum(y / dy**2)
    ) / denom
    b = (
        np.sum(x**2 / dy**2) * np.sum(y / dy**2)
        - np.sum(x / dy**2) * np.sum(x * y / dy**2)
    ) / denom
    dm = np.sqrt(np.sum(1 / dy**2) / denom)
    db = np.sqrt(np.sum(x / dy**2) / denom)
    return [m, dm, b, db]

# linear best fit division density with time
def bestFitUnwound(fileType="Unwound18h"):
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    filenames = util.getFilesType(fileType)[0]
    count = np.zeros([len(filenames), int(T / timeStep)])
    area = np.zeros([len(filenames), int(T / timeStep)])
    for k in range(len(filenames)):
        filename = filenames[k]
        t0 = util.findStartTime(filename)
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]
        for t in range(count.shape[1]):
            df1 = dfFile[dfFile["T"] > timeStep * t]
            df = df1[df1["T"] <= timeStep * (t + 1)]
            count[k, t] = len(df)

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int) / 255
        )
        for t in range(area.shape[1]):
            t1 = int(timeStep / 2 * t - t0 / 2)
            t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
            if t1 < 0:
                t1 = 0
            if t2 < 0:
                t2 = 0
            area[k, t] = np.sum(inPlane[t1:t2]) * scale**2
    time = []
    dd = []
    std = []
    for t in range(area.shape[1]):
        _area = area[:, t][area[:, t] > 0]
        _count = count[:, t][area[:, t] > 0]
        if len(_area) > 0:
            _dd, _std = weighted_avg_and_std(_count / _area, _area)
            dd.append(_dd)
            std.append(_std)
            time.append(t * 10 + timeStep / 2)
    time = np.array(time)
    dd = np.array(dd)
    std = np.array(std)
    bestfit = OLSfit(time, dd)
    (m, c) = (bestfit[0], bestfit[2])

    return m, c

# -------------------


scale = 123.26 / 512
fileTypes = ["Unwound18h", "WoundS18h", "WoundL18h"]
groupTitle = "wild type"
T = 180
timeStep = 10
R = 110
rStep = 10

# ------------ Divison density with time best fit Unwound18h ------------
fileType = "Unwound18h"
filenames = util.getFilesType(fileType)[0]
dfDivisions = pd.read_pickle(f"databases/dfDivisionsUnwound18h.pkl")
count = np.zeros([len(filenames), int(T / timeStep)])
area = np.zeros([len(filenames), int(T / timeStep)])

# measure the division density of each video
for k in range(len(filenames)):
    filename = filenames[k]
    t0 = util.findStartTime(filename)
    dfFile = dfDivisions[dfDivisions["Filename"] == filename]

    # count number of cell divisions in each timepoint
    for t in range(count.shape[1]):
        df1 = dfFile[dfFile["T"] > timeStep * t]
        df = df1[df1["T"] <= timeStep * (t + 1)]
        count[k, t] = len(df)

    # measure area of tissue visable in each timepoint
    inPlane = 1 - (
        sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int) / 255
    )
    for t in range(area.shape[1]):
        t1 = int(timeStep / 2 * t - t0 / 2)
        t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
        if t1 < 0:
            t1 = 0
        if t2 < 0:
            t2 = 0
        area[k, t] = np.sum(inPlane[t1:t2]) * scale**2

# Measure division density
time = []
dd = []
std = []
for t in range(area.shape[1]):
    _area = area[:, t][area[:, t] > 0]
    _count = count[:, t][area[:, t] > 0]
    if len(_area) > 0:
        _dd, _std = weighted_avg_and_std(_count / _area, _area)
        dd.append(_dd * 10000)
        std.append(_std * 10000)
        time.append(t * timeStep + timeStep / 2)

# plot the data
time = np.array(time)
dd = np.array(dd)
std = np.array(std)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
bestfit = OLSfit(time, dd, dy=std)
(m, c) = (bestfit[0], bestfit[2])
ax.plot(time, dd, marker="o", label=f"Unwounded", color="tab:blue")
ax.fill_between(time, dd - std, dd + std, alpha=0.15, color="tab:blue")
ax.plot(time, m * time + c, color="r", label=f"Linear model")
ax.set(xlabel="Time", ylabel=r"Divison density ($10^{-4}\mu m^{-2}$)")
ax.title.set_text(f"Divison density with time {fileType}")
ax.set_ylim([0, 7])

fileTitle = util.getBoldTitle("unwounded")
ax.title.set_text(f"Division density with \n time " + fileTitle)
if "Wound" in fileType:
    ax.set(
        xlabel="Time after wounding (mins)",
        ylabel=r"Divison density ($10^{-4}\mu m^{-2}$)",
    )
else:
    ax.set(xlabel="Time (mins)", ylabel=r"Divison density ($10^{-4}\mu m^{-2}$)")

ax.legend(loc="upper left", fontsize=12)
fig.savefig(
    f"output/figure/Divison density with time best fit {fileType}",
    transparent=True,
    bbox_inches="tight",
    dpi=300,
)
plt.close("all")

# ------------ Compare divison density with time ------------

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# labels = ["Unwound18h", "WoundS18h", "WoundL18h"]
# legend = ["Unwounded", "Small wound", "Large wound"]
# colors = ["tab:blue", "tab:orange", "tab:green"]
labels = ["WoundS18h", "WoundL18h"]
legend = ["Small wound", "Large wound"]
colors = ["tab:orange", "tab:green"]
dat_dd = []
total = 0
i = 0
for fileType in labels:
    filenames = util.getFilesType(fileType)[0]
    count = np.zeros([len(filenames), int(T / timeStep)])
    area = np.zeros([len(filenames), int(T / timeStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    total += len(dfDivisions)

    # measure the division density of each video
    for k in range(len(filenames)):
        filename = filenames[k]
        t0 = util.findStartTime(filename)
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]

        # count number of cell divisions in each timepoint
        for t in range(count.shape[1]):
            df1 = dfFile[dfFile["T"] > timeStep * t]
            df = df1[df1["T"] <= timeStep * (t + 1)]
            count[k, t] = len(df)

        # measure area of tissue visable in each timepoint
        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int) / 255
        )
        for t in range(area.shape[1]):
            t1 = int(timeStep / 2 * t - t0 / 2)
            t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
            if t1 < 0:
                t1 = 0
            if t2 < 0:
                t2 = 0
            area[k, t] = np.sum(inPlane[t1:t2]) * scale**2

    # Measure division density
    dat_dd.append(count / area)
    time = []
    dd = []
    std = []
    for t in range(area.shape[1]):
        _area = area[:, t][area[:, t] > 0]
        _count = count[:, t][area[:, t] > 0]
        if len(_area) > 0:
            _dd, _std = weighted_avg_and_std(_count / _area, _area)
            dd.append(_dd * 10000)
            std.append(_std * 10000)
            time.append(t * timeStep + timeStep / 2)

    dd = np.array(dd)
    std = np.array(std)
    ax.plot(time, dd, label=f"{legend[i]}", marker="o", color=colors[i])
    ax.fill_between(time, dd - std, dd + std, alpha=0.15, color=colors[i])
    i += 1

# plot the data
time = np.array(time)
ax.plot(time, m * time + c, color="r", label=f"Linear model")
ax.set(
    xlabel="Time after wounding (mins)",
    ylabel=r"Division density ($10^{-4}\mu m^{-2}$)",
)
ax.title.set_text(f"Division density with \n time " + r"$\bf{wounds}$")
ax.set_ylim([0, 7])
ax.legend(loc="upper left", fontsize=12)

fig.savefig(
    f"output/figure/Compared division density with time",
    transparent=True,
    bbox_inches="tight",
    dpi=300,
)
plt.close("all")

# ------------ Division correlations figure ------------
for fileType in fileTypes:
    # load division correlation data
    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    divCorr = df["divCorr"].iloc[0]
    oriCorr = df["oriCorr"].iloc[0]
    df = 0

    # plot heatmap of density division correlation function
    t, r = np.mgrid[5:105:10, 5:115:10]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        divCorr[:10] * 10000**2,
        cmap="RdBu_r",
        vmin=-4.1,
        vmax=4.1,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r$ $(\mu m)$")
    fileTitle = util.getFileTitle(fileType)
    fileTitle = util.getBoldTitle(fileTitle)
    ax.title.set_text(f"Division density \n correlation " + fileTitle)

    fig.savefig(
        f"output/figure/Division Correlation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    # plot heatmap of division orientation correlation function
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax.pcolor(
        t,
        r,
        oriCorr[:10],
        cmap="RdBu_r",
        vmin=-1.05,
        vmax=1.05,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r$ $(\mu m)$")
    fileTitle = util.getFileTitle(fileType)
    fileTitle = util.getBoldTitle(fileTitle)
    ax.title.set_text(f"Division orientation \n correlation " + fileTitle)

    fig.savefig(
        f"output/figure/Division orientation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")



# ------------ Change in divison density with distance from wound edge and time ------------
for fileType in fileTypes:
    filenames = util.getFilesType(fileType)[0]
    for filename in filenames:
        # load cell division 
        count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            # count the number of divisons in each timepoint and radial band
            for r in range(count.shape[2]):
                for t in range(count.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    count[k, t, r] = len(df)

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
                / 255
            )
            dist = (
                sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
                * scale
            )

            # measure area of tissue visable in each timepoint and radial band
            for r in range(area.shape[2]):
                for t in range(area.shape[1]):
                    t1 = int(timeStep / 2 * t - t0 / 2)
                    t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                    if t1 < 0:
                        t1 = 0
                    if t2 < 0:
                        t2 = 0
                    area[k, t, r] = (
                        np.sum(
                            inPlane[t1:t2][
                                (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                            ]
                        )
                        * scale**2
                    )

        # measure the division density of tissue visable in each timepoint and radial band
        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        sumArea = np.zeros([int(T / timeStep), int(R / rStep)])
        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                    sumArea[t, r] = np.sum(_area)
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        # get the linear best fit unwounded tissue 
        (m, c) = bestFitUnwound()
        time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
        # minus the unwounded dd from the wounded dd
        for r in range(dd.shape[1]):
            dd[:, r] = dd[:, r] - (m * time + c)

        dd[sumArea < 600 * len(filenames)] = np.nan
        dd = dd * 10000

        # plot the change in division density heatmap
        fileTitle = util.getFileTitle(fileType)
        t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd,
            vmin=-5,
            vmax=5,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        fileTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(
            f"Deviation in division density: \n " + fileTitle + " from linear model"
        )
        if "Wound" in fileType:
            ax.set(
                xlabel="Time after wounding (mins)", ylabel=r"Distance from wound $(\mu m)$"
            )
        else:
            ax.set(xlabel="Time (mins)", ylabel=r"Distance from virtual wound $(\mu m)$")

        fig.savefig(
            f"output/figure/Change in Division density heatmap {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")



# ------------ Divison orientation with respect to tissue ------------
for fileType in fileTypes:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    fileTitle = util.getFileTitle(fileType)
    fileTitle = util.getBoldTitle(fileTitle)

    # plot the histogram of orientation of divisions
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cm = plt.cm.get_cmap("RdBu_r")
    n, bins, patches = ax.hist(dfDivisions["Orientation"], color="g")

    ax.set(xlabel="Orientation", ylabel="Number of Divisions")
    ax.title.set_text(f"Divison orientation with respect \n to wing " + fileTitle)

    fig.savefig(
        f"output/figure/Divison orientation with respect to tissue {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# ------------ shift in orientation after division relative to tissue ------------
for fileType in fileTypes:
    filenames = util.getFilesType(fileType)[0]
    # load the tracks and shapes of cell shapes
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []

    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        Q = np.mean(dfShape["q"][dfShape["Filename"] == filename])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])
        R = util.rotation_matrix(-theta0)

        # only takes tracks that are longer than 20mins post division
        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
            t = 10

            # at 20 mins post wound finds the position of the daughter cells
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
            x0, y0 = util.centroid(polygon)
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[1]
            x1, y1 = util.centroid(polygon)
            # measure the angle between the cells
            phi = (np.arctan2(y0 - y1, x0 - x1) * 180 / np.pi) % 180

            ori = ori - theta0
            phi = phi - theta0
            if ori > 90:
                ori = 180 - ori
            if phi > 90:
                phi = 180 - phi

            # saves the angle of mitosis division and angle 20mins later
            _df.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "Shape Orientation": phi,
                    "Change Towards Tissue": ori - phi,
                }
            )

    df = pd.DataFrame(_df)

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[df["Orientation"] > theta]
        df2 = df1[df1["Orientation"] < theta + 10]
        deltaOri.append(np.mean(df2["Change Towards Tissue"]))
        deltaOristd.append(np.std(df2["Change Towards Tissue"]))

    deltaOri = np.array(deltaOri)
    deltaOristd = np.array(deltaOristd)

    # plot the angle of mitosis division and angle 20mins later
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.hist(
        df["Orientation"],
        alpha=0.4,
        density=True,
        label="Mitosis orientation",
        color="g",
    )
    ax.axvline(np.median(df["Orientation"]), color="g")
    ax.hist(
        df["Shape Orientation"],
        alpha=0.4,
        color="m",
        density=True,
        label="Post shuffling orientation",
    )
    ax.axvline(np.median(df["Shape Orientation"]), color="m")

    ax.set(xlabel="Division orientation relative to wing", ylabel="Frequency")
    fileTitle = util.getFileTitle(fileType)
    boldTitle = util.getBoldTitle(fileTitle)
    if "Wound" in fileType:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in tissue with "
            + boldTitle
        )
    else:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in " + boldTitle
        )

    ax.set_ylim([0, 0.024])
    ax.legend(fontsize=12, loc="upper left")


    fig.tight_layout()
    fig.savefig(
        f"output/figure/change in ori after division relative to tissue {fileType} figure",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# ------------ shift in orientation after division relative to wound ------------
for fileType in fileTypes:
    filenames = util.getFilesType(fileType)[0]
    # load the tracks and shapes of cell shapes
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []

    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]

        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()

        # only takes tracks that are longer than 20mins post division
        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 3]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        dist = (
            sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int) * scale
        )
        t0 = util.findStartTime(filename)

        for label in labels:
            dfDiv = df[df["Label"] == label]
            ori = dfDivision["Orientation"][dfDivision["Label"] == label].iloc[0] % 180
            t = dfDiv["Time"][dfDiv["Division Time"] == 0].iloc[0]
            x = dfDivision["X"][dfDivision["Label"] == label].iloc[0]
            y = dfDivision["Y"][dfDivision["Label"] == label].iloc[0]
            # measure angle of division relative to wound centre
            if "Wound" in filename:
                (xc, yc) = dfWound["Position"].iloc[t]
            else:
                (xc, yc) = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
            psi = np.arctan2(y - yc, x - xc) * 180 / np.pi
            ori = (ori - psi) % 180

            dfDiv = df[df["Label"] == label]
            T = 10
            # at 20 mins post wound finds the position of the daughter cells
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == T].iloc[0]
            x0, y0 = util.centroid(polygon)
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == T].iloc[1]
            x1, y1 = util.centroid(polygon)
            phi = (np.arctan2(y0 - y1, x0 - x1) * 180 / np.pi) % 180
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            # measure the angle between the cells
            psi = np.arctan2(ym - yc, xm - xc) * 180 / np.pi
            phi = (phi - psi) % 180
            r = dist[t, int(512 - ym), int(xm)]

            if ori > 90:
                ori = 180 - ori
            if phi > 90:
                phi = 180 - phi

            # saves the angle of mitosis division and angle 20mins later
            _df.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "T": int(t0 + t * 2),
                    "R": r,
                    "Nuclei Orientation": ori,
                    "Daughter Orientation": phi,
                    "Change Towards Wound": ori - phi,
                }
            )

    df = pd.DataFrame(_df)

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[df["Nuclei Orientation"] > theta]
        df2 = df1[df1["Nuclei Orientation"] < theta + 10]
        deltaOri.append(np.mean(df2["Change Towards Wound"]))
        deltaOristd.append(np.std(df2["Change Towards Wound"]))

    deltaOri = np.array(deltaOri)
    deltaOristd = np.array(deltaOristd)

    # plot the angle of mitosis division and angle 20mins later
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(
        df["Nuclei Orientation"],
        alpha=0.4,
        density=True,
        label="Mitosis orientation",
        color="g",
    )
    ax.axvline(np.median(df["Nuclei Orientation"]), color="g")
    ax.hist(
        df["Daughter Orientation"],
        alpha=0.4,
        color="m",
        density=True,
        label="Post shuffling orientation",
    )
    ax.axvline(np.median(df["Daughter Orientation"]), color="m")

    ax.set(xlabel="Division orientation relative \n to wound", ylabel="Frequency")
    fileTitle = util.getFileTitle(fileType)
    boldTitle = util.getBoldTitle(fileTitle)
    if "Wound" in fileType:
        ax.title.set_text(f"Shift in division orientation \n relative to " + boldTitle)
    else:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in " + boldTitle
        )
    ax.set_ylim([0, 0.024])
    ax.legend(fontsize=12, loc="upper left")


    fig.tight_layout()
    fig.savefig(
        f"output/figure/change in ori after division relative to wound {fileType} figure",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")