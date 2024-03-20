
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import skimage as sm
import skimage.io
import skimage.measure
import skimage.feature
from shapely.geometry import Polygon
import tifffile

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 12})

# -------------------
# This scripted re-organises and sorts data ready for analysis.
# -------------------

def dist(polygon, polygon0):
    [x1, y1] = cell.centroid(polygon)
    [x0, y0] = cell.centroid(polygon0)
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = 180 + diff

    return abs(diff)


def findtcj(polygon, img):

    centroid = cell.centroid(polygon)
    x, y = int(centroid[0]), int(centroid[1])
    img = 1 - img / 255
    img = np.asarray(img, "uint8")

    imgLabel = sm.measure.label(img, background=0, connectivity=1)
    label = imgLabel[x, y]
    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]

    # imgLabelrc = util.imgxyrc(imgLabel)
    # imgLabelrc[imgLabelrc == label] = round(1.25 * imgLabelrc.max())
    # imgLabelrc = np.asarray(imgLabelrc, "uint16")
    # tifffile.imwrite(f"results/imgLabel{filename}.tif", imgLabelrc)

    if label == 0:
        print("label == 0")

    zeros = np.zeros([512, 512])

    zeros[imgLabel == label] = 1
    for con in contour:
        zeros[int(con[0]), int(con[1])] = 1

    struct2 = sp.ndimage.generate_binary_structure(2, 2)
    dilation = sp.ndimage.morphology.binary_dilation(zeros, structure=struct2).astype(
        zeros.dtype
    )
    dilation[zeros == 1] = 0
    # dilationrc = util.imgxyrc(dilation)
    # dilationrc = np.asarray(dilationrc, "uint16")
    # tifffile.imwrite(f"results/dilation{filename}.tif", dilationrc)

    tcj = np.zeros([512, 512])
    diff = img - dilation
    tcj[diff == -1] = 1
    tcj[tcj != 1] = 0

    outerTCJ = skimage.feature.peak_local_max(tcj)
    tcjrc = util.imgxyrc(tcj)
    tcjrc = np.asarray(tcjrc, "uint16")
    tifffile.imwrite(f"results/tcj{filename}.tif", tcjrc)

    tcj = []
    for coord in outerTCJ:
        tcj.append(findtcjContour(coord, contour[0:-1]))

    if "False" in tcj:
        tcj.remove("False")
        print("removed")

    return tcj


def isBoundary(contour):

    boundary = False

    for con in contour:
        if con[0] == 0:
            boundary = True
        if con[1] == 0:
            boundary = True
        if con[0] == 511:
            boundary = True
        if con[1] == 511:
            boundary = True

    return boundary


def findtcjContour(coord, contour):

    close = []
    for con in contour:
        r = ((con[0] - coord[0]) ** 2 + (con[1] - coord[1]) ** 2) ** 0.5
        if r < 1.5:
            close.append(con)

    if len(close) == 1:
        tcj = close[0]
    elif len(close) == 0:
        tcj = "False"
    else:
        tcj = np.mean(close, axis=0)

    return tcj


def getSecondColour(track, colour):
    colours = track[np.all((track - colour) != 0, axis=1)]
    colours = colours[np.all((colours - np.array([255, 255, 255])) != 0, axis=1)]

    col = []
    count = []
    while len(colours) > 0:
        col.append(colours[0])
        count.append(len(colours[np.all((colours - colours[0]) == 0, axis=1)]))
        colours = colours[np.all((colours - colours[0]) != 0, axis=1)]

    maxm = np.max(count)
    colourD = col[count.index(maxm)]

    return colourD


def vectorBoundary(pts):

    n = len(pts)

    test = True

    for i in range(n):
        if pts[i][0] == 0 or pts[i][0] == 511:
            test = False
        elif pts[i][1] == 0 or pts[i][1] == 511:
            test = False
        else:
            continue
    return test


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 91

# Nucleus velocity relative to tissue
if True:
    _df2 = []
    _df = []
    for filename in filenames:
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(df["q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
        R = util.rotation_matrix(-theta0)

        df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")
        mig = np.zeros(2)

        for t in range(90):
            dft = df[df["T"] == t]
            V = np.mean(dft["Velocity"]) * scale
            V = np.matmul(R, V)
            _df.append(
                {
                    "Filename": filename,
                    "T": t,
                    "V": V,
                }
            )

    dfVelocityMean = pd.DataFrame(_df)
    dfVelocityMean.to_pickle(f"databases/dfVelocityMean{fileType}.pkl")

# Cell division relative to tissue and wound
if True:
    _df = []
    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])

        t0 = util.findStartTime(filename)
        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                (x_w, y_w) = dfWound["Position"].iloc[t]
                x = dfDivision["X"].iloc[i]
                y = dfDivision["Y"].iloc[i]
                ori = (dfDivision["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                theta = (np.arctan2(y - y_w, x - x_w) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori > 90:
                    ori = 180 - ori
                if ori_w > 90:
                    ori_w = 180 - ori_w
                theta = (np.arctan2(y - y_w, x - x_w) - theta0) * 180 / np.pi
                r = dist[t, 512 - y, x]
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every 2 minutes
                        "X": x * scale,
                        "Y": y * scale,
                        "R": r * scale,
                        "Theta": theta % 360,
                        "Orientation": ori,
                        "Orientation Wound": ori_w,
                    }
                )
        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
                xc = 256 * scale + mig[0]
                yc = 256 * scale + mig[1]
                x = dfDivision["X"].iloc[i] * scale
                y = dfDivision["Y"].iloc[i] * scale
                r = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                ori = (dfDivision["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                theta = (np.arctan2(y - yc, x - xc) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori_w > 90:
                    ori_w = 180 - ori_w
                if ori > 90:
                    ori = 180 - ori
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every t2 minutes
                        "X": x,
                        "Y": y,
                        "R": r,
                        "Theta": theta % 360,
                        "Orientation": ori,
                        "Orientation Wound": ori_w,
                    }
                )

    dfDivisions = pd.DataFrame(_df)
    dfDivisions.to_pickle(f"databases/dfDivisions{fileType}.pkl")

# Cell Shape relative to tissue
if True:
    _df2 = []
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        dfFilename = dfVelocityMean[dfVelocityMean["Filename"] == filename]
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        mig = np.zeros(2)
        Q = np.mean(df["q"])
        theta0 = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta0)

        for t in range(T):
            dft = df[df["Time"] == t]
            if len(dft) > 0:
                Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
                P = np.matmul(R, np.mean(dft["Polar"]))

            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0],
                    dft["Centroid"].iloc[i][1],
                ]
                r = dist[t, int(512 - y), int(x)]
                q = np.matmul(R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R)))
                dq = q - Q
                A = dft["Area"].iloc[i] * scale**2
                TrQdq = np.trace(np.matmul(Q, dq))
                dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                [x, y] = np.matmul(R, np.array([x, y]) * scale)
                p = np.matmul(R, dft["Polar"].iloc[i])

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x - mig[0],
                        "Y": y - mig[1],
                        "R": r,
                        "Centroid": np.array(dft["Centroid"].iloc[i]) * scale,
                        "dq": dq,
                        "q": q,
                        "TrQdq": TrQdq,
                        "Area": A,
                        "dp": dp,
                        "Polar": p,
                        "Shape Factor": dft["Shape Factor"].iloc[i],
                    }
                )

            if t < 90:
                mig += np.array(dfFilename["V"][dfFilename["T"] == t])[0]

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfShape{fileType}.pkl")

# Cells Divisions and Shape changes
if True:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []
    _dfTrack = []
    for filename in filenames:
        # print(filename)
        df = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)
        T, X, Y, C = tracks.shape

        binary = util.vidrcxy(binary)
        tracks = util.vidrcxyRGB(tracks)
        tracksDivisions = tracks

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            # if label == 41:
            #     print(0)
            ori = df["Orientation"].iloc[i] % 180
            tm = t = int(df["T"].iloc[i])
            x = df["X"].iloc[i]
            y = df["Y"].iloc[i]

            colour = tracks[t, int(x), int(y)]
            if np.all((colour - np.array([255, 255, 255])) == 0):
                continue

            track = tracks[t][np.all((tracks[t] - colour) == 0, axis=2)]
            if len(track) > 1500:
                continue

            finished = False
            t_i = t
            while finished == False:
                t_i += 1
                A0 = len(track)
                track = tracks[t_i][np.all((tracks[t_i - 1] - colour) == 0, axis=2)]
                A1 = len(track[np.all((track - colour) == 0, axis=1)])
                if A1 / A0 < 0.65:
                    finished = True
                    try:
                        colourD = getSecondColour(track, colour)
                    except:
                        continue

                if A1 > 1500:
                    finished = True

                if t_i == T - 1:
                    finished = True
                    colourD = getSecondColour(track, colour)
            if A1 > 1500:
                continue
            tc = t_i - 1

            if tc - tm < 3:

                if tc > 30:
                    time = np.linspace(tc, tc - 30, 31)
                else:
                    time = np.linspace(tc, 0, tc + 1)
                polyList = []

                t = int(time[0])
                contour = sm.measure.find_contours(
                    np.all((tracks[t] - colour) == 0, axis=2), level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                try:
                    polygon = Polygon(poly)
                    pts = list(polygon.exterior.coords)
                    if vectorBoundary(pts):
                        tcj = findtcj(polygon, binary[t])
                    else:
                        tcj = False
                    theta = cell.orientation_tcj(tcj)
                    q = cell.qTensor(polygon)
                    q_tcj = cell.qTensor_tcj(tcj)
                    _dfTrack.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Type": "parent",
                            "Colour": colour,
                            "Time": t,
                            "Division Time": int(t - tm),
                            "Polygon": polygon,
                            "TCJ": tcj,
                            "Area": polygon.area,
                            "Shape Factor": cell.shapeFactor(polygon),
                            "Orientation": (cell.orientation(polygon) * 180 / np.pi)
                            % 180,
                            "q": q,
                            "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                            "q_tcj": q_tcj,
                            "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2) ** 0.5,
                            "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                            "Orientation tcj": (cell.orientation_tcj(tcj) * 180 / np.pi)
                            % 180,
                        }
                    )
                    polyList.append(polygon)
                    polygon0 = polygon
                except:
                    continue

                for t in time[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colour) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            pts = list(polygon.exterior.coords)
                            if vectorBoundary(pts):
                                tcj = findtcj(polygon, binary[int(t)])
                            else:
                                tcj = False
                            q = cell.qTensor(polygon)
                            q_tcj = cell.qTensor_tcj(tcj)
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "parent",
                                    "Colour": colour,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "TCJ": tcj,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": (
                                        cell.orientation(polygon) * 180 / np.pi
                                    )
                                    % 180,
                                    "q": q,
                                    "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                                    "q_tcj": q_tcj,
                                    "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2)
                                    ** 0.5,
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": (
                                        cell.orientation_tcj(tcj) * 180 / np.pi
                                    )
                                    % 180,
                                }
                            )
                            polyList.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                if tc < T - 31:
                    timeD = np.linspace(tc + 1, tc + 31, 31)
                else:
                    timeD = np.linspace(tc + 1, T - 1, T - tc - 1)
                polyListD1 = []

                t = int(timeD[0])
                try:
                    contour = sm.measure.find_contours(
                        np.all((tracks[t] - colour) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    if polygon.area == 0:
                        continue
                    elif polygon.area > 1500:
                        continue
                    else:
                        pts = list(polygon.exterior.coords)
                        if vectorBoundary(pts):
                            tcj = findtcj(polygon, binary[t])
                        else:
                            tcj = False
                        theta = cell.orientation_tcj(tcj)
                        q = cell.qTensor(polygon)
                        q_tcj = cell.qTensor_tcj(tcj)
                        _dfTrack.append(
                            {
                                "Filename": filename,
                                "Label": label,
                                "Type": "daughter1",
                                "Colour": colour,
                                "Time": t,
                                "Division Time": int(t - tm),
                                "Polygon": polygon,
                                "TCJ": tcj,
                                "Area": polygon.area,
                                "Shape Factor": cell.shapeFactor(polygon),
                                "Orientation": (cell.orientation(polygon) * 180 / np.pi)
                                % 180,
                                "q": q,
                                "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                                "q_tcj": q_tcj,
                                "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2) ** 0.5,
                                "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                "Orientation tcj": (
                                    cell.orientation_tcj(tcj) * 180 / np.pi
                                )
                                % 180,
                            }
                        )
                        polyListD1.append(polygon)
                        polygon0 = polygon
                except:
                    continue

                for t in timeD[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colour) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            pts = list(polygon.exterior.coords)
                            if vectorBoundary(pts):
                                tcj = findtcj(polygon, binary[int(t)])
                            else:
                                tcj = False
                            q = cell.qTensor(polygon)
                            q_tcj = cell.qTensor_tcj(tcj)
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "daughter1",
                                    "Colour": colour,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "TCJ": tcj,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": (
                                        cell.orientation(polygon) * 180 / np.pi
                                    )
                                    % 180,
                                    "q": q,
                                    "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                                    "q_tcj": q_tcj,
                                    "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2)
                                    ** 0.5,
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": (
                                        cell.orientation_tcj(tcj) * 180 / np.pi
                                    )
                                    % 180,
                                }
                            )
                            polyListD1.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                polyListD2 = []

                t = int(timeD[0])
                try:
                    contour = sm.measure.find_contours(
                        np.all((tracks[t] - colourD) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    if polygon.area == 0:
                        continue
                    elif polygon.area > 1500:
                        continue
                    else:
                        pts = list(polygon.exterior.coords)
                        if vectorBoundary(pts):
                            tcj = findtcj(polygon, binary[t])
                        else:
                            tcj = False
                        theta = cell.orientation_tcj(tcj)
                        q = cell.qTensor(polygon)
                        q_tcj = cell.qTensor_tcj(tcj)
                        _dfTrack.append(
                            {
                                "Filename": filename,
                                "Label": label,
                                "Type": "daughter2",
                                "Colour": colourD,
                                "Time": t,
                                "Division Time": int(t - tm),
                                "Polygon": polygon,
                                "TCJ": tcj,
                                "Area": polygon.area,
                                "Shape Factor": cell.shapeFactor(polygon),
                                "Orientation": (cell.orientation(polygon) * 180 / np.pi)
                                % 180,
                                "q": q,
                                "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                                "q_tcj": q_tcj,
                                "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2) ** 0.5,
                                "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                "Orientation tcj": (
                                    cell.orientation_tcj(tcj) * 180 / np.pi
                                )
                                % 180,
                            }
                        )
                        polyListD2.append(polygon)
                        polygon0 = polygon
                except:
                    continue

                for t in timeD[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colourD) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            pts = list(polygon.exterior.coords)
                            if vectorBoundary(pts):
                                tcj = findtcj(polygon, binary[int(t)])
                            else:
                                tcj = False
                            q = cell.qTensor(polygon)
                            q_tcj = cell.qTensor_tcj(tcj)
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "daughter2",
                                    "Colour": colourD,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "TCJ": tcj,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": (
                                        cell.orientation(polygon) * 180 / np.pi
                                    )
                                    % 180,
                                    "q": q,
                                    "q0": (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5,
                                    "q_tcj": q_tcj,
                                    "q0_tcj": (q_tcj[0, 0] ** 2 + q_tcj[0, 1] ** 2)
                                    ** 0.5,
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": (
                                        cell.orientation_tcj(tcj) * 180 / np.pi
                                    )
                                    % 180,
                                }
                            )
                            polyListD2.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                _df.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "Times": np.array(time),
                        "Times daughters": np.array(timeD),
                        "Cytokineses Time": tc,
                        "Anaphase Time": tm,
                        "X": x,
                        "Y": y,
                        "Colour": colour,
                        "Daughter Colour": colourD,
                        "Polygons": polyList,
                        "Polygons daughter1": polyListD1,
                        "Polygons daughter2": polyListD2,
                        "Track length": len(polyList),
                        "Daughter length": np.min([len(polyListD1), len(polyListD2)]),
                    }
                )

    dfDivisionShape = pd.DataFrame(_df)
    dfDivisionShape.to_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.DataFrame(_dfTrack)
    dfDivisionTrack.to_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
