
import numpy as np
import pandas as pd
import skimage as sm
import shapely
from statistics import mode
import tifffile


def isWound(t):
    frames = [1, 2, 4, 5, 7, 8, 10]
    if t in frames:
        return True
    else:
        return False


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
    return (int(cx), int(cy))



# Eval compare different DL dephts and ground true
if True:
    _df = []
    T = 12
    model = "101"

    display = np.zeros([12, 1024, 1024, 3])
    for t0 in range(T):
        wound = isWound(t0)
        t = t0 + 48
        
        # load ground truth
        groundTrue = sm.io.imread(
            f"dat_boundary/test_ground_truth/mask_{t}.tif"
        ).astype(int)
        binaryGT = np.zeros([1024, 1024])
        binaryGT[groundTrue == 1] = 255
        imgGT = 255 - binaryGT
        imgGT = sm.morphology.dilation(imgGT, footprint=np.ones([3, 3]))
        
        # load watersheded model boundaries
        binaryDL = sm.io.imread(
            f"dat_boundary/test_watershed/landscape_{t}/handCorrection.tif"
        ).astype(int)[:, :, 0]
        outPlane = np.zeros([1024, 1024])
        outPlane[groundTrue == 3] = 255
        imgDL = 255 - binaryDL
        display[t0, :, :, 0] = imgDL
        display[t0, :, :, 1] = imgDL
        display[t0, :, :, 2] = imgDL

        # find and labels cells
        imgLabelGT = sm.measure.label(imgGT, background=0, connectivity=1)
        imgLabelsGT = np.unique(imgLabelGT)[1:]
        imgLabelDL = sm.measure.label(imgDL, background=0, connectivity=1)
        imgLabelsDL = np.unique(imgLabelDL)[1:]
        
        imgLabelGT = np.asarray(imgLabelGT, "uint16")
        tifffile.imwrite(
            f"results/imgLabelGT.tif", imgLabelGT, imagej=True
        )
        imgLabelDL = np.asarray(imgLabelDL, "uint16")
        tifffile.imwrite(
            f"results/imgLabelDL_{model}.tif",
            imgLabelDL,
            imagej=True,
        )

        for labelGT in imgLabelsGT:

            # tests if labelled "cell" is out of veiw (e.g. in wound or tissue fold)
            inPlane = outPlane[imgLabelGT == labelGT]
            inPlane = np.array(inPlane)
            inPlaneRatio = np.sum(inPlane == 255) / len(inPlane)
            if inPlaneRatio < 0.2:
                
                # test if there is a 80% overlap with prediation and ground truth cells
                # if not then it is counted as false negative 
                labelsDL = list(imgLabelDL[imgLabelGT == labelGT])
                labelsDL = list(filter(lambda a: a != 0, labelsDL))
                most = mode(labelsDL)
                labelsDL = np.array(labelsDL)
                ratio = np.sum(labelsDL == most) / len(labelsDL)

                if ratio > 0.8:
                    # test if there is a 80% overlap with ground truth cells and prediation
                    # if not then it is counted as false negative 
                    labelsGT = list(imgLabelGT[imgLabelDL == most])
                    labelsGT = list(filter(lambda a: a != 0, labelsGT))
                    mostLabel = mode(labelsGT)
                    labelsGT = np.array(labelsGT)
                    ratio = np.sum(labelsGT == mostLabel) / len(labelsGT)

                    inPlane = outPlane[imgLabelDL == most]
                    inPlane = np.array(inPlane)
                    inPlaneRatio = np.sum(inPlane == 255) / len(inPlane)

                    if inPlaneRatio < 0.2:
                        if ratio > 0.8:
                            # cells that pass all tests match the cell shape from model and ground truth cells
                            # these are counted as true positive
                            _df.append(
                                {
                                    "Model": model,
                                    "T": t,
                                    "Ground True Label": labelGT,
                                    "Label DL": most,
                                    "Result": "TP",
                                    "Wound": wound,
                                }
                            )
                            # mark display cell green as true postive
                            display[t0, :, :, 0][imgLabelDL == most] = 0
                            display[t0, :, :, 1][imgLabelDL == most] = 255
                            display[t0, :, :, 2][imgLabelDL == most] = 0
                        else:
                            _df.append(
                                {
                                    "Model": model,
                                    "T": t,
                                    "Ground True Label": labelGT,
                                    "Label DL": most,
                                    "Result": "FN",
                                    "Wound": wound,
                                }
                            )
                            # mark display cell red as false negative
                            display[t0, :, :, 0][imgLabelDL == most] = 255
                            display[t0, :, :, 1][imgLabelDL == most] = 0
                            display[t0, :, :, 2][imgLabelDL == most] = 0
                    else:
                        _df.append(
                            {
                                "Model": model,
                                "T": t,
                                "Ground True Label": labelGT,
                                "Label DL": most,
                                "Result": "FN",
                                "Wound": wound,
                            }
                        )
                else:
                    _df.append(
                        {
                            "Model": model,
                            "T": t,
                            "Ground True Label": labelGT,
                            "Label DL": most,
                            "Result": "FN",
                            "Wound": wound,
                        }
                    )

        labelsDL = np.unique(
            imgLabelDL[np.all(display[t0] == np.array([255, 255, 255]), axis=2)]
        )
        for labelDL in labelsDL:

            # cells in the predition that don't match a cell in the ground truth
            # are counted as a false positive 
            inPlane = outPlane[imgLabelDL == labelDL]
            inPlane = np.array(inPlane)
            inPlaneRatio = np.sum(inPlane == 255) / len(inPlane)

            if inPlaneRatio < 0.2:
                _df.append(
                    {
                        "Model": model,
                        "T": t,
                        "Label DL": labelDL,
                        "Result": "FP",
                        "Wound": wound,
                    }
                )
                # mark display cell blue as false positive
                display[t0, :, :, 0][imgLabelDL == labelDL] = 0
                display[t0, :, :, 1][imgLabelDL == labelDL] = 0
                display[t0, :, :, 2][imgLabelDL == labelDL] = 255

           
    # display confusion images of segmented cells
    display = np.asarray(display, "uint8")
    tifffile.imwrite(
        f"results/displayConfusion_{model}.tif", display
    )

    # save confusion database
    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfConfusionBoundary.pkl")

    print("")
    print("Model depth " + model)
    dfMod = df[df["Model"] == model]

    truePos = len(dfMod[dfMod["Result"] == "TP"])
    falsePos = len(dfMod[dfMod["Result"] == "FP"])
    falseNeg = len(dfMod[dfMod["Result"] == "FN"])
    print(
        f"True Pos: {truePos}",
        f"False Pos: {falsePos}",
        f"False Neg: {falseNeg}",
    )
    print(f"F1 score : {2*truePos/(2*truePos+falsePos+falseNeg)}")
