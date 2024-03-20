
import numpy as np
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from skimage.draw import circle_perimeter
from scipy import optimize


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


def inertia(polygon):
    """takes the polygon and finds its inertia tensor matrix"""

    polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
    pts = list(polygon.exterior.coords)
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sxx = 0
    sxy = 0
    syy = 0
    a = polygon.area
    cx, cy = centroid(polygon)
    for i in range(len(pts) - 1):
        sxx += (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * (
            x[i] * y[i + 1] - x[i + 1] * y[i]
        )
        syy += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2) * (
            x[i] * y[i + 1] - x[i + 1] * y[i]
        )
        sxy += (
            x[i] * y[i + 1]
            + 2 * x[i] * y[i]
            + 2 * x[i + 1] * y[i + 1]
            + x[i + 1] * y[i]
        ) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    Ixx = sxx / 12 - a * cy**2
    Iyy = syy / 12 - a * cx**2
    Ixy = sxy / 24 - a * cx * cy
    I = np.zeros(shape=(2, 2))
    I[0, 0] = Ixx
    I[1, 0] = -Ixy
    I[0, 1] = -Ixy
    I[1, 1] = Iyy
    I = I / a**2
    return I


def qTensor(polygon):

    S = -inertia(polygon)
    TrS = S[0, 0] + S[1, 1]
    I = np.zeros(shape=(2, 2))
    I[0, 0] = 1
    I[1, 1] = 1
    q = S - TrS * I / 2
    return q


def qTensor_tcj(tcj):

    S = -inertia_tcj(tcj)
    TrS = S[0, 0] + S[1, 1]
    I = np.zeros(shape=(2, 2))
    I[0, 0] = 1
    I[1, 1] = 1
    q = S - TrS * I / 2
    return q


# ----------------------------------------------------


def area(polygon):

    A = polygon.area
    return A


def perimeter(polygon):

    P = polygon.length
    return P


def orientation(polygon):
    """Using the inertia tensor matrix it products the orientation of the polygon"""

    I = inertia(polygon)
    D, V = linalg.eig(I)
    e1 = D[0]
    e2 = D[1]
    v1 = V[:, 0]
    v2 = V[:, 1]
    if e1 < e2:
        v = v1
    else:
        v = v2

    theta = np.arctan2(v[1], v[0])

    return theta


def circularity(polygon):

    A = polygon.area
    P = polygon.length
    Cir = 4 * np.pi * A / (P**2)
    return Cir


def ellipticity(polygon):

    I = inertia(polygon)
    D = linalg.eig(I)[0]
    e1 = D[0]
    e2 = D[1]
    A = polygon.area
    P = polygon.length
    z = (e1 / e2) ** 0.5
    frac1 = np.pi * (2 + z + 1 / z)
    frac2 = A / (P) ** 2
    ell = frac1 * frac2
    ell = ell.real
    return ell


def shapeFactor(polygon):
    """Using the inertia tensor matrix it products the shape factor of the polygon"""

    I = inertia(polygon)
    D = linalg.eig(I)[0]
    e1 = D[0]
    e2 = D[1]
    SF = abs((e1 - e2) / (e1 + e2))
    return SF


def traceS(polygon):

    S = inertia(polygon)
    TrS = S[0, 0] + S[1, 1]
    return TrS


def traceQQ(polygon):

    Q = qTensor(polygon)
    QQ = np.dot(Q, Q)
    TrQQ = QQ[0, 0] + QQ[1, 1]
    return TrQQ


def polarisation(polygon):
    """takes the polygon and finds its inertia tensor matrix"""

    polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
    pts = list(polygon.exterior.coords)
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    cx, cy = centroid(polygon)
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - cx
    y = y - cy
    theta = orientation(polygon)
    phi = np.pi / 2 - theta

    x1 = []
    y1 = []

    for i in range(len(pts)):
        x1.append(x[i] * np.cos(phi) - y[i] * np.sin(phi))
        y1.append(x[i] * np.sin(phi) + y[i] * np.cos(phi))

    x = np.asarray(x1)
    y = np.asarray(y1)

    sxxx = 0
    syyy = 0
    a = polygon.area

    for i in range(len(pts) - 1):
        sxxx += (
            y[i] ** 3 + (y[i] ** 2) * y[i + 1] + y[i] * (y[i + 1]) ** 2 + y[i + 1] ** 3
        ) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        syyy += (
            x[i] ** 3 + (x[i] ** 2) * x[i + 1] + x[i] * (x[i + 1]) ** 2 + x[i + 1] ** 3
        ) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    I_mayor = sxxx / 20
    I_minor = syyy / 20
    I = np.zeros(shape=(2))
    I[0] = I_mayor
    I[1] = I_minor
    I = I / (a ** (5 / 2))
    return I


def mayorPolar(polygon):

    I = polarisation(polygon)

    M = I[0]

    return M


def minorPolar(polygon):

    I = polarisation(polygon)

    m = I[1]

    return m


def polarOri(polygon):

    m = minorPolar(polygon)
    M = mayorPolar(polygon)

    theta = orientation(polygon)

    if m == 0 and M == 0:
        phi = 0
    else:
        phi = np.arctan(m / M)

    if M < 0:
        phi = phi - np.pi

    P_ori = theta + phi

    while P_ori < 0:
        P_ori = P_ori + 2 * np.pi

    return P_ori


def polarMag(polygon):

    m = minorPolar(polygon)
    M = mayorPolar(polygon)

    r = (m**2 + M**2) ** (0.5)

    return r


def polar(polygon):

    y = minorPolar(polygon)
    x = mayorPolar(polygon)
    P = np.array([x, y])

    theta = orientation(polygon)

    xr = P[0] * np.cos(theta) - P[1] * np.sin(theta)
    yr = P[0] * np.sin(theta) + P[1] * np.cos(theta)

    P = np.array([xr, yr])

    return P


def polarUnit(polygon):

    y = minorPolar(polygon)
    x = mayorPolar(polygon)
    P = np.array([x, y])

    theta = orientation(polygon)

    xr = P[0] * np.cos(theta) - P[1] * np.sin(theta)
    yr = P[0] * np.sin(theta) + P[1] * np.cos(theta)

    P = np.array([xr, yr])

    c = (P[0] ** 2 + P[1] ** 2) ** 0.5

    if c != 0:
        P = P / c

    return P


def inertia_tcj(tcj):
    """takes the tcj and finds its inertia tensor matrix"""

    x = []
    y = []
    for i in range(len(tcj)):
        x.append(tcj[i][0])
        y.append(tcj[i][1])
    Ixx = 0
    Iyy = 0
    Ixy = 0
    cx, cy = np.mean(x), np.mean(y)
    for i in range(len(tcj)):
        Ixx += (y[i] - cy) ** 2
        Iyy += (x[i] - cx) ** 2
        Ixy += (x[i] - cx) * (y[i] - cy)

    I = np.zeros(shape=(2, 2))
    I[0, 0] = Ixx
    I[1, 0] = -Ixy
    I[0, 1] = -Ixy
    I[1, 1] = Iyy
    return I


def shapeFactor_tcj(tcj):
    """Using the inertia tensor matrix it products the shape factor of the polygon"""

    I = inertia_tcj(tcj)
    D = linalg.eig(I)[0]
    e1 = D[0]
    e2 = D[1]
    SF = abs((e1 - e2) / (e1 + e2))
    return SF


def orientation_tcj(tcj):
    """Using the inertia tensor matrix it products the orientation of the polygon"""

    I = inertia_tcj(tcj)
    D, V = linalg.eig(I)
    e1 = D[0]
    e2 = D[1]
    v1 = V[:, 0]
    v2 = V[:, 1]
    if e1 < e2:
        v = v1
    else:
        v = v2
    theta = np.arctan2(v[1], v[0])

    return theta


def findContourCurvature(contour, m):

    n = len(contour)

    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    polygon = Polygon(poly)
    (Cx, Cy) = centroid(polygon)

    contourPlus = contour[n - int((m - 1) / 2) : n]
    contourMinus = contour[0 : int((m - 1) / 2)]

    con = np.concatenate([contourPlus, contour, contourMinus])

    class ComputeCurvature:
        def __init__(self):
            """Initialize some variables"""
            self.xc = 0  # X-coordinate of circle center
            self.yc = 0  # Y-coordinate of circle center
            self.r = 0  # Radius of the circle
            self.xx = np.array([])  # Data points
            self.yy = np.array([])  # Data points

        def calc_r(self, xc, yc):
            """calculate the distance of each 2D points from the center (xc, yc)"""
            return np.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2)

        def f(self, c):
            """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
            ri = self.calc_r(*c)
            return ri - ri.mean()

        def df(self, c):
            """Jacobian of f_2b
            The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
            xc, yc = c
            df_dc = np.empty((len(c), x.size))

            ri = self.calc_r(xc, yc)
            df_dc[0] = (xc - x) / ri  # dR/dxc
            df_dc[1] = (yc - y) / ri  # dR/dyc
            df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
            return df_dc

        def fit(self, xx, yy):
            self.xx = xx
            self.yy = yy
            center_estimate = np.r_[np.mean(xx), np.mean(yy)]
            center = optimize.leastsq(
                self.f, center_estimate, Dfun=self.df, col_deriv=True
            )[0]

            self.xc, self.yc = center
            ri = self.calc_r(*center)
            self.r = ri.mean()

            return (1 / self.r, center)  # Return the curvature

    curvature = []

    for i in range(n):
        x = np.r_[con[i : i + m][:, 0]]
        y = np.r_[con[i : i + m][:, 1]]
        comp_curv = ComputeCurvature()
        center = comp_curv.fit(x, y)[1]
        xMid = x[int((m - 1) / 2)]
        yMid = y[int((m - 1) / 2)]

        v0 = np.array([Cx - xMid, Cy - yMid])
        v1 = np.array([center[0] - xMid, center[1] - yMid])
        costheta = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

        if costheta < 0:
            curvature.append(-comp_curv.fit(x, y)[0])
        else:
            curvature.append(comp_curv.fit(x, y)[0])
    return curvature
