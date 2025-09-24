import copy, math
import numpy as np
from scipy.stats import chi2
from tensorly.tenalg import mode_dot
from scipy.special import sph_harm
from sklearn.linear_model import LinearRegression

def double_res(x):
    # double the spatial resolution of x
    coord = np.zeros((2 * x.shape[0]))
    for i, val in enumerate(x):
        if i == 0:
            coord[0] = val / 2
            coord[1] = val
        else:
            coord[2 * i] = (val + x[i - 1]) / 2
            coord[2 * i + 1] = val
    return coord


def UnitCoord(K):
    coord = np.linspace(start=0.0, stop=1.0, num=K + 2)
    return coord[1:-1]


def DoubleUnitCoord(K, base=5):
    # generate a length-K array of spatial coordinate by doubling the resolution of the length-base array
    x = UnitCoord(base)
    current_res = copy.deepcopy(base)

    while current_res < K:
        x = double_res(x)
        current_res *= 2
    return x


def RealSH(l, m, colat, lon):
    # obtain the real spherical harmonics (RSH)
    if m > 0:
        basis = sph_harm(m, l, lon, colat)
        return math.sqrt(2) * ((-1) ** m) * (basis.real)
    elif m == 0:
        return sph_harm(0, l, lon, colat).real
    else:
        basis = sph_harm(-m, l, lon, colat)
        return math.sqrt(2) * ((-1) ** m) * (basis.imag)


def BasisGen(M, N, R, basis_eta=3, return_basis=True):
    eta = basis_eta
    lon, colat = np.meshgrid(UnitCoord(N) * 2 * math.pi, UnitCoord(M) * math.pi)  # colatitude & longitude grid
    lon_long, colat_long = lon.reshape((M * N, 1), order="F"), colat.reshape((M * N, 1), order="F")
    angle = (np.sin(colat_long) @ np.sin(colat_long).T) * np.cos(lon_long - lon_long.T) + (
            np.cos(colat_long) @ np.cos(colat_long).T)
    angle = np.clip(angle, -1, 1)
    cov = (3 + basis_eta) / (12 * math.pi) - basis_eta * np.sqrt((1 - angle) / 2) / (8 * math.pi)

    basis = np.zeros((R, M, N))
    l, m = 0, 0  # spherical harmonics basis order
    eigen_seq = []

    if return_basis:
        while (l ** 2 + l + m + 1) <= R:
            # generate a spherical harmonics basis
            basis[l ** 2 + l + m, :, :] = RealSH(l, m, colat, lon)

            if l == 0:
                eigen_seq.append(1)
            else:
                eigen_seq.append(eta / ((4 * l * l - 1) * (2 * l + 3)))

            if m == l:
                l += 1
                m = -l
            else:
                m += 1

    return {"basis": basis, "eigen_seq": np.array(eigen_seq), "kernel_matrix": cov}


def KernelGen(M, N, basis_eta=3):
    lon, colat = np.meshgrid(UnitCoord(N) * 2 * math.pi, UnitCoord(M) * math.pi)
    lon_long, colat_long = lon.reshape((M * N, 1), order="F"), colat.reshape((M * N, 1), order="F")
    angle = (np.sin(colat_long) @ np.sin(colat_long).T) * np.cos(lon_long - lon_long.T) + (
            np.cos(colat_long) @ np.cos(colat_long).T)
    angle = np.clip(angle, -1, 1)
    cov = (3 + basis_eta) / (12 * math.pi) - basis_eta * np.sqrt((1 - angle) / 2) / (
            8 * math.pi)  # covariance matrix based on the Lebedev kernel

    return cov


def VecTensor23(X):
    # vectorize the 2nd & 3rd mode of a tensor X by column-major order
    a, b, c = X.shape
    X_trans = np.zeros((a, b * c))

    for i in range(a):
        X_trans[i, :] = np.reshape(X[i, :, :], (-1,), order="F")
    return X_trans


def MatTensor23(X, size):
    # matricize 2-mode tensor by unfolding the 2nd mode in column major order
    a = X.shape[0]
    X_trans = np.zeros((a, size[0], size[1]))

    for i in range(a):
        X_trans[i, :, :] = np.reshape(X[i, :], (size[0], size[1]), order="F")
    return X_trans