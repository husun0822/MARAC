import math,random, copy, pickle
import numpy as np
from scipy.special import sph_harm
from utils import *

def MaxEigen(A, B):
    P = A.shape[0]
    d = A.shape[1] * B.shape[1]
    L = np.zeros((d * P, d * P))

    for p in range(P):
        L[0:d, (p * d):((p + 1) * d)] = np.kron(B[p], A[p])

    L[d:(P * d), 0:((P - 1) * d)] = np.eye((P - 1) * d)
    w = np.linalg.eigvals(L)
    return np.abs(w).max()


def simulate(T=500, M=10, N=20, P=3, Q=3, D=3, R=20, basis_eta=3, random_state=42, noise_var=1.0, AB_bandwidth=0.2, auxiliary_multiplier=1.0, basis_gen_method="downsample"):
    # ----- generate simulated A ----- #
    A = np.zeros((P, M, M))
    x, y = np.meshgrid(UnitCoord(M), UnitCoord(M))
    # x, y = np.meshgrid(DoubleUnitCoord(M), DoubleUnitCoord(M))
    rho_A, rho_B, c = 0.5, 0.5, 0.5

    for p in range(P):
        # A[i,j] = (rho_A*(c^p))^(10*|s_i - s_j|)
        A[p] = (rho_A * (c ** p)) ** (10 * np.abs(x - y))
        A[p][np.abs(x-y)>=AB_bandwidth] = 0.0
        cp = np.sign(np.trace(A[p])) * np.linalg.norm(A[p])
        A[p] = A[p] / cp

    # ----- generate simulated B ----- #
    B = np.zeros((P, N, N))
    x, y = np.meshgrid(UnitCoord(N), UnitCoord(N))
    # x, y = np.meshgrid(DoubleUnitCoord(N), DoubleUnitCoord(N))
    for p in range(P):
        # B[i,j] = (rho_B*(c^p))^(10*|s_i - s_j|)
        B[p] = (rho_B * (c ** p)) ** (10 * np.abs(x - y))
        B[p][np.abs(x-y)>=AB_bandwidth] = 0.0

        # ----- generate stationary A, B ----- #
    while MaxEigen(A, B) > 1:
        eigen_AB = MaxEigen(A, B)
        for p in range(P):
            B[p] = B[p] * 0.9 / eigen_AB

    # ----- generate basis functions ----- #
    eta = basis_eta  # smoothness of the functional parameter
    lon, colat = np.meshgrid(UnitCoord(N) * 2 * math.pi,
                             UnitCoord(M) * math.pi)  # colatitude & longitude grid
    # lon, colat = np.meshgrid(DoubleUnitCoord(N) * 2 * math.pi, DoubleUnitCoord(M) * math.pi)  # colatitude & longitude grid
    basis = np.zeros((R, M, N))
    l, m = 0, 0  # spherical harmonics basis order
    eigen_seq = []

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

    # ----- generate ground truth basis ----- #
    ground_truth_basis = np.zeros((Q, D, M, N))  # spatial basis for each (q,d) combination
    lon_long, colat_long = lon.reshape((M * N, 1), order="F"), colat.reshape((M * N, 1), order="F")
    angle = (np.sin(colat_long) @ np.sin(colat_long).T) * np.cos(lon_long - lon_long.T) + (np.cos(colat_long) @ np.cos(colat_long).T)
    angle = np.clip(angle, -1, 1)
    cov = (3 + eta) / (12 * math.pi) - eta * np.sqrt((1 - angle) / 2) / (8 * math.pi)  # covariance matrix based on the Lebedev kernel

    np.random.seed(random_state)
    if basis_gen_method == "GP":
        for q in range(Q):
            for d in range(D):
                true_basis = np.random.multivariate_normal(mean=np.zeros((M * N,)), cov=cov, size=1).squeeze()
                ground_truth_basis[q, d] = np.reshape(true_basis, (M, N), order="F")
    elif basis_gen_method == "downsample":
        fine_lon, fine_colat = np.meshgrid(DoubleUnitCoord(40) * 2 * math.pi, DoubleUnitCoord(40) * math.pi)  # fine-resolution longitude/latitude grid
        fine_lon_long, fine_colat_long = fine_lon.reshape((40 * 40, 1), order="F"), fine_colat.reshape((40 * 40, 1), order="F")
        angle = (np.sin(fine_colat_long) @ np.sin(fine_colat_long).T) * np.cos(fine_lon_long - fine_lon_long.T) + (np.cos(fine_colat_long) @ np.cos(fine_colat_long).T)
        angle = np.clip(angle, -1, 1)
        fine_cov = (3 + eta) / (12 * math.pi) - eta * np.sqrt((1 - angle) / 2) / (8 * math.pi)  # covariance matrix based on the Lebedev kernel for the fine resolution grid
        row_sample_idx = np.arange(start=0, stop=40, step=int(40 / M)) + (int(40 / M) - 1)
        col_sample_idx = np.arange(start=0, stop=40, step=int(40 / N)) + (int(40 / N) - 1)

        for q in range(Q):
            for d in range(D):
                fine_res_basis = np.reshape(np.random.multivariate_normal(mean=np.zeros((40 * 40,)), cov=fine_cov, size=1).squeeze(), (40, 40), order="F")

                for i, m in enumerate(row_sample_idx):
                    for j, n in enumerate(col_sample_idx):
                        ground_truth_basis[q, d, i, j] = fine_res_basis[m, n]

    # ----- generate auxiliary time-series sequence ----- #
    # Z follows a VAR(1) process
    np.random.seed(random_state)
    Z = np.zeros((T, D))
    C_Z = np.array([[0.9, 0.05, -0.05], [0.05, 0.85, -0.05], [-0.05, -0.05, 0.8]])


    Z[0, :] = np.random.normal(loc=0.0, scale=0.1, size=(D,))
    for t in range(1, T):
        Z[t, :] = (C_Z @ Z[t - 1, :].reshape((D, 1))).squeeze() + np.random.normal(loc=0.0, scale=0.2, size=(D,))

        # ----- generate error covariance component ----- #
    # Sigma_c, Sigma_r = np.eye(N), np.eye(M)
    x, y = np.meshgrid(UnitCoord(M), UnitCoord(M))
    Sigma_r = (0.5) ** (10 * abs(x - y)) * noise_var

    x, y = np.meshgrid(UnitCoord(N), UnitCoord(N))
    Sigma_c = (0.5) ** (10 * abs(x - y)) * noise_var
    Sigma = np.kron(Sigma_c, Sigma_r)

    # ----- generate matrix time-series sequence ----- #
    np.random.seed(random_state)
    Et = np.random.multivariate_normal(mean=np.zeros((M * N,)), cov=Sigma, size=T)
    Et = MatTensor23(Et, (M, N))
    Et[0] = np.zeros((M, N))

    X = copy.deepcopy(Et)
    ground_truth_basis = auxiliary_multiplier * ground_truth_basis # apply a multiplier to scale the basis
    transposed_basis = np.transpose(ground_truth_basis, axes=(0, 2, 3, 1))

    np.random.seed(random_state)
    for t in range(max(P, Q)):
        X[t] += np.random.normal(loc=0.0, scale=0.5, size=(M, N))

    for t in range(max(P, Q), T):
        Xt = np.zeros((M, N))

        # add the auto-regressive part
        for p in range(P):
            Xt += A[p] @ X[t - 1 - p] @ B[p].T

        # add the non-parametric part
        for q in range(Q):
            Xt += transposed_basis[q] @ Z[t - 1 - q]

        X[t] += Xt

    # organize the output
    AR = {"A": A, "B": B}
    AC = {"basis": basis, "ground_truth_basis": ground_truth_basis, "eigen_seq": np.array(eigen_seq), "kernel_matrix": cov}
    ERR = {"Sigma_c": Sigma_c, "Sigma_r": Sigma_r}
    data = {"Matrix-TS": X, "Vector-TS": Z}

    return AR, AC, ERR, data