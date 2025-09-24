import copy, math
import numpy as np
from scipy.stats import chi2
from tensorly.tenalg import mode_dot
from scipy.special import sph_harm
from sklearn.linear_model import LinearRegression
from utils import *

class MARAC():

    def __init__(self, P=1, Q=1, R=20, method="Truncated_PMLE"):
        self.P, self.Q, self.R = P, Q, R
        self.method = method


    def fit(self, MatrixTS, VectorTS, KGram=None, Basis=None, BasisEigen=None, max_iter=200, tol=1e-3, lmbda=1e-3, print_freq=100):
        '''
        Fit the MARAC(P,Q) model

        Inputs:
        MatrixTS: (T,M,N) array, the matrix time series
        VectorTS: (T,D) array, the D-dimensional vector time series
        KGram: (MN,MN) array, the kernel gram matrix
        Basis: (R,M,N) array, the spatial bases functions
        BasisEigen: length-R array, the eigenvalue sequence of the bases functions
        max_iter: maximum number of alternating minimization iterations
        tol: convergence threshold
        lmbda: RKHS functional norm penalty tuning parameter
        '''

        if self.method == "Truncated_PMLE" and (Basis is None):
            raise Exception("Please input the spatial bases!")
        elif self.method == "PMLE" and (KGram is None):
            raise Exception("Please input the kernel gram matrix!")

        # -------------------------- #
        # -- Model Dimensionality -- #
        # -------------------------- #
        T, M, N = MatrixTS.shape
        D = VectorTS.shape[1]
        self.D, self.M, self.N = D, M, N

        R = self.R if self.method == "Truncated_PMLE" else (M * N)
        P, Q = self.P, self.Q
        PvQ = max(P, Q)

        self.KGram = KGram

        # --------------------------- #
        # ---- Regression Target ---- #
        # --------------------------- #
        Y = MatrixTS[PvQ:T]  # regression target
        Y_len = Y.shape[0]  # number of testing frames
        Y_index = np.array(list(range(PvQ, T)))  # target frame indices

        # ------------------------------- #
        # ---- Precompute Quantities ---- #
        # ------------------------------- #
        VectorTS_Moment = np.zeros((Q, D, D))
        for q in range(Q):
            VectorTS_Moment[q] = VectorTS[Y_index - q - 1].T @ VectorTS[Y_index - q - 1]

        if self.method == "Truncated_PMLE":
            Basis_Vec = VecTensor23(Basis)

        # ------------------------------ #
        # ---- Estimator Initialize ---- #
        # ------------------------------ #
        A, B, A_old, B_old = np.random.normal(size=(P, M, M)), np.random.normal(size=(P, N, N)), np.random.normal(
            size=(P, M, M)), np.random.normal(size=(P, N, N))
        gamma, gamma_old = np.random.normal(size=(Q, R, D)), np.random.normal(size=(Q, R, D))
        Sigma_c, Sigma_r, Sigma_c_old, Sigma_r_old = np.eye(N), np.eye(M), np.eye(N), np.eye(M)
        Sigma_r_inv, Sigma_c_inv = np.linalg.inv(Sigma_r), np.linalg.inv(Sigma_c)

        # ---------- Coordinate Penalized MLE ---------- #
        iter_counter, iter_delta = 0, 1
        loss_hist, delta_hist = [], []
        while (iter_counter < max_iter) and (iter_delta > tol):
            # compute the full matrix time series residual
            Y_pred = np.zeros_like(Y)
            for p in range(P):
                Y_pred += A[p] @ MatrixTS[Y_index - p - 1] @ B[p].T

            for q in range(Q):
                coef = VectorTS[Y_index - q - 1] @ gamma[q].T  # (T_test, R)
                if self.method == "Truncated_PMLE":
                    Y_pred += mode_dot(Basis, coef, mode=0)
                elif self.method == "PMLE":
                    Y_pred += MatTensor23(coef @ KGram, (M, N))
            Y_residual = Y - Y_pred

            # ----- Update Auto-regressive Coefficient (AR) ----- #
            for p in range(P):
                partial_residual = Y_residual + A[p] @ MatrixTS[Y_index - p - 1] @ B[p].T

                # A-step
                X_BT = MatrixTS[Y_index - p - 1] @ B[p].T
                RR = np.sum(X_BT @ Sigma_c_inv @ np.transpose(X_BT, axes=(0, 2, 1)), axis=0)
                LL = np.sum(partial_residual @ Sigma_c_inv @ np.transpose(X_BT, axes=(0, 2, 1)), axis=0)
                A[p] = LL @ np.linalg.inv(RR)

                # B-step
                A_X = A[p] @ MatrixTS[Y_index - p - 1]
                RR = np.sum(np.transpose(A_X, axes=(0, 2, 1)) @ Sigma_r_inv @ A_X, axis=0)
                LL = np.sum(np.transpose(partial_residual, axes=(0, 2, 1)) @ Sigma_r_inv @ A_X, axis=0)
                B[p] = LL @ np.linalg.inv(RR)

                # re-apply the updated parameter to the prediction residual
                Y_residual = partial_residual - A[p] @ MatrixTS[Y_index - p - 1] @ B[p].T

            # ----- Update Auxiliary Vector TS Coefficient (AC) ----- #
            for q in range(Q):
                if self.method == "Truncated_PMLE":
                    partial_residual = Y_residual + mode_dot(Basis, VectorTS[Y_index - q - 1] @ gamma[q].T,
                                                             mode=0)  # (T_test,M,N)

                    # memory-efficient implementation
                    Z_Xresid = np.zeros((D * M * N, 1))
                    for i, t in enumerate(Y_index):
                        # Z_Xresid += np.kron(np.expand_dims(VectorTS[t-q-1], axis=1), partial_residual_vec[i])
                        Z_Xresid += np.kron(np.expand_dims(VectorTS[t - q - 1], axis=1),
                                            np.reshape(Sigma_r_inv @ partial_residual[i] @ Sigma_c_inv, (-1, 1),
                                                       order="F"))
                    RR = np.kron(np.eye(D), Basis_Vec) @ Z_Xresid

                    # K_S_KT = Basis_Vec @ (np.kron(Sigma_c_inv, Sigma_r_inv)) @ Basis_Vec.T
                    # memory-efficient implementation
                    K_S_KT = np.zeros((self.R, self.R))
                    for i1 in range(self.R):
                        for j1 in range(i1, self.R):
                            K_S_KT[i1, j1] = np.trace(Basis[i1].T @ Sigma_r_inv @ Basis[j1] @ Sigma_c_inv)
                            K_S_KT[j1, i1] = K_S_KT[i1, j1]

                    LL = np.kron(VectorTS_Moment[q], K_S_KT) + lmbda * np.kron(np.eye(D), np.diag(BasisEigen ** (-1)))
                    # LL = np.kron(VectorTS_Moment[q], K_S_KT) + lmbda * np.kron(np.eye(D), np.eye(BasisEigen.shape[0]))
                    gamma_q_vec = np.linalg.inv(LL) @ RR
                    gamma[q] = np.reshape(gamma_q_vec, (R, D), order="F")
                    Y_residual = partial_residual - mode_dot(Basis, VectorTS[Y_index - q - 1] @ gamma[q].T, mode=0)
                elif self.method == "PMLE":
                    partial_residual = Y_residual + MatTensor23((VectorTS[Y_index - q - 1] @ gamma[q].T) @ KGram,
                                                                (M, N))
                    LL = np.kron(VectorTS_Moment[q],
                                 KGram @ np.kron(Sigma_c_inv, Sigma_r_inv) @ KGram) + lmbda * np.kron(np.eye(D), KGram)
                    RR = np.zeros((M * N * D, 1))

                    for i, t in enumerate(Y_index):
                        RR += np.kron(np.expand_dims(VectorTS[t - q - 1], axis=1),
                                      np.reshape(partial_residual[i], (M * N, 1), order="F"))
                    gamma_q_vec = np.linalg.inv(LL) @ np.kron(np.eye(D), KGram @ np.kron(Sigma_c_inv, Sigma_r_inv)) @ RR
                    gamma[q] = np.reshape(gamma_q_vec, (R, D), order="F")
                    Y_residual = partial_residual - MatTensor23((VectorTS[Y_index - q - 1] @ gamma[q].T) @ KGram,
                                                                (M, N))

            # ----- Update the Covariance Component ----- #
            Sigma_c = np.mean(np.transpose(Y_residual, axes=(0, 2, 1)) @ Sigma_r_inv @ Y_residual, axis=0) / M
            Sigma_r = np.mean(Y_residual @ np.linalg.inv(Sigma_c) @ np.transpose(Y_residual, axes=(0, 2, 1)), axis=0) / N
            Sigma_c_inv = np.linalg.inv(Sigma_c)
            Sigma_r_inv = np.linalg.inv(Sigma_r)

            # ----- Rescale the A, B factors ----- #
            for p in range(P):
                cp = np.sign(np.trace(A[p])) * np.linalg.norm(A[p])
                A[p] = A[p] / cp
                B[p] = B[p] * cp

            # ----- Check Convergence ----- #
            delta_AR = 0
            for p in range(P):
                delta_AR += (np.linalg.norm(B_old[p]) * np.linalg.norm(A[p] - A_old[p]) + np.linalg.norm(
                    A[p]) * np.linalg.norm(B[p] - B_old[p])) ** 2
            delta_AR = delta_AR / (P * M * M * N * N) if (P > 0) else 0
            delta_AC = (np.linalg.norm(gamma - gamma_old) ** 2) / (Q * R * D) if (Q > 0) else 0
            # delta_AC = np.linalg.norm(gamma - gamma_old)**2
            delta_Sigma = (np.linalg.norm(Sigma_c_old) * np.linalg.norm(Sigma_r - Sigma_r_old) + np.linalg.norm(
                Sigma_r) * np.linalg.norm(Sigma_c - Sigma_c_old)) ** 2 / (M * M * N * N)
            iter_delta = max(delta_AR, delta_AC, delta_Sigma)
            delta_hist.append(iter_delta)
            iter_counter += 1

            # recalculate the loss
            nll = 0.5 * Y_len * (M * np.log(np.linalg.det(Sigma_c)) + N * np.log(np.linalg.det(Sigma_r)))
            for t_pred in range(Y_len):
                nll += 0.5 * np.trace(Y_residual[t_pred].T @ Sigma_r_inv @ Y_residual[t_pred] @ Sigma_c_inv)
            newloss = copy.deepcopy(nll)
            for q in range(Q):
                if self.method == "Truncated_PMLE":
                    newloss += 0.5 * lmbda * np.trace(gamma[q].T @ np.diag(BasisEigen ** (-1)) @ gamma[q])
                elif self.method == "PMLE":
                    for d in range(D):
                        newloss += (0.5 * lmbda * np.expand_dims(gamma[q, :, d], axis=0) @ KGram @ np.expand_dims(
                            gamma[q, :, d], axis=1)).squeeze()
            loss_hist.append(newloss)

            # overwrite the old parameters
            A_old, B_old = copy.deepcopy(A), copy.deepcopy(B)
            Sigma_c_old, Sigma_r_old = copy.deepcopy(Sigma_c), copy.deepcopy(Sigma_r)
            gamma_old = copy.deepcopy(gamma)

            if iter_counter % print_freq == 0:
                print(f"Iter = {iter_counter}, Delta = {iter_delta}")

        # compute the model selection information criterion
        dof = (P+1) * (M*M + N*N - 1)  # model complexity
        for q in range(Q):
            if self.method == "PMLE":
                LL = np.kron(VectorTS_Moment[q], KGram @ np.kron(Sigma_c_inv, Sigma_r_inv) @ KGram) + lmbda * np.kron(np.eye(D), KGram)
                dof += np.trace(np.linalg.inv(LL) @ (LL - lmbda * np.kron(np.eye(D), KGram)))
            elif self.method == "Truncated_PMLE":
                K_S_KT = np.zeros((self.R, self.R))
                for i1 in range(self.R):
                    for j1 in range(i1, self.R):
                        K_S_KT[i1, j1] = np.trace(Basis[i1].T @ Sigma_r_inv @ Basis[j1] @ Sigma_c_inv)
                        K_S_KT[j1, i1] = K_S_KT[i1, j1]
                LL = np.kron(VectorTS_Moment[q], K_S_KT) + lmbda * np.kron(np.eye(D), np.diag(BasisEigen ** (-1)))
                dof += np.trace(np.linalg.inv(LL) @ (LL - lmbda * np.kron(np.eye(D), np.diag(BasisEigen ** (-1)))))
        AIC = (2 * nll + 2 * dof) / Y_len # Akaike Information Criterion
        BIC = (2 * nll + np.log(Y_len) * dof) / Y_len # Bayesian Information Criterion

        self.params = {"A": A, "B": B, "Sigma_r": Sigma_r, "Sigma_c": Sigma_c, "gamma": gamma, "loss": loss_hist,
                       "delta_hist": delta_hist}
        self.IC = {"AIC": AIC, "BIC": BIC, "nll": nll, "df": dof}

    def reconstruct(self, Basis=None):
        # reconstruct the patterns for each of the auxiliary covariate
        pattern = np.zeros((self.Q, self.D, self.M, self.N))

        if self.method == "Truncated_PMLE":
            for q in range(self.Q):
                for d in range(self.D):
                    pattern[q, d] = mode_dot(Basis, np.expand_dims(self.params["gamma"][q, :, d], axis=0),
                                             mode=0).squeeze()
        elif self.method == "PMLE":
            for q in range(self.Q):
                for d in range(self.D):
                    pattern[q, d] = np.reshape(self.KGram @ np.expand_dims(self.params["gamma"][q, :, d], axis=1),
                                               (self.M, self.N), order="F")

        self.params["pattern"] = pattern

    def predict(self, MatrixTS, VectorTS, Basis=None):
        T = MatrixTS.shape[0]
        PvQ = max(self.P, self.Q)
        Y_pred = np.zeros((T - PvQ, self.M, self.N))
        Y_index = np.array(list(range(PvQ, T)))

        for p in range(self.P):
            Y_pred += self.params["A"][p] @ MatrixTS[Y_index - p - 1] @ self.params["B"][p].T

        for q in range(self.Q):
            coef = VectorTS[Y_index - q - 1] @ self.params["gamma"][q].T
            if self.method == "Truncated_PMLE":
                Y_pred += mode_dot(Basis, coef, mode=0)
            elif self.method == "PMLE":
                Y_pred += MatTensor23(coef @ self.KGram, (self.M, self.N))

        return Y_pred

    def specification_test(self, MatrixTS, VectorTS, alpha=0.05, print_decision=True):
        """
            Conduct a specification test for the auxiliary covariates. The computation here is hard-coded.
        """
        inv_cov = np.kron(np.linalg.inv(self.params["Sigma_c"]), np.linalg.inv(self.params["Sigma_r"]))
        PvQ = max(self.P, self.Q)
        M, N, fitP, fitQ, fitD = self.M, self.N, self.P, self.Q, self.D

        # compute W_t
        total_T = MatrixTS.shape[0] - PvQ
        A_hat, B_hat, G_hat = self.params['A'], self.params['B'], self.params['pattern']

        # placeholder
        block_1, block_2, block_3 = np.zeros((total_T, M * N, fitP * M * M)), np.zeros(
            (total_T, M * N, fitP * N * N)), np.zeros((total_T, M * N, M * N * fitQ * fitD))
        for t in range(PvQ, MatrixTS.shape[0]):
            A_component, B_component = np.zeros((M * N, fitP * N * N)), np.zeros((M * N, fitP * M * M))

            # A/B component
            for p in range(fitP):
                B_component[:, (p * M * M):((p + 1) * M * M)] = np.kron(B_hat[p] @ MatrixTS[t - p].T, np.eye(M))
                A_component[:, (p * N * N):((p + 1) * N * N)] = np.kron(np.eye(N), A_hat[p] @ MatrixTS[t - p])

            # auxiliary component
            z_component = np.zeros((M * N, M * N * fitQ * fitD))

            for q in range(fitQ):
                z_component[:, (q * M * N * fitD):((q + 1) * M * N * fitD)] = np.kron(
                    np.expand_dims(VectorTS[t - q], axis=0), self.KGram)

            block_1[t - 1] = B_component
            block_2[t - 1] = A_component
            block_3[t - 1] = z_component

        W = np.concatenate([block_1, block_2, block_3], axis=2)

        # compute H
        parameter_dimension = fitP * (M**2 + N**2) + fitQ * fitD * M * N
        WT_S_W = np.zeros((parameter_dimension, parameter_dimension))
        for k in range(W.shape[0]):
            WT_S_W += W[k].T @ inv_cov @ W[k]
        WT_S_W = WT_S_W / W.shape[0]

        epsilon = np.zeros((WT_S_W.shape[0], 1))
        for p in range(fitP):
            epsilon[(p * M * M):((p + 1) * M * M), :] = np.reshape(A_hat[0], (-1, 1), order='F')

        H = WT_S_W + epsilon @ epsilon.T

        # compute the variance-covariance matrix of vec(G)
        Xi = np.linalg.inv(H) @ WT_S_W @ np.linalg.inv(H)
        left_multiplier = np.concatenate(
            [
                np.zeros((fitQ * fitD * M * N, fitP * (M * M + N * N))),
                np.kron(np.eye(fitQ * fitD), self.KGram)
            ],
            axis=1
        )
        Phi = left_multiplier @ Xi @ left_multiplier.T

        # compute the test statistics
        vecG = np.zeros((fitQ * fitD * M * N, 1))
        for q in range(fitQ):
            start_loc = q * (fitD * M * N)
            for d in range(fitD):
                vecG[(start_loc + d * M * N):(start_loc + (d + 1) * M * N), :] = np.reshape(G_hat[q, d], (M * N, 1),
                                                                                            order='F')

        test_statistics = total_T * (vecG.T @ np.linalg.inv(Phi) @ vecG)[0, 0]

        # conduct test
        dof = fitQ * fitD * M * N - 1
        p_value = 1.0 - chi2.cdf(test_statistics, df=dof)
        decision = "cannot reject the null" if p_value >= alpha else "reject the null"

        if print_decision:
            print(f"p_value = {p_value:.5f}. {decision}")

        return test_statistics, p_value


class PixAR():

    def __init__(self, P=1, Q=1):
        self.P, self.Q = P, Q

    def fit(self, X_train, Z_train, X_test, Z_test):
        P, Q, PvQ = self.P, self.Q, max(self.P, self.Q)
        T, M, N = X_train.shape
        D = Z_train.shape[1]
        T_test = X_test.shape[0]

        # fit pixel-wise linear model
        Y_pred = np.zeros((T_test - PvQ, M, N))
        Y_test = X_test[PvQ:T_test]

        for m in range(M):
            for n in range(N):
                # regression response
                y_train = X_train[PvQ:T, m, n]

                # regression predictor
                x_train = np.zeros((y_train.shape[0], P * 1 + Q * D))
                x_test = np.zeros((Y_test.shape[0], P * 1 + Q * D))

                # fill in autoregressive term
                for p in range(P):
                    x_train[:, p] = X_train[(PvQ - p - 1):(T - p - 1), m, n]
                    x_test[:, p] = X_test[(PvQ - p - 1):(T_test - p - 1), m, n]

                    # fill in auxiliary covariate term
                for q in range(Q):
                    x_train[:, (P + q * D):(P + (q + 1) * D)] = Z_train[(PvQ - p - 1):(T - p - 1), :]
                    x_test[:, (P + q * D):(P + (q + 1) * D)] = Z_test[(PvQ - p - 1):(T_test - p - 1), :]

                # fit linear model
                model = LinearRegression(fit_intercept=True)
                model.fit(X=x_train, y=np.reshape(y_train, (y_train.shape[0], 1)))
                Y_pred[:, m, n] = model.predict(X=x_test).squeeze()

        return Y_pred


class VAR():

    def __init__(self, P=1, Q=1):
        self.P, self.Q = P, Q

    def fit(self, X_train, Z_train, X_test, Z_test):
        P, Q, PvQ = self.P, self.Q, max(self.P, self.Q)
        T, M, N = X_train.shape
        D = Z_train.shape[1]
        T_test = X_test.shape[0]
        S = M * N

        # fit pixel-wise linear model
        Y_pred = np.zeros((T_test - PvQ, M, N))
        Y_test = X_test[PvQ:T_test]

        for m in range(M):
            for n in range(N):
                # regression response
                y_train = X_train[PvQ:T, m, n]

                # regression predictor
                x_train = np.zeros((y_train.shape[0], P * S + Q * D))
                x_test = np.zeros((Y_test.shape[0], P * S + Q * D))

                # fill in autoregressive term
                for p in range(P):
                    x_train[:, (p * S):((p + 1) * S)] = X_train[(PvQ - p - 1):(T - p - 1), :, :].reshape(
                        (y_train.shape[0], -1))
                    x_test[:, (p * S):((p + 1) * S)] = X_test[(PvQ - p - 1):(T_test - p - 1), :, :].reshape(
                        (Y_pred.shape[0], -1))

                # fill in auxiliary covariate term
                for q in range(Q):
                    x_train[:, (P * S + q * D):(P * S + (q + 1) * D)] = Z_train[(PvQ - p - 1):(T - p - 1), :]
                    x_test[:, (P * S + q * D):(P * S + (q + 1) * D)] = Z_test[(PvQ - p - 1):(T_test - p - 1), :]

                # fit linear model
                model = LinearRegression(fit_intercept=True)
                model.fit(X=x_train, y=np.reshape(y_train, (y_train.shape[0], 1)))
                Y_pred[:, m, n] = model.predict(X=x_test).squeeze()

        return Y_pred


class MAR_LM():

    def __init__(self, P=1, Q=1):
        self.P, self.Q = P, Q

    def fit(self, data, coef):
        D = data["Z_train"].shape[1]

        # fit the MAR model
        model = MARAC(P=self.P, Q=0, R=121, method="PMLE")
        model.fit(data["X_train"], data["Z_train"], KGram=coef["kernel_matrix"], Basis=coef["basis"],
                  BasisEigen=coef["eigen_seq"], max_iter=3000, tol=1e-4, lmbda=1e-2)
        model.reconstruct(coef["basis"])

        # get the residual time series
        Y_train_pred = model.predict(data["X_train"], data["Z_train"], coef["basis"])
        Y_test_pred = model.predict(data["X_test"], data["Z_test"], coef["basis"])
        Y_residual = data["X_train"][self.P:, :, :] - Y_train_pred
        T, T_test = data["X_train"].shape[0], data["X_test"].shape[0]

        # fit pixelwise linear model
        M, N = Y_residual.shape[1], Y_residual.shape[2]
        Y_pred = Y_test_pred
        for m in range(M):
            for n in range(N):
                y_train = Y_residual[:, m, n]  # regression response
                x_train = np.zeros((y_train.shape[0], self.Q * D))  # regression covariate
                x_test = np.zeros((Y_pred.shape[0], self.Q * D))  # test set regression covariate
                for q in range(self.Q):
                    x_train[:, (q * D):((q + 1) * D)] = data["Z_train"][(self.P - q - 1):(T - q - 1), :]
                    x_test[:, (q * D):((q + 1) * D)] = data["Z_test"][(self.P - q - 1):(T_test - q - 1), :]

                lm_model = LinearRegression(fit_intercept=True)
                lm_model.fit(X=x_train, y=np.reshape(y_train, (y_train.shape[0], 1)))
                Y_pred[:, m, n] += lm_model.predict(X=x_test).squeeze()

        return Y_pred

