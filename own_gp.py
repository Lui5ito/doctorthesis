"""
Trying to make my own implementation of Gaussian Process.
1- Define the kernel
2- Make the kernel Gram matrix
3- Compute the loglikelihood, as a function of the kernel's HP and sigma the noise.

The goal is to compute the alphas of the GP.
"""


from scipy.linalg import cho_factor, cho_solve
import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist, squareform


class Matern52(object):
    """
    """

    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, X, Y=None):
        if Y is None:
            dists = pdist(X/self.length_scale, metric="euclidean")
        else:
            dists = cdist(X/self.length_scale, Y/self.length_scale, metric="euclidean")

        K = dists * np.sqrt(5)
        K = (1.0+K+K**2/3.0) * np.exp(-K)

        if Y is None:
            # convert from upper-triangular matrix to square matrix.
            K = squareform(K)
            np.fill_diagonal(K, 1)

        return K

    def grad(self, X, Y=None):
        if Y is None:
            dists = pdist(X/self.length_scale, metric="euclidean")
        else:
            dists = cdist(X/self.length_scale, Y/self.length_scale, metric="euclidean")

        K = dists * np.sqrt(5)
        K_grad = (1.0 + K) * (K**2 / (3.0*self.length_scale)) * np.exp(-K)

        if Y is None:
            # convert from upper-triangular matrix to square matrix.
            K_grad = squareform(K_grad)
            np.fill_diagonal(K_grad, 1)

        return K_grad


class GaussianProcessRegressor(object):
    """
    """

    def __init__(self, kernel, num_restarts):
        self.kernel = kernel
        self.num_restarts = num_restarts
        self.sigma = None
        self.X_train = None
        self.gamma_posterior = None
        self.mean = None
        self.std = None

    def train(self, X, y):
        self.X_train = X

        # 1- Define objective function
        def log_likelihood(theta):
            # Unpack theta
            lengthscale = theta[0]
            sigma = theta[1]

            # Compute K
            self.kernel.lengthscale = lengthscale
            K = self.kernel(X)
            print(K)

            # Add the sigma noise to K
            K_noisy = K + sigma*np.eye(K.shape[0])

            # Compute Cholesky factorisation
            L, lower = cho_factor(K_noisy)

            # Compute alpha
            gamma_gp = cho_solve((L, lower), y)

            # Compute all three terms
            nlog_likelihood = (-1/2)*np.matmul(y.T, gamma_gp)
            nlog_likelihood -= np.log(np.diag(L)).sum()
            nlog_likelihood -= (K.shape[0]/2)*np.log(2*np.pi)

            # Compute derivaties
            gammagammaT = np.matmul(gamma_gp, gamma_gp.T)
            K_noisy_inv = cho_solve((L, lower), np.eye(K_noisy.shape[0]))
            eta = gammagammaT - K_noisy_inv

            # Compute derivative wrt lengthscale
            K_ls = self.kernel.grad(X)
            d_wrt_ls = 0.5 * np.trace(np.matmul(eta, K_ls))

            # Compute derivative wrt sigma
            d_wrt_sigma = 0.5 * np.trace(np.matmul(eta, 2*sigma*np.eye(K.shape[0])))

            return nlog_likelihood

        # 2- Optimize the objective function
        best_likelihood = np.inf  # Start with a very high likelihood
        best_theta = None  # Placeholder for the optimal theta

        for i in range(self.num_restarts):
            print("Restart number: ", i)
            # Initiate random theta
            initial_theta = np.random.uniform(0, 1, (1, 2))[0]
            # initial_theta = np.array([0.5, 0.5])
            print(initial_theta)

            optim_res = scipy.optimize.minimize(
                log_likelihood,
                initial_theta,
                method="L-BFGS-B",
                jac=False,
            )
            theta_opt, min_likelihood = optim_res.x, optim_res.fun
            print(min_likelihood)

            if min_likelihood < best_likelihood:
                best_likelihood = min_likelihood
                best_theta = theta_opt

        self.kernel.lengthscale = best_theta[0]
        self.sigma = best_theta[1]

        # Compute gamma_gp posterior
        K = self.kernel(X)
        K_noisy = K + self.sigma*np.eye(K.shape[0])
        L, lower = cho_factor(K_noisy)
        gamma_gp = cho_solve((L, lower), y)
        self.gamma_posterior = gamma_gp

        return best_theta

    def predict(self, X):
        K_test = self.kernel(X=self.X_train, Y=X).T
        preds = np.matmul(K_test, self.gamma_posterior)
        return preds


if __name__ == "__main__":
    import universalbands as ub
    # Generate the data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    X_train, y_train = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=sample_size + 1,
        sample_dim=sample_dim,
        seed=123,
    )

    my_kernel = Matern52(length_scale=1)
    gp_model = GaussianProcessRegressor(kernel=my_kernel, num_restarts=1)

    theta = gp_model.train(X=X_train, y=y_train)

    print("Done training! With theta = ", theta)
    import matplotlib.pyplot as plt

    X_test, y_test = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=3*sample_size + 1,
        sample_dim=sample_dim,
        seed=987,
    )

    gp_predictions = gp_model.predict(X=X_test)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    X_test_sq = X_test.squeeze()
    sorted_indices = np.argsort(X_test_sq)
    X_test_sorted = X_test_sq[sorted_indices].squeeze()
    gp_predictions_sq = gp_predictions[sorted_indices].squeeze()
    ax.scatter(X_test, y_test, s=35, c="black", alpha=0.6, marker="o")
    ax.plot(X_test_sorted, gp_predictions_sq, color="blue", linewidth=3)
    fig.savefig("figure1.pdf")
