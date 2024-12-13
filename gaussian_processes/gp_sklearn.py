import universalbands as ub
from GPy.kern import Matern52
from GPy.models.gp_regression import GPRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import os
import s3fs
import json
import matplotlib.pyplot as plt
import numpy as np

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

# Train the Gausian Process Regression.
ker_matern = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
ker_cste = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e5))
ker_white = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
kernel_1 = ker_matern + ker_white

gp_model_sk_1 = GaussianProcessRegressor(kernel=kernel_1, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None)

gp_model_sk_1.fit(X_train, y_train)

X_test, y_test = ub.data_generation.synthetic_data(
    case=f"case_{case_number}",
    sample_size=3*sample_size + 1,
    sample_dim=sample_dim,
    seed=987,
)

gp_predictions = gp_model_sk_1.predict(X=X_test)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
X_test_sq = X_test.squeeze()
sorted_indices = np.argsort(X_test_sq)
X_test_sorted = X_test_sq[sorted_indices].squeeze()
gp_predictions_sq = gp_predictions[sorted_indices].squeeze()
ax.scatter(X_test, y_test, s=35, c="black", alpha=0.6, marker="o")
ax.plot(X_test_sorted, gp_predictions_sq, color="blue", linewidth=3, label="from predict")
plt.legend()
fig.savefig("gp_from_sklearn.pdf")
