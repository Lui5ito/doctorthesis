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

# Train the Gausian Process Regression with GPy
kernel = Matern52(input_dim=1)
gp_model_gpy = GPRegression(
    X_train,
    y_train,
    kernel,
    Y_metadata=None,
    normalizer=None,
    noise_var=1.0,
    mean_function=None,
)
gp_model_gpy.kern.variance.fix()

gp_model_gpy.optimize_restarts(
    num_restarts=10, messages=False, verbose=True, max_iters=1000, parallel=True,
)
print("Posterior lengthscale: ", gp_model_gpy.kern.lengthscale[0])
print("Posterior variance: ", gp_model_gpy.kern.variance[0])
print("Posterior Gaussian noise?: ", gp_model_gpy.Gaussian_noise.variance[0])

X_test, y_test = ub.data_generation.synthetic_data(
    case=f"case_{case_number}",
    sample_size=3*sample_size + 1,
    sample_dim=sample_dim,
    seed=987,
)

gp_predictions, _ = gp_model_gpy.predict(Xnew=X_test)
gp_pred, _ = gp_model_gpy.predict_noiseless(Xnew=X_test)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
X_test_sq = X_test.squeeze()
sorted_indices = np.argsort(X_test_sq)
X_test_sorted = X_test_sq[sorted_indices].squeeze()
gp_predictions_sq = gp_predictions[sorted_indices].squeeze()
gp_pred_sq = gp_pred[sorted_indices].squeeze()
ax.scatter(X_test, y_test, s=35, c="black", alpha=0.6, marker="o")
ax.plot(X_test_sorted, gp_predictions_sq, color="blue", linewidth=6, label="from predict")
ax.plot(X_test_sorted, gp_pred_sq, color="red", linewidth=3, label="from predict_noiseless")
plt.legend()
fig.savefig("gp_from_gpy.pdf")
