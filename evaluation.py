import numpy as np


# assumes input of dimension (images, W, H, 2)
def area_under_curve(ab_pred, ab_true):
	thresholds = np.arange(0, 150)
	ab_pred = np.repeat(ab_pred[:, None, :, :, :], thresholds.size, axis=1)
	ab_true = np.repeat(ab_true[:, None, :, :, :], thresholds.size, axis=1)

	l2_norms = np.linalg.norm(ab_pred - ab_true, axis=4, keepdims=True)
	accuracies = np.mean(l2_norms <= thresholds, axis=(2, 3, 4))

	return np.trapz(accuracies, thresholds)[0] / (thresholds.size - 1)

