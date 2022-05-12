import numpy as np
from tqdm import tqdm
from config import parse_configs
from AuC_dataloader import create_dataloader
from baselines import to_gray, to_random


def area_under_curve(ab_pred, ab_true):
	# assumes input of dimension (images, W, H, 2)
	thresholds = np.arange(0, 150)
	ab_pred = np.repeat(ab_pred[:, None, :, :, :], thresholds.size, axis=1)
	ab_true = np.repeat(ab_true[:, None, :, :, :], thresholds.size, axis=1)

	l2_norms = np.linalg.norm(ab_pred - ab_true, axis=4, keepdims=True)
	accuracies = np.mean(l2_norms <= thresholds, axis=(2, 3, 4))

	return np.trapz(accuracies, thresholds)[0] / (thresholds.size - 1)


if __name__ == '__main__':
	configs = parse_configs()
	val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val_local", "tree.p")

	for X, ab_true, Weights, ii in tqdm(val_loader, leave=False):
		ab_true = ab_true.numpy().transpose([0, 2, 3, 1])
		ab_gray = to_gray(ab_true)
		ab_random = to_random(ab_true, val_loader)

		accuracy_gray = area_under_curve(ab_gray, ab_true)
		accuracy_random = area_under_curve(ab_random, ab_true)

		print("Baseline (gray) accuracy: " + str(accuracy_gray))
		print("Baseline (random) accuracy: " + str(accuracy_random))
