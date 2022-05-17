import numpy as np
import os
import torch
import pickle
from scipy.stats import norm
from tqdm import tqdm
from config import parse_configs
from AuC_dataloader import create_dataloader
from dataloaders import encode
from baselines import to_gray, to_random


def CalculateSaveW(dataloader):
	"""Class rebalancing"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sigma = 5
	with open("tree.p", 'rb') as pickle_file:
		tree = pickle.load(pickle_file)

	# ab color distribution (now int, but it will get transformed into the correct shape 1D)
	p = 0
	for i, (X, _, Weights, ii) in enumerate(tqdm(dataloader)):
		# y [batch_size, 322, 224, 224]
		X, y = encode(X, Weights, ii, device)
		p += y.mean(axis=(0, 2, 3))

	p /= len(dataloader)

	# smooth with gaussian filter
	p = p.cpu().numpy()

	p_smooth = np.zeros_like(p)
	for i in range(322):
		weights = norm.pdf(tree.data, loc=tree.data[i], scale=sigma)
		weights = weights[:, 0]*weights[:, 1]
		weights = weights/weights.sum()
		p_smooth[i] = np.dot(p, weights)
	# mix with uniform
	w = 1 / p_smooth

	# normalize
	w = w / np.dot(w, p_smooth)
	np.save("AuC_w/p_sports_cars.npy", p)
	np.save("AuC_w/p_smooth_sports_cars.npy", p_smooth)
	np.save("AuC_w/W_sports_cars.npy", w)


def area_under_curve(ab_pred, ab_true, dataloader=None):
	ab_true_orig = ab_true

	# ab_pred, ab_true [images, W, H, 2]
	thresholds = np.arange(0, 151)
	ab_pred = np.repeat(ab_pred[:, :, :, :, None], thresholds.size, axis=4)
	ab_true = np.repeat(ab_true[:, :, :, :, None], thresholds.size, axis=4)

	l2_norms = np.linalg.norm(ab_pred - ab_true, axis=3, keepdims=True)
	correctly_identified = l2_norms <= thresholds

	if dataloader is None:
		accuracies = np.mean(correctly_identified, axis=(1, 2, 3))

		return np.trapz(accuracies, thresholds)[0] / (thresholds.size - 1)
	else:
		w = np.load("AuC_w/W_sports_cars.npy")

		weights = np.repeat(w[val_loader.dataset.tree.query(ab_true_orig)[1]][:, :, :, None, None], thresholds.size, axis=4)
		weights /= np.sum(weights, axis=(1, 2, 3), keepdims=True)

		accuracies = np.sum(weights * correctly_identified, axis=(1, 2, 3))

		return np.trapz(accuracies, thresholds)[0] / (thresholds.size - 1)



if __name__ == '__main__':
	configs = parse_configs()
	val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "sports_cars/val", "tree.p")

	if not os.path.exists("AuC_w/W_sports_cars.npy"):
		CalculateSaveW(val_loader)

	accuracies_gray = np.zeros(len(val_loader))
	accuracies_gray_rebal = np.zeros(len(val_loader))
	accuracies_random = np.zeros(len(val_loader))
	accuracies_random_rebal = np.zeros(len(val_loader))

	for i, (X, ab_true, Weights, ii) in enumerate(tqdm(val_loader, leave=False)):
		ab_true = ab_true.numpy().transpose([0, 2, 3, 1])
		ab_gray = to_gray(ab_true)
		ab_random = to_random(ab_true, val_loader)

		accuracies_gray[i] = area_under_curve(ab_gray, ab_true)
		accuracies_gray_rebal[i] = area_under_curve(ab_gray, ab_true, val_loader)
		accuracies_random[i] = area_under_curve(ab_random, ab_true)
		accuracies_random_rebal[i] = area_under_curve(ab_random, ab_true, val_loader)

	print("Baseline, gray (non-rebal) accuracy: " + str(np.mean(accuracies_gray)))
	print("Baseline, gray (rebal) accuracy: " + str(np.mean(accuracies_gray_rebal)))
	print("Baseline, random (non-rebal) accuracy: " + str(np.mean(accuracies_random)))
	print("Baseline, random (rebal) accuracy: " + str(np.mean(accuracies_random_rebal)))
