from model import Model
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
import subprocess
from numpy import random
import copy

# import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import cifar10, cifar100

transform = None

parser = argparse.ArgumentParser(description='Continuum learning')
parser.add_argument('--outfile', default='temp_0.1.csv', type=str, help='Output file name')
parser.add_argument('--matr', default='results/acc_matr.npz', help='Accuracy matrix file name')
parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time', type=int)
parser.add_argument('--init_lr', default=0.1, type=float, help='Init learning rate')

parser.add_argument('--num_epochs', default=40, type=int, help='Number of epochs')

parser.add_argument('--batch_size', default=64, type=int, help='Mini batch size')
args = parser.parse_args()

num_classes = args.num_classes
#all_train = cifar100(root='./data',
#                  train=True,
#                  classes=range(100),
#                  download=True,
#                  transform=None)
#mean_image = all_train.mean_image
#np.save("cifar_mean_image.npy", mean_image)
mean_image = np.load("cifar_mean_image.npy")

total_classes = 100
perm_id = np.random.permutation(total_classes)
all_classes = np.arange(total_classes)
for i in range(len(all_classes)):
	all_classes[i] = perm_id[all_classes[i]]

n_cl_temp = 0
num_iters = total_classes//num_classes
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
	if cl not in class_map:
		class_map[cl] = int(n_cl_temp)
		n_cl_temp += 1

print ("Class map:", class_map)

for cl, map_cl in class_map.items():
	map_reverse[map_cl] = int(cl)

print ("Map Reverse:", map_reverse)

print ("all_classes:", all_classes)
# else:
	# perm_id = np.arange(args.total_classes)

with open(args.outfile, 'w') as file:
	print("Classes, Train Accuracy, Test Accuracy", file=file)


	#shuffle classes
	# random.shuffle(all_classes)
	# class_map = {j: int(i) for i, j in enumerate(all_classes)}
	# map_reverse = {i: int(j) for i, j in enumerate(all_classes)}
	# print('Map reverse: ', map_reverse)
	# print('Class map: ', class_map)
	# print('All classes: ', all_classes)

	model = Model(1, class_map, args)
	model.cuda()
	acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
	for s in range(0, num_iters, num_classes):
		# Load Datasets
		print('Iteration: ', s)
		#print('Algo running: ', args.algo)
		print("Loading training examples for classes", all_classes[s: s+num_classes])
		train_set = cifar100(root='./data',
							 train=True,
							 classes=all_classes[s:s+num_classes],
							 download=True,
							 transform=transform,
							 mean_image=mean_image)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
												   shuffle=True, num_workers=12)

		test_set = cifar100(root='./data',
							 train=False,
							 classes=all_classes[:s+num_classes],
							 download=True,
							 transform=None,
							 mean_image=mean_image)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
												   shuffle=False, num_workers=12)

		# Update representation via BackProp
		model.update(train_set, class_map, args)
		model.eval()

		model.n_known = model.n_classes
		print ("%d, " % model.n_known, file=file, end="")
		print ("model classes : %d, " % model.n_known)

		total = 0.0
		correct = 0.0
		for indices, images, labels in train_loader:
			images = Variable(images).cuda()
			preds = model.classify(images)
			preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
			total += labels.size(0)
			correct += (preds == labels.numpy()).sum()

		# Train Accuracy
		print ('%.2f ,' % (100.0 * correct / total), file=file, end="")
		print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

		total = 0.0
		correct = 0.0
		for indices, images, labels in test_loader:
			images = Variable(images).cuda()
			preds = model.classify(images)
			preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
			total += labels.size(0)
			correct += (preds == labels.numpy()).sum()

		# Test Accuracy
		print ('%.2f' % (100.0 * correct / total), file=file)
		print ('Test Accuracy : %.2f' % (100.0 * correct / total))

		# Accuracy matrix
		for i in range(model.n_known):
			test_set = cifar100(root='./data',
							 train=False,
							 classes=all_classes[i*num_classes: (i+1)*num_classes],
							 download=True,
							 transform=None,
							 mean_image=mean_image)
			test_loader = torch.utils.data.DataLoader(test_set, batch_size=min(500, len(test_set)),
												   shuffle=False, num_workers=12)


			total = 0.0
			correct = 0.0
			for indices, images, labels in test_loader:
				images = Variable(images).cuda()
				preds = model.classify(images)
				preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
				total += labels.size(0)
				correct += (preds == labels.numpy()).sum()
			acc_matr[i, int(s/num_classes)] = (100 * correct / total)

		print ("Accuracy matrix", acc_matr[:int(s/num_classes + 1), :int(s/num_classes + 1)])

		model.train()
		githash = subprocess.check_output(['git', 'describe', '--always'])
		np.savez(args.matr, acc_matr=acc_matr, hyper_params = args, githash=githash)
