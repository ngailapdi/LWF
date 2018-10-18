from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import cv2
import time

class cifar10(CIFAR10):
	def __init__(self, root,
				 classes=range(10),
				 train=True,
				 transform=None,
				 target_transform=None,
				 download=False,
				 mean_image=None):
		super(cifar10, self).__init__(root,
									   train=train,
									   transform=transform,
									   target_transform=target_transform,
									   download=download)
		
		self.tensorTranform = transforms.ToTensor()
		self.train = train
		self.img_size = 224
		if mean_image is not None:
			mean_image = mean_image.transpose(1,2,0)
			self.mean_image = cv2.resize(mean_image, (self.img_size, self.img_size))
			self.mean_image = self.mean_image.transpose(2,0,1)

		# Select subset of classes
		if self.train:
			train_data = []
			train_labels = []

			for i in range(len(self.train_data)):
				if self.train_labels[i] in classes:
					curr_img = cv2.resize(self.train_data[i], (self.img_size, self.img_size))
					curr_img = curr_img.transpose(2,0,1)
					if mean_image is None:
						train_data.append(curr_img/255.)
					else:
						train_data.append(curr_img/255. - self.mean_image)

					train_labels.append(int(self.train_labels[i]))
					
			self.train_data = np.array(train_data, dtype = np.float32)
			self.train_labels = np.array(train_labels)

			
			if mean_image is None:
				self.mean_image = np.mean(self.train_data, axis=0)

		else:
			test_data = []
			test_labels = []

			for i in range(len(self.test_data)):
				if self.test_labels[i] in classes:
					curr_img = cv2.resize(self.test_data[i], (self.img_size, self.img_size))
					curr_img = curr_img.transpose(2,0,1)
					test_data.append(curr_img/255. - self.mean_image)
					test_labels.append(int(self.test_labels[i]))
					
			self.test_data = np.array(test_data, dtype = np.float32)
			self.test_labels = test_labels


	def __getitem__(self, index):
		if self.train:
			image = self.train_data[index]
			random_cropped = np.zeros(image.shape, dtype=np.float32)
			padded = np.pad(image,((0,0),(4,4),(4,4)),mode='constant')
			crops = np.random.random_integers(0,high=8,size=(1,2))
			# Cropping and possible flipping
			if (np.random.randint(2) > 0):
				random_cropped[:,:,:] = padded[:,crops[0,0]:(crops[0,0]+self.img_size),crops[0,1]:(crops[0,1]+self.img_size)]
			else:
				random_cropped[:,:,:] = padded[:,crops[0,0]:(crops[0,0]+self.img_size),crops[0,1]:(crops[0,1]+self.img_size)][:,:,::-1]
			image = torch.FloatTensor(random_cropped)
			target = self.train_labels[index]
		else:
			image, target = self.test_data[index], self.test_labels[index]

		image = torch.FloatTensor(image)
		
		return index, image, target

	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)
		

class cifar100(cifar10):
	base_folder = 'cifar-100-python'
	url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
	filename = "cifar-100-python.tar.gz"
	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
	train_list = [
		['train', '16019d7e3df5f24257cddd939b257f8d'],
	]
	test_list = [
		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
	]
