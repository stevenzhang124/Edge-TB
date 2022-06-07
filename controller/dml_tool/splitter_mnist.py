"""
this file loads mnist from keras and cut the data into several pieces in sequence.
"""
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

if __name__ == '__main__':
	# load dataset
	trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	dataset_train = datasets.MNIST('../dataset/MNIST/', train=True, download=True, transform=trans_mnist)
	dataset_test = datasets.MNIST('../dataset/MNIST/', train=False, download=True, transform=trans_mnist)

	# print(len(dataset_train))



"""
this file loads mnist from keras and cut the data into several pieces in sequence.

import os

import numpy as np
from tensorflow.keras.datasets import mnist

from splitter_utils import split_data, save_data

if __name__ == '__main__':
	# all configurable parameters.
	train_batch = 100
	train_drop_last = True
	test_batch = 100
	test_drop_last = True
	one_hot = False
	# all configurable parameters.

	# load data from keras.
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data ()
	# normalize.
	train_images, test_images = train_images / 255.0, test_images / 255.0
	# convert to float32.
	train_images, test_images = train_images.astype (np.float32), test_images.astype (np.float32)

	# save in here.
	path = os.path.abspath (os.path.join (os.path.dirname (__file__), '../dataset/MNIST'))
	train_path = os.path.join (path, 'train_data')
	test_path = os.path.join (path, 'test_data')

	# split and save.
	train_images_loader, train_labels_loader = split_data (train_images, train_labels, train_batch, train_drop_last)
	save_data (train_images_loader, train_labels_loader, train_path, one_hot)

	test_images_loader, test_labels_loader = split_data (test_images, test_labels, test_batch, test_drop_last)
	save_data (test_images_loader, test_labels_loader, test_path, one_hot)
"""