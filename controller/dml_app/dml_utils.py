import io
import time
import collections
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import worker_utils
import requests

write = io.BytesIO ()
cur_index = 0
loss_fn = CrossEntropyLoss()


# def load_data (path, start_index, _len, input_shape):
# 	x_list = []
# 	y_list = []
# 	for i in range (_len):
# 		x_list.append (np.load (path + '/images_' + str (start_index + i) + '.npy')
# 			.reshape (input_shape))
# 		y_list.append (np.load (path + '/labels_' + str (start_index + i) + '.npy'))
# 	images = np.concatenate (tuple (x_list))
# 	labels = np.concatenate (tuple (y_list))
# 	return images, labels

class DatasetSplit(Dataset):
	def __init__(self, dataset, idxs):
		self.dataset = dataset
		self.idxs = list(idxs)

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, item):
		image, label = self.dataset[self.idxs[item]]
		return image, label

def load_data (path, start_index, _len, batch_size, train=True):
	trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	if train:
		dataset = datasets.MNIST(path, train=True, download=False, transform=trans_mnist)
		idxs = [i for i in range(start_index, start_index+_len)]
		local_dataset =  DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
		return local_dataset
	else:
		dataset = datasets.MNIST(path, train=False, download=False, transform=trans_mnist)
		return dataset

	



def train_all (model, images, labels, epochs, batch_size):
	h = model.fit (images, labels, epochs=epochs, batch_size=batch_size)
	return h.history ['loss']


# def train (model, images, labels, epochs, batch_size, train_len):
# 	global cur_index
# 	cur_images = images [cur_index * 500: (cur_index + 1) * 500]
# 	cur_labels = labels [cur_index * 500: (cur_index + 1) * 500]
# 	cur_index += 1
# 	if cur_index == train_len:
# 		cur_index = 0
# 	h = model.fit (cur_images, cur_labels, epochs=epochs, batch_size=batch_size)
# 	return h.history ['loss']

def train (net, local_dataset, epochs, batch_size):
	net.train()
	# train and update
	optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

	epoch_loss = []
	for iter in range(epochs):
		batch_loss = []
		for batch_idx, (images, labels) in enumerate(local_dataset):
			# images, labels = images.to(self.args.device), labels.to(self.args.device)
			optimizer.zero_grad()
			log_probs = net(images.float())
			loss = loss_fn(log_probs, labels.long())
			loss.backward()
			optimizer.step()
			if batch_idx % 10 == 0:
				print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						iter, batch_idx * len(images), len(local_dataset.dataset),
							   100. * batch_idx / len(local_dataset), loss.item()))

			batch_loss.append(loss.item())
		epoch_loss.append(sum(batch_loss)/len(batch_loss))
	return epoch_loss


def test (model, test_data, batch_size):
	acc, loss = test_img(model, test_data, batch_size)
	# loss, acc = model.test_on_batch (images, labels)
	return loss, acc

def test_img(net_g, datatest, batch_size):
	net_g.eval()
	# testing
	test_loss = 0
	correct = 0
	data_loader = DataLoader(datatest, batch_size=batch_size)
	l = len(data_loader)
	for idx, (data, target) in enumerate(data_loader):
		# if args.gpu != -1:
		# 	data, target = data.cuda(), target.cuda()
		log_probs = net_g(data)
		# sum up batch loss
		test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
		# get the index of the max log-probability
		y_pred = log_probs.data.max(1, keepdim=True)[1]
		correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

	test_loss /= len(data_loader.dataset)
	accuracy = 100.00 * correct / len(data_loader.dataset)
	# accuracy = correct / len(data_loader.dataset)
	# if args.verbose:
	# print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
	# 		test_loss, correct, len(data_loader.dataset), accuracy))
	return accuracy, test_loss


def client_train (net, local_dataset, conf):
	net.train()
	# train and update
	optimizer_client = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
	
	epoch_loss = []
	for iter in range(conf['epoch']):
		batch_loss = []
		for batch_idx, (images, labels) in enumerate(local_dataset):
			optimizer_client.zero_grad()
			#-------------  forward prop--------------------------------
			activation = net(images.float())
			
			#------------ send activations to server and receive gradients
			activations = (activation, labels)
			
			print("sending activations to aggregator")

			received = send_activation (activations, '/get_activation', conf ['father_node'], conf ['connect'], conf ['client_layers'])
			
			# de-serizelize
			loss = received.json()['loss']
			client_gradients = torch.Tensor(received.json()['client_gradients'])

			#-------------- backward prop
			activation.backward(client_gradients)
			optimizer_client.step()
			if batch_idx % 10 == 0:
				print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						iter, batch_idx * len(images), len(local_dataset.dataset),
							   100. * batch_idx / len(local_dataset), loss))

			batch_loss.append(loss)
		epoch_loss.append(sum(batch_loss)/len(batch_loss))

	return epoch_loss, net.state_dict()


def send_activation(activation, path, node_list, connect, clients_layers, forward=None, layer=-1):
	self = 0
	# torch.save(weights, '../dml_file/local_model.pkl')
	# np.save (write, weights)
	# write.seek (0)
	for node in node_list:
		if node == 'self':
			self = 1
			continue
		if node in connect:
			addr = 'http://' + connect [node] + path
			print(addr)
			data = {"activation": activation[0].tolist(), "labels": activation[1].view(-1, 1).tolist(), "client_layers": str(clients_layers)}
			res = requests.post(addr, json=data)
		elif forward:
			addr = 'http://' + connect [node] + path
			print(addr)
			data = {"activation": activation[0].tolist(), "labels": activation[1].view(-1, 1).tolist(), "client_layers": str(clients_layers)}
			res = requests.post(addr, json=data)
		else:
			Exception ('has not connect to ' + node)
		# write.seek (0)
	# write.truncate ()
	return res

# continue the forward and backward 
def server_split_train (net, activation, labels):
	net.train()
	optimizer_server = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

	#train and update
	optimizer_server.zero_grad()

	#------------------ forward prop ------------------------
	log_probs = net(activation)
	loss = loss_fn(log_probs, labels.long())
	loss.backward()
	client_gradients = activation.grad.clone().detach()
	optimizer_server.step()

	return loss.item(), client_gradients.tolist()


def parse_weights (weights):
	# w = np.load (weights, allow_pickle=True)
	# print('here')
	return w


# only store the weights at received_weights [0]
# and accumulate as soon as new weights are received to save space :-)
def store_weights (received_weights, new_weights, received_count):
	sum_weights = collections.OrderedDict()
	if received_count == 1:
		received_weights.append (new_weights)
		print("received one")
	else:
		for key in list(received_weights [0].keys()):
			sum_weights[key] = received_weights[0][key] + new_weights[key]
		received_weights.pop()
		received_weights.append(sum_weights) 
		print("received one")


def avg_weights (received_weights, received_count):
	for key in list(received_weights [0].keys()):
		received_weights [0][key] = received_weights [0][key] / received_count
	return received_weights [0]


def assign_weights (model, weights):
	# model.set_weights (weights)
	model.load_state_dict(weights)


def send_weights (weights, path, node_list, connect, forward=None, layer=-1):
	self = 0
	torch.save(weights, '../dml_file/local_model.pkl')
	# np.save (write, weights)
	# write.seek (0)
	for node in node_list:
		if node == 'self':
			self = 1
			continue
		if node in connect:
			addr = connect [node]
			data = {'path': path, 'layer': str (layer)}
			# send_weights_helper (write, data, addr, is_forward=False)
			send_weights_helper (weights, data, addr, is_forward=False)
		elif forward:
			addr = forward [node]
			data = {'node': node, 'path': path, 'layer': str (layer)}
			# send_weights_helper (write, data, addr, is_forward=True)
			send_weights_helper (weights, data, addr, is_forward=True)
		else:
			Exception ('has not connect to ' + node)
		# write.seek (0)
	# write.truncate ()
	return self


def send_weights_helper (weights, data, addr, is_forward):
	s = time.time ()
	if not is_forward:
		worker_utils.send_data ('POST', data ['path'], addr, data=data, files={'weights': open('../dml_file/local_model.pkl', 'rb')})
	else:
		worker_utils.log ('need ' + addr + ' to forward to ' + data ['node'] + data ['path'])
		worker_utils.send_data ('POST', '/forward', addr, data=data, files={'weights': open('../dml_file/local_model.pkl', 'rb')})
	e = time.time ()
	worker_utils.log ('send weights to ' + addr + ', cost=' + str (e - s))


def send_client_weights (weights, path, node_list, connect, clients_layers, forward=None, layer=-1):
	self = 0
	torch.save(weights, '../dml_file/client_weights.pkl')
	# np.save (write, weights)
	# write.seek (0)
	for node in node_list:
		if node == 'self':
			self = 1
			continue
		if node in connect:
			addr = 'http://' + connect [node] + '/' + path
			data = {'client_layers': str(client_layers)}
			res = requests.post(addr, json=data, files={'weights': open(client_weight_file, 'rb')})
		elif forward:
			addr = 'http://' + connect [node] + '/' + path
			data = {'client_layers': str(client_layers)}
			res = requests.post(addr, json=json.dumps(data), files={'weights': open(client_weight_file, 'rb')})
		else:
			Exception ('has not connect to ' + node)
		# write.seek (0)
	# write.truncate ()
	return self


def send_weights_split (weights, path, node_list, connect, clients_layers, forward=None, layer=-1):
	self = 0
	# torch.save(weights, '../dml_file/local_model.pkl')
	# np.save (write, weights)
	# write.seek (0)
	for node in node_list:
		if node == 'self':
			self = 1
			continue
		if node in connect:
			addr = connect [node]
			client_layer = clients_layers[node]
			client_weight_file = get_client_weights(weights, node, client_layer)
			data = {'path': path, 'layer': str (layer)}
			send_weights_split_helper (client_weight_file, data, addr, is_forward=False)
		elif forward:
			addr = forward [node]
			client_layers = int(clients_layers[node])
			client_weight_file = get_client_weights(weights, node, client_layers)
			data = {'path': path, 'layer': str (layer)}
			send_weights_split_helper (client_weight_file, data, addr, is_forward=True)
		else:
			Exception ('has not connect to ' + node)
		# write.seek (0)
	# write.truncate ()
	return self

def get_client_weights(weights, node, client_layer):
	if client_layer == 1:
		cleint_weight = {k:v for k, v in weights.items() if 'conv1' in k}

	if client_layer == 2:
		cleint_weight = {k:v for k, v in weights.items() if 'conv1' in k or 'conv2' in k}

	if client_layer == 3:
		cleint_weight = {k:v for k, v in weights.items() if 'conv1' in k or 'conv2' in k or 'fc1' in k}

	if client_layer == 4:
		cleint_weight = weights

	client_weight_file_addr = '../dml_file/' + node + '_local_model.pkl'
	torch.save(cleint_weight, client_weight_file_addr)

	return client_weight_file_addr


def send_weights_split_helper (client_weight_file, data, addr, is_forward):
	s = time.time ()
	if not is_forward:
		worker_utils.send_data ('POST', data ['path'], addr, data=data, files={'weights': open(client_weight_file, 'rb')})
	else:
		worker_utils.log ('need ' + addr + ' to forward to ' + data ['node'] + data ['path'])
		worker_utils.send_data ('POST', '/forward', addr, data=data, files={'weights': open(client_weight_file, 'rb')})
	e = time.time ()
	worker_utils.log ('send weights to ' + addr + ', cost=' + str (e - s))



def random_selection (node_list, number):
	return np.random.choice (node_list, number, replace=False)


def log_loss (loss, _round):
	"""
	we left a comma at the end for easy positioning and extending.
	this message can be parse by controller/ctl_utils.py, parse_log ().
	"""
	message = 'Train: loss={}, round={},'.format (loss, _round)
	worker_utils.log (message)
	return message


def log_acc (acc, _round, layer=-1):
	"""
	we left a comma at the end for easy positioning and extending.
	this message can be parsed by controller/ctl_utils.py, parse_log ().
	"""
	if layer != -1:
		message = 'Aggregate: accuracy={}, round={}, layer={},'.format (acc, _round, layer)
	else:
		message = 'Aggregate: accuracy={}, round={},'.format (acc, _round)
	worker_utils.log (message)
	return message

print("hello world")