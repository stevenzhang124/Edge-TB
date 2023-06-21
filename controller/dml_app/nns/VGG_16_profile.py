# VGG profile on cifar-10 dataset
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
# import sys
#from logger import Logger
from itertools import chain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true',
                        help='use gpu or not')
args = parser.parse_args()

# 定义超参数
batch_size = 128        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 1        # 遍历训练集的次数

# 数据类型转换，转换成numpy类型
#def to_np(x):
#    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
# Transform data
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load training data
train_dataset = datasets.CIFAR10(root='../../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load testing data
test_dataset = datasets.CIFAR10(root='../../dataset/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




class VGG16(nn.Module):
	def __init__(self):
		super(VGG16, self).__init__()

		# VGG16 architecture
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu3_3 = nn.ReLU(inplace=True)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu5_3 = nn.ReLU(inplace=True)
		self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fc6 = nn.Linear(512, 4096)
		self.relu6 = nn.ReLU(inplace=True)
		self.dropout6 = nn.Dropout()

		self.fc7 = nn.Linear(4096, 4096)
		self.relu7 = nn.ReLU(inplace=True)
		self.dropout7 = nn.Dropout()

		self.fc8 = nn.Linear(4096, 10)

	def forward(self, x):
		start_time = time.time()
		x = self.relu1_1(self.conv1_1(x))
		end_time = time.time()
		print("forward time Conv1_1: ", end_time-start_time)
		print("Output size Conv1_1: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu1_2(self.conv1_2(x))
		x = self.maxpool1(x)
		end_time = time.time()
		print("forward time Conv1_2: ", end_time-start_time)
		print("Output size Conv1_2: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.relu2_1(self.conv2_1(x))
		end_time = time.time()
		print("forward time Conv2_1: ", end_time-start_time)
		print("Output size Conv2_1: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu2_2(self.conv2_2(x))
		x = self.maxpool2(x)
		end_time = time.time()
		print("forward time Conv2_2: ", end_time-start_time)
		print("Output size Conv2_2: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.relu3_1(self.conv3_1(x))
		end_time = time.time()
		print("forward time Conv3_1: ", end_time-start_time)
		print("Output size Conv3_1: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu3_2(self.conv3_2(x))
		end_time = time.time()
		print("forward time Conv3_2: ", end_time-start_time)
		print("Output size Conv3_2: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu3_3(self.conv3_3(x))
		x = self.maxpool3(x)
		end_time = time.time()
		print("forward time Conv3_3: ", end_time-start_time)
		print("Output size Conv3_3: ", x.element_size() * x.nelement())
		
		start_time = time.time()
		x = self.relu4_1(self.conv4_1(x))
		end_time = time.time()
		print("forward time Conv4_1: ", end_time-start_time)
		print("Output size Conv4_1: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu4_2(self.conv4_2(x))
		end_time = time.time()
		print("forward time Conv4_2: ", end_time-start_time)
		print("Output size Conv4_2: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu4_3(self.conv4_3(x))
		x = self.maxpool4(x)
		end_time = time.time()
		print("forward time Conv4_3: ", end_time-start_time)
		print("Output size Conv4_3: ", x.element_size() * x.nelement())
		
		start_time = time.time()
		x = self.relu5_1(self.conv5_1(x))
		end_time = time.time()
		print("forward time Conv5_1: ", end_time-start_time)
		print("Output size Conv5_1: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu5_2(self.conv5_2(x))
		end_time = time.time()
		print("forward time Conv5_2: ", end_time-start_time)
		print("Output size Conv5_2: ", x.element_size() * x.nelement())
		start_time = time.time()
		x = self.relu5_3(self.conv5_3(x))
		x = self.maxpool5(x)
		end_time = time.time()
		print("forward time Conv5_3: ", end_time-start_time)
		print("Output size Conv5_3: ", x.element_size() * x.nelement())
		
		start_time = time.time()
		x = x.view(x.size(0), -1)  # Flatten the tensor

		x = self.relu6(self.fc6(x))
		x = self.dropout6(x)
		end_time = time.time()
		print("forward time fc6: ", end_time-start_time)
		print("Output size fc6: ", x.element_size() * x.nelement())
		
		start_time = time.time()
		x = self.relu7(self.fc7(x))
		x = self.dropout7(x)
		end_time = time.time()
		print("forward time fc7: ", end_time-start_time)
		print("Output size fc7: ", x.element_size() * x.nelement())
		
		start_time = time.time()
		x = self.fc8(x)
		end_time = time.time()
		print("forward time fc8: ", end_time-start_time)
		print("Output size fc8: ", x.element_size() * x.nelement())


		return x


def layer_sizes(model):
	sizes = {}
	layer_sizes = {}
	for name, param in chain(model.named_parameters(), model.named_buffers()):
		sizes[name] = param.numel() * 4

	for layer in layer_name:
		layer_sizes[layer] = sizes[layer+'.weight'] + sizes[layer+'.bias']

	return layer_sizes

def train(model, use_gpu, index, layer_name, criterion, optimizer):
	# Training loop
	for epoch in range(num_epoches):
		print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
		print('*' * 10)
		running_loss = 0.0

		for i, batch in enumerate(train_loader, 1):
			batch_start = time.time()

			inputs, labels = batch
			if use_gpu:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			optimizer.zero_grad()

			print("For batch {}".format(i))
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			start_time = time.time()
			loss.backward()
			end_time = time.time()

			print("Backward time for layers {}: {}".format(' '.join(layer_name[index+1:]), end_time - start_time))

			update_start = time.time()
			optimizer.step()
			update_end = time.time()
			print("Weights update time for layers {}: {}".format(' '.join(layer_name[index:]), update_end - update_start))

			running_loss += loss.item()

			batch_end = time.time()
			print("Batch {} consumes: {}".format(i, batch_end - batch_start))

			# for small test
			if index == 0:
				if i == 10:
					break
			else:
				if i == 5:
					break

		print('Finish {} epoch, Loss: {:.6f}'.format(
			epoch + 1, running_loss / (len(train_dataset))))

if __name__ == '__main__':
	layer_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']

	# profile the size of each layer
	model = VGG16()
	# print(model)
	# for name,param in model.named_parameters():
	# 	print(name)

	
	layer_sizes = layer_sizes(model)

	for name, size in layer_sizes.items():
		print(f"Size of {name} in bytes: {size}")

	print(f"Total size of the model (including buffers) in bytes: {sum(layer_sizes.values())}")

	for i in range(len(layer_name)):
		print("For Layer ", i)
		freeze_layer = layer_name[0:i]
		# Instantiate the model and optimizer
		model = VGG16()
		if args.gpu:
			use_gpu = True
		else:
			use_gpu = False

		if use_gpu:
			model = model.cuda()

		for name, param in model.named_parameters():
			for layer in freeze_layer:
				if layer in name:
					param.requires_grad = False

		# start profile

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# model = VGG16()
	# use_gpu = False
	# if use_gpu:
	# 	model = model.cuda()

	# # start profile

	# criterion = nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		train(model, use_gpu, i, layer_name, criterion, optimizer)
	
