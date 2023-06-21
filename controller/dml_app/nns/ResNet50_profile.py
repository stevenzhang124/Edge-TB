# ResNet-50 profile on FashionMNIST dataset
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
transform = transforms.Compose([
	transforms.Resize(224),  # ResNet-50 requires the input size of 224x224
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))  # FashionMNIST has only 1 channel
])

# Load training data
train_dataset = datasets.FashionMNIST(root='../../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load testing data
# test_dataset = datasets.FashionMNIST(root='../../dataset/', train=False, download=True, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a single block in a ResNet
class Block(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(Block, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels*4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = nn.Sequential()

		if stride != 1 or in_channels != out_channels * 4:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels*4)
			)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

# Define a layer in a ResNet (containing multiple blocks)
def make_layer(block, in_channels, out_channels, blocks, stride=1):
	layers = []
	layers.append(block(in_channels, out_channels, stride))
	in_channels = out_channels * 4

	for _ in range(1, blocks):
		layers.append(block(in_channels, out_channels))

	return nn.Sequential(*layers)

# Define the ResNet50 model
class ResNet50(nn.Module):
	def __init__(self, num_classes=10):  # FashionMNIST has 10 classes
		super(ResNet50, self).__init__()
		self.in_channels = 64
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = make_layer(Block, self.in_channels, 64, 3)
		self.layer2 = make_layer(Block, 256, 128, 4, stride=2)
		self.layer3 = make_layer(Block, 512, 256, 6, stride=2)
		self.layer4 = make_layer(Block, 1024, 512, 3, stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512*4, num_classes)

	def forward(self, x):
		start_time = time.time()
		x = self.conv1(x)
		end_time = time.time()
		print("forward time Conv1: ", end_time-start_time)
		print("Output size Conv1: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		end_time = time.time()
		print("forward time Bn1: ", end_time-start_time)
		print("Output size Bn1: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.layer1(x)
		end_time = time.time()
		print("forward time layer1: ", end_time-start_time)
		print("Output size layer1: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.layer2(x)
		end_time = time.time()
		print("forward time layer2: ", end_time-start_time)
		print("Output size layer2: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.layer3(x)
		end_time = time.time()
		print("forward time layer3: ", end_time-start_time)
		print("Output size layer3: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.layer4(x)
		end_time = time.time()
		print("forward time layer4: ", end_time-start_time)
		print("Output size layer4: ", x.element_size() * x.nelement())

		start_time = time.time()
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		end_time = time.time()
		print("forward time fc: ", end_time-start_time)
		print("Output size fc: ", x.element_size() * x.nelement())

		return x




def layer_sizes(model):
	sizes = {}
	layer_sizes = {}
	for name, param in chain(model.named_parameters(), model.named_buffers()):
		sizes[name] = param.numel() * 4

	for layer in layer_name:
		size = 0
		for name in sizes.keys():
			if layer in name:
				size += sizes[name]

		layer_sizes[layer] = size

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
	layer_name = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

	# profile the size of each layer
	# model = ResNet50()
	# print(model)
	# names = []
	# layer_name = []
	# for name,param in model.named_parameters():
	# 	names.append(name)
	# # print(names)
	# for name in names:
	# 	layer = name.split('.')[0]
	# 	if layer not in layer_name:
	# 		layer_name.append(layer)
	# print(layer_name)

	
	# layer_sizes = layer_sizes(model)

	# for name, size in layer_sizes.items():
	# 	print(f"Size of {name} in bytes: {size}")

	# print(f"Total size of the model (including buffers) in bytes: {sum(layer_sizes.values())}")

	# for i in range(len(layer_name)):
	i = 0
	print("For Layer ", i)
	freeze_layer = layer_name[0:i]
	# Instantiate the model and optimizer
	model = ResNet50()
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
	
