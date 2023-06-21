# ResNet-101 profile on CIFAR-10 dataset
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


# Define the ResNet Block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# Define the ResNet Layer
def make_layer(block, num_residual_blocks, in_channels, out_channels, stride):
    identity_downsample = None
    layers = []

    # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels change, we need to adapt the Identity (skip connection) so it will be able to be added
    # to the layer that's ahead
    if stride != 1 or in_channels != out_channels * 4:
        identity_downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                                             nn.BatchNorm2d(out_channels * 4))

    layers.append(block(in_channels, out_channels, identity_downsample, stride))

    # The expansion size is always 4 for ResNet 50,101,152
    in_channels = out_channels * 4

    for i in range(num_residual_blocks - 1):
        layers.append(block(in_channels, out_channels))  # 256 will be mapped to 64, then 64*4 (256 again)...

    return nn.Sequential(*layers)

# Define the ResNet101
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = make_layer(block, layers[0], self.in_channels, out_channels=64, stride=1)
        self.in_channels = 64 * 4 # Update in_channels to prepare for next layer
        self.layer2 = make_layer(block, layers[1], self.in_channels, out_channels=128, stride=2)
        self.in_channels = 128 * 4 # Update in_channels to prepare for next layer
        self.layer3 = make_layer(block, layers[2], self.in_channels, out_channels=256, stride=2)
        self.in_channels = 256 * 4 # Update in_channels to prepare for next layer
        self.layer4 = make_layer(block, layers[3], self.in_channels, out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

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
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        end_time = time.time()
		print("forward time fc: ", end_time-start_time)
		print("Output size fc: ", x.element_size() * x.nelement())

        return x

def ResNet101(img_channel=3, num_classes=10):
    return ResNet101(Block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=10):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


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
	model = ResNet101()
	# print(model)
	# names = []
	# layer_name = []
	# for name,param in model.named_parameters():
	# 	names.append(name)
	# print(names)
	# for name in names:
	# 	layer = name.split('.')[0]
	# 	if layer not in layer_name:
	# 		layer_name.append(layer)
	# print(layer_name)

	
	layer_sizes = layer_sizes(model)

	for name, size in layer_sizes.items():
		print(f"Size of {name} in bytes: {size}")

	print(f"Total size of the model (including buffers) in bytes: {sum(layer_sizes.values())}")

	for i in range(len(layer_name)):
		print("For Layer ", i)
		freeze_layer = layer_name[0:i]
		# Instantiate the model and optimizer
		model = ResNet101()
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
	
