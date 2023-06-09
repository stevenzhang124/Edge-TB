# VGG profile on MNIST dataset
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
trainset = torchvision.datasets.CIFAR10(root='../../dataset/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Load testing data
testset = torchvision.datasets.CIFAR10(root='../../dataset/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)




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
		x = self.relu1_1(self.conv1_1(x))
		x = self.relu1_2(self.conv1_2(x))
		x = self.maxpool1(x)

		x = self.relu2_1(self.conv2_1(x))
		x = self.relu2_2(self.conv2_2(x))
		x = self.maxpool2(x)

		x = self.relu3_1(self.conv3_1(x))
		x = self.relu3_2(self.conv3_2(x))
		x = self.relu3_3(self.conv3_3(x))
		x = self.maxpool3(x)

		x = self.relu4_1(self.conv4_1(x))
		x = self.relu4_2(self.conv4_2(x))
		x = self.relu4_3(self.conv4_3(x))
		x = self.maxpool4(x)

		x = self.relu5_1(self.conv5_1(x))
		x = self.relu5_2(self.conv5_2(x))
		x = self.relu5_3(self.conv5_3(x))
		x = self.maxpool5(x)

		x = x.view(x.size(0), -1)  # Flatten the tensor

		x = self.relu6(self.fc6(x))
		x = self.dropout6(x)

		x = self.relu7(self.fc7(x))
		x = self.dropout7(x)

		x = self.fc8(x)

		return x

def train(model, use_gpu, criterion, optimizer):
	# Training loop
	for epoch in range(num_epoches):
		print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
		print('*' * 10)
		running_loss = 0.0

		for i, batch in enumerate(train_loader, 1):
			inputs, labels = batch
			if use_gpu:
				img = img.cuda()
				label = label.cuda()
			
			optimizer.zero_grad()

			print("For batch {}".format(i))
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			start_time = time.time()
			loss.backward()
			end_time = time.time()

			# print("Backward time for layers {}: {}".format(' '.join(layer_name[index+1:]), end_time - start_time))

			optimizer.step()

			running_loss += loss.item()

		print('Finish {} epoch, Loss: {:.6f}'.format(
			epoch + 1, running_loss / (len(train_dataset))))

if __name__ == '__main__':
	# layer_name = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
	# for i in range(len(layer_name)-1):
	#     print("For Layer ", i+1)
	#     freeze_layer = layer_name[0:i+1]
	#     # Instantiate the model and optimizer
	#     model = LeNet()
	#     use_gpu = False
	#     if use_gpu:
	#         model = model.cuda()

	#     for name, param in model.named_parameters():
	#         for layer in freeze_layer:
	#             if layer in name:
	#                 param.requires_grad = False

	#     # start profile

	#     criterion = nn.CrossEntropyLoss()
	#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	model = VGG16()
	use_gpu = False
	if use_gpu:
		model = model.cuda()

	# start profile

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	train(model, use_gpu, criterion, optimizer)

