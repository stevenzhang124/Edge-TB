import torch
from torch import nn, optim
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
#from logger import Logger

# 定义超参数
batch_size = 128        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 20        # 遍历训练集的次数

# 数据类型转换，转换成numpy类型
#def to_np(x):
#    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='../../dataset/MNIST/', train=True, transform=transforms.ToTensor(), download=False)

test_dataset = datasets.MNIST(
    root='../../dataset/MNIST/', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        start_time = time.time()
        x = self.pool1(F.relu(self.conv1(x)))
        end_time = time.time()
        print("forward time Conv1: ", end_time-start_time)
        print("Output size Conv1: ", sizeof(x))

        start_time = time.time()
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        end_time = time.time()
        print("forward time Conv2: ", end_time-start_time)
        print("Output size Conv2: ", sizeof(x))

        start_time = time.time()
        x = F.relu(self.fc1(x))
        end_time = time.time()
        print("forward time FC1: ", end_time-start_time)
        print("Output size FC1: ", sizeof(x))

        start_time = time.time()
        x = F.relu(self.fc2(x))
        end_time = time.time()
        print("forward time FC2: ", end_time-start_time)
        print("Output size FC2: ", sizeof(x))

        start_time = time.time()
        x = self.fc3(x)
        end_time = time.time()
        print("forward time FC3: ", end_time-start_time)
        print("Output size FC3: ", sizeof(x))

        return x



# Instantiate the model and optimizer
model = LeNet()
use_gpu = False
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# "Freeze" the first two layers
for name, param in model.named_parameters():
    if 'conv1' in name or 'conv2' in name:
        param.requires_grad = False

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

        print("Forward time in batch {}".format(i))
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        start_time = time.time()
        loss.backward()
        end_time = time.time()

        print("Backward time for layers fc1, fc2, fc3 in batch {}: {} seconds".format(i, end_time - start_time))

        optimizer.step()

        running_loss += loss.item()

    print('Finish {} epoch, Loss: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset))))