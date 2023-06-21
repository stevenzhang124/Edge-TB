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
        print("Output size Conv1: ", x.element_size() * x.nelement())

        start_time = time.time()
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        end_time = time.time()
        print("forward time Conv2: ", end_time-start_time)
        print("Output size Conv2: ", x.element_size() * x.nelement())

        start_time = time.time()
        x = F.relu(self.fc1(x))
        end_time = time.time()
        print("forward time FC1: ", end_time-start_time)
        print("Output size FC1: ", x.element_size() * x.nelement())

        start_time = time.time()
        x = F.relu(self.fc2(x))
        end_time = time.time()
        print("forward time FC2: ", end_time-start_time)
        print("Output size FC2: ", x.element_size() * x.nelement())

        start_time = time.time()
        x = self.fc3(x)
        end_time = time.time()
        print("forward time FC3: ", end_time-start_time)
        print("Output size FC3: ", x.element_size() * x.nelement())

        return x


# layer_name = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
def layer_sizes(model):
    sizes = {}
    layer_sizes = {}
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        sizes[name] = param.numel() * 4

    for layer in layer_name:
        layer_sizes[layer] = sizes[layer+'.weight'] + sizes[layer+'.bias']

    return layer_sizes

# lenet = LeNet()

# layer_sizes = layer_sizes(lenet)

# for name, size in layer_sizes.items():
#     print(f"Size of {name} in bytes: {size}")

# print(f"Total size of the model (including buffers) in bytes: {sum(layer_sizes.values())}")


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

            print("Backward time for layers {}: {}".format(' '.join(layer_name[index:]), end_time - start_time))

            update_start = time.time()
            optimizer.step()
            update_end = time.time()
            print("Weights update time for layers {}: {}".format(' '.join(layer_name[index:]), update_end - update_start))


            running_loss += loss.item()

            batch_end = time.time()
            print("Batch {} consumes: {}".format(i, batch_end - batch_start))

            # for small test
            if index != 0:
                if i == 5:
                    break

        print('Finish {} epoch, Loss: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset))))

if __name__ == '__main__':
    layer_name = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']

    # profile the size of each layer
    model = LeNet()

    layer_sizes = layer_sizes(model)

    for name, size in layer_sizes.items():
        print(f"Size of {name} in bytes: {size}")

    print(f"Total size of the model (including buffers) in bytes: {sum(layer_sizes.values())}")



    for i in range(len(layer_name)):
        print("For Layer ", i)
        freeze_layer = layer_name[0:i]
        # Instantiate the model and optimizer
        model = LeNet()
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

        train(model, use_gpu, i, layer_name, criterion, optimizer)



