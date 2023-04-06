# import torch
# from torch import nn
# import torch.nn.functional as F

# import requests
# import json

# class LeNet_client_side(nn.Module):
#     def __init__(self, client_layers):
#         super(LeNet_client_side, self).__init__()
#         self.client_layers = client_layers

#         if self.client_layers == 1:
#         	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

#         if self.client_layers == 2:
#         	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         	self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         	self.conv2_drop = nn.Dropout2d()

#         if self.client_layers == 3:
#         	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
# 	        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
# 	        self.conv2_drop = nn.Dropout2d()
# 	        self.fc1 = nn.Linear(320, 50)
        
#         if self.client_layers == 4: 
# 	        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
# 	        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
# 	        self.conv2_drop = nn.Dropout2d()
# 	        self.fc1 = nn.Linear(320, 50)
# 	        self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         if self.client_layers == 1:
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             return x
#         if self.client_layers == 2:
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#             return x
#         if self.client_layers == 3:
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             return x
#         if self.client_layers == 4:
#             x = F.relu(F.max_pool2d(self.conv1(x), 2))
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             x = self.fc2(x)
#             return x

# client_model = LeNet_client_side(2)
# client_model.eval()
# batch_size = 1
# x = torch.randn(batch_size, 1, 28, 28)
# client_output = client_model(x)
# print(type(client_output))

# def send_activation(activation):
#     addr = "http://localhost:5001/get_activation"
#     # print(activation)
#     # activation_np = activation.detach().numpy()
#     # data = {"data": activation_np.tolist()}
#     # data_json = json.dumps(data)

#     test_data = {'path': 'path', 'client_layers': 2}
#     res = requests.post(addr, data=test_data)
#     return res.text

# send_activation(client_output)


import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import requests

# Define the Model1 class
class Model1(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = x.view(-1, 16*4*4)
        return x

model1 = Model1()
# Define the IP address and port of the Model2 server
model2_url = 'http://localhost:5001/'

# Load the MNIST dataset
mnist_trainset = datasets.MNIST(root='../../dataset', train=True, download=True, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)

# Define a criterion and an optimizer for Model1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)

# Train Model1 on the MNIST dataset
model1 = Model1()
for i, (images, labels) in enumerate(data_loader):
    optimizer.zero_grad()
    activations = model1(images)
    labels = labels.view(-1, 1)
    response = requests.post(model2_url, json={"activations": activations.tolist(), "labels": labels.tolist()})
    gradients = torch.Tensor(response.json()["gradients"])
    loss = response.json()["loss"]
    activations.backward(gradients)
    optimizer.step()
    if (i+1) % 10 == 0:
        print("Step [{}/{}], Loss: {:.4f}".format(i+1, len(data_loader), loss))
