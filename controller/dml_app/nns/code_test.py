import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import random

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class LeNet_client_side(nn.Module):
    def __init__(self, client_layers):
        super(LeNet_client_side, self).__init__()
        self.client_layers = client_layers

        if self.client_layers == 1:
        	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        if self.client_layers == 2:
        	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        	self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        	self.conv2_drop = nn.Dropout2d()

        if self.client_layers == 3:
        	self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
	        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
	        self.conv2_drop = nn.Dropout2d()
	        self.fc1 = nn.Linear(320, 50)
        
        if self.client_layers == 4: 
	        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
	        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
	        self.conv2_drop = nn.Dropout2d()
	        self.fc1 = nn.Linear(320, 50)
	        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        if self.client_layers == 1:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            return x
        if self.client_layers == 2:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            return x
        if self.client_layers == 3:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return x
        if self.client_layers == 4:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

class LeNet_server_side(nn.Module):
    def __init__(self, client_layers):
        super(LeNet_server_side, self).__init__()
        self.client_layers = client_layers

        if self.client_layers == 1:
        	self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
	        self.conv2_drop = nn.Dropout2d()
	        self.fc1 = nn.Linear(320, 50)
	        self.fc2 = nn.Linear(50, 10)

        if self.client_layers == 2:
        	self.fc1 = nn.Linear(320, 50)
        	self.fc2 = nn.Linear(50, 10)
        
        if self.client_layers == 3:
        	self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        if self.client_layers == 1:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x
        if self.client_layers == 2:
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x
        if self.client_layers == 3:
            x = self.fc2(x)
            return x


class LeNet_full_model(nn.Module):
    def __init__(self):
        super(LeNet_full_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x



client_model = LeNet_client_side(2)
server_model = LeNet_server_side(2)
full_model = LeNet_full_model()

# client_model.eval()
# server_model.eval()
# full_model.eval()

initial_weight = full_model.state_dict()
client_weight = {k:v for k, v in initial_weight.items() if 'conv1' in k or 'conv2' in k}
client_model.load_state_dict(client_weight)
server_weight = {k:v for k, v in initial_weight.items() if 'fc1' in k or 'fc2' in k}
server_model.load_state_dict(server_weight)

print(client_weight)
# print(initial_weight)

batch_size = 1
x = torch.randn(batch_size, 1, 28, 28)
client_output = client_model(x)
print(client_model.state_dict())

server_output = server_model(client_output)

print("loss is ", server_output)
# print(client_weight)

full_model_loss = full_model(x)
print("full model loss is ", full_model_loss)
# print(initial_weight)





































##### check the output and intermediate parameters of the torch model ####
# class CNNMnist(nn.Module):
#     def __init__(self):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x




# class LeNet5(nn.Module):
    
#     def __init__(self):
#         super(LeNet5,self).__init__()
        
#         # 卷积层
#         self.conv_layer = nn.Sequential(
#             # input:torch.Size([batch_size, 3, 32, 32])
#             nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0),  # output:torch.Size([batch_size, 6, 28, 28])
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # output:torch.Size([batch_size, 6, 14, 14])
#             nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0), # output:torch.Size([batch_size, 16, 10, 10])
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # output:torch.Size([batch_size, 16, 5, 5])
#         )
#         # output:torch.Size([batch_size, 16, 5, 5])
        
#         # 全连接层
#         self.fullconn_layer = nn.Sequential(
#             # input:torch.Size([batch_size, 16*5*5])
#             nn.Linear(16*5*5,120),
#             nn.ReLU(),
#             # input:torch.Size([batch_size, 120])
#             nn.Linear(120,84),
#             nn.ReLU(),
#             # input:torch.Size([batch_size, 84])
#             nn.Linear(84,10),           
#         )
#         # output:torch.Size([10, 10])
        
#     def forward(self,x):
        
#         output = self.conv_layer(x)           # output:torch.Size([batch_size, 16, 5, 5])
#         # output = output.view(batch_size,-1)   # output:torch.Size([10, 16*5*5])
#         output = output.view(x.size(0),-1)
#         output = self.fullconn_layer(output)  # output:torch.Size([10, 10])
#         return output

# net = LeNet5()


# # net = CNNMnist()
# # check structure of LeNet
# print(net)
# print(type(net.state_dict()))
# print(net.state_dict())
# for name in net.state_dict():
# 	print(name)
# # print(net.state_dict()['0.bias'])
# # print(net.state_dict()['3.bias'])
# print(type(net.named_parameters()))
# for name,param in net.named_parameters():
# 	print(name, param)



# '''
# conv_layer.0.weight
# conv_layer.0.bias
# conv_layer.2.weight
# conv_layer.2.bias
# fullconn_layer.0.weight
# fullconn_layer.0.bias
# fullconn_layer.2.weight
# fullconn_layer.2.bias
# fullconn_layer.4.weight
# fullconn_layer.4.bias

# '''

# net_2 = CNNMnist()


# # net = CNNMnist()
# # check structure of LeNet
# print(net_2)
# print(type(net_2.state_dict()))
# print(net_2.state_dict())
# for name in net_2.state_dict():
# 	print(name)
# # print(net.state_dict()['0.bias'])
# # print(net.state_dict()['3.bias'])
# print(type(net_2.named_parameters()))
# for name,param in net_2.named_parameters():
# 	print(name, param)

# '''
# conv1.weight
# conv1.bias
# conv2.weight
# conv2.bias
# fc1.weight
# fc1.bias
# fc2.weight
# fc2.bias

# '''

