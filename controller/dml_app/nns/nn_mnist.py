# from torch.nn import Module
# from torch import nn


# class Model(Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(256, 120)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu4 = nn.ReLU()
#         self.fc3 = nn.Linear(84, 10)
#         self.relu5 = nn.ReLU()

#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.relu1(y)
#         y = self.pool1(y)
#         y = self.conv2(y)
#         y = self.relu2(y)
#         y = self.pool2(y)
#         y = y.view(y.shape[0], -1)
#         y = self.fc1(y)
#         y = self.relu3(y)
#         y = self.fc2(y)
#         y = self.relu4(y)
#         y = self.fc3(y)
#         y = self.relu5(y)
#         return y

# net = Model()



import torch
from torch import nn
import torch.nn.functional as F

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

# net = CNNMnist()


# the following code is for client-server splitting learning
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


# from tensorflow.keras import Input, layers, Model, losses


# class Mnist (object):
# 	def __init__ (self):
# 		self.input_shape = [-1, 28 * 28]  # -1 means no matter how much data

# 		inputs = Input (shape=tuple (self.input_shape) [1:])
# 		x = layers.Dense (512, activation='sigmoid') (inputs)
# 		x = layers.Dense (256, activation='sigmoid') (x)
# 		x = layers.Dense (128, activation='sigmoid') (x)
# 		outputs = layers.Dense (10) (x)

# 		self.model = Model (inputs, outputs)
# 		self.model.compile (optimizer='adam', loss=losses.SparseCategoricalCrossentropy (from_logits=True),
# 			metrics=['accuracy'])

# 		self.size = 4 * self.model.count_params ()  # 4 byte per np.float32


# nn = Mnist ()
