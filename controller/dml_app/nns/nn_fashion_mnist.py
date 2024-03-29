import torch
from torch import nn
import time

#construct neural network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#three methods to define a neural network 1.forward function 2.Sequential 3. ModuleList
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):          #for execute the neural network computation
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
net = LeNet()
# print(net)


# from tensorflow.keras import Input, layers, Model, losses


# class FashionMnist (object):
# 	def __init__ (self):
# 		self.input_shape = [-1, 28, 28, 1]  # -1 means no matter how much data

# 		# Block 1
# 		inputs = Input (shape=tuple (self.input_shape) [1:])
# 		x = layers.Conv2D (32, (3, 3), padding='same', activation='relu') (inputs)
# 		x = layers.BatchNormalization () (x)
# 		x = layers.Conv2D (32, (3, 3), padding='same', activation='relu') (x)
# 		x = layers.BatchNormalization () (x)
# 		x = layers.MaxPooling2D ((2, 2)) (x)
# 		# Block 2
# 		x = layers.Conv2D (64, (3, 3), padding='same', activation='relu') (x)
# 		x = layers.BatchNormalization () (x)
# 		x = layers.MaxPooling2D ((2, 2)) (x)
# 		# Classification block
# 		x = layers.Flatten () (x)
# 		x = layers.Dense (64, activation='relu') (x)
# 		outputs = layers.Dense (10) (x)

# 		self.model = Model (inputs, outputs)
# 		self.model.compile (optimizer='adam', loss=losses.SparseCategoricalCrossentropy (from_logits=True),
# 			metrics=['accuracy'])

# 		self.size = 4 * self.model.count_params ()  # 4 byte per np.float32


# nn = FashionMnist ()

