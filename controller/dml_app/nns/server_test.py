# import torch
# from torch import nn
# import torch.nn.functional as F

# from flask import Flask, request
# import json
# import numpy as np


# class LeNet_server_side(nn.Module):
#     def __init__(self, client_layers):
#         super(LeNet_server_side, self).__init__()
#         self.client_layers = client_layers

#         if self.client_layers == 1:
#         	self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
# 	        self.conv2_drop = nn.Dropout2d()
# 	        self.fc1 = nn.Linear(320, 50)
# 	        self.fc2 = nn.Linear(50, 10)

#         if self.client_layers == 2:
#         	self.fc1 = nn.Linear(320, 50)
#         	self.fc2 = nn.Linear(50, 10)
        
#         if self.client_layers == 3:
#         	self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         if self.client_layers == 1:
#             x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#             x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             x = self.fc2(x)
#             return x
#         if self.client_layers == 2:
#             x = F.relu(self.fc1(x))
#             x = F.dropout(x, training=self.training)
#             x = self.fc2(x)
#             return x
#         if self.client_layers == 3:
#             x = self.fc2(x)
#             return x

# server_model = LeNet_server_side(2)


# app = Flask(__name__)

# @app.route('/get_activation', methods=['POST'])
# def get_activation():
#     # activation_json = request.get_json()
#     # activation = json.loads(activation_json)['data']
#     # print(type(activation))
#     data = request.get_data()
#     print(data)
#     data = json.loads(data)
#     print(data['client_layers'])


#     # activation_torch = torch.Tensor(np.array(activation))
#     # print(type(activation_torch))
#     # print(activation_torch)

#     # y = server_model(activation_torch)
#     # print(y)

#     return "OK"

# if __name__ == '__main__':

#     app.run(host='localhost', port=5001, threaded=True, debug=True)


# import torch
# import torch.nn as nn
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Define the second model (Model2)
# class Model2(nn.Module):
#     def __init__(self):
#         super(Model2, self).__init__()
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = x.view(-1, 16 * 5 * 5)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x

# # Initialize the model
# model2 = Model2()

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)

# # Define the route for receiving requests
# @app.route('/', methods=['POST'])
# def train():
#     # Receive the activations and labels from the client program
#     data = request.get_json()
#     activations = torch.tensor(data['activations'])
#     labels = torch.tensor(data['labels']).squeeze()

#     # Execute the forward propagation of Model2
#     output = model2(activations)

#     # Calculate the loss and execute the backward propagation of Model2
#     loss = criterion(output, labels)
#     optimizer.zero_grad()
#     loss.backward()

#     # Get the gradients of Model2 and send them back to the client program
#     gradients = [p.grad for p in model2.parameters()]
#     return jsonify({'gradients': [grad.tolist() for grad in gradients]})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)

#----------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import requests
# import json
# from flask import Flask, request

# # Define the Model2 class
# class Model2(torch.nn.Module):
#     def __init__(self, num_classes=10):
#         super(Model2, self).__init__()
#         self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
#         self.fc2 = nn.Linear(in_features=120, out_features=84)
#         self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

#     def forward(self, x):
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Load the trained weights of Model2
# model2 = Model2()
# # model2.load_state_dict(torch.load('model2_weights.pth'))

# # Define the Flask app
# app = Flask(__name__)

# # Define the endpoint for receiving requests from the client
# @app.route('/', methods=['POST'])
# def forward_and_backward():
#     data = request.json
#     activations = torch.Tensor(data['activations'])
#     activations = activations.requires_grad_(True)
#     labels = torch.tensor(data['labels']).view(-1)
#     optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)

#     # Forward propagation
#     outputs = model2(activations)

#     # Calculate loss and gradients
#     loss = torch.nn.functional.cross_entropy(outputs, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     # gradients = model2.fc1.weight.grad.clone().detach()
#     gradients = activations.grad.clone().detach()

#     # Update the parameters of Model2
#     optimizer.step()

#     # Send the gradients to the client
#     response_data = {'gradients': gradients.tolist(), 'loss': loss.item()}
#     return json.dumps(response_data)

# if __name__ == '__main__':
#     app.run(host='localhost', port=5001)

# from flask import Flask, request

# app = Flask(__name__)

# @app.route('/endpoint', methods=['POST'])
# def handle_request():
#     data = request.form.get('key')
#     file = request.files.get('file')
#     # do something with the data and the file
#     print(data)
#     print(file.read())

#     return 'success'

# if __name__ == '__main__':
#     app.run(host='localhost', port=5001, debug=True)


# the following code is only for profiling the communication capabilities
import torch
from flask import Flask, request

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def handle_request():
    weights_rb = request.files.get ('weights')
    # weights_rb.save('model2.pkl')
    # weights = torch.load('model2.pkl')
    weights = torch.load(weights_rb)
    # weights = request.get_json()['weights']
    print("parse weights")

    return 'success'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)



