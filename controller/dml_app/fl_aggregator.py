import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, request

import torch
import dml_utils
import worker_utils
#from nns.nn_mnist import net  # configurable parameter, from nns.whatever import net.
from nns.nn_mnist import LeNet_server_side, LeNet_full_model

dirname = os.path.abspath (os.path.dirname (__file__))

# listen on port 4444.
# we do not recommend changing this port number.
dml_port = 4444

ctl_addr = os.getenv ('NET_CTL_ADDRESS')
agent_addr = os.getenv ('NET_AGENT_ADDRESS')
node_name = os.getenv ('NET_NODE_NAME')

net = LeNet_full_model()
initial_weights = net.state_dict()
# input_shape = nn.input_shape
log_file = os.path.abspath (os.path.join (dirname, '../dml_file/log/',
	node_name + '.log'))
worker_utils.set_log (log_file)
conf = {}
trainer_list = []
trainer_per_round = 0
# configurable parameter, specify the dataset path.
test_path = os.path.join (dirname, '../dataset/MNIST/')


app = Flask (__name__)
weights_lock = threading.Lock ()
executor = ThreadPoolExecutor (1)


# if this is container, docker will send a GET to here every 30s
# this ability is defined in controller/class_node.py, Emulator.save_yml (), healthcheck.
@app.route ('/hi', methods=['GET'])
def route_hi ():
	# send a heartbeat to the agent.
	# when the agent receives the heartbeat of a container for the first time,
	# it will deploy the container's tc settings.
	# please ensure that your app implements this function, i.e.,
	# receiving docker healthcheck and sending heartbeat to the agent.
	worker_utils.heartbeat (agent_addr, node_name)
	return 'this is node ' + node_name + '\n'


@app.route ('/conf/dataset', methods=['POST'])
def route_conf_d ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	print ('POST at /conf/dataset')

	global test_data
	test_data = dml_utils.load_data (test_path, conf ['test_start_index'], conf ['test_len'], conf ['batch_size'], train=False)

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_dataset.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))
	return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	global trainer_per_round
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	conf ['current_round'] = 0
	conf ['received_number'] = 0
	conf ['received_weights'] = []
	trainer_list.extend (conf ['child_node'])
	trainer_per_round = int (len (trainer_list) * conf ['trainer_fraction'])
	print ('POST at /conf/structure')

	# define the server side and full model here
	global net_1, net_2, net_3
	net_1 = LeNet_server_side(1)
	net_2 = LeNet_server_side(2)
	net_3 = LeNet_server_side(3)

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))
	return ''


# for customized selection >>>

total_time = {}
send_time = {}
name_list = []
prob_list = []
prob_lock = threading.Lock ()


@app.route ('/ttime', methods=['GET'])
def route_ttime ():
	print ('GET at /ttime')
	node = request.args.get ('node')
	_time = request.args.get ('time', type=float)
	print ('train: ' + node + ' use ' + str (_time))
	total_time [node] = _time

	if len (total_time) == len (trainer_list):
		prob_lock.acquire ()
		if len (total_time) == len (trainer_list):
			file_path = os.path.join (dirname, '../dml_file/ttime.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (total_time))
				print ('ttime collection completed, saved on ' + file_path)
		prob_lock.release ()
	return ''


@app.route ('/stest', methods=['POST'])
def route_stest ():
	print ('POST at /stest')
	# just get the weights to test the time.
	_ = dml_utils.parse_weights (request.files.get ('weights'))
	return ''


@app.route ('/stime', methods=['GET'])
def route_stime ():
	print ('GET at /stime')
	node = request.args.get ('node')
	_time = request.args.get ('time', type=float)
	print ('send: ' + node + ' use ' + str (_time))
	send_time [node] = _time

	if len (send_time) == len (trainer_list):
		prob_lock.acquire ()
		if len (send_time) == len (trainer_list):
			file_path = os.path.join (dirname, '../dml_file/stime.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (send_time))
				print ('stime collection completed, saved on ' + file_path)

			count = 0
			for node in total_time:
				total_time [node] += send_time [node]
			file_path = os.path.join (dirname, '../dml_file/totaltime.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (total_time))
				print ('totaltime collection completed, saved on ' + file_path)
			for node in total_time:
				total_time [node] = 1 / (total_time [node] ** 0.5)
				count += total_time [node]
			for node in total_time:
				name_list.append (node)
				prob_list.append (round (total_time [node] / count, 3) * 1000)
			count = 0
			for i in range (len (prob_list)):
				count += prob_list [i]
			prob_list [-1] += 1000 - count
			for i in range (len (prob_list)):
				prob_list [i] /= 1000
			print ('prob_list = ')
			print (prob_list)
		prob_lock.release ()
	return ''


def customized_selection (number):
	return np.random.choice (name_list, number, p=prob_list, replace=False)


# <<< for customized selection

@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/start', methods=['GET'])
def route_start ():
	print("aggregator start")
	# if ctl_addr:
	# 	print(ctl_addr)
	# else:
	# 	print("ctl_addr is null")
	# print(ctl_addr, agent_addr, node_name)
	_, initial_acc = dml_utils.test (net, test_data, conf ['batch_size'])
	msg = dml_utils.log_acc (initial_acc, 0)
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	executor.submit (on_route_start)
	return ''


def on_route_start ():
	# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
	# trainers = customized_selection (trainer_per_round)
	# trainers = ['p1','p2','n3']
	trainers = ['p3', 'n1','n2']
	print(trainers)
	# dml_utils.send_weights (initial_weights, '/train', trainers, conf ['connect'])
	dml_utils.send_weights_split (initial_weights, '/train', trainers, conf ['connect'], conf['clients_layers'])
	worker_utils.send_print (ctl_addr, 'start FL')


# continue the forward calculation
@app.route('/get_activation', methods=['POST'])
def get_activation ():
    data = request.json
    client_layers = int(data['client_layers'])
    activation = torch.Tensor(data['activation']).requires_grad_(True)
    labels = torch.Tensor(data['labels']).view(-1)

    # need to know the labels of this batch of training data
    loss, client_gradients = server_train(client_layers, activation, labels)

    # have to serilize the variables, otherwise there may be wrong


    return json.dumps({'loss': loss, 'client_gradients': client_gradients})


def server_train (client_layers, activation_torch, labels_torch):
	if client_layers == 1:
		loss, client_gradients = dml_utils.server_split_train (net_1, activation_torch, labels_torch)

	if client_layers == 2:
		loss, client_gradients = dml_utils.server_split_train (net_2, activation_torch, labels_torch)

	if client_layers == 3:
		loss, client_gradients = dml_utils.server_split_train (net_3, activation_torch, labels_torch)

	return loss, client_gradients


# combine request from clients for split learning
@app.route ('/combine_split', methods=['POST'])
def combine_split ():
	print ('POST at /combine')
	# weights = dml_utils.parse_weights (request.files.get ('weights'))
	weights_rb = request.files.get ('weights')
	data = request.json
	client_layers = data['client_layers']

	weights = torch.load(weights_rb)

	# executor.submit (on_route_combine, weights)
	on_route_combine_split (weights, client_layers)
	return ''

def on_route_combine_split (client_weights, client_layers):
	if client_layers == 1:
		weights = client_weights.update(net_1.state_dict())

	if client_layers == 2:
		weights = client_weights.update(net_2.state_dict())

	if client_layers == 3:
		weights = client_weights.update(net_3.state_dict())


	weights_lock.acquire ()
	conf ['received_number'] += 1
	dml_utils.store_weights (conf ['received_weights'], weights,
		conf ['received_number'])
	weights_lock.release ()

	if conf ['received_number'] == trainer_per_round:
		combine_weights ()


# combine request from the lower layer node.
@app.route ('/combine', methods=['POST'])
def route_combine ():
	print ('POST at /combine')
	# weights = dml_utils.parse_weights (request.files.get ('weights'))
	weights_rb = request.files.get ('weights')
	weights = torch.load(weights_rb)
	# executor.submit (on_route_combine, weights)
	on_route_combine (weights)
	return ''


def on_route_combine (weights):
	weights_lock.acquire ()
	conf ['received_number'] += 1
	dml_utils.store_weights (conf ['received_weights'], weights,
		conf ['received_number'])
	weights_lock.release ()

	if conf ['received_number'] == trainer_per_round:
		combine_weights ()


def combine_weights ():
	weights = dml_utils.avg_weights (conf ['received_weights'],
		conf ['received_number'])
	# dml_utils.assign_weights (net, weights)
	net.load_state_dict(weights)
	conf ['received_weights'].clear ()
	conf ['received_number'] = 0
	conf ['current_round'] += 1

	print('Testing')
	# _, acc = dml_utils.test (net, test_images, test_labels)
	_, acc = dml_utils.test (net, test_data, conf ['batch_size'])
	msg = dml_utils.log_acc (acc, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

	if conf ['current_round'] == conf ['sync']:
		worker_utils.log ('>>>>>training ended<<<<<')
		worker_utils.send_data ('GET', '/finish', ctl_addr)
	# send down to train.
	else:
		# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
		# trainers = customized_selection (trainer_per_round)
		trainers = ['p3', 'n1','n2']
		#dml_utils.send_weights (weights, '/train', trainers, conf ['connect'])
		dml_utils.send_weights_split (weights, '/train', trainers, conf ['connect'], conf['clients_layers'])


app.run (host='0.0.0.0', port=dml_port, threaded=True, debug=True)
