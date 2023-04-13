import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, request

import torch
import dml_utils
import worker_utils
# from nns.nn_mnist import net  # configurable parameter, from nns.whatever import net.
from nns.nn_mnist import LeNet_client_side

dirname = os.path.abspath (os.path.dirname (__file__))

# listen on port 4444.
# we do not recommend changing this port number.
dml_port = 4444

ctl_addr = os.getenv ('NET_CTL_ADDRESS')
agent_addr = os.getenv ('NET_AGENT_ADDRESS')
node_name = os.getenv ('NET_NODE_NAME')

log_file = os.path.abspath (os.path.join (dirname, '../dml_file/log/',
	node_name + '.log'))
worker_utils.set_log (log_file)
conf = {}
# configurable parameter, specify the dataset path.
train_path = os.path.join (dirname, '../dataset/MNIST/')


app = Flask (__name__)
weights_lock = threading.Lock ()
executor = ThreadPoolExecutor (1)


# if this is container, docker will send a GET to here every 30s, as we set the healthcheck option in the yml file, the default interval is 30s
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

	# global train_images, train_labels
	global local_data
	local_data = dml_utils.load_data (train_path, conf ['train_start_index'], conf ['train_len'], conf ['batch_size'], train=True)

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_dataset.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))
	return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	conf ['current_round'] = 0
	print ('POST at /conf/structure')

	global net
	net = LeNet_client_side(conf['client_layers'])
	print(net)

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))

	# for customized selection >>>
	# executor.submit (perf_eval)
	# <<< for customized selection
	return ''


# for customized selection >>>

def perf_eval ():
	s = time.time ()
	dml_utils.train (net, train_images, train_labels, 1, conf ['batch_size'], conf ['train_len'])
	e = time.time () - s
	addr = conf ['connect'] [conf ['father_node'] [0]]
	path = '/ttime?node=' + node_name + '&time=' + str (e)
	worker_utils.log (node_name + ': train time=' + str (e))
	worker_utils.send_data ('GET', path, addr)

	s = time.time ()
	dml_utils.send_weights (net.get_weights (), '/stest', conf ['father_node'], conf ['connect'])
	e = time.time () - s
	path = '/stime?node=' + node_name + '&time=' + str (e)
	worker_utils.log (node_name + ': send time=' + str (e))
	worker_utils.send_data ('GET', path, addr)


# <<< for customized selection

@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/train', methods=['POST'])
def route_train ():
	print ('POST at /train')
	# weights = dml_utils.parse_weights (request.files.get ('weights')) # suppose we get the weights in a right way
	weights_rb = request.files.get ('weights')
	weights = torch.load(weights_rb)
	print("parse weights")
	# executor.submit (on_route_train, weights)
	executor.submit(on_route_train_split, weights)
	return ''


def on_route_train (received_weights):
	net.load_state_dict(received_weights)
	print("updated local weights")
	loss_list = dml_utils.train (net, local_data, conf ['epoch'], conf ['batch_size'])
	conf ['current_round'] += 1

	last_epoch_loss = loss_list [-1]
	msg = dml_utils.log_loss (last_epoch_loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	dml_utils.send_weights (net.state_dict(), '/combine', conf ['father_node'], conf ['connect'])

# modify here to send the activations to server
def on_route_train_split (received_weights):
	net.load_state_dict(received_weights)
	print("updated local weights")
	loss_list, client_weight = dml_utils.client_train (net, local_data, conf)
	conf ['current_round'] += 1

	last_epoch_loss = loss_list [-1]
	msg = dml_utils.log_loss (last_epoch_loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	dml_utils.send_client_weights (client_weight, '/combine_split', conf ['father_node'], conf ['connect'], conf['client_layers'])

app.run (host='0.0.0.0', port=dml_port, threaded=True, debug=True)
