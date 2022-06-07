import os
import random

from flask import Flask

import ctl_utils
import fl_manager
from class_node import Net

if __name__ == '__main__':
	dirname = os.path.abspath (os.path.dirname (__file__))
	ip = '192.168.1.107'
	port = 3333
	net = Net (ip, port)
	net.add_nfs (tag='dml_app', ip='192.168.1.1', mask=24,
		path=os.path.join (dirname, 'dml_app'))
	net.add_nfs (tag='dataset', ip='192.168.1.1', mask=24,
		path=os.path.join (dirname, 'dataset'))
	ctl_utils.restore_nfs ()
	ctl_utils.export_nfs (net)

	dml_port = 4444

	p4 = net.add_physical_node ('p4', 'wlan0', '192.168.1.102')
	p4.mount_nfs (tag='dml_app', mount_point='./dml_app')
	p4.mount_nfs (tag='dataset', mount_point='./dataset')
	p4.set_cmd (working_dir='dml_app', cmd=['python3', 'fl_aggregator.py'])

	p1 = net.add_physical_node (name='p1', nic='wlan0', ip='192.168.1.100')
	p1.mount_nfs (tag='dml_app', mount_point='./dml_app')
	p1.mount_nfs (tag='dataset', mount_point='./dataset')
	p1.set_cmd (working_dir='dml_app', cmd=['python3', 'fl_trainer.py'])
	net.asymmetrical_link (p4, p1, bw=50, unit='mbps')
	net.asymmetrical_link (p1, p4, bw=random.randint (20, 50), unit='mbps')

	p2 = net.add_physical_node (name='p2', nic='wlan0', ip='192.168.1.111')
	p2.mount_nfs (tag='dml_app', mount_point='./dml_app')
	p2.mount_nfs (tag='dataset', mount_point='./dataset')
	p2.set_cmd (working_dir='dml_app', cmd=['python3', 'fl_trainer.py'])
	net.asymmetrical_link (p4, p2, bw=50, unit='mbps')
	net.asymmetrical_link (p2, p4, bw=random.randint (20, 50), unit='mbps')

	p3 = net.add_physical_node (name='p3', nic='wlan0', ip='192.168.1.108')
	p3.mount_nfs (tag='dml_app', mount_point='./dml_app')
	p3.mount_nfs (tag='dataset', mount_point='./dataset')
	p3.set_cmd (working_dir='dml_app', cmd=['python3', 'fl_trainer.py'])
	net.asymmetrical_link (p4, p3, bw=50, unit='mbps')
	net.asymmetrical_link (p3, p4, bw=random.randint (20, 50), unit='mbps')

	emu = net.add_emulator ('3990x', '192.168.1.103')
	emu.mount_nfs ('dml_app')
	emu.mount_nfs ('dataset')
	for i in range (13):
		#cpu = (i % 4) * 2
		cpu = (i % 3)
		if cpu == 0:
			cpu = 1
		n = emu.add_node ('n' + str (i + 1), 'eth0', '/home/worker/dml_app',
			['python3', 'fl_trainer.py'], 'dml:v1.0', cpu=cpu, memory=4, unit='G')
		n.add_volume ('./dml_file', '/home/worker/dml_file')
		n.add_nfs ('dml_app', '/home/worker/dml_app')
		n.add_nfs ('dataset', '/home/worker/dataset')
		n.add_port (dml_port, 8001 + i)
		net.asymmetrical_link (p4, n, bw=50, unit='mbps')
		net.asymmetrical_link (n, p4, bw=random.randint (20, 50), unit='mbps')

	net.save_node_ip () #node_ip.json

	net.save_bw () #links.json

	# links_json = ctl_utils.read_json (os.path.join (dirname, 'links.json'))
	# net.load_bw (links_json)

	net.save_yml ()

	app = Flask (__name__)

	ctl_utils.print_listener (app)

	ctl_utils.docker_tc_listener (app, net)
	ctl_utils.send_docker_address (net)     # this two lines start the listener process in th controller and send the controller address to the agent
	# path_dockerfile = os.path.join (dirname, 'dml_app/Dockerfile')
	# path_req = os.path.join (dirname, 'dml_app/dml_req.txt')
	# ctl_utils.deploy_dockerfile (net, 'dml:v1.0', path_dockerfile, path_req)

	ctl_utils.send_device_nfs (net) #agent (physical) received nfs configuration and mount the nfs
	ctl_utils.send_device_env (net)
	# path_req = os.path.join (dirname, 'dml_app/dml_req.txt')
	# ctl_utils.sent_device_req (net, path_req)

	if net.tcLinkNumber > 0:
		ctl_utils.send_docker_tc (net) #update tc configuration, haven't execution
		ctl_utils.send_device_tc (net) #update tc configuration and execute
	else:
		print ('tc finish')

	ctl_utils.update_tc_listener (app, net)

	ctl_utils.docker_controller_listener (app, net)
	ctl_utils.device_controller_listener (app, net)
	ctl_utils.clear_nfs_listener (app)

	fl_manager.manager (app, net)

	ctl_utils.deploy_all_device (net) #start physical node
	ctl_utils.deploy_all_yml (net) #start container

	app.run (host='0.0.0.0', port=port, threaded=True)
