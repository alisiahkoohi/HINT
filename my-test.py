import torch
from FrEIA.framework import *
from FrEIA.modules import *
from hint import *
import numpy as np



if __name__ == '__main__':

	ndim_y = 2
	ndim_y = 2

	y_lane = [InputNode(ndim_y, name='y')]
	x_lane = [InputNode(ndim_x, name='x')]

	for i in range(n_blocks):
	    if i > 0:
	        y_lane.append(Node(y_lane[-1],
	                           HouseholderPerm,
	                           {'fixed': False, 'n_reflections': ndim_y},
	                           name=f'perm_y_{i}'))
	        x_lane.append(Node(x_lane[-1],
	                           HouseholderPerm,
	                           {'fixed': False, 'n_reflections': ndim_x},
	                           name=f'perm_x_{i}'))

	    x_lane.append(Node(x_lane[-1],
	                       HierarchicalAffineCouplingBlock,
	                       {'c_internal': [hidden_layer_sizes, hidden_layer_sizes//2, hidden_layer_sizes//4]},
	                       name=f'hac_x_{i+1}'))

	    if i < n_blocks-1:
	        x_lane.append(Node(x_lane[-1],
	                           ExternalAffineCoupling,
	                           {'F_class': F_fully_connected,
	                            'F_args': {'internal_size': hidden_layer_sizes}},
	                           conditions=y_lane[-1],
	                           name=f'ac_y_to_x_{i+1}'))

	    y_lane.append(Node(y_lane[-1],
	                       AffineCoupling,
	                       {'F_class': F_fully_connected,
	                        'F_args': {'internal_size': hidden_layer_sizes}},
	                       name=f'ac_y_{i+1}'))

	y_lane.append(OutputNode(y_lane[-1], name='z_y'))
	x_lane.append(OutputNode(x_lane[-1], name='z_x'))

	model = ReversibleGraphNet(y_lane + x_lane, verbose=False)
	model.to(device)


