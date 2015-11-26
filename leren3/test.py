import numpy as np

def a_function(activation_layer, t_layer):
    return 1 / (1 + np.exp(-activation_layer.dot(t_layer.T)))

# Computes the activations for all nodes in all layers (last activation should be h0, the output value)
def forward_function(activation_x_layer, t_all, node_architecture):
    activation_all = []
    activation_all.append(activation_x_layer) # Initialize activations with the input variables
    for layerid, layer in enumerate(node_architecture): # Loop trough layers
        activation_layer = []
        for nodeid in range(layer): # Loop trough nodes in next layer
            t_node = np.asarray(t_all[layerid][nodeid])
            current_activation_node = np.array(activation_all[layerid])
            activation_node = a_function(current_activation_node, t_node)
            activation_layer.append(activation_node)
        activation_all.append(activation_layer)
    return activation_all

#print(forward_function(np.asarray([2, 3]), np.asarray([[[1,2], [3,4]], [[1],[2]]]), [2,2,1]))

def forward_functions(activation_x_layer, t_all):
    activations = []
    # Add input variables to activation vector list
    activations.append(activation_x_layer)

    # Compute activaton in hidden layer
    a_layer_in_hidden = []
    for t_node_in_hidden in t_all[0]:
        a_node_in_hidden = a_function(np.asarray(activations[0]), np.asarray(t_node_in_hidden))
        a_layer_in_hidden.append(a_node_in_hidden)
        print(a_layer_in_hidden)
    activations.append(a_layer_in_hidden)

    a_layer_in_output = []
    for t_node_in_output in t_all[1]:
        a_node_in_output = a_function(np.asarray(activations[1]), np.asarray(t_node_in_output))
        a_layer_in_output.append(a_node_in_output)
        print(a_layer_in_output)
    activations = activations.append(a_layer_in_output)

    return activations

forward_functions([1,2], [[[1,2], [3,4]], [[1,1]]])