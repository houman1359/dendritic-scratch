

#from units import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import floor 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopKLinear(nn.Linear):
    """ Custom Linear layer that only uses the strongest K weights """
    def __init__(self, in_features, out_features, keep_in_features, bias=False):
        super(TopKLinear, self).__init__(in_features, out_features, bias=bias)
        self.keep_in_features = keep_in_features
        self.topk_indices = None

    def forward(self, x):
        mask = torch.zeros_like(self.weight)  # Create mask dynamically on the correct device
        net_weight = torch.zeros_like(self.weight)  # Create net_weight dynamically on the correct device

        topk_indices = torch.topk(self.weight, self.keep_in_features, 1, largest=True)[1]
        mask[torch.arange(self.weight.shape[0])[:, None], topk_indices] = 1
        net_weight = mask * self.weight
        w_times_x = torch.mm(x, net_weight.t())
        return w_times_x
    
class BlockLinear(nn.Linear):
    """ Custom Linear layer that uses block structures """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        block_size = torch.div(in_features, out_features, rounding_mode='floor')
        assert block_size * out_features == in_features
        self.block_size = block_size

    def forward(self, x):
        mask = torch.zeros(self.out_features, self.in_features, device=self.weight.device)

        for counter in range(self.out_features):
            mask[counter, torch.arange(counter * self.block_size, (counter + 1) * self.block_size)] = 1
        net_weight = self.weight * mask
        w_times_x = torch.mm(x, net_weight.t())
        return w_times_x
    
class DendriteBranchLayer(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                 excitatory_synapses_per_dendrite_branch, inhibitory_synapses_per_dendrite_branch, neuron_type, 
                 branch_input=False, input_dendrite_branch_n=None):
        super(DendriteBranchLayer, self).__init__()

        self.branch_input = branch_input
        self.neuron_type = neuron_type

        self.inputs_to_dendrite_branch = TopKLinear(excitatory_input_size, dendrite_branch_n, excitatory_synapses_per_dendrite_branch)
        self.inhibition_to_dendrite_branch = TopKLinear(inhibitory_input_size, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch)

        if self.branch_input:
            self.dendrite_branch_inputs_to_dendrite_branch = BlockLinear(input_dendrite_branch_n, dendrite_branch_n)
            self.dendrite_branch_inputs_to_dendrite_branch.requires_grad_(False)  # We don't train this. All dendrites have the same weight

        self.batch_norm = nn.BatchNorm1d(dendrite_branch_n)

    def forward(self, excitatory_input, inhibitory_input, dendrite_branch_outputs=None):
        dendrite_excitation = self.inputs_to_dendrite_branch(excitatory_input)
        dendrite_inhibition = self.inhibition_to_dendrite_branch(inhibitory_input)

        if self.branch_input:
            dendrite_depolarization = self.dendrite_branch_inputs_to_dendrite_branch(dendrite_branch_outputs)
            total_excitatory_input = dendrite_excitation + dendrite_depolarization
        else:
            total_excitatory_input = dendrite_excitation

        if self.neuron_type == 'excitatory':
            dendrite_act = total_excitatory_input - e_to_i_ratio * dendrite_inhibition
        else:  # Inhibitory neuron
            dendrite_act = dendrite_inhibition - e_to_i_ratio * dendrite_excitation

        dendrite_act = self.batch_norm(dendrite_act)

        return torch.sigmoid(dendrite_act)
    
    
    
class Neuron(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, 
                 excitatory_synapses_per_dendrite_branch, neuron_type='excitatory'):
        super(Neuron, self).__init__()

        self.neuron_type = neuron_type

        # Each neuron has a dendritic branch layer
        self.dendritic_layer = DendriteBranchLayer(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                   excitatory_synapses_per_dendrite_branch, inhibitory_synapses_per_dendrite_branch, 
                                                   neuron_type)

    def forward(self, excitatory_input, inhibitory_input):
        return self.dendritic_layer(excitatory_input, inhibitory_input)
    
    

class Layer(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, N_e, N_i, dendrite_branch_n, 
                 inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch):
        super(Layer, self).__init__()

        self.excitatory_neurons = nn.ModuleList([Neuron(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                        inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch, 
                                                        'excitatory') for _ in range(N_e)])
        
        self.inhibitory_neurons = nn.ModuleList([Neuron(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                        inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch, 
                                                        'inhibitory') for _ in range(N_i)])

    def forward(self, excitatory_input, inhibitory_input):

        excitatory_outputs = torch.stack([neuron(excitatory_input, inhibitory_input) for neuron in self.excitatory_neurons])        
        inhibitory_outputs = torch.stack([neuron(excitatory_input, inhibitory_input) for neuron in self.inhibitory_neurons])

        return excitatory_outputs, inhibitory_outputs
    
    
class Network(nn.Module):
    def __init__(self, input_size, layer_configs, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch):
        super(Network, self).__init__()

        self.first_layer = Layer(input_size, input_size, layer_configs[0][0], layer_configs[0][1], dendrite_branch_n, 
                                 inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch)

        self.layers = nn.ModuleList()
        for i in range(1, len(layer_configs)):
            excitatory_input_size = layer_configs[i-1][0]  # Number of excitatory neurons from the previous layer
            inhibitory_input_size = layer_configs[i-1][1]  # Number of inhibitory neurons from the previous layer
            self.layers.append(Layer(excitatory_input_size, inhibitory_input_size, layer_configs[i][0], layer_configs[i][1], 
                                     dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch))

        self.final_output = nn.Linear(layer_configs[-1][0], num_classes)  # Only excitatory neurons contribute to the final prediction

    def forward(self, excitatory_input, inhibitory_input):
        excitatory_output, inhibitory_output = self.first_layer(excitatory_input, inhibitory_input)

        for layer in self.layers:
            excitatory_output, inhibitory_output = layer(excitatory_output, inhibitory_output)

        final_output = self.final_output(excitatory_output[-1])  # Use the excitatory output from the last layer
        return final_output
    
    
def train_model(train_loader, model, optimizer, criterion, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        for images_raw, labels_raw in train_loader:
            images = images_raw.view(-1, 28*28).to(device)
            labels = labels_raw.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        
        
        
        
        








# Example hyperparameters
input_size = 784  # For MNIST mod 10
num_classes = 10
num_epochs = 50
lr = 1e-2

excitatory_synapses_per_dendrite_branch = 50
inhibitory_synapses_per_dendrite_branch = 1
dendrite_layers_n = 5
dendrite_branch_per_dendrite = [4, 4, 4, 4, 4]
dendrite_per_soma = 10
e_to_i_ratio = excitatory_synapses_per_dendrite_branch / inhibitory_synapses_per_dendrite_branch
soma_n = num_classes
inhibitory_units_n = 100

# Layer configuration: (N_e excitatory, N_i inhibitory neurons) per layer
layer_configs = [(100, 20), (80, 16), (60, 12), (40, 8), (20, 4)]  # Example configuration: (N_e, N_i) for each layer

# Initialize the network
model = Network(input_size, layer_configs, dendrite_branch_per_dendrite, inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch).to(device)

# Initialize the network
# Define batch_size
batch_size = 64  # Example batch size

# Example MNIST inputs
excitatory_input = torch.randn(batch_size, input_size).to(device)  # Excitatory input on the correct device
inhibitory_input = torch.randn(batch_size, input_size).to(device)  # Inhibitory input on the correct device

# Forward pass through the model
outputs = model(excitatory_input, inhibitory_input)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Example training loop (train_loader needs to be defined)
# train_model(train_loader, model, optimizer, criterion, num_epochs)