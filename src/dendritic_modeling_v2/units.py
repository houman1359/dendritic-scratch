import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define layer class that only keeps K top weights
class TopKLinear(nn.Linear):
    """ Custom Linear layer that only uses the strongest K weights """
    def __init__(self, in_features, out_features, keep_in_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.keep_in_features = keep_in_features
        self.topk_indices = []
        self.mask = 0 * self.weight.data
        self.net_weight = 0 * self.weight.data

    def forward(self, x):
        self.mask = 0 * self.weight.data
        self.topk_indices = torch.topk(self.weight, self.keep_in_features, 1, largest=True)[1]
        self.mask[torch.arange(self.weight.shape[0])[:, None], self.topk_indices] = 1
        self.net_weight = self.mask * self.weight
        w_times_x = torch.mm(x, self.net_weight.t())
        return w_times_x


    
# Define layer class that uses a block structure
class BlockLinear(nn.Linear):
    """ Custom Linear layer that uses block structures """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        block_size = floor(in_features / out_features)
        assert(block_size * out_features == in_features)
        self.block_size = block_size
        self.mask = torch.zeros(out_features, in_features, device=device)
        for counter in range(self.out_features):
            self.mask[counter, torch.arange(counter * self.block_size, (1 + counter) * self.block_size)] = 1

    def forward(self, x):
        self.net_weight = self.weight * self.mask
        w_times_x = torch.mm(x, self.net_weight.t())
        return w_times_x
    
    
class DendriteBranchLayer(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                 excitatory_synapses_per_dendrite_branch, inhibitory_synapses_per_dendrite_branch, neuron_type, 
                 branch_input=False, input_dendrite_branch_n=None):
        super(DendriteBranchLayer, self).__init__()

        self.branch_input = branch_input
        self.neuron_type = neuron_type

        # Separate layers for excitatory and inhibitory inputs
        self.inputs_to_dendrite_branch = TopKLinear(excitatory_input_size, dendrite_branch_n, excitatory_synapses_per_dendrite_branch)
        self.inhibition_to_dendrite_branch = TopKLinear(inhibitory_input_size, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch)

        if self.branch_input:
            self.dendrite_branch_inputs_to_dendrite_branch = BlockLinear(input_dendrite_branch_n, dendrite_branch_n)
            self.dendrite_branch_inputs_to_dendrite_branch.requires_grad_(False)  # We don't train this. All dendrites have the same weight

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(dendrite_branch_n)

    def forward(self, excitatory_input, inhibitory_input, dendrite_branch_outputs=None):
        # Process excitatory and inhibitory inputs separately
        dendrite_excitation = self.inputs_to_dendrite_branch(excitatory_input)
        dendrite_inhibition = self.inhibition_to_dendrite_branch(inhibitory_input)

        if self.branch_input:
            dendrite_depolarization = self.dendrite_branch_inputs_to_dendrite_branch(dendrite_branch_outputs)
            total_excitatory_input = dendrite_excitation + dendrite_depolarization
        else:
            total_excitatory_input = dendrite_excitation

        # Depending on neuron type, process excitatory and inhibitory input
        if self.neuron_type == 'excitatory':
            dendrite_act = total_excitatory_input - e_to_i_ratio * dendrite_inhibition
        else:  # Inhibitory neuron
            dendrite_act = dendrite_inhibition - e_to_i_ratio * dendrite_excitation

        # Apply batch normalization
        dendrite_act = self.batch_norm(dendrite_act)

        # Apply activation function (Sigmoid or ReLU)
        return torch.sigmoid(dendrite_act)
    
    
    
# Define a single neuron with dendritic structure
class Neuron(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, 
                 excitatory_synapses_per_dendrite_branch, neuron_type='excitatory'):
        super(Neuron, self).__init__()

        # Store neuron type (excitatory or inhibitory)
        self.neuron_type = neuron_type

        # Each neuron has a dendritic branch layer
        self.dendritic_layer = DendriteBranchLayer(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                   excitatory_synapses_per_dendrite_branch, inhibitory_synapses_per_dendrite_branch, 
                                                   neuron_type)

    def forward(self, excitatory_input, inhibitory_input):
        # Use the dendritic layer for processing excitatory and inhibitory inputs
        return self.dendritic_layer(excitatory_input, inhibitory_input)
    
    

class Layer(nn.Module):
    def __init__(self, excitatory_input_size, inhibitory_input_size, N_e, N_i, dendrite_branch_n, 
                 inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch):
        super(Layer, self).__init__()

        # Create excitatory neurons
        self.excitatory_neurons = nn.ModuleList([Neuron(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                        inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch, 
                                                        'excitatory') for _ in range(N_e)])
        
        # Create inhibitory neurons
        self.inhibitory_neurons = nn.ModuleList([Neuron(excitatory_input_size, inhibitory_input_size, dendrite_branch_n, 
                                                        inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch, 
                                                        'inhibitory') for _ in range(N_i)])

    def forward(self, excitatory_input, inhibitory_input):
        # Get outputs from excitatory neurons
        excitatory_outputs = torch.stack([neuron(excitatory_input, inhibitory_input) for neuron in self.excitatory_neurons])
        
        # Get outputs from inhibitory neurons
        inhibitory_outputs = torch.stack([neuron(excitatory_input, inhibitory_input) for neuron in self.inhibitory_neurons])

        # Return both excitatory and inhibitory outputs
        return excitatory_outputs, inhibitory_outputs
    
    
class Network(nn.Module):
    def __init__(self, input_size, layer_configs, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch):
        super(Network, self).__init__()

        # First layer will take MNIST data as input
        self.first_layer = Layer(input_size, input_size, layer_configs[0][0], layer_configs[0][1], dendrite_branch_n, 
                                 inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch)

        # Define subsequent layers based on configuration (using outputs from previous layers as inputs)
        self.layers = nn.ModuleList([
            Layer(layer_configs[i][0], layer_configs[i][1], layer_configs[i+1][0], layer_configs[i+1][1], 
                  dendrite_branch_n, inhibitory_synapses_per_dendrite_branch, excitatory_synapses_per_dendrite_branch)
            for i in range(1, len(layer_configs) - 1)
        ])

        # Final output layer (using only excitatory outputs)
        self.final_output = nn.Linear(layer_configs[-1][0], num_classes)  # Only excitatory neurons contribute to the final prediction

    def forward(self, excitatory_input, inhibitory_input):
        # First layer: both excitatory and inhibitory inputs come from MNIST
        excitatory_output, inhibitory_output = self.first_layer(excitatory_input, inhibitory_input)

        # Subsequent layers: inputs are the outputs from the previous layer
        for layer in self.layers:
            excitatory_output, inhibitory_output = layer(excitatory_output, inhibitory_output)

        # Final output: only use excitatory output for the final prediction
        final_output = self.final_output(excitatory_output[-1])  # Use the excitatory output from the last layer
        return final_output
    
    
# Training the model
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
        
        
        
        
        
        
