import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# Parameters
d = 1  # Mean difference for S=1
sigma = 1.0  # Standard deviation of X1 and X2
alpha_range = torch.linspace(0, 2, 100)  # Range of alpha for inhibitory scaling
rho_in_values = [0.0]  # Different rho_in values to test
num_samples = 100000  # Number of samples
epsilon = 1e-10
# Function to compute mutual information
def mutual_information(Y, S):
    num_bins = 30
    p_y, _ = torch.histogram(Y, bins=num_bins, range=(float(Y.min()), float(Y.max())))
    p_y = p_y / p_y.sum()  # Normalize to get density
    
    p_y_s0, _ = torch.histogram(Y[S == 0], bins=num_bins, range=(float(Y.min()), float(Y.max())))
    p_y_s0 = p_y_s0 / p_y_s0.sum()
    
    p_y_s1, _ = torch.histogram(Y[S == 1], bins=num_bins, range=(float(Y.min()), float(Y.max())))
    p_y_s1 = p_y_s1 / p_y_s1.sum()
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    p_y = torch.clamp(p_y, min=epsilon)
    p_y_s0 = torch.clamp(p_y_s0, min=epsilon)
    p_y_s1 = torch.clamp(p_y_s1, min=epsilon)
    
    I_Y_S = torch.sum(p_y * (torch.log(p_y) - (torch.log(p_y_s0) + torch.log(p_y_s1)) / 2))
    return I_Y_S

# Generate samples for binary variable S
S = torch.randint(0, 2, (num_samples,))  # Random S values

# Results storage for each condition
results = {'Condition 1': [], 'Condition 2': [], 'Condition 3': [], 'Condition 4': []}

# Test different correlation coefficients for X1 and X2 (rho) and inhibitory input (rho_in)
correlation_values = [0.0, 0.9]
for rho in correlation_values:
    # Covariance matrix for X1 and X2 with correlation rho
    cov_matrix = torch.tensor([[sigma**2, rho*sigma**2],
                               [rho*sigma**2, sigma**2]])
    mean_S0 = torch.tensor([0.0, 0.0])
    mean_S1 = torch.tensor([d, d])
    mean = torch.where(S.unsqueeze(1) == 0, mean_S0, mean_S1)
    ep = MultivariateNormal(torch.zeros(2), cov_matrix).sample((num_samples,))
    X = ep + mean
    ofset = 2

    for alpha in alpha_range:
        for rho_inn in rho_in_values:
            rho_in = 0
            # Condition 1: I_in independent from S and uncorrelated with X1 and X2
            I_in_1 = alpha * torch.randn(num_samples) * sigma + ofset
            
            rho_in = 0
            # Condition 2: I_in informative about S, uncorrelated with X1 and X2
            I_in_2 = torch.where(S == 0, torch.randn(num_samples), torch.randn(num_samples) + d)
            I_in_2 = I_in_2 * alpha + ofset

            rho_in = 0.9
            # Condition 3: I_in independent from S but correlated with X1 and X2
            I_in_3 = alpha * (rho_in * (ep[:, 0] + ep[:, 1]) + torch.randn(num_samples) * sigma) + ofset

            # Condition 4: I_in informative about S and correlated with X1 and X2
            rho_in = 0.9
            I_in_4 = torch.where(S == 0, torch.randn(num_samples), torch.randn(num_samples) + d) + rho_in * (ep[:, 0] + ep[:, 1])
            I_in_4 = I_in_4 * alpha  + ofset
            
            # I_in_1[I_in_1<0]=0
            # I_in_2[I_in_2<0]=0
            # I_in_3[I_in_3<0]=0
            # I_in_4[I_in_4<0]=0


            # Compute dendritic output D and mutual information for each condition
            D_1 = X[:, 0] + X[:, 1] - I_in_1
            D_2 = X[:, 0] + X[:, 1] - I_in_2
            D_3 = X[:, 0] + X[:, 1] - I_in_3
            D_4 = X[:, 0] + X[:, 1] - I_in_4            

            # exc = X[:, 0] + X[:, 1]
            # D_1 = exc / (exc + I_in_1 + epsilon)
            # D_2 = exc / (exc + I_in_2 + epsilon)
            # D_3 = exc / (exc + I_in_3 + epsilon)
            # D_4 = exc / (exc + I_in_4 + epsilon)
            
            # # Ensure D values are finite
            # D_1 = torch.sigmoid(torch.clamp(D_1, min=0.0, max=1e6))
            # D_2 = torch.sigmoid(torch.clamp(D_2, min=0.0, max=1e6))
            # D_3 = torch.sigmoid(torch.clamp(D_3, min=0.0, max=1e6))
            # D_4 = torch.sigmoid(torch.clamp(D_4, min=0.0, max=1e6))
            
            I_Y_S_1 = mutual_information(D_1, S)
            I_Y_S_2 = mutual_information(D_2, S)
            I_Y_S_3 = mutual_information(D_3, S)
            I_Y_S_4 = mutual_information(D_4, S)
            
            # Store results
            results['Condition 1'].append((alpha.item(), rho, rho_inn, I_Y_S_1.item()))
            results['Condition 2'].append((alpha.item(), rho, rho_inn, I_Y_S_2.item()))
            results['Condition 3'].append((alpha.item(), rho, rho_inn, I_Y_S_3.item()))
            results['Condition 4'].append((alpha.item(), rho, rho_inn, I_Y_S_4.item()))

# Plotting the results for each condition and correlation value
alpha_values = alpha_range.numpy()
plt.figure(figsize=(14, 12))

# Plot for each condition
for i, (condition, mi_values) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    for rho_in in rho_in_values:
        for rho in correlation_values:
            filtered_values = [v[3] for v in mi_values if v[1] == rho and v[2] == rho_in]
            plt.plot(alpha_values, filtered_values, label=f'rho = {rho}, rho_in = {rho_in}')
    plt.title(f'Mutual Information ({condition})')
    plt.xlabel('Alpha')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


######## visualizationimport torchimport matplotlib.pyplot as pltimport matplotlib.pyplot as plt

# Example parameters
d_example = 1.0  # Mean difference for S=1
sigma_example = 1.0  # Standard deviation
rho_example = 0.5
rho_in_example = 0.5
alpha_example = 1.0
colors = ['blue', 'red']

# Generate S
S_example = torch.randint(0, 2, (num_samples,))

# Define means for S=0 and S=1
mean_S0 = torch.tensor([0.0, 0.0])
mean_S1 = torch.tensor([d_example, d_example])

# Covariance matrix
cov_matrix_example = torch.tensor([[sigma_example**2, rho_example*sigma_example**2],
                                   [rho_example*sigma_example**2, sigma_example**2]])

# Generate X1 and X2 based on S
mean_example = torch.where(S_example.unsqueeze(1) == 0, mean_S0, mean_S1)
ep_example = MultivariateNormal(torch.zeros(2), cov_matrix_example).sample((num_samples,))
X_example = ep_example + mean_example

rho_in = 0
# Conditions for the example case
# Condition 1: I_in independent from S and uncorrelated with X1 and X2
I_in_1_example = alpha_example * torch.randn(num_samples) * sigma_example

rho_in = 0
# Condition 2: I_in informative about S, uncorrelated with X1 and X2
I_in_2_example = torch.where(S_example == 0, torch.randn(num_samples), torch.randn(num_samples) + d_example)
I_in_2_example = I_in_2_example * alpha_example

rho_in = 0.9
# Condition 3: I_in independent from S but correlated with X1 and X2
I_in_3_example = alpha_example * (rho_in_example * (ep_example[:, 0] + ep_example[:, 1]) + torch.randn(num_samples) * sigma_example)

rho_in = 0.9
# Condition 4: I_in informative about S and correlated with X1 and X2
I_in_4_example = torch.where(S_example == 0, torch.randn(num_samples), torch.randn(num_samples) + d_example)
I_in_4_example = I_in_4_example * alpha_example + rho_in_example * (X_example[:, 0] + X_example[:, 1])

# Finding the global minimum and maximum y-values for consistent y-axis range
y_min = float('inf')
y_max = float('-inf')

for mi_values in results.values():
    y_values = [v[3] for v in mi_values]
    y_min = min(y_min, min(y_values))
    y_max = max(y_max, max(y_values))

# Adding a margin to the y-axis limits for better visualization
y_margin = (y_max - y_min) * 0.05
y_min -= y_margin
y_max += y_margin

# Plotting the mutual information results with consistent y-axis range
plt.figure(figsize=(14, 12))

for i, (condition, mi_values) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    for rho_in in rho_in_values:
        for rho in correlation_values:
            filtered_values = [v[3] for v in mi_values if v[1] == rho and v[2] == rho_in]
            plt.plot(alpha_values, filtered_values, label=f'rho = {rho}, rho_in = {rho_in}')
    plt.title(f'Mutual Information ({condition})')
    plt.xlabel('Alpha')
    plt.ylabel('Mutual Information')
    plt.ylim(y_min, y_max)  # Setting the same y-axis range for all subplots
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Scatter plots for each condition
plt.figure(figsize=(18, 16))

# Sample subset of points for better visibility
num_points = 1000  # Reduced number of points for better visibility
dot_size = 5  # Small dot size for clarity
indices = torch.randperm(num_samples)[:num_points]

# Define common axis limits for square plots
axis_limits = [-10, 10]

# Condition 1: Independent and uncorrelated
plt.subplot(4, 3, 1)
plt.scatter(X_example[indices, 0].numpy(), X_example[indices, 1].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 1: Scatter (X1 vs X2)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 2)
plt.scatter(X_example[indices, 0].numpy(), I_in_1_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 1: Scatter (X1 vs I_in)')
plt.xlabel('X1')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 3)
plt.scatter(X_example[indices, 1].numpy(), I_in_1_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 1: Scatter (X2 vs I_in)')
plt.xlabel('X2')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

# Continue with the other conditions using the same approach
plt.subplot(4, 3, 4)
plt.scatter(X_example[indices, 0].numpy(), X_example[indices, 1].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 2: Scatter (X1 vs X2)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 5)
plt.scatter(X_example[indices, 0].numpy(), I_in_2_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 2: Scatter (X1 vs I_in)')
plt.xlabel('X1')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 6)
plt.scatter(X_example[indices, 1].numpy(), I_in_2_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 2: Scatter (X2 vs I_in)')
plt.xlabel('X2')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 7)
plt.scatter(X_example[indices, 0].numpy(), X_example[indices, 1].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 3: Scatter (X1 vs X2)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 8)
plt.scatter(X_example[indices, 0].numpy(), I_in_3_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 3: Scatter (X1 vs I_in)')
plt.xlabel('X1')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 9)
plt.scatter(X_example[indices, 1].numpy(), I_in_3_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 3: Scatter (X2 vs I_in)')
plt.xlabel('X2')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 10)
plt.scatter(X_example[indices, 0].numpy(), X_example[indices, 1].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 4: Scatter (X1 vs X2)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 11)
plt.scatter(X_example[indices, 0].numpy(), I_in_4_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 4: Scatter (X1 vs I_in)')
plt.xlabel('X1')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.subplot(4, 3, 12)
plt.scatter(X_example[indices, 1].numpy(), I_in_4_example[indices].numpy(),
            c=[colors[s] for s in S_example[indices].numpy()], alpha=0.5, s=dot_size)
plt.title('Condition 4: Scatter (X2 vs I_in)')
plt.xlabel('X2')
plt.ylabel('I_in')
plt.grid(True)
plt.axis('square')
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.tight_layout()
plt.show()