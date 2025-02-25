from math import sqrt, exp, log, erf
from scipy.stats import norm
import torch
from torch.nn.init import _no_grad_normal_


def compute_expectation_truncated_log_normal(
    mean, std, upper_quantile,
):
    log_alpha = mean + std * norm.ppf(1 - upper_quantile).item()

    numerator = exp(mean + 0.5 * std**2) * (
        1 - erf((log_alpha - mean - std**2) / (std * sqrt(2))))
    denominator = 1 - erf((log_alpha - mean) / (std * sqrt(2)))

    expectation = numerator / denominator
    return expectation


def initialize_dendritic_branch_layer(dbl):
    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0

        if dbl.input_excitatory:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_mean = 0
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))

            if not dbl.input_inhibitory and not dbl.input_branches:
                print('exc method 1', flush = True)
                step = 0.01
                done = False
                while not done:
                    E_w_exc = compute_expectation_truncated_log_normal(
                        mean = exc_mean, std = exc_std,
                        upper_quantile = exc_upper_quantile,
                    )
                    if 0.5 * dbl.branch_excitation.K * E_w_exc <= 1:
                        done = True
                    else:
                        exc_mean -= step
            
            else:
                print('exc method 2', flush = True)
                E_w_exc = compute_expectation_truncated_log_normal(
                    mean = exc_mean, std = exc_std, 
                    upper_quantile = exc_upper_quantile,
                )
            
            _no_grad_normal_(dbl.branch_excitation.pre_w, exc_mean, exc_std)
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc
            print('exc', E_g_exc, E_w_exc, flush = True)

        
        if dbl.input_inhibitory:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim            

            if dbl.input_excitatory:
                print('inh method 1', flush = True)
                ie_ratio = dbl.branch_inhibition.K / dbl.branch_excitation.K

                inh_std = exc_std
                step = .01
                done = False
                while not done:
                    E_w_inh = compute_expectation_truncated_log_normal(
                        mean = 0, std = inh_std,
                        upper_quantile = inh_upper_quantile,
                    )
                    if E_w_exc <= ie_ratio * E_w_inh:
                        done = True
                    else:
                        inh_std += step
                
            else:
                print('inh method 2', flush = True)
                inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
                E_w_inh = compute_expectation_truncated_log_normal(
                    mean = 0, std = inh_std,
                    upper_quantile = inh_upper_quantile,
                )
            
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh
            print('inh', E_g_inh, E_w_inh, flush = True)
            
        
        E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)

        if dbl.input_branches:
            E_Vinf = max(0.49, 0.5 * E_Vinf + 0.25)

            sum_g_branch = ((E_g_exc + E_g_inh + 1) * E_Vinf - E_g_exc) / (0.5 - E_Vinf)
            g_branch = sum_g_branch / dbl.branches_to_output.block_size

            dbl.branches_to_output.initialize(g_branch)
        
        print('E_Vinf', E_Vinf, flush = True)

        if dbl.reactivate:
            print('init react', flush = True)
            dbl.reactivation.initialize(1.8, log(E_Vinf))
    
    print('dbl initialized', flush = True)


