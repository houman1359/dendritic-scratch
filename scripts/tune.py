
import os
from datetime import datetime


from dendritic_modeling.tuning import DendriNetTuner
from dendritic_modeling.models import EINetClassifier
from dendritic_modeling.utils import load_MNIST, accuracy_score

print('\nimports complete\n', flush = True)


if __name__ == '__main__':

    args = {
        'name' : 'tune',
        'train-valid split' : 0.8,
        'epochs' : 500,
        'batch size' : 512,
    }

    hp_space = {
        'branch_factors_space' : ([1,3],),
        'n_inh_cells_space' : (20,),
        'syn_per_inh_cell_space' : (200,),
        'inh_syn_per_branch_space' : (1,2,),
        'exc_syn_per_branch_space' : (50,100,150),
        'shunting_space' : (True,),
        'reactivate_space' : (False,),
        'b_trainable_space' : (False,),
        'log_lr_space' : (-4,-3),
    }
    
    data = load_MNIST(train_valid_split = 0.8)
    

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'results', f'{args['name']}_{now}')
    os.makedirs(save_path, exist_ok = True)

    print('tuning.....\n')
    # this tuning class needs to be adjusted for EINet
    tuner = DendriNetTuner(**hp_space)
    tuner.tune(
        dendrinet = EINetClassifier, input_dim = 784, output_dim = 10,
        train_data = data['train'], valid_data = data['valid'], test_data = data['test'],
        epochs = args['epochs'], batch_size = args['batch size'],
        metric_fn = accuracy_score,
        save_root = save_path,
    )
    





