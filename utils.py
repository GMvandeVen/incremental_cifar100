import os
import pickle
import torch
from torch import nn
from models.fc import excitability_modules as em

##-------------------------------------------------------------------------------------------------------------------##

######################
## Random utilities ##
######################

def checkattr(args, attr):
    '''Check whether attribute exists, whether it's a boolean and whether its value is True.'''
    return hasattr(args, attr) and type(getattr(args, attr))==bool and getattr(args, attr)


##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


##-------------------------------------------------------------------------------------------------------------------##

#########################################
## Model-saving and -loading functions ##
#########################################

def save_checkpoint(model, model_dir, verbose=True, name=None):
    '''Save state of [model] as dictionary to [model_dir] (if name is None, use "model.name").'''
    # -path to store the checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # -save the checkpoint
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict()}, path)
    # -notify that we successfully saved the checkpoint
    if verbose:
        print('=> saved model {name} to {path}'.format(name=name, path=model_dir))


def load_checkpoint(model, model_dir, verbose=True, name=None):
    '''Load saved state (in form of dictionary) at [model_dir] (if name is None, use "model.name") to [model].'''
    # -path from where to load checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # load parameters (i.e., [model] will now have the state of the loaded model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    # notify that we succesfully loaded the checkpoint
    if verbose:
        print('=> loaded checkpoint of {name} from {path}'.format(name=name, path=model_dir))


##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("Model-name: \"" + model.name + "\"")
    print(40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-" + "\n\n")



##-------------------------------------------------------------------------------------------------------------------##

########################################
## Parameter-initialization functions ##
########################################

def weight_reset(m):
    '''Reinitializes parameters of [m] according to default initialization scheme.'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, em.LinearExcitability):
        m.reset_parameters()


def weight_init(model, strategy="xavier_normal", std=0.01):
    '''Initialize weight-parameters of [model] according to [strategy].
    [xavier_normal]     "normalized initialization" (Glorot & Bengio, 2010) with Gaussian distribution
    [xavier_uniform]    "normalized initialization" (Glorot & Bengio, 2010) with uniform distribution
    [normal]            initialize with Gaussian(mean=0, std=[std])
    [...]               ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() >= 2:
            if strategy=="xavier_normal":
                nn.init.xavier_normal_(p)
            elif strategy=="xavier_uniform":
                nn.init.xavier_uniform_(p)
            elif strategy=="normal":
                nn.init.normal_(p, std=std)
            else:
                raise ValueError("Invalid weight-initialization strategy {}".format(strategy))


def bias_init(model, strategy="constant", value=0.01):
    '''Initialize bias-parameters of [model] according to [strategy].
    [zero]      set them all to zero
    [constant]  set them all to [value]
    [positive]  initialize with Uniform(a=0, b=[value])
    [any]       initialize with Uniform(a=-[value], b=[value])
    [...]       ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() == 1:
            ## NOTE: be careful if excitability-parameters are added to the model!!!!
            if strategy == "zero":
                nn.init.constant_(p, val=0)
            elif strategy == "constant":
                nn.init.constant_(p, val=value)
            elif strategy == "positive":
                nn.init.uniform_(p, a=0, b=value)
            elif strategy == "any":
                nn.init.uniform_(p, a=-value, b=value)
            else:
                raise ValueError("Invalid bias-initialization strategy {}".format(strategy))