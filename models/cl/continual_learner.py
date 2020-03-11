import abc
import torch
from torch import nn


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract  module to add continual learning (CL) capabilities to a classifier.

    Adds methods/attributes for various CL methods (SI, LwF, BI-R) to its subclasses.'''

    def __init__(self):
        super().__init__()

        # SI:
        self.si_c = 0            #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1       #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0
        self.si_param_dict = {}  #-> dict with parts of network whose params should be controlled by SI

        # LwF / Brain-Inspired Replay:
        self.distill_temp = 2.       #-> temperature for distillation loss
        self.previous_model = None   #-> keep a copy of model after finishing training on previous task

        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass


    #------------- "Synaptic Intelligenc"-specifc functions -------------#

    def register_start_values(self):
        '''At very beginning, register starting values of all params controlled by SI.'''
        for key in self.si_param_dict:
            for n, p in self.si_param_dict[key].named_parameters():
                n = key + "." + n
                n = n.replace('.', '__')
                self.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

    def initiate_W(self):
        '''Before each new episode/task, initiate new W matrices.'''
        W = {}
        p_old = {}
        for key in self.si_param_dict:
            for n, p in self.si_param_dict[key].named_parameters():
                n = key + "." + n
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def update_W(self, W, p_old):
        '''After each model-update, update the W matrices.'''
        for key in self.si_param_dict:
            for n, p in self.si_param_dict[key].named_parameters():
                if p.requires_grad:
                    n = key + "." + n
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad * (p.detach() - p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for key in self.si_param_dict:
            for n, p in self.si_param_dict[key].named_parameters():
                n = key + "." + n
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)


    def si_loss(self):
        '''Calculate SI's regularization loss.'''
        try:
            losses = []
            for key in self.si_param_dict:
                for n, p in self.si_param_dict[key].named_parameters():
                    if p.requires_grad:
                        n = key + "." + n
                        n = n.replace('.', '__')
                        # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                        prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        # Calculate SI's surrogate loss, sum over all parameters
                        losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses) if len(losses)>0 else torch.tensor(0., device=self._device())
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
