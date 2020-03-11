import numpy as np
import torch
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.cl.continual_learner import ContinualLearner


class Classifier(ContinualLearner):
    '''Model for encoding (i.e., feature extraction) and classifying images, enriched as "ContinualLearner"--object.'''

    def __init__(self, image_size, image_channels, classes, target_name, only_active=False,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=16, reducing_layers=4, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=2000, h_dim=2000, fc_drop=0, fc_bn=False, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False):

        # model configurations
        super().__init__()
        self.classes = classes
        self.target_name = target_name
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # whether always all classes can be predicted or only those seen so far
        self.only_active = only_active
        self.active_classes = []

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers<2:
            self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
        elif fc_layers==2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
        self.units_before_classifier = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        #--> fully connected layers
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated)
        #--> classifier
        self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none', drop=fc_drop)


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list


    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.fc_layer_sizes[0], "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def update_active_classes(self, y):
        '''Given labels of newly observed batch, update list of all classes seen so far.'''
        for i in y.cpu().numpy():
            if not i in self.active_classes:
                self.active_classes.append(i)


    def forward(self, x):
        '''Return tensors required for updating weights and a dict (with key self.target_name) of predicted labels.'''
        # Perform forward pass
        logits = self.classifier(self.fcE(self.flatten(self.convE(x))))
        # Get predictions
        if self.only_active and len(self.active_classes)>0:
            # -restrict predictions to those classes listed in [self.active_classes]
            logits_for_prediction = logits[:, self.active_classes]
            predictions_shifted = logits_for_prediction.cpu().data.numpy().argmax(1)
            predictions = {self.target_name: np.array([self.active_classes[i] for i in predictions_shifted])}
        else:
            # -all classes can be predicted (even those not yet observed)
            predictions = {self.target_name: logits.cpu().data.numpy().argmax(1)}
        return logits, predictions


    def update_weights(self, logits, y, x=None, rnt=None, update=True, **kwargs):
        '''Train model for one batch ([x],[y]).

        [logits]  <tensor> batch of logits returned by model for inputs [x]
        [y]       <tensor> batch of corresponding ground-truth labels'''

        # Set model to training-mode
        if update:
            self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Hack to make it work if batch-size is 1
        y = y.expand(1) if len(y.size())==0 else y

        # If requested, restrict predictions to those classes seen so far
        if self.only_active:
            # -update "active classes" (i.e., list of all classes seen so far)
            self.update_active_classes(y)
            # -remove predictions for classes not yet seen
            logits = logits[:, self.active_classes]
            # -update indeces of ground-truth labels to match those in the "active classes"-list
            y = torch.tensor([self.active_classes.index(i) for i in y]).to(self._device())

        # print(self.active_classes)
        # print(y)
        # print(logits.shape)

        # Calculate multiclass prediction loss
        predL = F.cross_entropy(input=logits, target=y, reduction='none') # -> summing over classes is "implicit"
        predL = lf.weighted_average(predL, weights=None, dim=0)           # -> average over batch
        loss_cur = predL

        # Calculate training-precision
        precision = (y == logits.max(1)[1]).sum().item() / logits.size(0)

        # If doing LwF, add 'replayed' data & calculate loss on it
        if self.previous_model is not None:
            # -generate the labels to 'replay' the current inputs with
            with torch.no_grad():
                target_logits, _ = self.previous_model.forward(x)
            if self.only_active:
                target_logits = target_logits[:, self.previous_model.active_classes]
            # -evaluate replayed data
            loss_replay = lf.loss_fn_kd(scores=logits, target_scores=target_logits, T=self.distill_temp)
        else:
            loss_replay = None

        # Calculate total loss
        loss_total = loss_cur if (self.previous_model is None) else rnt*loss_cur+(1-rnt)*loss_replay

        # Add SI-loss (Zenke et al., 2017)
        si_loss = self.si_loss()
        if self.si_c>0:
            loss_total += self.si_c * si_loss

        if update:
            # Backpropagate errors
            loss_total.backward()

            # Take optimization-step
            self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item(),
            'loss_replay': loss_replay.item() if (loss_replay is not None) else 0.,
            'si_loss': si_loss.item(),
            'precision': precision,
        }

