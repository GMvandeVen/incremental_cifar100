import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers
from models.fc.nets import MLP, MLP_gates
from models.fc.layers import fc_layer,fc_layer_split, fc_layer_fixed_gates
from models.cl.continual_learner import ContinualLearner



class IntegratedReplayModel(ContinualLearner):
    """Class for brain-inspired replay (BI-R) models."""

    def __init__(self, image_size, image_channels, classes, target_name, only_active=False,
                 # -conv-layers
                 conv_type="standard", depth=5, start_channels=16, reducing_layers=4, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=2000, h_dim=2000, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=False,
                 fc_gated=False,
                 # -prior
                 prior="GMM", z_dim=100, per_class=True, n_modes=1,
                 # -decoder
                 recon_loss='MSEnorm', dg_gates=True, dg_prop=0.5, device='cpu',
                 # -training-specific settings (can be changed after setting up model)
                 lamda_pl=1., lamda_rcl=1., lamda_vl=1., **kwargs):

        # Set configurations for setting up the model
        super().__init__()
        self.target_name = target_name
        self.label = "BIR"
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth

        # whether always all classes can be predicted or only those seen so far
        self.only_active = only_active
        self.active_classes = []

        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss # options: BCE|MSE|MSEnorm
        self.network_output = "sigmoid" if self.recon_loss in ("MSE", "BCE") else "none"
        # -settings for class-specific gates in fully-connected hidden layers of decoder
        self.dg_prop = dg_prop
        self.dg_gates = dg_gates if dg_prop>0. else False
        self.gate_size = classes if self.dg_gates else 0

        # Prior-related parameters
        self.prior = prior
        self.per_class = per_class
        self.n_modes = n_modes*classes if self.per_class else n_modes
        self.modes_per_class = n_modes if self.per_class else None

        # Components deciding how to train / run the model (i.e., these can be changed after setting up the model)
        # -options for prediction loss
        self.lamda_pl = lamda_pl   # weight of classification-loss
        # -how to compute the loss function?
        self.lamda_rcl = lamda_rcl     # weight of reconstruction-loss
        self.lamda_vl = lamda_vl       # weight of variational loss

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")


        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = ConvLayers(conv_type=conv_type, block_type="basic", num_blocks=num_blocks,
                                image_channels=image_channels, depth=self.depth, start_channels=start_channels,
                                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
                                output="none" if no_fnl else "normal", global_pooling=global_pooling,
                                gated=conv_gated)
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
        real_h_dim = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                       excit_buffer=excit_buffer, gated=fc_gated)
        # to z
        self.toZ = fc_layer_split(real_h_dim, z_dim, nl_mean='none', nl_logvar='none')#, drop=fc_drop)

        ##>----Classifier----<##
        self.units_before_classifier = real_h_dim
        self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none')

        ##>----Decoder (= p[x|z])----<##
        out_nl = True if fc_layers > 1 else (True if (self.depth > 0 and not no_fnl) else False)
        real_h_dim_down = h_dim if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        if self.dg_gates:
            self.fromZ = fc_layer_fixed_gates(
                z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none",
                gate_size=self.gate_size, gating_prop=dg_prop,
            )
        else:
            self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none")
        fc_layer_sizes_down = self.fc_layer_sizes
        fc_layer_sizes_down[0] = self.convE.out_units(image_size, ignore_gp=True)
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        if self.dg_gates:
            self.fcD = MLP_gates(
                size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                gate_size=self.gate_size, gating_prop=dg_prop, device=device,
                output=self.network_output,
            )
        else:
            self.fcD = MLP(
                size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                gated=fc_gated, output=self.network_output,
            )
        # to image-shape
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth > 0 else image_channels)
        # through deconv-layers
        self.convD = modules.Identity()

        ##>----Prior----<##
        # -if using the GMM-prior, add its parameters
        if self.prior=="GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()



    ##------ NAMES --------##

    def get_name(self):
        convE_label = "{}{}_".format(self.convE.name, "H") if self.depth>0 else ""
        fcE_label = "{}_".format(self.fcE.name) if self.fc_layers>1 else "{}{}_".format("h" if self.depth>0 else "i",
                                                                                        self.conv_out_units)
        z_label = "z{}{}".format(self.z_dim, "" if self.prior=="standard" else "-{}{}{}".format(
            self.prior, self.n_modes, "pc" if self.per_class else ""
        ))
        class_label = "_c{}".format(self.classes)
        decoder_label = "_{}{}".format("cg", self.dg_prop) if self.dg_gates else ""
        return "{}={}{}{}{}{}".format(self.label, convE_label, fcE_label, z_label, class_label, decoder_label)

    @property
    def name(self):
        return self.get_name()



    ##------ LAYERS --------##

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        return list



    ##------ FORWARD FUNCTIONS --------##

    def update_active_classes(self, y):
        '''Given labels of newly observed batch, update list of all classes seen so far.'''
        for i in y.cpu().numpy():
            if not i in self.active_classes:
                self.active_classes.append(i)

    def get_logits(self, x, hidden=False):
        '''Perform feedforward pass solely to get predicted logits.'''
        hidden_x = self.convE(x) if not hidden else x
        logits = self.classifier(self.fcE(self.flatten(hidden_x)))
        return logits

    def forward(self, x, hidden=False):
        '''Return tensors required for updating weights and a dict (with key self.target_name) of predicted labels.'''
        # Perform forward pass
        hidden_x = self.convE(x) if not hidden else x
        hE = self.fcE(self.flatten(hidden_x))
        logits = self.classifier(hE)
        # Get predictions
        if self.only_active and len(self.active_classes)>0:
            # -restrict predictions to those classes listed in [self.active_classes]
            logits_for_prediction = logits[:, self.active_classes]
            predictions_shifted = logits_for_prediction.cpu().data.numpy().argmax(1)
            predictions = {self.target_name: np.array([self.active_classes[i] for i in predictions_shifted])}
        else:
            # -all classes can be predicted (even those not yet observed)
            predictions = {self.target_name: logits.cpu().data.numpy().argmax(1)}
        # Create tuple of tensors required for updating weights
        tensors_for_weight_update = (hidden_x, hE, logits)
        return tensors_for_weight_update, predictions

    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z, gate_input=None):
        '''Decode latent variable activations.

        INPUT:  - [z]            <2D-tensor>; latent variables to be decoded
                - [gate_input]   <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-/taskID  ---OR---
                                 <2D-tensor>; for each batch-element in [x] a probability for every class-/task-ID

        OUTPUT: - [image_recon]  <4D-tensor>'''

        # -if needed, convert [gate_input] to one-hot vector
        if self.dg_gates and (gate_input is not None) and (type(gate_input)==np.ndarray or gate_input.dim()<2):
            gate_input = lf.to_one_hot(gate_input, classes=self.gate_size, device=self._device())

        # -put inputs through decoder
        hD = self.fromZ(z, gate_input=gate_input) if self.dg_gates else self.fromZ(z)
        image_features = self.fcD(hD, gate_input=gate_input) if self.dg_gates else self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def continued(self, hE, gate_input=None, reparameterize=True, **kwargs):
        '''Forward function to propagate [hE] furhter through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[out_channels]x[out_size]x[outsize]
               - [gate_input] <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-ID (eg, [y]) ---OR---
                              <2D-tensor>; for each batch-element in [x] a probability for each class-ID (eg, [y_hat])

        Output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x]
        - [z_mean]      <2D-tensor> with either [z] or the estimated mean of [z]
        - [z_logvar]    <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction'''

        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        # -reparameterize
        z = self.reparameterize(z_mean, z_logvar) if reparameterize else z_mean
        # -decode
        gate_input = gate_input if self.dg_gates else None
        x_recon = self.decode(z, gate_input=gate_input)
        # -return
        return (x_recon, z_mean, z_logvar, z)


    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, allowed_classes=None, class_probs=None, sample_mode=None, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_classes]     <list> of [class_ids] from which to sample
                - [class_probs]         <list> with for each class the probability it is sampled from it
                - [sample_mode]         <int> to sample from specific mode of [z]-distr'n, overwrites [allowed_classes]

        OUTPUT: - [X]  <4D-tensor> generated image-features of shape [batch_size]x[out_channels]x[out_size]x[outsize]'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # sample for each sample the prior-mode to be used
        if self.prior=="GMM":
            if sample_mode is None:
                if (allowed_classes is None and class_probs is None) or (not self.per_class):
                    # -randomly sample modes from all possible modes (and find their corresponding class, if applicable)
                    sampled_modes = np.random.randint(0, self.n_modes, size)
                    y_used = np.array(
                        [int(mode / self.modes_per_class) for mode in sampled_modes]
                    ) if self.per_class else None
                else:
                    if allowed_classes is None:
                        allowed_classes = [i for i in range(len(class_probs))]
                    # -sample from modes belonging to [allowed_classes], possibly weighted according to [class_probs]
                    allowed_modes = []     # -collect all allowed modes
                    unweighted_probs = []  # -collect unweighted sample-probabilities of those modes
                    for index, class_id in enumerate(allowed_classes):
                        allowed_modes += list(range(class_id * self.modes_per_class, (class_id+1)*self.modes_per_class))
                        if class_probs is not None:
                            for i in range(self.modes_per_class):
                                unweighted_probs.append(class_probs[index].item())
                    mode_probs = None if class_probs is None else [p / sum(unweighted_probs) for p in unweighted_probs]
                    sampled_modes = np.random.choice(allowed_modes, size, p=mode_probs, replace=True)
                    y_used = np.array([int(mode / self.modes_per_class) for mode in sampled_modes])
            else:
                # -always sample from the provided mode
                sampled_modes = np.repeat(sample_mode, size)
                y_used = np.repeat(int(sample_mode / self.modes_per_class), size) if self.per_class else None
        else:
            y_used = None

        # sample z
        if self.prior=="GMM":
            prior_means = self.z_class_means
            prior_logvars = self.z_class_logvars
            # -for each sample to be generated, select the previously sampled mode
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim).to(self._device())

        # if no classes are selected yet, but they are needed for the "decoder-gates", select classes to be sampled
        if (y_used is None) and (self.dg_gates):
            if allowed_classes is None and class_probs is None:
                y_used = np.random.randint(0, self.classes, size)
            else:
                if allowed_classes is None:
                    allowed_classes = [i for i in range(len(class_probs))]
                y_used = np.random.choice(allowed_classes, size, p=class_probs, replace=True)

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z, gate_input=y_used if self.dg_gates else None)

        # set model back to its initial mode
        self.train(mode=mode)

        # return samples as [batch_size]x[out_channels]x[out_size]x[outsize] tensor
        return X



    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, average=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''

        batch_size = x.size(0)
        if self.recon_loss in ("MSE", "MSEnorm"):
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss=="BCE":
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                            reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError("Wrong choice for type of reconstruction-loss!")
        # --> if [average]=True, reconstruction loss is averaged over all pixels/elements (otherwise it is summed)
        #       (averaging over all elements in the batch will be done later)
        return reconL


    def calculate_variat_loss(self, z, mu, logvar, y=None, y_prob=None, allowed_classes=None):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers, corresponding to actual class-IDs)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            # --> calculate analytically
            # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        elif self.prior=="GMM":
            # --> calculate "by estimation"

            ## Get [means] and [logvars] of all (possible) modes
            allowed_modes = list(range(self.n_modes))
            # -if we don't use the specific modes of a target, we could select modes based on list of classes
            if (y is None) and (allowed_classes is not None) and self.per_class:
                allowed_modes = []
                for class_id in allowed_classes:
                    allowed_modes += list(range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
            # -calculate/retireve the means and logvars for the selected modes
            prior_means = self.z_class_means[allowed_modes, :]
            prior_logvars = self.z_class_logvars[allowed_modes, :]
            # -rearrange / select for each batch prior-modes to be used
            z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
            means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
            logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            n_modes = self.modes_per_class if (
                ((y is not None) or (y_prob is not None)) and self.per_class
            ) else len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            # --> for each element in batch, calculate log-likelihood for all pseudoinputs: [batch_size] x [n_modes]
            if (y is not None) and self.per_class:
                modes_list = list()
                for i in range(len(y)):
                    target = y[i].item()
                    modes_list.append(list(range(target * self.modes_per_class, (target + 1) * self.modes_per_class)))
                modes_tensor = torch.LongTensor(modes_list).to(self._device())
                a = a.gather(dim=1, index=modes_tensor)
                # --> reduce [a] to size [batch_size]x[modes_per_class] (ie, per batch only keep modes of [y])
                #     but within the batch, elements can have different [y], so this reduction couldn't be done before
            a_max, _ = torch.max(a, dim=1)  # [batch_size]
            # --> for each element in batch, take highest log-likelihood over all pseudoinputs
            #     this is calculated and used to avoid underflow in the below computation
            a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
            if (y is None) and (y_prob is not None) and self.per_class:
                batch_size = y_prob.size(0)
                y_prob = y_prob.view(-1, 1).repeat(1, self.modes_per_class).view(batch_size, -1)
                # ----> extend probabilities per class to probabilities per mode; y_prob: [batch_size] x [n_modes]
                a_logsum = torch.log(torch.clamp(torch.sum(y_prob * a_exp, dim=1), min=1e-40))
            else:
                a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))  # -> sum over modes: [batch_size]
            log_p_z = a_logsum + a_max  # [batch_size]

            ## Calculate "log_q_z" (entropy of "reparameterized" [z] given [x])
            log_q_z = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
            # ----->  log_q_z: [batch_size]

            ## Combine
            variatL = -(log_p_z - log_q_z)

        return variatL


    def loss_function(self, x, y, x_recon, y_hat, scores, mu, z, logvar, allowed_classes=None, batch_weights=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [y]           <1D-tensor> with target-classes (as integers, corresponding to [allowed_classes])
                - [x_recon]     <4D-tensor> reconstructed image in same shape as [x]
                - [y_hat]       <2D-tensor> with predicted "logits" for each class (corresponding to [allowed_classes])
                - [scores]         <2D-tensor> with target "logits" for each class (corresponding to [allowed_classes])
                                     (if len(scores)<len(y_hat), 0 probs are added during distillation step at the end)
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)
                - [allowed_classes]None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
                - [predL]        prediction loss indicating how well targets [y] are predicted
                - [distilL]      knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                     match the target "logits" ([scores])'''

        ###-----Reconstruction loss-----###
        batch_size = x.size(0)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True,
                                           x_recon=x_recon.view(batch_size, -1)) # -> average over pixels
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)       # -> average over batch

        ###-----Variational loss-----###
        actual_y = torch.tensor([allowed_classes[i.item()] for i in y]).to(self._device()) if (
            (allowed_classes is not None) and (y is not None)
        ) else y
        if (y is None and scores is not None):
            y_prob = F.softmax(scores / self.distill_temp, dim=1)
            if allowed_classes is not None and len(allowed_classes) > y_prob.size(1):
                n_batch = y_prob.size(0)
                zeros_to_add = torch.zeros(n_batch, len(allowed_classes) - y_prob.size(1))
                zeros_to_add = zeros_to_add.to(self._device())
                y_prob = torch.cat([y_prob, zeros_to_add], dim=1)
        else:
            y_prob = None
        # ---> if [y] is not provided but [scores] is, calculate variational loss using weighted sum of prior-modes
        variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar, y=actual_y, y_prob=y_prob,
                                             allowed_classes=allowed_classes)
        variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)  # -> average over batch
        variatL /= (self.image_channels * self.image_size ** 2)               # -> divide by # of input-pixels

        ###-----Prediction loss-----###
        if y is not None and y_hat is not None:
            predL = F.cross_entropy(input=y_hat, target=y, reduction='none')
            #--> no reduction needed, summing over classes is "implicit"
            predL = lf.weighted_average(predL, weights=batch_weights, dim=0)  # -> average over batch
        else:
            predL = torch.tensor(0., device=self._device())

        ###-----Distilliation loss-----###
        if scores is not None and y_hat is not None:
            # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes would be added to [scores]!
            n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
            distilL = lf.loss_fn_kd(scores=y_hat[:, :n_classes_to_consider], target_scores=scores, T=self.distill_temp,
                                    weights=batch_weights)  #--> summing over classes & averaging over batch in function
        else:
            distilL = torch.tensor(0., device=self._device())

        # Return a tuple of the calculated losses
        return reconL, variatL, predL, distilL



    ##------ TRAINING FUNCTIONS --------##


    def update_weights(self, tensors_for_weight_update, y, rnt=None, update=True, **kwargs):
        '''Train model for one batch ([x],[y]), with [x] transformed to [tensor_for_weight_update] by forward-pass.

        [tensors_for_weight_update]  <tuple> containing (hidden_x, hE, logits)
        [y]                          <tensor> batch of corresponding ground-truth labels'''

        # Set model to training-mode
        if update:
            self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Unpack [tensor_for_weight_update]
        hidden_x = tensors_for_weight_update[0]
        hE = tensors_for_weight_update[1]
        logits = tensors_for_weight_update[2]


        ##--(1)-- CURRENT DATA --##

        # Hack to make it work if batch-size is 1
        y = y.expand(1) if len(y.size())==0 else y

        # Continue running the model
        recon_batch, mu, logvar, z = self.continued(hE, gate_input=y if self.dg_gates else None)

        # If requested, restrict predictions to those classes seen so far
        if self.only_active:
            # -update "active classes" (i.e., list of all classes seen so far)
            self.update_active_classes(y)
            # -remove predictions for classes not yet seen
            logits = logits[:, self.active_classes]
            # -update indeces of ground-truth labels to match those in the "active classes"-list
            y = torch.tensor([self.active_classes.index(i) for i in y]).to(self._device())

        # Calculate all losses
        reconL, variatL, predL, _ = self.loss_function(
            x=hidden_x, y=y, x_recon=recon_batch, y_hat=logits, scores=None, mu=mu, z=z, logvar=logvar,
            allowed_classes=self.active_classes if self.only_active else None
        )

        # Weigh losses as requested
        loss_cur = self.lamda_rcl * reconL + self.lamda_vl * variatL + self.lamda_pl * predL

        # Calculate training-precision
        precision = (y == logits.max(1)[1]).sum().item() / logits.size(0)


        ##--(2)-- REPLAYED DATA --##

        # If model from previous episode is stored, generate replay (at the hidden/latent level!)
        if self.previous_model is not None:
            batch_size_replay = z.size(0)

            # -generate the hidden representations to be replayed
            x_ = self.previous_model.sample(
                batch_size_replay, allowed_classes=self.previous_model.active_classes if self.only_active else None,
            )

            # -generate labels for this replay
            with torch.no_grad():
                target_logits = self.previous_model.get_logits(x_, hidden=True)
            if self.only_active:
                # -remove targets for classes not yet seen (they will be set to zero prob later)
                target_logits = target_logits[:, self.previous_model.active_classes]

            # -run current model on the replayed data
            (_, hE_, logits_), _ = self.forward(x_, hidden=True)
            target_probs = F.softmax(target_logits / self.distill_temp, dim=1)
            if self.only_active:
                # for those classes not in [self.previous_model.active_classes], set target_prob to zero
                new_target_probs = None
                for i in range(self.classes):
                    if i in self.previous_model.active_classes:
                        tensor_to_add = target_probs[:, self.previous_model.active_classes.index(i)].unsqueeze(1)
                    else:
                        tensor_to_add = target_probs[:, 0].zero_().unsqueeze(1)

                    if new_target_probs is None:
                        new_target_probs = tensor_to_add
                    else:
                        new_target_probs = torch.cat([new_target_probs, tensor_to_add], dim=1)
                target_probs = new_target_probs
            recon_x_, mu_, logvar_, z_ = self.continued(hE_, gate_input=target_probs if self.dg_gates else None)

            # -if requested, restrict predictions to classes seen so far
            if self.only_active:
                # -remove predictions for classes not yet seen, in both predictions and targets
                logits_ = logits_[:, self.active_classes]

            # -evaluate replayed data
            reconL_r, variatL_r, _, distilL_r = self.loss_function(
                x=x_, y=None, x_recon=recon_x_, y_hat=logits_, scores=target_logits, mu=mu_, z=z_, logvar=logvar_,
                allowed_classes=self.active_classes if self.only_active else None
            )
            # -weigh losses as requested
            loss_replay = self.lamda_rcl*reconL_r + self.lamda_vl*variatL_r + self.lamda_pl*distilL_r
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