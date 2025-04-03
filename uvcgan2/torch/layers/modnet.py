# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import os
import torch
from torch import nn

import numpy as np

from uvcgan2.torch.select import get_activ_layer
from .cnn  import get_upsample_x2_layer
from .unet import UNetEncBlock

# Ref: https://arxiv.org/pdf/1912.04958.pdf

def get_demod_scale(mod_scale, weights, eps = 1e-6):
    # Ref: https://arxiv.org/pdf/1912.04958.pdf
    #
    # demod_scale[alpha] = 1 / sqrt(sigma[alpha]^2 + eps)
    #
    # sigma[alpha]^2
    #   = sum_{beta i} (mod_scale[alpha]  * weights[alpha, beta, i])^2
    #   = sum_{beta} (mod_scale[alpha])^2 * sum_i (weights[alpha, beta, i])^2
    #

    # mod_scale : (N, C_in)
    # weights   : (C_out, C_in, h, w)

    # w_sq : (C_out, C_in)
    w_sq = torch.sum(weights.square(), dim = (2, 3))

    # w_sq : (C_out, C_in) -> (1, C_in, C_out)
    w_sq = torch.swapaxes(w_sq, 0, 1).unsqueeze(0)

    # mod_scale_sq : (N, C_in, 1)
    mod_scale_sq = mod_scale.square().unsqueeze(2)

    # sigma : (N, C_out)
    sigma_sq = torch.sum(mod_scale_sq * w_sq, dim = 1)

    # result : (N, C_out)
    return 1 / torch.sqrt(sigma_sq + eps)

class ModulatedConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride = 1, padding = 0, dilation = 1, groups = 1, eps = 1e-6,
        demod = True, device = None, dtype = None, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, device = device, dtype = dtype
        )

        self._eps   = eps
        self._demod = demod

    def forward(self, x, s):
        # x : (N, C_in, H_in, W_in)
        # s : (N, C_in)

        # x_mod : (N, C_in, H_in, W_in)
        x_mod = x * s.unsqueeze(2).unsqueeze(3)

        # y : (N, C_out, H_out, W_out)
        y = self.conv(x_mod)

        if self._demod:
            # s_demod : (N, C_out)
            s_demod = get_demod_scale(s, self.conv.weight)
            y_demod = y * s_demod.unsqueeze(2).unsqueeze(3)

            return y_demod

        return y

class StyleBlock(nn.Module):

    def __init__(
        self, mod_features, style_features, rezero = True, bias = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.affine_mod = nn.Linear(mod_features, style_features, bias = bias)
        self.rezero     = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, mod):
        # mod : (N, mod_features)
        # s   : (N, style_features)
        s = 1 + self.re_alpha * self.affine_mod(mod)

        return s

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ModNetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, mod_features,
        mid_features = None,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.style_block1 = StyleBlock(
            mod_features, in_features, style_rezero, style_bias
        )
        self.conv_block1 = ModulatedConv2d(
            in_features, mid_features, kernel_size = 3, padding = 1,
            demod = demod
        )
        self.activ1 = get_activ_layer(activ)

        self.style_block2 = StyleBlock(
            mod_features, mid_features, style_rezero, style_bias
        )
        self.conv_block2 = ModulatedConv2d(
            mid_features, out_features, kernel_size = 3, padding = 1,
            demod = demod
        )
        self.activ2 = get_activ_layer(activ)

    def forward(self, x, mod):
        # x   : (N, C_in, H_in, W_in)
        # mod : (N, mod_features)

        # mod_scale1 : (N, C_out)
        mod_scale1 = self.style_block1(mod)

        # y1 : (N, C_mid, H_mid, W_mid)
        y1 = self.conv_block1(x, mod_scale1)
        y1 = self.activ1(y1)

        # mod_scale2 : (N, C_out)
        mod_scale2 = self.style_block2(mod)

        # result : (N, C_out, H_out, W_out)
        result = self.conv_block2(y1, mod_scale2)
        result = self.activ2(result)

        return result

class ModNetDecBlock(nn.Module):

    def __init__(
        self, input_shape, output_features, skip_features, mod_features,
        activ, upsample,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        (input_features, H, W) = input_shape
        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_features
        )

        self.block = ModNetBasicBlock(
            skip_features + input_features, output_features, activ,
            mod_features = mod_features,
            mid_features = max(input_features, input_shape[0]),
            demod        = demod,
            style_rezero = style_rezero,
            style_bias   = style_bias,
        )

        self._output_shape = (output_features, 2 * H, 2 * W)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, r, mod):
        # x   : (N, C, H_in, W_in)
        # r   : (N, C, H_out, W_out)
        # mod : (N, mod_features)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y, mod)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ModNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.output_shape
        self.inner_module = None

        self.deconv = ModNetDecBlock(
            input_shape     = self.inner_shape,
            output_features = image_shape[0],
            skip_features   = self.inner_shape[0],
            mod_features    = mod_features,
            activ           = activ,
            upsample        = upsample,
            rezero          = rezero,
            demod           = demod,
            style_rezero    = style_rezero,
            style_bias      = style_bias,
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y   : (N, C_inner, H_inner, W_inner)
        # mod : (N, mod_features)
        # y_hat: inner_most image
        y, mod = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r, mod)
        
        return (y, mod)

    def perturb_style(self, x, label, outer, disc):
        """
        perturbs the style feature of the image to attack
        or shift the discrimniator probability
        args:
          x: input image (A/B)
          label: the target label (B/A)
          disc: discriminator (disc_b/disc_a)
        """
        # init criterion and softmax
        import torch.nn.functional as F
        from uvcgan2.base.losses import GANLoss
        criterion = GANLoss('lsgan').to(x.device) # GANLoss(config.loss).to(self.device)
        
        # forward to inner like usual
        (y, r) = self.conv(x)
        y, mod, y_hat = self.inner_module(y)
        y, mod = y.detach(), mod.detach()

        # compute initial probability
        y.requires_grad = True
        # generate output image and calculate the gradient of style
        y_init = self.deconv(y, r, mod)
        y_init_out = outer(y_init)
        y_disc_out = disc(y_init_out)
        loss_init = criterion(y_disc_out, True)
        # calculate gradient
        dzdxp = torch.autograd.grad((loss_init), y)[0]
        
        lam_cache = {}
        def compute_shift(lam):
            if lam not in lam_cache:
                xpp = self.deconv(y+dzdxp*lam, r, mod).detach()
                pred = F.sigmoid(criterion(disc(outer(xpp)), False)).detach().cpu().numpy()
                lam_cache[lam] = xpp, pred
            return lam_cache[lam]
        
        _, init_pred = compute_shift(0)
        

    ### modified code to shift or attack the image
    def shift_forward(self, x, label=None, classifier=None, shift='shift'):
        """
        args:
          x: input image (N, C, H, W)
          shift: [shift, attack]
            'shift' for latent shift
            'attack' for adversarial attack
            'random' for random normal noise
          s_lambda: magnitude of the attack or shift
        """
        # encoding step
        (y, r) = self.conv(x)
        y, mod = self.inner_module(y)
        
        # attack or shift step
        mod_s = self.push_boundary_latent(mod, label=label, classifier=classifier, shift=shift)
        
        # if lambda shifted generate all of them
        if type(mod_s) == dict and shift == 'decision':
            y_s = {k: self.deconv(y, r, torch.tensor(mod_s[k]['step'].clone().detach(), device=mod.device)) for k in mod_s.keys()}
            mod_s = {k: torch.tensor(mod_s[k]['step'].clone().detach(), device=mod.device) for k in mod_s.keys()}
        elif type(mod_s) == dict:
            y_s = {k: {j: self.deconv(y, r, torch.tensor(mod_s[k][j], device=mod.device)) for j in mod_s[k].keys()} for k in mod_s.keys()}
            mod_s = {k: {j: torch.tensor(mod_s[k][j], device=mod.device) for j in mod_s[k].keys()} for k in mod_s.keys()}
        elif type(mod_s) == list:
            y_s = [self.deconv(y, r, torch.tensor(k, device=mod.device)) for k in mod_s]
            mod_s = [torch.tensor(m, device=mod.device) for m in mod_s]
        else:
            # decode step
            y_s = self.deconv(y, r, mod_s)
            
        return (y_s, mod_s)
    
    def push_boundary_latent(self, mod, label, classifier=None, shift='random'):
        """
        args:
          mod: input modnet features
          classifier: classifier to use to shift or attack
          shift: [shift, attack]
            'shift' for latent shift
            'attack' for adversarial attack
            'random' for random normal noise
          s_lambda: magnitude of the attack or shift
        """
        if shift == 'random':
            mod_s = mod * torch.randn(mod.shape, device=mod.device)
        elif shift == 'zeros':
            mod_s = torch.zeros(mod.shape, device=mod.device)
        elif shift == 'flip':
            mod_s = mod * -1
        elif shift == 'avgft':
            mod_s = {}
            for f in ['a', 'b', 'c', 'd']:
                avg_ft_path = '/media/Datacenter_storage/jason_notebooks/decisionGAN/uvcgan2/features/breast_{}_fwd_ft_avg.npy'.format(f)
                # figure out direction after - maybe use classifier to check
                avg_mod = np.load(avg_ft_path)
                mod_s[f] = {'avg_mod({})'.format(i):mod.detach().cpu().numpy()*(1-i) + avg_mod*i for i in np.arange(0.0, 1.1, 0.25)}
        elif shift == 'decision':
            mod_s = self.perturb_latent(model=classifier, features=mod, labels=label)
                    
        return mod_s
        
    def perturb_latent(self, model, features, labels, lam_step=100, p=0.0, verbose=False):
        """
        args:
            attack (bool):
                True: attack to flip the label
                False: push towards the boundary
            guided (bool):
                True: guiding the gradient like gifsplanation
                False: vanilla adversarial attack (gradient ascent)
            p (float): percentage to mix the original features
            alpha (float): maximum magnitude of modification
            max_iter (int): maximum number of iterations
        """
        avg_ft_dir = '/media/Datacenter_storage/jason_notebooks/decisionGAN/uvcgan2/features/'
        criterion = nn.CrossEntropyLoss() 
        softmax = nn.Softmax(dim=1)
        orig_ft = features.clone().detach()
        # get input feature prediction
        in_ft = features.clone().detach()
        in_ft.requires_grad = True
        logits = model(in_ft)
        loss = criterion(logits, labels)
        dzdxp = torch.autograd.grad(loss, in_ft)[0]
        initial_pred = softmax(logits)[0][labels].item()
        print('initial_pred: {} {:.3f}, true: {}'.format(softmax(logits), initial_pred, labels)) if verbose else None
        
        if initial_pred >= 0.99:
            avg_ft_path = os.path.join(avg_ft_dir, 'breast_{}_fwd_ft_avg.npy'.format('a' if labels == 0 else 'd'))
            avg_ft = np.load(avg_ft_path)
            print('inital pred too strong, using average') if verbose else None
            in_ft = torch.tensor(avg_ft, device=in_ft.device).unsqueeze(dim=0)
            in_ft.requires_grad = True
            logits = model(in_ft)
            loss = criterion(logits, labels)
            dzdxp = torch.autograd.grad((loss), in_ft)[0]
            initial_pred = softmax(logits)[0][labels].item()
            print('new initial_pred: {} {:.3f}, true: {}'.format(softmax(logits), initial_pred, labels)) if verbose else None
        
        cache = {}
        def compute_shift(lam, label):
            if lam not in cache:
                step = (in_ft+dzdxp*lam).detach()
                logit = model((in_ft*p + step*(1-p))).detach().cpu()
                pred = softmax(logit)[0][label].item()
                cache[lam] = step, pred
            return cache[lam]
            
        # make conditions for lower and upper bounds
        if initial_pred > 0.9:
            lbound, rbound = [0, 2500]
        else:
            lbound, rbound = [-25, 75]

        lambdas = np.arange(lbound, rbound, np.abs((lbound-rbound)/lam_step))
        preds = []
        steps = []
        for lam in lambdas:
            step, pred = compute_shift(lam, labels.item())
            preds.append(pred)
            steps.append(step.detach().cpu())
        
        # calculate several lambdas ['cls+', 'cls-', '-cls-', '-cls+']
        lam_dict = {'cls+': {'prob':0.95, 'lam': None, 'step': None}, 'cls-': {'prob':0.55, 'lam': None, 'step': None}, 
                '-cls-': {'prob':0.45, 'lam': None, 'step': None}, '-cls+': {'prob':0.05, 'lam': None, 'step': None}}
        # for each range, get the lambda that is closest to it
        for lam_r in lam_dict.keys():
            idx, val = min(enumerate(preds), key=lambda x: abs(x[1]-lam_dict[lam_r]['prob']))
            lam_dict[lam_r]['lam'] = int(lambdas[idx])
            lam_dict[lam_r]['step'] = steps[idx]
        
        lam_dict['initial'] = {'prob':initial_pred, 'lam':0, 'step': orig_ft}
        
        if verbose:
            print(lbound, rbound)
            for lam_r in lam_dict.keys():
                print('lambda:{}, prob:{}, actual_prob:{:.3f}'.format(lam_dict[lam_r]['lam'], lam_dict[lam_r]['prob'] ,lam_dict[lam_r]['actual_prob']))
        
        return lam_dict


class ModNetLinearDecoder(nn.Module):

    def __init__(
        self, features_list, input_shape, output_shape, skip_shapes,
        mod_features, activ, upsample,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.net           = nn.ModuleList()
        self._input_shape  = input_shape
        self._output_shape = output_shape
        curr_shape         = input_shape

        for features, skip_shape in zip(
            features_list[::-1], skip_shapes[::-1]
        ):
            layer = ModNetDecBlock(
                input_shape     = curr_shape,
                output_features = features,
                skip_features   = skip_shape[0],
                mod_features    = mod_features,
                activ           = activ,
                upsample        = upsample,
                rezero          = rezero,
                demod           = demod,
                style_rezero    = style_rezero,
                style_bias      = style_bias,
            )
            curr_shape = layer.output_shape

            self.net.append(layer)

        self.output = nn.Conv2d(
            curr_shape[0], output_shape[0], kernel_size = 1
        )
        curr_shape = (output_shape[0], *curr_shape[1:])

        assert tuple(output_shape) == tuple(curr_shape)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, skip_list, mod):
        # x   : (N, C, H, W)
        # mod : (N, mod_features)
        # skip_list : List[(N, C_i, H_i, W_i)]

        y = x

        for layer, skip in zip(self.net, skip_list[::-1]):
            y = layer(y, skip, mod)

        return self.output(y)

class ModNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        return_mod   = False,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.return_mod    = return_mod

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = ModNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                mod_features, rezero, demod, style_rezero, style_bias
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])
            
        self.modnet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.image_shape[0], kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.modnet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.layer_input(x)
        y, mod = self.modnet(y)
        y = self.layer_output(y)

        if self.return_mod:
            return (y, mod)

        return y

    def shift_forward(self, x, label, classifier=None, shift='shift'):
        y = self.layer_input(x)
        y, mod = self.modnet.shift_forward(y, label=label, classifier=classifier, shift=shift)
        if len(y) > 1:
            if type(y) == list:
                y = [self.layer_output(i) for i in y]
            else:
                if shift == 'decision':
                    y = {k: self.layer_output(y[k]) for k in y.keys()}
                else:
                    y = {k: {j: self.layer_output(y[k][j]) for j in y[k].keys()} for k in y.keys()}
        else:
            y = self.layer_output(y)

        if self.return_mod:
            return (y, mod)

        return y
        
    def perturb_style(self, x, label, disc):
        y = self.layer_input(x)
        y, mod = self.modnet.perturb_style(y, label, self.layer_output, disc)
        y = self.layer_output(y)

        if self.return_mod:
            return (y, mod)

        return y
        
### STAR versions ###

class ModNetBlockStar(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.output_shape
        self.inner_module = None

        self.deconv = ModNetDecBlock(
            input_shape     = self.inner_shape,
            output_features = image_shape[0],
            skip_features   = self.inner_shape[0],
            mod_features    = mod_features,
            activ           = activ,
            upsample        = upsample,
            rezero          = rezero,
            demod           = demod,
            style_rezero    = style_rezero,
            style_bias      = style_bias,
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y   : (N, C_inner, H_inner, W_inner)
        # mod : (N, mod_features)
        # y_hat: inner_most image
        y, mod, y_hat_in, y_hat_out = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r, mod)
        
        return (y, mod, y_hat_in, y_hat_out)

class ModNetStar(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        return_mod   = False,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.return_mod    = return_mod

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = ModNetBlockStar(
                features, activ, norm, curr_image_shape, downsample, upsample,
                mod_features, rezero, demod, style_rezero, style_bias
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])
            
        self.modnet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.image_shape[0], kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.modnet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.layer_input(x)
        y, mod, y_hat_in, y_hat_out = self.modnet(y)
        y = self.layer_output(y)

        if self.return_mod:
            return (y, mod, y_hat_in, y_hat_out)

        return y