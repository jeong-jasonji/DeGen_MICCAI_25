# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch
import torch.nn as nn

from torchvision.transforms import GaussianBlur, Resize

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, BatchHeadWrapperCond, get_batch_head
from uvcgan2.base.losses             import GANLoss
from uvcgan2.torch.gradient_penalty  import GradientPenalty
from uvcgan2.models.discriminator    import construct_discriminator
from uvcgan2.models.generator        import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input, set_two_domain_label

def construct_consistency_model(consist, device):
    name, kwargs = extract_name_kwargs(consist)

    if name == 'blur':
        return GaussianBlur(**kwargs).to(device)

    if name == 'resize':
        return Resize(**kwargs).to(device)

    raise ValueError(f'Unknown consistency type: {name}')

def queued_forward(batch_head_model, input_image, queue, update_queue = True, true_disc=True):
    if true_disc:
        output_body, output_cls, body = batch_head_model.forward(
            input_image, extra_bodies = queue.query(), return_body = True,
        )
    
        if update_queue:
            queue.push(body)
    
        return output_body, output_cls
    else:
        output, pred_body = batch_head_model.forward(
            input_image, extra_bodies = queue.query(), return_body = True,
        )
    
        if update_queue:
            queue.push(pred_body)
    
        return output

class UVCGAN2_DecisionStar(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a', 'fake_b',
            'reco_a', 'reco_b',
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
        ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b', ]

        return NamedDict(*images)
    
    # add labels and latent vectors
    def _setup_labels(self, _config):
        labels = [
            'real_a', 'real_b',
        ]

        return NamedDict(*labels)
    
    def _setup_latent_preds(self, _config):
        latent_preds = [
            'pred_ab_fwd', 'pred_ba_fwd',
            'pred_ab_bwd', 'pred_ba_bwd'
        ]

        if self.is_train and self.lambda_idt > 0:
            latent_preds += [ 'pred_aa_fwd', 'pred_bb_fwd', ]

        return NamedDict(*latent_preds)
    
    # need _latent_feats to make the disc_l move forward
    def _setup_latent_feats(self, _config):
        latent_feats = [
            'feat_ab_fwd', 'feat_ba_fwd',
            'feat_ab_bwd', 'feat_ba_bwd'
        ]
        
        if self.is_train and self.lambda_idt > 0:
            latent_feats += [ 'feat_aa_fwd', 'feat_bb_fwd', ]

        return NamedDict(*latent_feats)

    def _setup_latent_imgs(self, _config):
        latent_imgs = [
            'y_hat_ab_fwd', 'y_hat_ab_bwd',
            'y_hat_ba_fwd', 'y_hat_ba_bwd',  
        ]
        
        if self.is_train and self.lambda_idt > 0:
            latent_imgs += [ 'y_hat_aa_fwd', 'y_hat_bb_fwd', ]

        return NamedDict(*latent_imgs)

    def _construct_batch_head_disc(self, model_config, input_shape, true_disc=True):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        disc_cls = get_batch_head(self.cls_config)
        disc_cls = prepare_model(disc_cls, self.device)
        if true_disc:
            return BatchHeadWrapperCond(disc_body, disc_head, disc_cls)
        else:
            return BatchHeadWrapper(disc_body, disc_cls)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        # style disc
        models['disc_style'] = nn.Linear(config.l_discriminator.model_args['latent_dim'], config.l_discriminator.model_args['n_classes'])
        models['disc_style'] = models['disc_style'].to(self.device)
        # yhat disc
        models['disc_yhat'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape, true_disc=False
            )
        models['disc_yhat'] = models['disc_yhat'].to(self.device)

        if self.avg_momentum is not None:
            models['avg_gen'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['avg_gen'].load_state_dict(models['gen'].state_dict())
            
            models['avg_disc_style'] = nn.Linear(config.l_discriminator.model_args['latent_dim'], config.l_discriminator.model_args['n_classes'])
            models['avg_disc_style'].load_state_dict(models['disc_style'].state_dict())
            
            models['avg_disc_yhat'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape, true_disc=False
            )
            models['avg_disc_yhat'] = models['disc_yhat'].to(self.device)
            
        if self.is_train:
            models['disc'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen', 'cycle', 'disc',
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt' ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp' ]

        if self.consist_model is not None:
            losses += [ 'consist' ]
        # add prediction losses if auxiliary classifier - both forward and backward classification
        if self.models['disc_style'] is not None:
            losses += ['style']
            
        if self.models['disc_yhat'] is not None:
            losses += ['yhat']

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen.parameters(),
                self.models.disc_style.parameters(),
                self.models.disc_yhat.parameters(),
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None, cls_config = None,
        lambda_a        = 10.0,
        lambda_b        = 10.0,
        lambda_idt      = 0.5,
        lambda_consist  = 0,
        lambda_l_fwd    = 5.0,
        lambda_l_bwd    = 5.0,
        head_queue_size = 3,
        cls_queue_size  = 3,
        avg_momentum    = None,
        consistency     = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a  # cycle loss a
        self.lambda_b       = lambda_b  # cycle loss b
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.lambda_l_fwd   = lambda_l_fwd  # latent prediction loss a
        self.lambda_l_bwd   = lambda_l_bwd  # latent prediction loss b
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.cls_config     = cls_config or {}  # cls config head
        self.consist_model  = None

        if (lambda_consist > 0) and (consistency is not None):
            self.consist_model \
                = construct_consistency_model(consistency, device)

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = GANLoss(config.loss).to(self.device)
        self.criterion_cycle   = torch.nn.L1Loss()
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_consist = torch.nn.L1Loss()
        self.criterion_latent  = torch.nn.CrossEntropyLoss() # added latent criterion
        self.criterion_cls     = GANLoss(config.disc_clf_loss).to(self.device) # specific for class

        if self.is_train:
            # figure out queues after everything else
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_a', 'real_b', 'fake_a', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

    def _set_input(self, samples, domain):
        # set the inputs (images and labels) correctly
        try:
            inputs_a, inputs_b = samples[0], samples[1]
            inputs = [inputs_a[0], inputs_b[0]]
            labels = [inputs_a[1], inputs_b[1]]
        except:  # for eval, it's not paired so one input
            inputs = samples[0]
            labels = samples[1]
        
        set_two_domain_input(self.images, inputs, domain, self.device)
        set_two_domain_label(self.labels, labels, domain, self.device)

        if self.images.real_a is not None:
            if self.consist_model is not None:
                self.images.consist_real_a \
                    = self.consist_model(self.images.real_a)

        if self.images.real_b is not None:
            if self.consist_model is not None:
                self.images.consist_real_b \
                    = self.consist_model(self.images.real_b)

    def cycle_forward_image(self, real, real_label, gen, fake_label, disc_l, disc_yhat):
        # pylint: disable=no-self-use
        
        # (N, C, H, W)
        fake, fwd_mod_ft, fwd_y_hat_in, fwd_y_hat_out = gen(real)
        # detach and make new gradient for the cycle
        fake = fake.detach().clone().requires_grad_(True)
        reco, bwd_mod_ft, bwd_y_hat_in, bwd_y_hat_out = gen(fake)
        
        # after getting the mod features get the losses between them
        fwd_style = disc_l(fwd_mod_ft)
        bwd_style = disc_l(bwd_mod_ft)
        
        fwd_yhat = disc_yhat(fwd_y_hat_in)
        bwd_yhat = disc_yhat(bwd_y_hat_in)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)
        
        return (fake, reco, consist_fake, fwd_cls, bwd_cls, fwd_mod_ft, bwd_mod_ft)

    def idt_forward_image(self, real, real_label, gen, disc_l):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        idt, mod_ft = gen(real)
        cls = disc_l(mod_ft)
        return idt, cls, mod_ft

    def forward_dispatch(self, direction):
        if direction == 'ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b,
                self.latent_preds.pred_ab_fwd, self.latent_preds.pred_ab_bwd,  # latent predictions
                self.latent_feats.feat_ab_fwd, self.latent_feats.feat_ab_bwd   # latent features
            ) = self.cycle_forward_image(
                self.images.real_a, self.labels.real_a, self.models.gen, self.labels.real_b, self.models.disc_l, self.models.disc_yhat
            )

        elif direction == 'ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a,
                self.latent_preds.pred_ba_fwd, self.latent_preds.pred_ba_bwd,  # latent predictions
                self.latent_feats.feat_ba_fwd, self.latent_feats.feat_ba_bwd   # latent features
            ) = self.cycle_forward_image(
                self.images.real_b, self.labels.real_b, self.models.gen, self.labels.real_a, self.models.disc_l
            )

        elif direction == 'aa':
            # latent predictions
            self.images.idt_a, self.latent_preds.pred_aa_fwd, self.latent_feats.feats_aa_fwd = \
                self.idt_forward_image(self.images.real_a, self.labels.real_a, self.models.gen, self.models.disc_l)

        elif direction == 'bb':
            # latent predictions
            self.images.idt_b, self.latent_preds.pred_bb_fwd, self.latent_feats.feats_bb_fwd = \
                self.idt_forward_image(self.images.real_b, self.labels.real_b , self.models.gen, self.models.disc_l)

        elif direction == 'avg-ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b,
                self.latent_preds.pred_ab_fwd, self.latent_preds.pred_ab_bwd,  # latent predictions
                self.latent_feats.feat_ab_fwd, self.latent_feats.feat_ab_bwd   # latent features
            ) = self.cycle_forward_image(
                self.images.real_a, self.labels.real_a,
                self.models.avg_gen, self.labels.real_b, self.models.disc_l
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a,
                self.latent_preds.pred_ba_fwd, self.latent_preds.pred_ba_bwd,  # latent predictions
                self.latent_feats.feat_ba_fwd, self.latent_feats.feat_ba_bwd   # latent features
            ) = self.cycle_forward_image(
                self.images.real_b, self.labels.real_b,
                self.models.avg_gen, self.labels.real_a, self.models.disc_l
            )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ab')
            else:
                self.forward_dispatch(direction = 'ab')

        if self.images.real_b is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ba')
            else:
                self.forward_dispatch(direction = 'ba')

    def eval_consist_loss(
        self, consist_real_0, consist_fake_1, lambda_cycle_0
    ):
        return lambda_cycle_0 * self.lambda_consist * self.criterion_consist(
            consist_fake_1, consist_real_0
        )

    def eval_loss_of_cycle_forward(
        self, disc_1, real_0, fake_1, reco_0, fake_queue_1, lambda_cycle_0, 
        pred_ab_fwd, label_a, pred_ab_bwd, label_b, lambda_l_fwd, lambda_l_bwd
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_1, _ = queued_forward(
            disc_1, fake_1, fake_queue_1, update_queue = False
        )
        breakpoint()
        loss_gen   = self.criterion_gan(disc_pred_fake_1, True)
        loss_cycle = lambda_cycle_0 * self.criterion_cycle(reco_0, real_0)
        # make queue for style
        
        # style loss
        loss_style_fwd = lambda_l_fwd * self.criterion_latent(pred_ab_fwd, label_a) # added latent criteria
        loss_style_bwd = lambda_l_bwd * self.criterion_latent(pred_ab_bwd, label_b) # added latent criteria - not sure if needed
        # yhat loss
        loss_class_fwd = lambda_l_fwd * self.criterion_cls(pred_ab_fwd, label_a) # added latent criteria
        loss_class_bwd = lambda_l_bwd * self.criterion_cls(pred_ab_fwd, label_a) # added latent criteria
        
        loss_class = (loss_class_fwd + loss_class_bwd) * 0.5
        loss = loss_gen + loss_cycle + loss_class
        return (loss_gen, loss_cycle, loss_class, loss_class_fwd, loss_class_bwd, loss)

    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0, 
            pred_style, pred_cls, label, lambda_l_fwd,
    ):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)         
        )
        # latent prediction loss
        loss_idt_cls = (
              lambda_l_fwd
            * self.lambda_idt
            * self.criterion_latent(pred, label)
        )
        # yhat prediction loss
        loss_idt_yhat = (
              lambda_yhat_fwd
            * self.lambda_idt
            * self.criterion_yhat(pred, label)
        )

        loss = loss_idt + loss_idt_cls

        return (loss_idt, loss_idt_cls, loss)

    def backward_gen(self, direction):
        if direction == 'ab':
            (self.losses.gen_ab, self.losses.cycle_a, self.losses.pred_cls, self.losses.pred_ab_fwd, self.losses.pred_ab_bwd, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    self.queues.fake_b, self.lambda_a,
                    # latent vector predictions
                    self.latent_preds.pred_ab_fwd, self.labels.real_a, 
                    self.latent_preds.pred_ab_bwd, self.labels.real_b, 
                    self.lambda_l_fwd, self.lambda_l_bwd
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )

                loss += self.losses.consist_a

        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, self.losses.pred_cls, self.losses.pred_ba_fwd, self.losses.pred_ba_bwd, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b,
                    # latent vector predictions
                    self.latent_preds.pred_ba_fwd, self.labels.real_b, 
                    self.latent_preds.pred_ba_bwd, self.labels.real_a, 
                    self.lambda_l_fwd, self.lambda_l_bwd
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )

                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, self.losses.pred_aa, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a,
                    self.latent_preds.pred_aa_fwd, self.labels.real_a, self.lambda_l_fwd
                )

        elif direction == 'bb':
            (self.losses.idt_b, self.losses.pred_bb, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b, self.lambda_b,
                    self.latent_preds.pred_bb_fwd, self.labels.real_b, self.lambda_l_fwd
                )
        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

        loss.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()
        
        pred_real, pred_real_cls = queued_forward(
            model, real, queue_real, update_queue = True
        )
        
        loss_real = self.criterion_gan(pred_real, True)
        loss_real_cls = self.criterion_cls(pred_real_cls, self.labels.real_a)

        pred_fake, pred_fake_cls = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)
        loss_fake_cls = self.criterion_cls(pred_fake_cls, self.labels.real_b)

        real_fake_loss = (loss_real + loss_fake) * 0.5
        cls_loss = (loss_real_cls + loss_fake_cls) * 0.5
        
        loss = (real_fake_loss + cls_loss) * 0.5
        
        loss.backward()
        
        return (loss_gp, real_fake_loss, cls_loss, loss)

    def backward_discriminators(self):
        fake_a = self.images.fake_a.detach()
        fake_b = self.images.fake_b.detach()

        loss_gp_b, self.losses.disc_b_rf, self.losses.disc_b_cls, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a_rf, self.losses.disc_a_cls, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a
            )

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        dir_list = [ 'ab', 'ba' ]
        if self.lambda_idt > 0:
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen, self.models.gen, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()

