# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torchvision.transforms import GaussianBlur, Resize

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
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

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

class UVCGAN2_Decision_s2(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a_t', 'fake_b_t', # targets
            'reco_a_t', 'reco_b_t', # reconstruct inputs
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
        ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a_t', 'idt_b_t']

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
            'feat_ab_fwd_input', 'feat_ab_fwd_target', 
            'feat_ba_fwd_input', 'feat_ba_fwd_target',
            'feat_ab_bwd_input', 'feat_ab_bwd_target', 
            'feat_ba_bwd_input', 'feat_ba_bwd_target', 
        ]
        
        if self.is_train and self.lambda_idt > 0:
            latent_feats += [ 
                'feat_aa_t', 'feat_aa_i', 
                'feat_bb_t', 'feat_bb_i', 
                ]

        return NamedDict(*latent_feats)
        
    def _setup_latent_imgs(self, _config):
        latent_imgs = [
            'y_hat_ab_fwd', 'y_hat_ab_bwd',
            'y_hat_ba_fwd', 'y_hat_ba_bwd',  
        ]
        
        if self.is_train and self.lambda_idt > 0:
            latent_imgs += [ 'y_hat_aa_fwd', 'y_hat_bb_fwd', ]

        return NamedDict(*latent_imgs)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        
        models['gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )
            
        models['disc_l'] = nn.Linear(config.l_discriminator.model_args['latent_dim'], config.l_discriminator.model_args['n_classes'])
        models['disc_l'] = models['disc_l'].to(self.device)

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            
            models['avg_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['avg_gen_ba'].load_state_dict(models['gen_ba'].state_dict())
            models['avg_disc_l'] = nn.Linear(config.l_discriminator.model_args['latent_dim'],config.l_discriminator.model_args['n_classes'])
            models['avg_disc_l'] = models['avg_disc_l'].to(self.device)
            models['avg_disc_l'].load_state_dict(models['disc_l'].state_dict())
      
        # load the discriminators regardless of train/eval
        models['disc_a'] = self._construct_batch_head_disc(
            config.discriminator, config.data.datasets[0].shape
        )
        
        models['disc_b'] = self._construct_batch_head_disc(
            config.discriminator, config.data.datasets[1].shape
        )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b',
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp_a', 'gp_b' ]

        if self.consist_model is not None:
            losses += [ 'consist_a', 'consist_b' ]
        # add prediction losses if auxiliary classifier - both forward and backward classification
        if self.models['disc_l'] is not None:
            losses += ['loss_ab_ft', 'loss_ba_ft',
                      'loss_aa_ft', 'loss_bb_ft']

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc', 'disc_l')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters(),
            ),
            config.discriminator.optimizer
        )
        
        optimizers.disc_l = select_optimizer(
            itertools.chain(
                self.models.disc_l.parameters()  # added latent discriminator optimization
            ),
            config.l_discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_a        = 10.0,
        lambda_b        = 10.0,
        lambda_idt      = 0.5,
        lambda_consist  = 0,
        lambda_ft_reco    = 5.0,
        head_queue_size = 3,
        avg_momentum    = None,
        consistency     = None,
        eval_cycle_mode = 'vanilla',
        softmax_out     = 0
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a  # cycle loss a
        self.lambda_b       = lambda_b  # cycle loss b
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.lambda_ft_reco   = lambda_ft_reco  # input output loss
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.consist_model  = None
        self.eval_cycle_mode= eval_cycle_mode
        self.push_decision  = False

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
        self.criterion_latent  = torch.nn.BCEWithLogitsLoss() # added latent criterion

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_a', 'real_b', 
                                  'fake_a_t', 'fake_b_t']
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

    def cycle_forward_image(self, real, label, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use
        
        # (N, C, H, W)
        fake_target, fwd_target, fwd_input = gen_fwd(real)
        reco_target, bwd_target, bwd_input = gen_bkw(fake_target)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)
        
        return (fake_target, fwd_target, fwd_input, reco_target, bwd_target, bwd_input)

    def idt_forward_image(self, real, gen):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        idt_target, fwd_target, fwd_input = gen(real)
        
        return idt_target, fwd_target, fwd_input

    def forward_dispatch(self, direction):
        if direction == 'ab':
            (
                self.images.fake_b_t, self.latent_feats.feat_ab_fwd_target, self.latent_feats.feat_ab_fwd_input,
                self.images.reco_a_t, self.latent_feats.feat_ba_bwd_target, self.latent_feats.feat_ba_bwd_input
                
            ) = self.cycle_forward_image(
                self.images.real_a, self.labels.real_a, self.models.gen_ab, self.models.gen_ba
            )

        elif direction == 'ba':
            (
                self.images.fake_a_t, self.latent_feats.feat_ba_fwd_target, self.latent_feats.feat_ba_fwd_input,
                self.images.reco_b_t, self.latent_feats.feat_ab_bwd_target, self.latent_feats.feat_ab_bwd_input
                
            ) = self.cycle_forward_image(
                self.images.real_b, self.labels.real_b, self.models.gen_ba, self.models.gen_ab
            )

        elif direction == 'aa':
            # latent predictions
            (   self.images.idt_a_t, self.latent_feats.feat_aa_t, self.latent_feats.feat_aa_i
            ) = self.idt_forward_image(
                self.images.real_a, self.models.gen_ba
            )

        elif direction == 'bb':
            # latent predictions
            (   self.images.idt_b_t, self.latent_feats.feat_bb_t, self.latent_feats.feat_bb_i
            ) = self.idt_forward_image(
                self.images.real_b, self.models.gen_ab
            )

        elif direction == 'avg-ab':
            (
                self.images.fake_b_t, self.latent_feats.feat_ab_fwd_target, self.latent_feats.feat_ab_fwd_input,
                self.images.reco_a_t, self.latent_feats.feat_ba_bwd_target, self.latent_feats.feat_ba_bwd_input
            ) = self.cycle_forward_image(
                self.images.real_a, self.labels.real_a,
                self.models.avg_gen_ab, self.models.avg_gen_ba
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a_t, self.latent_feats.feat_ba_fwd_target, self.latent_feats.feat_ba_fwd_input,
                self.images.reco_b_t, self.latent_feats.feat_ab_bwd_target, self.latent_feats.feat_ab_bwd_input
            ) = self.cycle_forward_image(
                self.images.real_b, self.labels.real_b,
                self.models.avg_gen_ba, self.models.avg_gen_ab
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
        ab_fwd_t, ab_fwd_i, ba_bwd_t, ba_bwd_i, lambda_ft_reco
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_1 = queued_forward(
            disc_1, fake_1, fake_queue_1, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_1, True)
        loss_cycle = lambda_cycle_0 * self.criterion_cycle(reco_0, real_0)
        # feature reconstruction loss
        loss_a_ft = lambda_ft_reco * self.criterion_cycle(ab_fwd_t, ba_bwd_i)
        loss_b_ft = lambda_ft_reco * self.criterion_cycle(ba_bwd_t, ab_fwd_i)
        loss_ft = (loss_a_ft + loss_b_ft) * 0.5
        
        loss = loss_gen + loss_cycle + loss_ft
            
        return (loss_gen, loss_cycle, loss_ft, loss)

    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0, 
            xx_t, xx_i, lambda_ft_reco,
    ):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)         
        )
        
        # latent similarity loss
        loss_idt_ft = (
              lambda_ft_reco
            * self.criterion_idt(xx_t, xx_i)
        )

        loss = loss_idt + loss_idt_ft

        return (loss_idt, loss_idt_ft, loss)

    def backward_gen(self, direction):
        if direction == 'ab':
            (self.losses.gen_ab, self.losses.cycle_a, self.losses.loss_ab_ft, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_b,
                    self.images.real_a, self.images.fake_b_t, self.images.reco_a_t,
                    self.queues.fake_b_t, self.lambda_a,
                    # feature similarity
                    self.latent_feats.feat_ab_fwd_target, self.latent_feats.feat_ab_fwd_input,
                    self.latent_feats.feat_ba_bwd_target, self.latent_feats.feat_ba_bwd_input,
                    self.lambda_ft_reco
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )

                loss += self.losses.consist_a

        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, self.losses.loss_ba_ft, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a_t, self.images.reco_b_t,
                    self.queues.fake_a_t, self.lambda_b,
                    # feature similarity
                    self.latent_feats.feat_ba_fwd_target, self.latent_feats.feat_ba_fwd_input,
                    self.latent_feats.feat_ab_bwd_target, self.latent_feats.feat_ab_bwd_input,
                    self.lambda_ft_reco
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )

                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, self.losses.loss_aa_ft, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a_t, self.lambda_a,
                    self.latent_feats.feat_aa_t, self.latent_feats.feat_aa_i, self.lambda_ft_reco
                )

        elif direction == 'bb':
            (self.losses.idt_b, self.losses.loss_bb_ft, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b_t, self.lambda_b,
                    self.latent_feats.feat_bb_t, self.latent_feats.feat_bb_i, self.lambda_ft_reco
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

        pred_real = queued_forward(
            model, real, queue_real, update_queue = True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()
        
        return (loss_gp, loss)

    # backward_discriminator_base - for latent features
    def backward_l_discriminator_base(
        self, model, real_a, feat_ab_fwd, real_b, feat_ba_fwd,
    ):
        
        onehot_b = nn.functional.one_hot(real_b, num_classes=2).float()
        onehot_a = nn.functional.one_hot(real_a, num_classes=2).float()
        
        # forward class prediction loss
        pred_ab_fwd = model.forward(feat_ab_fwd)
        loss_ab_fwd = self.criterion_latent(pred_ab_fwd, onehot_b)
        
        pred_ba_fwd = model.forward(feat_ba_fwd)
        loss_ba_fwd = self.criterion_latent(pred_ba_fwd, onehot_a)
        
        loss = (loss_ab_fwd + loss_ba_fwd) * 0.5
        loss.backward()
        
        return loss

    def backward_discriminators(self):
        fake_a = self.images.fake_a_t.detach()
        fake_b = self.images.fake_b_t.detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b_t
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc_a, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a_t
            )

        if loss_gp_a is not None:
            self.losses.gp_a = loss_gp_a
        
        # compare against real and fake labels
        feat_ab_fwd_target = self.latent_feats.feat_ab_fwd_target.detach()
        feat_ba_fwd_target = self.latent_feats.feat_ba_fwd_target.detach()
        
        self.losses.disc_l = \
            self.backward_l_discriminator_base(
                self.models.disc_l,
                self.labels.real_a, feat_ab_fwd_target,
                self.labels.real_b, feat_ba_fwd_target,
            )

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b, self.models.disc_l], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        dir_list = [ 'ab', 'ba' ]
        if self.lambda_idt > 0:
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)
        # optimize discriminator
        self.set_requires_grad([self.models.disc_l], True)
        self.optimizers.disc_l.zero_grad(set_to_none = True) # disc_l

        self.backward_discriminators()

        self.optimizers.disc.step()
        self.optimizers.disc_l.step() # disc_l

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )
        update_average_model(
            self.models.avg_gen_ba, self.models.gen_ba, self.avg_momentum
        )
        update_average_model(
            self.models.avg_disc_l, self.models.disc_l, self.avg_momentum
        ) # disc_l

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()

