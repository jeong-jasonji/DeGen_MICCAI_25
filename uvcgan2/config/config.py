import json
import logging
import os

from uvcgan2.consts    import CONFIG_NAME

from .config_base     import ConfigBase
from .data_config     import parse_data_config
from .model_config    import ModelConfig, ClfModelConfig
from .transfer_config import TransferConfig

LOGGER = logging.getLogger('uvcgan2.config')

class Config(ConfigBase):
    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'batch_size',
        'architecture', # architecture
        'data',
        'epochs',
        'discriminator',
        'l_discriminator', # added latent discriminator
        'generator',
        'model',
        'model_args',
        'loss',
        'disc_clf_loss', # added
        'gradient_penalty',
        'seed',
        'scheduler',
        'steps_per_epoch',
        'transfer',
        'eval_cycle_mode', # added eval_cycle_mode
    ]

    def __init__(
        self,
        batch_size       = 32,
        architecture     = 'star',
        data             = None,
        data_args        = None,
        epochs           = 100,
        image_shape      = None,
        discriminator    = None,
        l_discriminator  = None,
        generator        = None,
        model            = 'cyclegan',
        model_args       = None,
        loss             = 'lsgan',
        disc_clf_loss    = 'clf',
        gradient_penalty = None,
        seed             = 0,
        scheduler        = None,
        steps_per_epoch  = 250,
        transfer         = None,
        workers          = None,
        eval_cycle_mode  = 'vanilla',
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.data = parse_data_config(data, data_args, image_shape, workers)

        self.batch_size      = batch_size
        self.architecture    = architecture
        self.model           = model
        self.model_args      = model_args or {}
        self.seed            = seed
        self.loss            = loss
        self.disc_clf_loss   = disc_clf_loss
        self.epochs          = epochs
        self.scheduler       = scheduler
        self.steps_per_epoch = steps_per_epoch

        if discriminator is not None:
            discriminator = ModelConfig(**discriminator)
        
        if l_discriminator is not None:
            l_discriminator = ClfModelConfig(**l_discriminator)

        if generator is not None:
            generator = ModelConfig(**generator)

        if gradient_penalty is True:
            gradient_penalty = {}

        if transfer is not None:
            if isinstance(transfer, list):
                transfer = [ TransferConfig(**conf) for conf in transfer ]
            else:
                transfer = TransferConfig(**transfer)

        self.discriminator    = discriminator
        self.l_discriminator  = l_discriminator
        self.generator        = generator
        self.gradient_penalty = gradient_penalty
        self.transfer         = transfer
        self.eval_cycle_mode  = eval_cycle_mode

        Config._check_deprecated_args(image_shape, workers)

        if image_shape is not None:
            self._validate_image_shape(image_shape)

    @staticmethod
    def _check_deprecated_args(image_shape, workers):
        if image_shape is not None:
            LOGGER.warning(
                "Deprecation Warning: Deprecated `image_shape` configuration "
                "parameter detected."
            )

        if workers is not None:
            LOGGER.warning(
                "Deprecation Warning: Deprecated `workers` configuration "
                "parameter detected."
            )

    def _validate_image_shape(self, image_shape):
        assert all(d.shape == image_shape for d in self.data.datasets), (
            f"Value of the deprecated `image_shape` parameter {image_shape}"
            f"does not match shapes of the datasets."
        )

    def get_savedir(self, outdir, label = None):
        if label is None:
            label = self.get_hash()

        discriminator = None
        if self.discriminator is not None:
            discriminator = self.discriminator.model

        generator = None
        if self.generator is not None:
            generator = self.generator.model

        savedir = 'model_m(%s)_d(%s)_g(%s)_%s' % (
            self.model, discriminator, generator, label
        )

        savedir = savedir.replace('/', ':')
        path    = os.path.join(outdir, savedir)

        os.makedirs(path, exist_ok = True)
        return path

    def save(self, path):
        # pylint: disable=unspecified-encoding
        with open(os.path.join(path, CONFIG_NAME), 'wt') as f:
            f.write(self.to_json(sort_keys = True, indent = '    '))

    @staticmethod
    def load(path):
        # pylint: disable=unspecified-encoding
        with open(os.path.join(path, CONFIG_NAME), 'rt') as f:
            return Config(**json.load(f))

