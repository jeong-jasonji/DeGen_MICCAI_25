from uvcgan2.base.networks import select_base_discriminator
from uvcgan2.models.funcs  import default_model_init

def select_discriminator(name, **kwargs):
    return select_base_discriminator(name, **kwargs)

def construct_discriminator(model_config, image_shape, device):
    if model_config.model == 'latent':
        model = select_discriminator(
            model_config.model,
            **model_config.model_args,
        )
    else:
        model = select_discriminator(
            model_config.model, image_shape = image_shape,
            **model_config.model_args,
        )

    return default_model_init(model, model_config, device)

