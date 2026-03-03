"""
Diffusion VAE-GMM models with latent-space denoising.
Encoder/Decoder are local copies (standalone, no cross-model imports).
"""

DATASET_TO_SIZE = {
    'mnist': '28x28',
    'cardiac': '128x128',
    'oasis': '128x128',
}


def get_model(config):
    """
    Get Diffusion VAE-GMM model with appropriate encoder/decoder for the dataset.

    Args:
        config: Configuration dict containing dataset and diffusion info

    Returns:
        DiffusionGMMVAE model instance
    """
    from .model import DiffusionGMMVAE

    dataset_name = config['dataset']['name']
    size = DATASET_TO_SIZE.get(dataset_name)
    if size is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if size == '28x28':
        from .encoder_28x28 import Encoder
        from .decoder_28x28 import Decoder
    elif size == '128x128':
        from .encoder_128x128 import Encoder
        from .decoder_128x128 import Decoder

    encoder = Encoder(latent_dim=config['model']['latent_dim'])
    decoder = Decoder(latent_dim=config['model']['latent_dim'])

    diffusion_config = config.get('diffusion', {})

    model = DiffusionGMMVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config['model']['latent_dim'],
        num_clusters=config['model']['num_clusters'],
        diffusion_config=diffusion_config,
    )
    return model
