"""
VAE-GMM models with size-based encoder/decoder variants.
"""

DATASET_TO_SIZE = {
    'mnist': '28x28',
    'cardiac': '128x128',
    'oasis': '128x128',
}


def get_model(config):
    """
    Get VAE-GMM model with appropriate encoder/decoder for the dataset.

    Args:
        config: Configuration dict containing dataset info

    Returns:
        GMMVAE model instance
    """
    from .model import GMMVAE

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

    model = GMMVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config['model']['latent_dim'],
        num_clusters=config['model']['num_clusters']
    )
    return model
