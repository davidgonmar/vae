import jax
import jax.numpy as jnp
import flax.linen as nn


class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mean, logvar


class Decoder(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(256)(z)
        z = nn.relu(z)
        z = nn.Dense(512)(z)
        z = nn.relu(z)
        z = nn.Dense(self.output_dim)(z)
        return z


class VAE(nn.Module):
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mean, logvar = Encoder(self.latent_dim)(x)
        z = mean + jax.random.normal(self.make_rng(), mean.shape) * jnp.exp(
            0.5 * logvar
        )
        x_recon = Decoder(self.output_dim)(z)
        return x_recon, mean, logvar
