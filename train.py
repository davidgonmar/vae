import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, orbax_utils
import tensorflow_datasets as tfds
import numpy as np
from model import VAE
import flax.linen as nn
import functools
import orbax.checkpoint


def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def binary_cross_entropy(x, x_recon):
    return jnp.sum(optax.sigmoid_binary_cross_entropy(x_recon, x))


def loss_fn(params, fn, x, rng):
    rng, return_rng = jax.random.split(rng)
    x_recon, mean, logvar = fn({"params": params}, x, rngs={"params": rng})
    recon_loss = binary_cross_entropy(x, x_recon)
    kl_loss = kl_divergence(mean, logvar)
    return recon_loss + kl_loss, return_rng


@jax.jit
def train_step(state: train_state.TrainState, x, rng):
    (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.apply_fn, x, rng
    )
    return state.apply_gradients(grads=grads), loss, rng


@functools.partial(jax.jit, static_argnums=(1,))
def eval_step(params, vae, x, rng):
    rng, return_rng = jax.random.split(rng)
    x_recon, mean, logvar = vae.apply({"params": params}, x, rngs={"params": rng})
    recon_loss = binary_cross_entropy(x, x_recon)
    kl_loss = kl_divergence(mean, logvar)
    return recon_loss + kl_loss, return_rng


def create_train_state(rng, vae, learning_rate, decay_steps, decay_rate):
    rng, init_rng = jax.random.split(rng)
    params = vae.init(init_rng, jnp.ones([1, 28 * 28]))["params"]
    schedule_fn = optax.exponential_decay(
        init_value=learning_rate, transition_steps=decay_steps, decay_rate=decay_rate
    )
    tx = optax.adam(learning_rate=schedule_fn)
    return train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=tx)


ds_builder = tfds.builder("mnist")
ds_builder.download_and_prepare()
train_ds = tfds.as_numpy(
    ds_builder.as_dataset(split="train", batch_size=96, shuffle_files=True)
)
test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=128))

vae = VAE(latent_dim=20, output_dim=28 * 28)
rng = jax.random.PRNGKey(0)
state = create_train_state(
    rng, vae, learning_rate=1e-4, decay_steps=1000, decay_rate=0.99
)

for epoch in range(100):
    for batch in train_ds:
        images = batch["image"].reshape(-1, 28 * 28) / 255.0
        state, loss, rng = train_step(state, images, rng)
    test_loss = 0
    total = 0
    for batch in test_ds:
        images = batch["image"].reshape(-1, 28 * 28) / 255.0
        loss, rng = eval_step(state.params, vae, images, rng)
        test_loss += loss
        total += 1
    print(f"Epoch {epoch}, Test Loss: {test_loss / total}")

    """ckpt = state
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('./vae.ckpt', save_args)
    """
