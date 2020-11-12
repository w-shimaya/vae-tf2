import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L


class SamplingGaussian(L.Layer):
    """reparameterization trick"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        normal = tf.random.normal(tf.shape(z_mean))
        return z_mean + normal * tf.exp(0.5 * z_log_var)


class Encoder(L.Layer):
    def __init__(self, latent_dim, hidden_units, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # define weights/layers
        self.projection = L.Dense(hidden_units, activation="relu")
        self.dense_z_mean = L.Dense(latent_dim)
        self.dense_z_log_var = L.Dense(latent_dim)

    def call(self, inputs):
        x = self.projection(inputs)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        return z_mean, z_log_var


class Decoder(L.Layer):
    def __init__(self, output_dim, hidden_units, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.projection = L.Dense(hidden_units, activation="relu")
        self.dense_out = L.Dense(output_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.projection(inputs)
        o = self.dense_out(x)
        return o


class VAE(K.Model):
    """'Vanilla' Variational Autoencoder.

    reference: https://www.tensorflow.org/guide/keras/custom_layers_and_models

    made some changes to the original codes to make applying the model to
    different datasets or adding new features easy (and just to fit my preference).
    """

    def __init__(self, data_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.enc = Encoder(latent_dim, 512)
        self.dec = Decoder(data_dim, 512)
        self.sample = SamplingGaussian()

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = self.enc(inputs)
        z_sample = self.sample((z_mean, z_log_var))
        reconstruction = self.dec(z_sample)

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1.
        )
        self.add_loss(kl_loss)
        return reconstruction
