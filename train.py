import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2
from vae import VAE
import argparse
import os


class MyCallback(K.callbacks.Callback):
    def __init__(self):
        super(MyCallback, self).__init__()


def preprocess_mnist(x):
    x = tf.reshape(x["image"], (-1, ))
    x = tf.cast(x, tf.float32) / 255.
    return (x, x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", "-l", type=int, required=True)
    parser.add_argument("--learning-rate", "-r", type=float, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    # load mnist
    train_ds = tfds.load("mnist", split="train", shuffle_files=True)
    test_ds = tfds.load("mnist", split="test", shuffle_files=False)
    data_dim = 28 * 28

    # preprocess (cast and normalize)
    train_ds = train_ds.map(preprocess_mnist)

    # prepare dataset
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(60000)
    train_ds = train_ds.batch(128)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # VAE
    model = VAE(data_dim=data_dim, latent_dim=args.latent_dim)

    # prepare tensorboard logging
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=args.logdir,
                                                   update_freq="batch")

    # use built-in training loop
    optimizer = K.optimizers.Adam(learning_rate=args.learning_rate)
    # model.compile(optimizer, K.losses.BinaryCrossentropy())
    model.compile(
        optimizer, K.losses.MeanSquaredError(
            reduction=K.losses.Reduction.SUM))
    model.fit(train_ds, epochs=5, callbacks=[tensorboard_callback])

    # analysis/evaluation
    rng = 1.5
    nums = 10

    # latent travarsal analysis
    #
    # --- uncomment for 2dim latents ---
    lin = tf.linspace(-rng * tf.ones(nums), rng * tf.ones(nums), nums, axis=0)
    x = tf.reshape(lin, [nums, nums, 1])
    y = tf.reshape(tf.transpose(lin), [nums, nums, 1])
    z = tf.concat([x, y], axis=-1)
    z = tf.reshape(z, [nums * nums, -1])

    # --- uncomment for n-dim latents ---
    # z = tf.random.normal((nums * nums, 32)) * rng

    # --- common ---
    img = model.dec(z)

    img = tf.reshape(img, (-1, 28, 28, 1)).numpy()

    tile = np.empty((1, nums * 28, nums * 28, 1))
    for i in range(nums * nums):
        x = i // nums
        y = i % nums
        tile[0, x * 28:(x + 1) * 28, y * 28:(y + 1) * 28, :] = img[i, :, :, :]

    # save image
    cv2.imwrite(os.path.join(args.logdir, "traverse.png"),
                tile[0, :, :, :] * 255.)

    # embed test images in the latent space
    x_list = [[], [], [], [], [], [], [], [], [], []]
    y_list = [[], [], [], [], [], [], [], [], [], []]
    for x in test_ds:
        img = x["image"]
        label = x["label"]
        img = tf.cast(img[tf.newaxis, :, :, :], tf.float32) / 255.
        img = tf.reshape(img, [1, 28 * 28])
        z, _ = model.enc(img)  # use means as representative points
        # 2-dim
        x_list[label.numpy()].append(z[0, 0])
        y_list[label.numpy()].append(z[0, 1])
        # TODO: n-dim -> dimension reduction like PCA

    # plot the latent space
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    color = ["red", "green", "blue", "cyan", "magenta",
             "yellow", "black", "gray", "orange", "purple"]
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        ax.scatter(np.array(x), np.array(y),
                   marker=".", c=color[i], label=str(i))

    ax.legend()
    plt.savefig(os.path.join(args.logdir, "embed.png"))
