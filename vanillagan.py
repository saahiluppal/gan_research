import tensorflow as tf
import matplotlib.image as mplim
import time
import os

BUFFER_SIZE = 600_000
BATCH_SIZE = 128
EPOCHS = 1000
NOISE_DIM = 100
K = 2


def prepare_dataset():
    (data, _), (_, _) = tf.keras.datasets.mnist.load_data()
    batch = tf.shape(data)[0].numpy()
    data = data.reshape(batch, -1).astype('float32')
    data = (data - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset


class Generator(tf.keras.Model):
    def __init__(self, noise_dim=NOISE_DIM):
        super(Generator, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(noise_dim,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(784, activation='tanh'),
        ])

    def call(self, x):
        y_hat = self.model(x)
        return y_hat


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape=784):
        super(Discriminator, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape, )),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(1)
        ])

    def call(self, x):
        y_hat = self.model(x)
        return y_hat


def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(real_output), real_output
        )
    )

    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(fake_output), fake_output
        )
    )

    return real_loss + fake_loss


def generator_loss(fake_output):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(fake_output), fake_output
        )
    )


class VanillaGAN(object):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.dataset = prepare_dataset()

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4)

    @tf.function
    def train_discriminator(self, images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as tape:
            generated_images = self.generator(noise)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            loss = discriminator_loss(real_output, fake_output)

        gradient = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradient, self.discriminator.trainable_variables))

        return loss

    @tf.function
    def train_generator(self):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        with tf.GradientTape as tape:
            generated_images = self.generator(noise, training=True)

            fake_output = self.discriminator(generated_images)

            loss = generator_loss(fake_output)

        gradient = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables))

        return loss

    def train(self):
        write_dir = './images'
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   discriminator=self.discriminator)
        manager = tf.train.CheckpointManager(
            ckpt, './checkpoints', max_to_keep=3)

        for epoch in range(EPOCHS):
            start = time.time()

            for index, images in enumerate(self.dataset):
                d_loss = self.train_discriminator(images)
                if index % K == 0:
                    g_loss = self.train_generator()

        print(f'E: {epoch + 1}, G: {g_loss}, D: {d_loss}, T: {time.time() - start}')

        if (epoch + 1) % 10 == 0:
            generated_image = self.generator(
                tf.random.normal([1, NOISE_DIM]))
            generated_image = tf.reshape(generated_image[0], (28, 28))
            mplim.imsave(os.path.join(
                write_dir, f'{epoch + 1}.png'), generated_image.numpy())

            manager.save()
            print('Checkpoint Saved...')
