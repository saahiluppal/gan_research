import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os

NOISE_DIM = 100
BATCH_SIZE = 256
BUFFER_SIZE = 100
EPOCHS = 20000
NUM_CLASSES = 10
K = 1


def prepare_dataset():
    (data, labels), (_, _) = tf.keras.datasets.mnist.load_data()
    batch = tf.shape(data)[0].numpy()
    data = data.reshape(batch, -1).astype('float32')
    data = (data - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    return dataset


class Generator(tf.keras.Model):
    def __init__(self, noise_dim=NOISE_DIM, num_classes=NUM_CLASSES):
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

        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_classes,
            output_dim=noise_dim,
        )

    def call(self, x, y):
        y = self.embedding(y)
        z = tf.multiply(x, y)
        y_hat = self.model(z)
        return y_hat


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape=784, num_classes=NUM_CLASSES):
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

        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_classes,
            output_dim=input_shape,
        )

    def call(self, x, y):
        y = self.embedding(y)
        x = tf.multiply(x, y)
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


class CGAN(object):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.dataset = prepare_dataset()
        self.write_dir = './images'
        self.checkpoint_dir = './checkpoints'

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4)

    @tf.function
    def train_discriminator(self, images, labels):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, labels)

            real_output = self.discriminator(images, labels, training=True)
            fake_output = self.discriminator(
                generated_images, labels, training=True)

            loss = discriminator_loss(real_output, fake_output)

        gradient = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradient, self.discriminator.trainable_variables))

        return loss

    @tf.function
    def train_generator(self, labels):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, labels, training=True)

            fake_output = self.discriminator(generated_images, labels)

            loss = generator_loss(fake_output)

        gradient = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables))

        return loss

    def train(self):
        if not os.path.exists(self.write_dir):
            os.mkdir(self.write_dir)

        self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                        discriminator=self.discriminator)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=3)

        for epoch in range(EPOCHS):
            start = time.time()

            for index, (images, labels) in enumerate(self.dataset):
                d_loss = self.train_discriminator(images, labels)
                if index % K == 0:
                    g_loss = self.train_generator(labels)

            print(
                f'E: {epoch + 1}, G: {g_loss}, D: {d_loss}, T: {time.time() - start}')
            if (epoch + 1) % 100 == 0:
                self.checkpoint_and_save(epoch + 1)

    def checkpoint_and_save(self, epoch):
        r, c = 2, 5
        noise = tf.random.normal([r * c, NOISE_DIM])
        labels = tf.range(10)

        generated_images = self.generator(noise, labels)
        generated_images = 0.5 * generated_images + 0.5

        fig, ax = plt.subplots(r, c)
        count = 0

        for i in range(2):
            for j in range(5):
                ax[i, j].imshow(tf.reshape(
                    generated_images[count], (28, 28)), cmap='gray')
                ax[i, j].set_title(f'Digit {count}')
                ax[i, j].axis('off')
                count += 1

        fig.savefig(f"images/{epoch}.png")
        plt.close()

        self.manager.save()
        print('Checkpoint and PNG Saved...')


if __name__ == '__main__':
    gan = CGAN()
    gan.train()