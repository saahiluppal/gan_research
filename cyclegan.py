import tensorflow as tf
import time
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 420_000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10
EPOCHS = 100


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess(image, label):
    image = random_jitter(image)
    image = normalize(image)

    return image


def create_dataset():
    dataset, _ = tfds.load('cycle_gan/horse2zebra',
                           with_info=True, as_supervised=True)
    horses, zebras = dataset['trainA'], dataset['trainB']

    horses = horses.map(preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    zebras = zebras.map(preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    return horses, zebras


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(output_channels, norm_type='batchnorm'):
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),
        downsample(128, 4, norm_type),
        downsample(256, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type, apply_dropout=True),
        upsample(512, 4, norm_type),
        upsample(256, 4, norm_type),
        upsample(128, 4, norm_type),
        upsample(64, 4, norm_type),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate(
            [inp, tar])

    down1 = downsample(64, 4, norm_type, False)(x)
    down2 = downsample(128, 4, norm_type)(down1)
    down3 = downsample(256, 4, norm_type)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


LOSS_OBJ = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = LOSS_OBJ(tf.ones_like(real), real)
    generated_loss = LOSS_OBJ(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return LOSS_OBJ(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


class CycleGAN(object):
    def __init__(self):
        self.horses, self.zebras = create_dataset()
        self.generator_g = unet_generator(3, norm_type='instancenorm')
        self.generator_f = unet_generator(3, norm_type='instancenorm')

        self.discriminator_x = discriminator(
            norm_type='instancenorm', target=False)
        self.discriminator_y = discriminator(
            norm_type='instancenorm', target=False)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(
                real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + \
                total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + \
                total_cycle_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

    def train(self):
        for epoch in range(EPOCHS):
            start = time.time()

            for (images, target) in zip(self.horses, self.zebras):
                loss = self.train_step(images, target)
                print('.', end='')

            print(f'E: {epoch + 1}, L: {loss}, T: {time.time() - start}')


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train()
