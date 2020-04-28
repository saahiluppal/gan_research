import tensorflow as tf

_all_losses = [
    'minimax_discriminator_loss',
    'minimax_generator_loss',
    'tf_minimax_discriminator_loss',
    'tf_minimax_generator_loss'
]

# Minimax Custom Implementation ->>


def minimax_discriminator_loss(real_outputs,
                               generated_outputs,
                               real_weights=1.,
                               generated_weights=1.,
                               smoothing=0.25):
    """
    Args:
    real_outputs: Discriminator output on real data.
    generated_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.

     L = real_weights * - log(sigmoid(D(x)))
        generated_weights * - log(1 - sigmoid(D(G(z))))

    """
    if smoothing > 0:
        real_outputs = (real_outputs * (1 - smoothing) +
                        0.5 * smoothing)

    loss_on_real = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.ones_like(real_outputs),
            logits=real_outputs,
            pos_weight=real_weights,
        )
    )

    loss_on_generated = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.zeros_like(generated_outputs),
            logits=generated_outputs,
            pos_weight=generated_weights,
        )
    )

    loss = loss_on_real + loss_on_generated
    return tf.reduce_mean(loss)


def minimax_generator_loss(generated_outputs,
                           weights=1.,
                           smoothing=0.):
    """
    Args:
    generated_outputs: Discriminator output on generated data. Expected 
        to be in range of (-inf, inf)
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to
        `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
    smoothing: The amount of smoothing for positive labels.

    L = log(sigmoid(D(x))) + log(1 - sigmoid(D(G(z))))
    """
    loss = - minimax_discriminator_loss(
        tf.ones_like(generated_outputs),
        generated_outputs,
        weights,
        weights,
        smoothing
    )

    return tf.reduce_mean(loss)

# Minimax Tensorflow's Implementation ->>


def tf_minimax_discriminator_loss(real_outputs,
                                  generated_outputs,
                                  smoothing=0.25,
                                  real_weights=1.0,
                                  generated_weights=1.0,
                                  reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """
    Args:
    real_outputs: Discriminator output on real data.
    generated_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    reduction: A `tf.losses.Reduction` to apply to loss.

     L = real_weights * - log(sigmoid(D(x)))
        generated_weights * - log(1 - sigmoid(D(G(z))))

    """
    loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_outputs),
        logits=real_outputs,
        weights=real_weights,
        label_smoothing=smoothing,
        reduction=reduction,
    )

    loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_outputs),
        logits=generated_outputs,
        weights=generated_weights,
        reduction=reduction,
    )

    loss = loss_on_real + loss_on_generated
    return loss


def tf_minimax_generator_loss(generated_outputs,
                              smoothing=0.,
                              weights=1.,
                              reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """
    Args:
    generated_outputs: Discriminator output on generated data. Expected 
        to be in range of (-inf, inf)
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to
        `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
    smoothing: The amount of smoothing for positive labels.
    reduction: A `tf.losses.Reduction` to apply to loss.

    L = log(sigmoid(D(x))) + log(1 - sigmoid(D(G(z))))
    """
    loss = - tf_minimax_discriminator_loss(
        tf.ones_like(generated_outputs),
        generated_outputs,
        smoothing,
        weights,
        weights,
        reduction,
    )

    return loss