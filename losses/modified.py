import tensorflow as tf

_all_losses = [
    'modified_discriminator_loss',
    'modified_generator_loss',
    'tf_modified_discriminator_loss',
    'tf_modified_generator_loss'
]

# Modified Custom Implementation ->>


def modified_discriminator_loss(real_outputs,
                                generated_outputs,
                                real_weights=1.0,
                                generated_weights=1.0,
                                smoothing=0.25):
    """
    Same as minimax discriminator loss.

    Args:
    real_outputs: Discriminator output on real data.
    generated_outputs: Discriminator output on generated data. Expected
        to be in range of (-inf, inf)
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
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


def modified_generator_loss(generated_outputs,
                            smoothing=0.0,
                            weights=1.0):
    """
    Args:
    generated_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to `labels` (i.e.,
      all dimensions must be either `1`, or the same as the corresponding
      dimension).

    L = - log(sigmoid(D(G(z))))
    """
    if smoothing > 0:
        generated_outputs = (generated_outputs * (1 - smoothing) +
                             0.5 * smoothing)

    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.ones_like(generated_outputs),
        logits=generated_outputs,
        pos_weight=weights,
    )

    return tf.reduce_mean(loss)


def tf_modified_discriminator_loss(real_outputs,
                                   generated_outputs,
                                   smoothing=0.25,
                                   real_weights=1.0,
                                   generated_weights=1.0,
                                   reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """
    Same as tf minimax discriminator loss
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


def tf_modified_generator_loss(generated_outputs,
                               smoothing=0.0,
                               weights=1.0,
                               reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """
    Args:
    generated_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to `labels` (i.e.,
      all dimensions must be either `1`, or the same as the corresponding
      dimension).
    reduction: A `tf.losses.Reduction` to apply to loss.

    L = -log(sigmoid(D(G(z))))
    """
    loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(generated_outputs),
        logits=generated_outputs,
        weights=weights,
        label_smoothing=smoothing,
        reduction=reduction,
    )
    return loss