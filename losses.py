import tensorflow as tf
from tensorflow.python.framework import smart_cond

_all_losses = [
    'minimax_discriminator_loss',
    'minimax_generator_loss',
]

# TODO
# Test smoothing here


def minimax_discriminator_loss(real_outputs,
                    generated_outputs,
                    real_weights=1.0,
                    generated_weights=1.0,
                    smoothing=0.25):
    """
    Args:
    real_outputs: Discriminator output on real data
    generated_outputs: Discriminator output on generated data. Expected 
        to be in range of (-inf, inf)
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `real_data`, and must be broadcastable to `real_data` (i.e., all
        dimensions must be either `1`, or the same as the corresponding
        dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    smoothing: The amount of smoothing for positive labels.

    L = - real_weights * log(sigmoid(D(x)))
      - generated_weights * log(1 - sigmoid(D(G(z))))
    """
    smoothing = tf.convert_to_tensor(smoothing, dtype=tf.float32)

    def _smooth_labels():
        output = (real_outputs * (1.0 - smoothing) + 0.5 * smoothing)
        return output

    real_outputs = smart_cond.smart_cond(smoothing,
                                         _smooth_labels, lambda: real_outputs)

    loss_on_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_outputs,
            labels=tf.ones_like(real_outputs) * smoothing,
        )
    )
    loss_on_generated = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generated_outputs,
            labels=tf.zeros_like(generated_outputs)
        )
    )
    loss = ((real_weights * loss_on_real) +
            (generated_weights * loss_on_generated))

    return loss


def minimax_generator_loss(generated_outputs,
                        weights = 1., 
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
    
    return loss

real_output = tf.ones(10)
generated_output = tf.random.uniform((10, 2))

print(minimax_discriminator_loss(real_output, generated_output))
print(minimax_generator_loss(generated_output))