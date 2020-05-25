import tensorflow as tf

# wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_pred*y_true)

# adds a discriminator block
def add_disc_block(old_model, n_input_layers=3):
    # weights initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    # constraints
    const = tf.keras.constraints.max_norm(1.0)

    # shape of existing model
    in_shape = list(old_model.input.shape)

    input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)

    in_image = tf.keras.layers.Input(shape=input_shape)

    # new input layer
    d = tf.keras.layers.Conv2D(128, (1,1), padding="same", kernel_intializer=init, kernel_constraint=const)(in_image)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    d = tf.keras.layers.Conv2D(128, (3,3), padding="same", kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyRelu(alpha=0.2)(d)

    d = tf.keras.layers.Conv2D(128, (3,3), padding="same", kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyRelu(alpha=0.2)(d)

    d = tf.keras.layers.AveragePooling2D()(d)

    new_block = d

    