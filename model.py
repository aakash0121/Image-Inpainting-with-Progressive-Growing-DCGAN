import tensorflow as tf
from custom_layers_func import MiniBatchStdev, WeightedSum, PixelNorm

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

    # skip the input, 1X1 and activation for old model
    

# define discriminator model
# input image start with 4X4 RGB
# add layers until it reaches required resolution
def discriminator(n_blocks, input_shape=(4,4,3)):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    # weight constraint
    const = tf.keras.constraints.max_norm(1.0)

    # initializing empty list for storing models
    model_list = list()

    # starting model input
    in_image = tf.keras.layers.Input(input_shape)

    # conv 1X1
    d = tf.keras.layers.Conv2D(128, (1,1), padding="same", kernel_initializer=init, kernel_constraint=const)(in_image)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    # conv 3X3
    d = MiniBatchStdev()(d)
    d = tf.keras.layers.Conv2D(128, (1,1), padding="same", kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    # conv 4X4
    d = tf.keras.layers.Conv2D(128, (4,4), padding="same", kernel_initializer=init, kernel_constraint=const)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

    # dense output layer
    d = tf.keras.layers.Flatten()(d)
    out_class = tf.keras.layers.Dense(1)(d)

    # define model
    model = tf.keras.layers.Model(in_image, out_class)

    # compile the model
    model.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    # store model
    model_list.append([model, model])

    # creating submodels
    for i in range(0, n_blocks-1):
        # get model without the fade-on
        old_model = model_list[i][0]
        models = add_disc_block(old_model)

        # store model
        model_list.append(models)
    
    return model_list

