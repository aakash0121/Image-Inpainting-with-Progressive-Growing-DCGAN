import tensorflow as tf 

# weighted function class for PGGAN
# It is used for merging two input layers 
# weighted sum = ((1.0 – alpha) * input1) + (alpha * input2)
class WeightedSum():
    def __init__(self, alpha=0.0, **kwargs):
        # setting the alpha value
        self.alpha = tf.keras.backend.variable(alpha, name="ws_alpha")
    
    def _merge_func(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)

        out = (1.0 - self.alpha)*input[0] + (self.alpha*input[1])

        return out

# used to provide statistical summary of batch of activations
# only used in the output block of discriminator layer
class MiniBatchStdev():
    def __init__(self, **kwargs):
        pass

    def call(self, inputs):
        # mean value for each pixel across channels
        mean = tf.keras.backend.mean(inputs, axis=0, keepdims=True)

        # calculate stdev
        squared_diff = tf.keras.backend.square(inputs - mean)
        mean_squared_diff = tf.keras.backend.mean(squared_diff, axis=0, keepdims=True) + 1e-8
        stdev = tf.keras.backend.sqrt(mean_squared_diff)

        # mean stdev for each pixel position
        mean_pixel = tf.keras.backend.mean(stdev, keepdims=True)

        # concatenate the output with inputs
        shape = tf.keras.backend.shape(inputs)
        output = tf.keras.backend.tile(mean_pixel, (shape[0], shape[1], shape[2], 1))

        output = tf.keras.backend.concatenate([inputs, output], axis=-1)

        return output
    
    # returns output shape of layer (basically just increases by 1 due to concatenation of output with inputs)
    def output_shape(self, input_shape):
        input_shape = list(input_shape)

        input_shape[-1] += 1
        return tuple(input_shape)

# a local response normalization
# in paper it referred as called pixelwise feature vector normalization
# it is only used in generator not in discriminator
class PixelNorm():
    def __init__(self, **kwargs):
        pass
    
    def call(self, inputs):
        # normalize inputs using L2 norm
        values = inputs**2.0

        mean_values = 1.0e-8 + tf.keras.backend.mean(values, axis=-1, keepdims=True)

        l2 = tf.keras.backend.sqrt(mean_values)
        normalized = mean_values/l2

        return normalized

    # unlike MiniBatchStdev it doesn't change the size of inputs
    def output_shape(self, input_shape):
        return input_shape