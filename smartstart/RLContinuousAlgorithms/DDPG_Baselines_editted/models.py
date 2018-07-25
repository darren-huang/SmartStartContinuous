import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor_Editted(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, h1=64, h2=64):
        """
        :param nb_actions: number of actions (size of vector)
        :param name: name of the model
        :param layer_norm: whether or not to normalize the layers (see tensorflow.contrib.layers.layer_norm)
        :param h1: size of 1st hidden layer
        :param h2: size of 2nd hidden layer
        """
        super(Actor_Editted, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.h1 = h1
        self.h2 = h2

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, self.h1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.h2)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic_Editted(Model):
    def __init__(self, name='critic', layer_norm=True, h1=64, h2=64):
        """
        :param name: name of the model
        :param layer_norm: whether or not to normalize the layers (see tensorflow.contrib.layers.layer_norm)
        :param h1: size of 1st hidden layer
        :param h2: size of 2nd hidden layer
        """
        super(Critic_Editted, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.h1 = h1
        self.h2 = h2

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, self.h1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, self.h2)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
