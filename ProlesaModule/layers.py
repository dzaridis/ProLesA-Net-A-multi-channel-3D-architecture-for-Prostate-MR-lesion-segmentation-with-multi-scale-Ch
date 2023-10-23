import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv3D, BatchNormalization, ReLU, Dropout, Concatenate, Add, Multiply, UpSampling3D

class ConvLayer3D(tf.keras.layers.Layer):
    def __init__(self, filters,**kwargs):
        """Convolutional Layer, consists of Conv -> BN -> ReLU

        Args:
            filters (int): number of Conv Filters
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(ConvLayer3D,self).__init__()
        self.conv = Conv3D(
            filters = filters,
            kernel_size = self.kernel_size,
            padding = 'same',
            activation = None,
            kernel_initializer='he_normal'
        )
        self.bn   = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.relu = ReLU()
        self.lyrs = [self.conv, self.bn, self.relu]

    def call(self, input_tensor):

        output_tensor = input_tensor

        for layer in self.lyrs:
            output_tensor = layer(output_tensor)

        return output_tensor
    
class ConvModule3D(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        """Convolutional Layer, consists of Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        Args:
            filters (int): number of Conv Filters
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(ConvModule3D,self).__init__()
        self.conv1  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.conv2  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.bn    = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.act   = ReLU()
        
    def call(self, input_tensor):

        intermediate_tensor = self.conv1(input_tensor)
        conv_op = self.conv2(intermediate_tensor)
        output_tensor = self.act(conv_op)
        return output_tensor
    
class SqueezeAndExcitation3D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SqueezeAndExcitation3D, self).__init__()
        self.filters = filters
        self.reshape = tf.keras.layers.Reshape([-1, 1, 1, self.filters])
        self.global_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc1 = tf.keras.layers.Dense(units=self.filters // 8, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=self.filters, activation='sigmoid')
        self.mul = tf.keras.layers.Multiply()
    def call(self, input_tensor):
        x = self.global_pool(input_tensor)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        x = self.mul([input_tensor,x])
        return x

class AttentionGate3D(tf.keras.layers.Layer):
    def __init__(self, filters, unpool_size):
        super(AttentionGate3D, self).__init__()
        self.filters = filters

        self.theta = tf.keras.layers.Conv3D(filters, 1, activation='relu', kernel_initializer='he_normal')
        self.phi = tf.keras.layers.Conv3D(filters, 1, activation='relu', kernel_initializer='he_normal')
        self.psi = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', kernel_initializer='he_normal')
        self.upsample = tf.keras.layers.UpSampling3D(size=unpool_size)

    def call(self, input_tensor, skip):
        theta = self.theta(skip)
        phi = self.upsample(self.phi(input_tensor))

        attn = tf.keras.activations.softmax(theta * phi, axis=[1, 2, 3])

        psi = self.psi(attn * skip)

        return psi
    
