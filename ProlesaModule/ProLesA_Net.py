import tensorflow as tf
from ProlesaModule import layers

class EncoderBlock(tf.keras.Model):

    def __init__(self, filters, pool_size, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(EncoderBlock,self).__init__()
        self.convnet = layers.ConvModule3D(filters, kernel_size = self.kernel_size)
        self.convnetsq = layers.ConvModule3D(filters, kernel_size = (1,1,1))
        self.squeeze = layers.SqueezeAndExcitation3D(filters)
        self.maxpoolres = tf.keras.layers.MaxPool3D(pool_size=pool_size, padding='same')
        self.conc = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, downscaled = None):
        lyrs = {}
        cnv = self.convnet(input_tensor)
        if downscaled is not None:
            cnv = self.conc([cnv, downscaled])
            cnv = self.convnetsq(cnv)
        squeeze_res = self.squeeze(cnv)
        squeeze = self.maxpoolres(squeeze_res)

        lyrs.update({"residual": squeeze_res, "Downsampling": squeeze})
        return lyrs

class Bottleneck2(tf.keras.Model):

    def __init__(self, filters, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(Bottleneck2,self).__init__()
        self.convnet = layers.ConvModule3D(filters,kernel_size = self.kernel_size)

    def call(self, input_tensor):
        cnv1 = self.convnet(input_tensor)
        return cnv1
    
class DecoderBlock(tf.keras.Model):
    def __init__(self, filters, up_size, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(DecoderBlock,self).__init__()
        self.transpose = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size= self.kernel_size,
            strides=up_size,
            padding="same")
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv2 = layers.ConvModule3D(filters, kernel_size = self.kernel_size)
        self.attention_gate = layers.AttentionGate3D(filters,up_size)

    def call(self, input_tensor, residual):
        at = self.attention_gate(input_tensor, residual)
        x = self.transpose(input_tensor)
        x = self.concat([x, at])
        x = self.conv2(x)
        return x

class Classifier(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv3D(filters=1,
                                            kernel_size=1,
                                            padding='same',
                                            activation="sigmoid")

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        return x
    
class ProlesaNet(tf.keras.Model):
        def __init__(self, **kwargs):
            super(ProlesaNet,self).__init__()

            self.enc1 = EncoderBlock(filters=32, pool_size = (1,1,1), kernel_size=(1,3,3))
            self.enc2 = EncoderBlock(filters=64, pool_size = (1,2,2), kernel_size=(1,3,3)) 
            self.enc3 = EncoderBlock(filters=128, pool_size = (2,2,2), kernel_size=(3,3,3)) 
            self.enc4 = EncoderBlock(filters=256, pool_size = (2,2,2), kernel_size=(3,3,3)) 

            self.Bottleneck = Bottleneck2(filters=320,kernel_size=(3,3,3))

            self.dec1 = DecoderBlock(filters=256, up_size = (2,2,2), kernel_size=(3,3,3))
            self.dec2 = DecoderBlock(filters=128, up_size = (2,2,2), kernel_size=(3,3,3)) 
            self.dec3 = DecoderBlock(filters=64, up_size = (1,2,2), kernel_size=(1,3,3)) 
            self.dec4 = DecoderBlock(filters=32, up_size = (1,1,1), kernel_size=(1,3,3)) 

            self.clasf = Classifier()

        def call(self, input_tensor):
            tmp1 = self.enc1(input_tensor)
            sq1, d1 = tmp1["residual"], tmp1["Downsampling"]


            tmp2 = self.enc2(d1, d1)
            sq2, d2 = tmp2["residual"], tmp2["Downsampling"]

            tmp3 = self.enc3(d2, d2)
            sq3, d3 = tmp3["residual"], tmp3["Downsampling"]

            tmp4 = self.enc4(d3, d3)
            sq4, d4 = tmp4["residual"], tmp4["Downsampling"]

            btn = self.Bottleneck(d4)

            dec1 = self.dec1(btn, sq4)

            dec2 = self.dec2(dec1, sq3)

            dec3 = self.dec3(dec2, sq2)

            dec4 = self.dec4(dec3, sq1)

            out = self.clasf(dec4)

            return out