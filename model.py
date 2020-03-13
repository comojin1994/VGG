import tensorflow as tf

# Conv Unit
class ConvUnit(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(ConvUnit, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')

    def call(self, x, training=False, mask=None):
        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h)
        return h

# Conv Layer
class ConvLayer(tf.keras.Model):
    def __init__(self, filters, num_units, kernel_size):
        super(ConvLayer, self).__init__()
        self.sequence = list()
        for _ in range(num_units):
            self.sequence.append(ConvUnit(filters, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

# VGG16
class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__(name='VGG16')
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.c1 = ConvLayer(64, 1, (3, 3))
        self.do1 = tf.keras.layers.Dropout(0.25)
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.c2 = ConvLayer(128, 2, (3, 3))
        self.do2 = tf.keras.layers.Dropout(0.25)
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.c3 = ConvLayer(256, 3, (3, 3))
        self.do3 = tf.keras.layers.Dropout(0.25)
        self.pool3 = tf.keras.layers.MaxPool2D()
        self.c4 = ConvLayer(512, 3, (3, 3))
        self.do4 = tf.keras.layers.Dropout(0.25)
        self.pool4 = tf.keras.layers.MaxPool2D()
        self.c5 = ConvLayer(512, 3, (3, 3))
        self.do5 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.c1(x, training=training)
        x = self.do1(x)
        x = self.pool1(x)
        x = self.c2(x, training=training)
        x = self.do2(x)
        x = self.pool2(x)
        x = self.c3(x, training=training)
        x = self.do3(x)
        x = self.pool3(x)
        x = self.c4(x, training=training)
        x = self.do4(x)
        x = self.pool4(x)
        x = self.c5(x, training=training)
        x = self.do5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# VGG19
class VGG19(tf.keras.Model):
    def __init__(self):
        super(VGG19, self).__init__(name='VGG19')
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.c1 = ConvLayer(64, 1, (3, 3))
        self.do1 = tf.keras.layers.Dropout(0.25)
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.c2 = ConvLayer(128, 2, (3, 3))
        self.do2 = tf.keras.layers.Dropout(0.25)
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.c3 = ConvLayer(256, 4, (3, 3))
        self.do3 = tf.keras.layers.Dropout(0.25)
        self.pool3 = tf.keras.layers.MaxPool2D()
        self.c4 = ConvLayer(512, 4, (3, 3))
        self.do4 = tf.keras.layers.Dropout(0.25)
        self.pool4 = tf.keras.layers.MaxPool2D()
        self.c5 = ConvLayer(512, 4, (3, 3))
        self.do5 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.c1(x, training=training)
        x = self.do1(x)
        x = self.pool1(x)
        x = self.c2(x, training=training)
        x = self.do2(x)
        x = self.pool2(x)
        x = self.c3(x, training=training)
        x = self.do3(x)
        x = self.pool3(x)
        x = self.c4(x, training=training)
        x = self.do4(x)
        x = self.pool4(x)
        x = self.c5(x, training=training)
        x = self.do5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Architecture
if __name__ == '__main__':
    name = input('Model : ')
    if name == 'VGG16': model = VGG16()
    elif name == 'VGG19': model = VGG19()
    else: print('Wrong name')
    model.build((32, 32, 32, 3))
    model.summary()