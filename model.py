import tensorflow as tf

class ConvNet:
    def __init__(self, H, W, sess, learning_rate=1e-4, keep_prob=0.5):
        self.image = tf.placeholder('float', shape=[None, H, W, 3], name='X')
        self.labels = tf.placeholder('float', shape=[None, H, W, 1], name='Y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        l2reg = tf.contrib.layers.l2_regularizer(0.00005)
        x = self.image

        x = tf.layers.conv2d(x, 48, 11, (4,6), padding='valid', activation=tf.nn.relu, kernel_regularizer=l2reg)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.layers.max_pooling2d(x, 5, 2, padding='same')

        x = tf.layers.conv2d(x, 128, 5, 2, padding='valid', activation=tf.nn.relu, kernel_regularizer=l2reg)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.layers.max_pooling2d(x, 3, 2, padding='same')

        x = tf.layers.conv2d(x, 192, 3, 1, padding='same', activation=tf.nn.relu, kernel_regularizer=l2reg)
        x = tf.layers.conv2d(x, 192, 3, 1, padding='same', activation=tf.nn.relu, kernel_regularizer=l2reg)
        x = tf.layers.conv2d(x,  96, 3, 1, padding='same', activation=tf.nn.relu, kernel_regularizer=l2reg)
        x = tf.layers.flatten(x)

        x = tf.layers.dropout(x, rate=1.-keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 2048, activation=tf.nn.relu, kernel_regularizer=l2reg)

        x = tf.layers.dropout(x, rate=1.-keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 1764, activation=tf.nn.relu, kernel_regularizer=l2reg)

        x = tf.reshape(x, [-1, int(H/10), int(W/10), 1])
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=5, kernel_regularizer=l2reg)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2, kernel_regularizer=l2reg)
        x = tf.layers.conv2d(x, 1, 5, 1, padding='same', kernel_regularizer=l2reg)

        self.scores = x
        probs = tf.sigmoid(self.scores)

        self.xent_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = self.xent_loss + self.reg_loss

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.predictions = tf.cast(tf.argmax(tf.stack([1.-probs, probs], axis=3), axis=3), tf.float32)
        self.accuracy = self.get_accuracy()
        self.metrics = [self.get_precision(), self.get_recall(), self.get_jaccard()]

    def get_accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))

    def get_precision(self):
        p_scores = tf.reduce_sum(self.predictions*self.labels, [1,2,3]) / tf.reduce_sum(self.predictions, [1,2,3])
        return tf.reduce_mean(p_scores)

    def get_recall(self):
        r_scores = tf.reduce_sum(self.predictions*self.labels, [1,2,3]) / tf.reduce_sum(self.labels, [1,2,3])
        return tf.reduce_mean(r_scores)

    def get_jaccard(self):
        intersection = self.predictions*self.labels
        union = self.predictions + self.labels - intersection
        return tf.reduce_mean( tf.reduce_sum(intersection, [1,2,3]) / tf.reduce_sum(union, [1,2,3]) )
