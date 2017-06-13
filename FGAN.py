import tensorflow as tf
import numpy as np
from skimage.io import imsave
import os
from datetime import datetime
from util import  file_reader
import tensorflow.contrib.slim as slim

class FGAN():
    def __init__(self):
        self.img_height = 28
        self.img_width = 28
        self.img_size = self.img_height * self.img_width
        self.epochs = 500
        self.batch_size = 100
        self.z_dim = 128
        self.learning_rate = 0.0003
        self.beta1 = 0.5

    def _build_generator(self,z, out_channel_dim, is_train=True):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """
        alpha = 0.2

        with tf.variable_scope('generator', reuse=False if is_train == True else True):
            # Fully connected
            fc1 = tf.layers.dense(z, 7 * 7 * 512)
            fc1 = tf.reshape(fc1, (-1, 7, 7, 512))
            fc1 = tf.maximum(alpha * fc1, fc1)

            # Starting Conv Transpose Stack
            deconv2 = tf.layers.conv2d_transpose(fc1, 256, 3, 1, 'SAME')
            batch_norm2 = tf.layers.batch_normalization(deconv2, training=is_train)
            lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

            deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 3, 1, 'SAME')
            batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
            lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

            deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 3, 2, 'SAME')
            batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
            lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

            # Logits
            logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 3, 2, 'SAME')

            # Output
            out = tf.tanh(logits)
            img =tf.reshape(out,[-1,self.img_height,self.img_width,1])
            tf.summary.image('fake',img)
            return out

    def _build_discriminator(self,images,reuse=False):
        """
        Create the discriminator network
        :param image: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """
        alpha = 0.2

        with tf.variable_scope('discriminator',reuse=reuse):
            # using 4 layer network as in DCGAN Paper

            # Conv 1
            conv1 = tf.layers.conv2d(images, 64, 5, 2, 'SAME')
            lrelu1 = tf.maximum(alpha * conv1, conv1)

            # Conv 2
            conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME')
            batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
            lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

            # Conv 3
            conv3 = tf.layers.conv2d(lrelu2, 256, 5, 1, 'SAME')
            batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
            lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

            # Conv 4
            conv4 = tf.layers.conv2d(lrelu3, 512, 5, 1, 'SAME')
            batch_norm4 = tf.layers.batch_normalization(conv4, training=True)
            lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

            # Flatten
            flat = tf.reshape(lrelu4, (-1, 7 * 7 * 512))

            # Logits
            logits = tf.layers.dense(flat, 1)

            # Output
            out = tf.sigmoid(logits)

            return out, logits

    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs
        :param image_width: The input image width
        :param image_height: The input image height
        :param image_channels: The number of image channels
        :param z_dim: The dimension of Z
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
        """
        inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height,image_channels),
                                     name='input_real')
        tf.summary.image('real',inputs_real)
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return inputs_real, inputs_z, learning_rate

    def model_loss(self,input_real, input_z, out_channel_dim):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """

        g_model = self._build_generator(input_z, out_channel_dim)
        d_model_real, d_logits_real = self._build_discriminator(input_real)
        d_model_fake, d_logits_fake = self._build_discriminator(g_model,reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                    labels=tf.ones_like(d_model_real) * 0.9))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.zeros_like(d_model_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.ones_like(d_model_fake)))

        return d_loss, g_loss

    def model_opt(self,d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

    def train(self):
        """
        Train the GAN
        :param epoch_count: Number of epochs
        :param batch_size: Batch Size
        :param z_dim: Z dimension
        :param learning_rate: Learning Rate
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :param get_batches: Function to get batches
        :param data_shape: Shape of the data
        :param data_image_mode: The image mode to use for images ("RGB" or "L")
        """
        tf.reset_default_graph()
        input_real, input_z, _ = self.model_inputs(self.img_height, self.img_width, 1, self.z_dim)
        d_loss, g_loss = self.model_loss(input_real, input_z, 1)
        d_opt, g_opt = self.model_opt(d_loss, g_loss, self.learning_rate, self.beta1)
        MORPH = file_reader.FileReader('/home/bingzhang/Documents/Dataset/MORPH/MORPH/', 'MORPH_Info.mat')
        steps = 0
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter('./logs/112358',sess.graph)
            for epoch_i in range(self.epochs):
                for j in range(600):
                    batch_images,_ = MORPH.next_batch(self.batch_size)
                    batch_images = np.asarray(batch_images)[:,:,:,np.newaxis]
                    batch_images = batch_images * 2
                    steps += 1

                    batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                    sum_,_ = sess.run([summary_op,d_opt], feed_dict={input_real: batch_images, input_z: batch_z})
                    _ = sess.run(g_opt, feed_dict={input_z: batch_z})
                    summary_writer.add_summary(sum_,epoch_i*600+j)
                    if steps % 100 == 0:
                        # At the end of every 10 epochs, get the losses and print them out
                        train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                        train_loss_g = g_loss.eval({input_z: batch_z})

                        print("Epoch {}/{}...".format(epoch_i + 1, self.epochs),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))

if __name__ == '__main__':
    fgan = FGAN()

    fgan.train()
