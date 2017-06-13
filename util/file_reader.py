# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import scipy.io as sio
import random as random
import numpy as np
from scipy import ndimage
from scipy import misc
import tensorflow as tf


class FileReader():
    def __init__(self, data_dir, data_info):

        self.data = sio.loadmat(data_dir+data_info)
        self.prefix = data_dir
        self.age = np.squeeze(self.data['MORPH_Info']['age'][0][0])
        self.path = np.squeeze(self.data['MORPH_Info']['file'][0][0])
        self.image_num = len(self.path)
        self.shuffled_index = range(self.image_num)
        np.random.shuffle(self.shuffled_index)
        self.current_index = 0

    def __str__(self):
        return 'Data directory:\t' + self.prefix + '\nPic Num:\t' + str(self.image_num)


    def next_batch(self,batch_size):
        if self.current_index+batch_size>self.image_num:
            np.random.shuffle(self.shuffled_index)
            self.current_index = 0
        image = []
        age_label = []
        for i in xrange(self.current_index,self.current_index+batch_size):
            idx = self.shuffled_index[i]
            image_path = self.prefix+self.path[idx]+'.jpg'
            image_data = self.read_jpeg_image(image_path)
            image.append(image_data)
            age_label.append(self.age[idx])
        self.current_index+=batch_size
        return image,age_label

    def read_jpeg_image(self, path):
        content = ndimage.imread(path,mode='L')
        content = misc.imresize(content,[28,28])
        mean_v = np.mean(content)
        adjustied_std = np.maximum(np.std(content),1.0/np.sqrt(250*250))
        content = (content-mean_v)/adjustied_std
        return content