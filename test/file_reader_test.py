from unittest import TestCase
from util import file_reader
import scipy.io as sio
class FileReaderTest(TestCase):

    MORPH = file_reader.FileReader('/home/bingzhang/Documents/Dataset/MORPH/MORPH/','MORPH_Info.mat')
    # file_reader.__str__()
    print MORPH.__str__()
    data,_ =  MORPH.next_batch(20)
    sio.savemat('test.mat',{'data':data})