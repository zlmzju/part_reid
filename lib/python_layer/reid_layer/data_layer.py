# imports
import caffe
import numpy as np
import os
import sys
import cv2
import random
from Queue import Queue
from threading import Thread
import time

class DataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a Detection model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        # === Read input parameters ===
        # params is a python dictionary with layer parameters.
        params=self.check_params(self.param_str)
        # store input as class variables
        self.batch_size = params['batch_size']
        self.input_shape = params['shape']
        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.input_shape[0], self.input_shape[1])
        top[1].reshape(self.batch_size)
        # Create a batch loader to load the images.
        self.queue=Queue(4)    #prefech up to 8 batches, with 4 workers
        params_p=params.copy()
        params_p['root_folder']=params['root_folder_p']
        params_p['source']=params['source_p']
        self.batch_prefechers = [BatchLoader(self.queue, params if i>0 else params_p) for i in range(3)]
        for worker in self.batch_prefechers:
            worker.start()
            time.sleep(0.25)

    def forward(self, bottom, top):
        while True:
            if self.queue.empty():
                continue
            image,label=self.queue.get()
            break
        top[0].data[...] = image
        top[1].data[...] = label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

    def check_params(self, param_str):
        params = eval(param_str)
        if 'shape' not in params:
            params['shape'] = (160, 80)
        if 'mean' not in params:
            params['mean'] = [104, 117, 123]
        if 'mirror' not in params:
            params['mirror'] = False
        if 'trans' not in params:
            params['trans'] = False
        if 'pad' not in params:
            params['pad'] = 0
        if 'max_per_id' not in params:
            params['max_per_id'] = 10   #max images per id within one batch
        if 'root_folder_p' not in params:
            params['root_folder_p'] = params['root_folder']
        if 'source_p' not in params:
            params['source_p'] = params['source']
        return params

class BatchLoader(Thread):
    def __init__(self, queue, params):
        super(BatchLoader, self).__init__()
        self.queue = queue
        self.batch_size = params['batch_size']
        self.id_dict = self.process_list(params['source'])
        self.max_per_id = params['max_per_id']
        self.all_id=self.id_dict.keys()
        self.index=0
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(params)

    def process_list(self, filename):
        list_file=open(filename)
        try:
            content = list_file.read( )
        finally:
            list_file.close( )
        lines = content.split('\n')
        all_id_dict={}
        for line in lines:
            if len(line.split())<2:
                continue
            file_name=line.split()[0]
            label_id=line.split()[1]
            if all_id_dict.has_key(label_id):
                all_id_dict[label_id].append(file_name)
            else:
                all_id_dict[label_id]=[file_name]
        return all_id_dict

    def get_batch_list(self):
        count=0
        cur_range=xrange(self.index,self.index+self.batch_size)
        for idx in cur_range:
            if idx>=len(self.all_id):
                random.shuffle(self.all_id)
            id=self.all_id[idx%len(self.all_id)]
            for file_name in self.id_dict[id][:self.max_per_id]:
                if count < self.batch_size:
                    count = count+1
                    yield (file_name,id)
                else:
                    self.index=idx%len(self.all_id)
                    return

    def run(self):
        while True:
            if self.queue.full():
                continue
            image_list=[]
            label_list=[]
            batch_list=self.get_batch_list()
            for file_name,label in batch_list:
                image=self.transformer.preprocess(file_name)
                image_list.append(image)
                label_list.append(label)
            blobs=(np.array(image_list),np.array(label_list))
            self.queue.put(blobs)


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, params):
        self.mean = params['mean']
        self.pad = params['pad']
        self.is_mirror = params['mirror']
        self.do_trans = params['trans']
        self.img_h, self.img_w = params['shape']
        self.root_folder=params['root_folder']

    def rand_transform(self, image):
        M=self.img_h;N=self.img_w
        pts1 = np.float32([[0,0],[N,0],[0,M]])
        ratio = 0.02
        ratio_s = 0.02
        #points
        dx = random.uniform(N*(-ratio), N*(ratio))
        dy = random.uniform(M*(-ratio), M*(ratio))
        ds = random.uniform(-ratio_s,ratio_s)
        ds_x = (N-(1+ds)*N)/2
        ds_y = (M-(1+ds)*M)/2
        if random.uniform(0,1) > 0.3:
            pts2 = np.float32([[dx+ds_x,dy+ds_y],[N+dx-ds_x,dy+ds_y],[dx+ds_x,M+dy-ds_y]])
        else:
            pts2 = np.float32([[N+dx-ds_x,dy+ds_y],[dx+ds_x,dy+ds_y],[N+dx-ds_x,M+dy-ds_y]])
        rows,cols,c = image.shape
        matrix = cv2.getAffineTransform(pts1,pts2)
        dst_img = cv2.warpAffine(image,matrix,(cols,rows))
        return dst_img

    def rand_pad_crop(self, image):
        padded=np.zeros((3,self.img_h+self.pad*2,self.img_w+self.pad*2))
        padded[:,self.pad:self.pad+self.img_h,self.pad:self.pad+self.img_w]=image
        left, top = np.random.randint(self.pad*2+1), np.random.randint(self.pad*2+1)
        return padded[:, top:top+self.img_h, left:left+self.img_w]

    def preprocess(self, file_name):
        """
        preprocess() emulate the pre-processing occuring in the vgg16
        """
        full_name=os.path.join(self.root_folder, file_name)
        if not os.path.isfile(full_name):
            print "Image file %s not exist!"%full_name
            return None
        image = cv2.imread(full_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(self.img_w,self.img_h))
        if self.do_trans:
            image=self.rand_transform(image)
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))

        if self.is_mirror and np.random.random() < 0.5:
            image=image[:,:,::-1]

        if self.pad>0:
            image=self.rand_pad_crop(image)

        return image

if __name__ == '__main__':
    print 'Hello world!'
