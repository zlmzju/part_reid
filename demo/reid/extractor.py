#
#Person ReID using the PersonPartNet model trained on Market-1501
#Author: Liming Zhao (zlmzju@gmail.com)
#
import numpy as np
import sys
import caffe
import time

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

class Extractor():
    N=1;C=3;H=160;W=80
    # ### 1. Use PyCaffe to extract features and rank
    def __init__(self,gpu_id=1,
                 proto_name = './model/deploy.prototxt',
                 model_name='./model/market.caffemodel'):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(proto_name, caffe.TEST, weights=model_name)        
        Extractor.N,Extractor.C,Extractor.H,Extractor.W=self.net.blobs['data'].data.shape
        self.transformer=self.get_transformer()
        
    def update_shape(self,N,C,H,W):
        self.net.blobs['data'].reshape(N,C,H,W)
        Extractor.N,Extractor.C,Extractor.H,Extractor.W=self.net.blobs['data'].data.shape
        self.transformer=self.get_transformer()
   
    @staticmethod
    def get_transformer():
        shape=(Extractor.N,Extractor.C,Extractor.H,Extractor.W)
        transformer = caffe.io.Transformer({'data': shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([ 104,  117,  123])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        return transformer

    def readImages(self,images):
        imageLen=len(images)
        imageDataList=[]
        for imageIdx in range(imageLen):
            imageName=images[imageIdx]
            imageImage=self.transformer.preprocess('data', caffe.io.load_image(imageName))
            imageDataList.append(imageImage) #center crop
            imageIdx+=1
        #imageData and imageData
        imageData=np.asarray(imageDataList)
        return imageData

    def extract_features(self,file_list):
        file_len=len(file_list)
        features=[]
        batch_size=200
        iterations=file_len/batch_size+1 if file_len%batch_size>0 else file_len/batch_size
        for batch_idx in range(iterations):
            cur_len=batch_size if batch_idx <file_len/batch_size else file_len%batch_size
            cur_list=file_list[batch_idx*batch_size+0:batch_idx*batch_size+cur_len]
            image_data=self.readImages(cur_list)
            self.net.blobs['data'].reshape(cur_len,Extractor.C,Extractor.H,Extractor.W)
            self.net.blobs['data'].data[:] = image_data
            self.net.forward()
            normed_features=self.net.blobs['normed_feature'].data.copy()
            from sklearn.preprocessing import normalize
            for idx in range(cur_len):
                cur_feature=np.squeeze(normed_features[idx,:])
                features.append(cur_feature)
        return features


class DataProcessor():
    # ### 2. Read the dataset given a data directory, which contains `jpg` images
    def __init__(self,data_dir='./data/',feat_folder='./feature/'):
        self.data_dir=data_dir
        self.image_list=self.readDir(data_dir)
        self.feat_folder=feat_folder
        self.score_list=[]
        
    #dataset related
    def readDir(self,folder):
        import fnmatch
        import os
        file_list = []
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                file_list.append(os.path.join(root, filename))
        return file_list

    def save_name(self,image_name):
        import os
        save_name=os.path.abspath(self.feat_folder+image_name[:-4])
        save_path=os.path.dirname(save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_name+'.txt'

    def save_features(self,image_features):
        import numpy as np
        for idx in range(len(self.image_list)):
            image_name=self.image_list[idx]
            feature=image_features[idx]
            feat_name=self.save_name(image_name)
            np.savetxt(feat_name,feature, fmt='%.18f')
               

    def rank_features(self,query_features,gallery_features):
        all_score_list=[]
        for query_idx in range(len(query_features)):
            query_feature=query_features[query_idx]
            cur_score_list=[]
            for gallery_idx in range(len(gallery_features)):
                gallery_feature=gallery_features[gallery_idx]
                dist = np.sum((query_feature-gallery_feature)**2)
                similar_score=1.0/(1.0+dist)
                cur_score_list.append(similar_score)
            #we get scoreList, then cal predictLists
            all_score_list.append(cur_score_list)
        return all_score_list
    
    def load_features(self,rank=True):
        import numpy as np
        image_features=[]
        for image_name in self.image_list:
            feat_name=self.save_name(image_name)
            feature=np.loadtxt(feat_name)
            image_features.append(feature)
        print 'Successfully loaded %d features'%(len(image_features))
        if rank:
            score_list=self.rank_features(image_features,image_features)
            self.score_list=score_list
            return score_list
        else:
            return image_features

    # ### 3. the feature extraction pipeline

    def extract_features(self,gpu_id=1):
        ###################init self.net##############################
        self.extractor=Extractor(gpu_id)
        ###################dataset###############################
        print 'Start extracting features for %d images in %s'%(len(self.image_list),self.data_dir)
        ################extract features#########################
        time1=time.time()
        image_features=self.extractor.extract_features(self.image_list)
        time2=time.time()
        print 'Finished within %.4f seconds, each costs %.4f second'%(time2-time1,
            (time2-time1)*1.0/len(image_features))
        #################save to txt files#####################
        self.save_features(image_features)
        image_path=self.image_list[0][:self.image_list[0].rfind('/')]
        print 'Features are saved to %s'%(self.feat_folder)
        
class Plotter():
    
    @staticmethod
    def show_image(name):
        plt.imshow(Plotter.read_image(name))
        
    
    @staticmethod
    def person_id(name):
        end=name.rfind('.')
        end=end if end<name.rfind('\\')+8 else name.rfind('\\')+8
        pid=name[name.rfind('\\')+1:end]
        return pid

    @staticmethod
    def read_image(name):
        transformer=Extractor.get_transformer()
        image=transformer.preprocess('data', caffe.io.load_image(name))
        img=transformer.deprocess('data', image)
        return img
    
    @staticmethod
    def show_maps(extractor, file_names):
        net=extractor.net
        extractor.update_shape(1,3,260,130)
        query_features=extractor.extract_features(file_names)
        
        for idx in range(len(file_names)):
            Plotter.show_one_map(net, idx, file_names)
            
    @staticmethod   
    def show_one_map(net, idx, file_names):
        fig=plt.figure(figsize=(12, 6))
        print file_names[idx]
        #draw input image
        transformer = Extractor.get_transformer()
        image=net.blobs['data'].data[idx,:]
        ax=plt.subplot(1,10,1)
        ax.tick_params(labelbottom='off', labelleft='off')
        plt.imshow(transformer.deprocess('data',image))
        #draw weight maps
        map_names=[name for name in net.blobs.keys() if 'mask_map' in name]
        all_maps=[net.blobs[name].data[idx,0,:] for name in map_names]
        ax=plt.subplot(1,10,2)
        ax.tick_params(labelbottom='off', labelleft='off')
        sum_map=np.average(np.array(all_maps),axis=0)
        plt.imshow(sum_map,cmap='gray')
        for i in range(len(all_maps)):
            ax=plt.subplot(1,10,i+3)
            ax.tick_params(labelbottom='off', labelleft='off')
            plt.imshow(all_maps[i],cmap='gray')
        plt.show()