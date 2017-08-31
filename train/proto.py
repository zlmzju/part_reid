#!/usr/bin/env python
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
import argparse

def base_net(prototxt):
    prototxt_file=open(prototxt)
    base_net_str=prototxt_file.read()
    prototxt_file.close()
    return base_net_str

# part map unit
def mask_unit(net,input_name,idx,feature_dim,each_dim):
    #map_num att_map
    net['mask_conv'+idx]=L.Convolution(net[input_name],kernel_size=1,num_output=1, \
                              param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],\
                              weight_filler=dict(type="xavier", variance_norm=2), \
                              bias_filler=dict(type="constant"))
    #input ~ (-1,1), rescale to range (0,1)
    net['mask_map'+idx]=L.Sigmoid(net['mask_conv'+idx])
    net['tile_map'+idx]=L.Tile(net['mask_map'+idx],tile_param=dict(tiles=feature_dim))
    net['masked'+idx]=L.Eltwise(net[input_name],net['tile_map'+idx],\
                                 eltwise_param=dict(operation=0))
    net['pooled'+idx]=L.Pooling(net['masked'+idx],pooling_param=dict(pool=1,global_pooling=1))
    net['linear'+idx]=L.InnerProduct(net['pooled'+idx], num_output=each_dim, \
                                    param = [dict(lr_mult=1, decay_mult=1),  \
                                             dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type="xavier"),
                                    bias_filler=dict(type="constant"))
    return net['linear'+idx]

# after the conv net
def mask_net(bottom_name='inception_4e/output',map_num=8,feature_dim=512):
    net=caffe.NetSpec()
    net[bottom_name],net['label']=L.Data(ntop=2)
    net['input_feat']=L.Convolution(net[bottom_name],kernel_size=1,num_output=feature_dim, \
                          param = [dict(lr_mult=1, decay_mult=1)],\
                          weight_filler=dict(type="xavier"), \
                          bias_filler=dict(type="constant"))
    bottom_list=[]
    for i in range(map_num):
        unit=mask_unit(net,'input_feat','_%d'%(i+1),feature_dim,feature_dim/map_num)
        bottom_list.append(unit)
    net['concat_features']=L.Concat(*bottom_list)
    net['normed_feature']=L.L2Norm(net['concat_features'])
    net['triplet_loss'],net['triplet_acc']=L.OnlineTripletLoss(net['normed_feature'],net['label'],
                                                               ntop=2,triplet_loss_param= \
                                                               dict(margin=0.2,negative=0,positive=0))
    proto_str='%s'%net.to_proto()
    return proto_str[proto_str.find('}')+1:]

#make the net protofile
def make_net(base_prototxt,map_num=8):
    base_str=base_net(base_prototxt)
    last_str=base_str[base_str.rfind('top'):]
    last_str=last_str[last_str.find('\"')+1:]
    feature_name=last_str[:last_str.find('\"')]
    att_str=mask_net(feature_name,map_num=map_num,feature_dim=512)
    proto_str=base_str+att_str
    return proto_str

#########################################################################################################
# other file operations
def write_file(file_name,proto_str):
    text_file = open(file_name, "w")
    text_file.write(proto_str)
    text_file.close()

def get_test(train_str):
    end_idx=train_str.find('OnlineTripletLoss')-40
    start_idx=train_str.find('######')
    base_str=train_str[start_idx:end_idx]
    test_str=train_str[:train_str.find('layer')]+base_str
    return test_str.replace('#input','input')

def get_train(proto_str,dataset):
    train_str=proto_str
    if 'cuhk03' in dataset:
        dataset_dir=r'%s/dataset/cuhk03/cuhk03_release/data/'%(args.root_dir)
        dataset_txt=r'%s/dataset/cuhk03/source/train.txt'%(args.root_dir)
    if 'market' in dataset:
        dataset_dir=r'%s/dataset/market/Market-1501/bounding_box_train/'%(args.root_dir)
        dataset_txt=r'%s/dataset/market/source/train.txt'%(args.root_dir)
    if 'viper' in dataset:
        dataset_dir=r'%s/dataset/viper/VIPeR/'%(args.root_dir)
        dataset_txt=r'%s/dataset/viper/source/train.txt'%(args.root_dir)
    if 'cuhk01' in dataset:
        dataset_dir=r'%s/dataset/cuhk01/campus/'%(args.root_dir)
        dataset_txt=r'%s/dataset/cuhk01/source/train.txt'%(args.root_dir)
    train_str=train_str.replace('DATASET_ROOT_DIR',dataset_dir)
    train_str=train_str.replace('DATASET_TXT_FILENAME',dataset_txt)
    #train_str=train_str.replace('\\','/')
    return train_str

def copy_file(src_name,dst_name):
    src_file = open(src_name, "r")
    content=src_file.read()
    content=content.replace("PROJECT_DIR",args.root_dir)
    content=content.replace("PROJECT_NAME","%s_%s"%(args.dataset,args.exp_name))
    src_file.close()
    write_file(dst_name,content)

def post_step(map_num,exp_name,proto_str,dataset,base_dir):
    exp_dir=exp_name+'/'
    import os
    snapshot_dir=exp_dir+'snapshot'
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    copy_file(base_dir+'training_script.sh',exp_dir+'run_%s.sh'%exp_name[exp_name.rfind('/')+1:])
    copy_file(base_dir+'solver.prototxt',exp_dir+'solver.prototxt')
    train_str=get_train(proto_str,dataset)
    test_str=get_test(train_str)
    write_file(exp_dir+'train.prototxt',train_str)
    write_file(exp_dir+'test.prototxt',test_str)
    print "Files have been writen to folder %s"%exp_dir


def gen_proto(exp_dir,map_num,base_network,dataset,exp_name):
    base_dir=args.root_dir+'/models/basenet/%s/'%base_network
    exp_name='%s/%s/%s'%(exp_dir,dataset,exp_name)
    proto_str=make_net(base_dir+'train_val.prototxt',map_num)
    if 'num' in exp_name:
        exp_name=exp_name+'/num%d'%map_num
    post_step(map_num,exp_name,proto_str,dataset,base_dir)

def main(args):
    gen_proto(args.exp_dir,args.map_num,args.base_net, args.dataset,args.exp_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make caffe prototxt.')
    parser.add_argument('--map-num', type=int, default=8,help='weighting unit number')
    parser.add_argument('--base-net', type=str, default='googlenet',help='base network')
    parser.add_argument('--dataset', type=str, default='market',help='dataset name')
    parser.add_argument('--exp-dir', type=str, default='.',help='experiment directory')
    parser.add_argument('--exp-name', type=str, default='partnet',help='experiment name')
    parser.add_argument('--root-dir', type=str, help='the root directory which contains caffe')
    args = parser.parse_args()

    import os
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = root_dir[:root_dir.rfind('/')]
    args.root_dir=root_dir

    main(args)
