#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse

# BASE_DIR = os.path.join(os.path.dirname(__file__),'../../../data/Stanford3dDataset_v1.2_Aligned_Version')

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f',  help='Path to data folder')
parser.add_argument("--dest", '-d')
args = parser.parse_args()
BASE_DIR = args.folder
print('CC',BASE_DIR)
object_dict = {
            'clutter':   0,
            'ceiling':   1,
            'floor':     2,
            'wall':      3,
            'beam':      4,
            'column':    5,
            'door':      6,
            'window':    7,
            'table':     8,
            'chair':     9,
            'sofa':     10,
            'bookcase': 11,
            'board':    12}

path_Dir_Areas =  os.listdir(BASE_DIR)

for Area in path_Dir_Areas:
    if not os.path.isdir(os.path.join(BASE_DIR,Area)):
        continue
    path_Dir_Rooms = os.listdir(os.path.join(BASE_DIR,Area))
    for Room in path_Dir_Rooms:


        xyz_Room = np.zeros((1,6))
        label_Room = np.zeros((1,1))
        path_Annotations = os.path.join(BASE_DIR,Area,Room,"Annotations")
        if not os.path.exists(path_Annotations):
            print("Error {} does not exists".format(path_Annotations))
            continue
        
        print(path_Annotations)
        # make store directories
        # path_prepare_label = os.path.join("../../../data/S3DIS/prepare_label_rgb",Area,Room)
        path_prepare_label = os.path.join(args.dest,Area,Room)
        print('dst file:',path_prepare_label)
        if not os.path.exists(path_prepare_label):
            os.makedirs(path_prepare_label)
            #############################
            path_objects = os.listdir(path_Annotations)
            for Object in path_objects:
                if Object.split("_",1)[0] in object_dict:
                    print(Object.split("_",1)[0] + " value:" ,object_dict[Object.split("_",1)[0]])
                    xyz_object = np.loadtxt(os.path.join(path_Annotations,Object))[:,:]#(N,6)
                    label_object = np.tile([object_dict[Object.split("_",1)[0]]],(xyz_object.shape[0],1))#(N,1)
                else:
                    continue

                xyz_Room = np.vstack((xyz_Room,xyz_object))
                label_Room = np.vstack((label_Room,label_object))

            xyz_Room = np.delete(xyz_Room,[0],0)
            label_Room = np.delete(label_Room,[0],0)

            np.save(path_prepare_label+"/xyzrgb.npy",xyz_Room.astype(np.float16))
            np.save(path_prepare_label+"/label.npy",label_Room)


