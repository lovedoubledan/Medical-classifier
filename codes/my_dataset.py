# -*- coding: utf-8 -*-
"""
@author: Wu
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import csv
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

import random
from glob import glob
import numpy as np
from PIL import Image

import json

class Xinguan(Dataset):
    def __init__(self, img_path_list, target_list, patient_ids, img_ids):
        super(Xinguan, self).__init__()
        self.img_path_list = img_path_list
        self.target_list = target_list
        self.patient_ids = patient_ids
        self.img_ids = img_ids
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        target = self.target_list[index]
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img = self.preprocess(img)
        if self.patient_ids is None:
            return {'image': torch.from_numpy(img).float(), 'target': torch.Tensor([target])}
        else:
            patient_id = self.patient_ids[index]
            img_id = self.img_ids[index]
            return {'image': torch.from_numpy(img).float(), 'target': torch.Tensor([target]), 
            'patient_id': torch.Tensor([patient_id]), 'img_id': torch.Tensor([img_id])}

    def __len__(self): 
        return len(self.target_list)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

def get_data(root_path, split_info_path):
    if not os.path.exists(os.path.join(split_info_path, 'train.json')):
        print('spliting data')
        split_data(root_path, split_info_path)
        print('data split over')
    split_info_paths = []
    split_info_paths.append(os.path.join(split_info_path, 'train.json'))
    split_info_paths.append(os.path.join(split_info_path, 'val.json'))
    split_info_paths.append(os.path.join(split_info_path, 'test.json')) 
    img_paths = [[],[],[]]
    targets = [[],[],[]]  
    patient_ids = [[],[],[]]
    img_ids = [[],[],[]]
    for i in range(3): 
        with open(split_info_paths[i], 'r') as json_file:
            dict_dataset = json.load(json_file)
            img_paths[i] = dict_dataset['img_paths']
            targets[i] = dict_dataset['targets']
            patient_ids[i] = dict_dataset['patient_ids']
            img_ids[i] = dict_dataset['img_ids']
    return img_paths, targets, patient_ids, img_ids

def split_data(root_path, output_folder):
    negative_folders = [root_path + '/normal' + '/source_crop', root_path + '/normal2' + '/source_crop']
    positive_folders = [root_path + '/sysu_san' + '/source_crop', root_path + '/wuhan' + '/source_crop']
    negative_path_list, negative_ids, img_ids_neg, num_of_patients = get_splited_path_list(negative_folders)
    positive_path_list, positive_ids, img_ids_pos, num_of_patients = get_splited_path_list(positive_folders, num_of_patients, len(img_ids_neg))
    mydicts = [dict(), dict(), dict()]
    for i in range(3):
        mydicts[i]['img_paths'] = negative_path_list[i] + positive_path_list[i]
        mydicts[i]['patient_ids'] = negative_ids[i] + positive_ids[i]
        mydicts[i]['img_ids'] = img_ids_neg[i] + img_ids_pos[i]
        mydicts[i]['targets'] = np.zeros(len(negative_path_list[i]), dtype=int).tolist() + np.ones(len(positive_path_list[i]), dtype=int).tolist()
    output_files = []
    output_files.append(os.path.join(output_folder, 'train.json'))
    output_files.append(os.path.join(output_folder, 'val.json'))
    output_files.append(os.path.join(output_folder, 'test.json'))
    for i in range(3):
        with open(output_files[i], 'w') as output_file:
            json.dump(mydicts[i], output_file)


def get_splited_path_list(folder_list, start_id=0, img_id=0):
    # get patient notes
    patient_note_set = set()
    for folder in folder_list:
        file_names = os.listdir(folder)
        for file_name in file_names:
            patient_note = file_name.split('_')[0]
            patient_note_set.add(patient_note)

    # get patient ids
    patient_id = 0
    patient_id_dict = dict()
    for patient_note in patient_note_set:
        patient_id_dict[patient_note] = patient_id
        patient_id += 1

    # split patient ids
    id_list = [i for i in range(patient_id)]
    train_ids, test_ids = train_test_split(id_list, test_size=0.4)
    test_ids, val_ids = train_test_split(test_ids, test_size=0.25)
    train_val_test_dict = dict()
    for i in train_ids:
        train_val_test_dict[i] = 0
    for i in val_ids:
        train_val_test_dict[i] = 1
    for i in test_ids:
        train_val_test_dict[i] = 2

    # split img path
    splited_paths = [[], [], []] # 0 for train, 1 for valid, 2 for test
    img_ids = [[], [], []]
    patient_ids = [[],[],[]]
    for folder in folder_list:
        file_names = os.listdir(folder)
        for file_name in file_names:
            patient_note = file_name.split('_')[0]
            patient_id = patient_id_dict[patient_note]
            train_val_test = train_val_test_dict[patient_id]
            splited_paths[train_val_test].append(os.path.join(folder, file_name))
            img_ids[train_val_test].append(img_id)
            img_id += 1
            patient_ids[train_val_test].append(patient_id + start_id)

            if train_val_test == 0:
                additional_file = folder.split('/')
                additional_file[-1] = 'filled'
                additional_file = '/'.join(additional_file)
                splited_paths[0].append(os.path.join(additional_file, file_name))
                patient_ids[0].append(patient_id + start_id)
                img_ids[0].append(img_id)

    return splited_paths, patient_ids, img_ids, len(id_list)
