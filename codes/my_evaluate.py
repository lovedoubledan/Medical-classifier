import torchvision
from torch.autograd import Variable
import torch.optim as optim
import torch
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
from my_dataset import Xinguan
from tqdm import tqdm
from sklearn import metrics
import numpy as np

def evaluate(model,testset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    model.eval() 
    
    tp =0;fp=0;fn=0;tn=0
    scores = []
    targets = []
    id_list = []
    sig = nn.Sigmoid()
    
    for batch_i, batch_data in enumerate(dataloader):
            
        inputs = Variable(batch_data['image'].to(device),requires_grad=True)
        labels = Variable(batch_data['target'].to(device),requires_grad=False)
        id_list += batch_data['patient_id'].tolist()
        
        outputs = model(inputs)
        
        predicted = outputs.data        
        predicted = sig(predicted)
        scores += predicted.tolist()
        targets += labels.tolist()
        predicted[predicted>=0.5] = 1   # 大于0.5的取为正样本
        predicted[predicted<0.5] = 0
        
        # 对于阳性样本中，分类的准确性。
        tp += (predicted * labels).sum().item() 
        tn += ((1-predicted)*(1-labels)).sum().item()
        fp += (predicted.sum().item() - (predicted * labels).sum().item())
        fn += (labels.sum().item() - (predicted * labels).sum().item()) # 应该召回的总数-TP =FN        
    print(f'per img  tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}')      
    precision = tp/(fp+tp+1e-6)
    recall = tp/(fn+tp+1e-6)
    acc = (tp+tn)/(tp + fn + fp + tn)
    F1score = 2*precision*recall/(precision+recall+1e-6)
    auc = metrics.roc_auc_score(targets, scores)
    
    log_str = f"precision:{precision}\n recall:{recall}\n acc:{acc}\n F1 score:{F1score}\n auc:{auc}"
    print('--'*20,'*'*40,'--'*20)
    print('per img evaluate:')
    print(log_str)
    print('*'*20,'='*40,'*'*20)

    per_img_res = dict()
    per_img_res['precision'] = precision
    per_img_res['recall'] = recall
    per_img_res['acc'] = acc
    per_img_res['F1score'] = F1score
    per_img_res['auc'] = auc

    return per_img_res, evaluate_per_patient(id_list, scores, targets)

def evaluate_per_patient(id_list, scores, targets):

    id_list = np.array(id_list).reshape(-1)
    scores = np.array(scores).reshape(-1)
    targets = np.array(targets).reshape(-1)
    new_scores = []
    new_targets = []

    idx = np.argsort(id_list)
    scores = scores[idx]
    targets = targets[idx]
    id_list = id_list[idx]


    id_prev = -1
    target_prev = -1
    score_list = []
    for score, my_id, target in zip(scores, id_list, targets):
        if id_prev == -1:
            id_prev = my_id
            score_list.append(score)
            target_prev = target
            continue
        if my_id == id_prev:
            score_list.append(score)
        else:
            new_scores.append(np.mean(score_list))
            new_targets.append(target_prev)
            id_prev = my_id
            score_list = [score]
            target_prev = target
    new_scores.append(np.mean(score_list))
    new_targets.append(target_prev)
    new_scores = np.array(new_scores)
    new_targets = np.array(new_targets)

    auc = metrics.roc_auc_score(new_targets, new_scores)

    new_scores[new_scores>=0.5] = 1   # 大于0.5的取为正样本
    new_scores[new_scores<0.5] = 0

    tp = (new_scores * new_targets).sum()
    tn = ((1-new_scores)*(1-new_targets)).sum()
    fp = new_scores.sum() - (new_scores * new_targets).sum()
    fn = new_targets.sum() - (new_scores * new_targets).sum() # 应该召回的总数-TP =FN     
    print(f'per patient  tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}')
    precision = tp/(fp+tp+1e-6)
    recall = tp/(fn+tp+1e-6)
    acc = (tp+tn)/(tp + fn + fp + tn)
    F1score = 2*precision*recall/(precision+recall+1e-6)
    
    per_patient_res = dict()
    per_patient_res['precision'] = precision
    per_patient_res['recall'] = recall
    per_patient_res['acc'] = acc
    per_patient_res['F1score'] = F1score
    per_patient_res['auc'] = auc

    log_str = f"precision:{precision}\n recall:{recall}\n acc:{acc}\n F1 score:{F1score}\n auc:{auc}"
    print('--'*20,'*'*40,'--'*20)
    print('per patient evaluate:')
    print(log_str)
    print('*'*20,'='*40,'*'*20)
    return per_patient_res
       
