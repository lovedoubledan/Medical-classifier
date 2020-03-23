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
from my_dataset import Xinguan, get_data
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import json

def save_res(id_list, scores, targets, img_id_list):
    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier3_2/split_info/test.json'
    output_path1 = 'res.csv'
    res = dict()
    id_list = np.array(id_list).reshape(-1).tolist()
    img_id_list = np.array(img_id_list).reshape(-1).tolist()
    scores = np.array(scores).reshape(-1).tolist()
    res['patient_ids_'] = id_list
    res['img_ids'] = img_id_list
    res['scores_'] = scores
    res['targets_'] = targets
    with open(split_info_path, 'r') as splitfile:
        dataset = json.load(splitfile)

    res_df = pd.DataFrame(res)
    split_df = pd.DataFrame(dataset)
    pd.merge(res_df, split_df).to_csv(output_path1)

def evaluate(model,testset, thresh=0.5):
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
    img_id_list = []
    sig = nn.Sigmoid()
    
    for batch_i, batch_data in enumerate(dataloader):
            
        inputs = Variable(batch_data['image'].to(device),requires_grad=True)
        labels = Variable(batch_data['target'].to(device),requires_grad=False)
        id_list += batch_data['patient_id'].tolist()
        img_id_list += batch_data['img_id'].tolist()
        
        outputs = model(inputs)
        
        predicted = outputs.data        
        predicted = sig(predicted)
        scores += predicted.tolist()
        targets += labels.tolist()
        predicted[predicted>=thresh] = 1   # 大于0.5的取为正样本
        predicted[predicted<thresh] = 0
        
        # 对于阳性样本中，分类的准确性。
        tp += (predicted * labels).sum().item() 
        tn += ((1-predicted)*(1-labels)).sum().item()
        fp += (predicted.sum().item() - (predicted * labels).sum().item())
        fn += (labels.sum().item() - (predicted * labels).sum().item()) # 应该召回的总数-TP =FN        

    save_res(id_list, scores, targets, img_id_list)
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

    return per_img_res, evaluate_per_patient(id_list, scores, targets, thresh)

def evaluate_per_patient(id_list, scores, targets, thresh=0.5):

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

    new_scores[new_scores>=thresh] = 1   # 大于0.5的取为正样本
    new_scores[new_scores<thresh] = 0

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights",type=str, help="if specified starts from checkpoint model")

    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print('--------------------loading data--------------------')
    root_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/data/data'
    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier3_2/split_info'
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])

    # init model, torchvision---resnet
    model = torchvision.models.resnet50(num_classes=1)
    model = model.to(device)    # 转gpu
    # If specified we start from checkpoint

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pretrained_weights))
    per_img_res, per_patient_res = evaluate(model,valset, 0.5)


      
