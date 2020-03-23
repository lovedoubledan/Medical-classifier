from my_dataset import Xinguan, get_data
import torchvision 
import torch.nn as nn
import torch
import time
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from my_evaluate import evaluate

from torch.utils.tensorboard import SummaryWriter


root_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/data/data'
split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier3_2/split_info'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--pretrained_weights",type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--num_workers", type=int, default=0, help="interval between saving model weights")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    print('--------------------loading data--------------------')
    # xinguan
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    trainset = Xinguan(img_paths[0], targets[0], patient_ids[0], img_ids[0])
    validset = Xinguan(img_paths[1], targets[1], patient_ids[1], img_ids[0])

    dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        
    print('-------------------data have loaded------------------')
    
    # init model, torchvision---resnet
    model = torchvision.models.resnet50(num_classes=1)
    model = model.to(device)
    writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batch_size}')     
    
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))
       
    # 数据多个GPU并行运算！（单机多卡~）
    model = torch.nn.DataParallel(model)
      
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)
    criterion = nn.BCEWithLogitsLoss()
    max_recall = 0
    loss_list = []
    sig = nn.Sigmoid()
    global_step = 0
    for epoch in range(opt.epochs):
        model.train() 
        start_time = time.time()
        
        total = 0
        correct = 0 

        # start training.
        for batch_i, batch_data in enumerate(dataloader):
            
            inputs = Variable(batch_data['image'].to(device),requires_grad=True)
            labels = Variable(batch_data['target'].to(device),requires_grad=False)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            
            predicted = outputs.data

            
            predicted = sig(predicted)
            predicted[predicted>=0.5] = 1   # 大于0.5的取为正样本
            predicted[predicted<0.5] = 0
            #print('=====predicted--after:',predicted)
            #print('=====label:',labels)
            
            # 对于阳性样本中，分类的准确性。
            tp = (predicted * labels).sum().item() 
            #print('tp:',tp)
            fp = predicted.sum().item() - tp
            fn = labels.sum().item() - tp # 应该召回的总数-TP =FN

            # log
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch + 1, opt.epochs, batch_i, len(dataloader))
            log_str += f"\n loss: {loss.item()}"
            log_str += f"\n TP: {tp}"
            log_str += f"\n presion: {float(tp/(fp+tp+1e-6))}"    # +1e-6避免出现分母为0
            log_str += f"\n recall: {float(tp/(fn+tp+1e-6))}"
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Presion/train', float(tp/(fp+tp+1e-6)), global_step)
            writer.add_scalar('Recall/train', float(tp/(fn+tp+1e-6)), global_step)
            global_step += 1

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)   

        # 验证的时候单卡。
        per_img_res, per_patient_res = evaluate(model,validset)  
        ############
        precision = per_img_res['precision']
        recall =  per_img_res['recall']
        acc =  per_img_res['acc']
        F1score =  per_img_res['F1score']
        auc =  per_img_res['auc']
        writer.add_scalar('Presion/test per img', float(precision), epoch)
        writer.add_scalar('Recall/test per img', float(recall), epoch) 
        writer.add_scalar('acc/test per img', float(acc), epoch)
        writer.add_scalar('F1score/test per img', float(F1score), epoch)
        writer.add_scalar('auc/test per img', float(auc), epoch)
        ############
        precision = per_patient_res['precision']
        recall =  per_patient_res['recall']
        acc =  per_patient_res['acc']
        F1score =  per_patient_res['F1score']
        auc =  per_patient_res['auc']
        writer.add_scalar('Presion/test per patient', float(precision), epoch)
        writer.add_scalar('Recall/test per patient', float(recall), epoch) 
        writer.add_scalar('acc/test per patient', float(acc), epoch)
        writer.add_scalar('F1score/test per patient', float(F1score), epoch)
        writer.add_scalar('auc/test per patient', float(auc), epoch)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")
    writer.close()
    testset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    evaluate(model,testset)
          