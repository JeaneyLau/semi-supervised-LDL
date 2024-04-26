import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datapreprocess import *
from sklearn.metrics import precision_score,f1_score,recall_score
import os
import sys
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn import metrics
import shutil
from Net import *
import setproctitle
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


iter_num = 0

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=8, help='number of labeled data per batch')
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--save',default='./result/SS_ALDL')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
    parser.add_argument('--labeled_num', type=int, default=58*2, help='number of labeled')
    parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
    parser.add_argument('--consistency', type=float,  default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
    parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--global_step', type=int,  default=0, help='global_step')
    
    args = parser.parse_known_args()[0]
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    
    setproctitle.setproctitle(args.save)
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss()
    criterion_cos = nn.CosineSimilarity(dim=1).cuda(0)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    
    iter_num = args.global_step

    #  ACNE04
    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])
    
    trainTransform = TransformTwice(transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        normalize
        ]))
    testTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    
    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 1165))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    BCdata_trian = PublicSkinDataset_distribution(root=r'./data/ACNE04_train.txt', transform=trainTransform)
    BCdata_test = PublicSkinDataset_test(root=r'./data/ACNE04_test.txt', transform=testTransform)

    trainLoader = DataLoader(BCdata_trian, batch_sampler=batch_sampler, num_workers=16)
    testLoader = DataLoader(BCdata_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=16)
    
    net = ResNet50_semi_distribution()
    net = net.to(device) 
    
    net_ema = ResNet50_semi_distribution()
    net_ema = net_ema.to(device) 
    for param in net_ema.parameters():
        param.detach_()     
            
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr,betas=(0.9, 0.999), weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)
    
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    testF_ema = open(os.path.join(args.save, 'test_ema.csv'), 'w')
    
    test_temp_acc = 0
    for epoch in range(1, args.nEpochs + 1):
        train(args, epoch, net, net_ema, trainLoader, optimizer, criterion,criterion_mse,criterion_kl,criterion_cos,scheduler, trainF)
        test(args, epoch, net, net_ema, testLoader,optimizer,criterion,criterion_mse,criterion_kl,criterion_cos,testF,testF_ema)
        torch.save(net, os.path.join(args.save, str(epoch)+'.pth'))
        torch.save(net_ema, os.path.join(args.save, str(epoch)+'_ema.pth'))


    trainF.close()
    testF.close()
    
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch,args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

#     mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    mse_loss = (input_softmax-target_softmax)**2 
    return mse_loss

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

def plot_matrix(Y_true,Y_predict,epoch,args):
    plt.clf()
    matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
#     classes = [1,2,3,4,5,6,7,8]    # X/Y label
    classes = [1,2,3,4]    # X/Y label
    confusion = confusion_matrix(Y_true,Y_predict,normalize='true')
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes,fontproperties='Times New Roman', size=8)
    plt.yticks(indices, classes,fontproperties='Times New Roman', size=8)
    plt.colorbar()
    plt.xlabel('Predicted label',fontdict={'family':'Times New Roman','size':8})
    plt.ylabel('True label',fontdict={'family':'Times New Roman','size':8})
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index,second_index,"%0.2f"%(confusion[second_index][first_index],),fontdict={'family':'Times New Roman', 'size': 8},va='center',ha='center')
#     plt.savefig(r'./draw/confusion_matrix_method.png',bbox_inches='tight', dpi=600)
    savepath = os.path.join(args.save, 'ConMatrix')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    plt.savefig(os.path.join(savepath,str(epoch)+'.jpg'),bbox_inches='tight', dpi=600)
    plt.ioff()



def genLD(label, sigma, loss, class_num):
    label_set = np.array(range(class_num))
    if loss == 'klloss':
        ld_num = len(label_set)
        dif_age = np.tile(label_set.reshape(ld_num, 1), (1, len(label))) - np.tile(label, (ld_num, 1))
        ld = 1.0 / np.tile(np.sqrt(2.0 * np.pi) * sigma, (ld_num, 1)) * np.exp(-1.0 * np.power(dif_age, 2) / np.tile(2.0 * np.power(sigma, 2), (ld_num, 1)))
        ld = ld / np.sum(ld, 0)
        return ld.transpose()


def train(args, epoch, net, net_ema, trainLoader, optimizer, criterion,criterion_mse,criterion_kl,criterion_cos,scheduler, trainF):
    net.train()                                       # 设置网络为训练模式
    print('\n epoch   **********   ',epoch)
    nProcessed = 0
    total_loss = 0
    lam= 0.5
    beta=1
    total_correct = 0
    nTrain = args.labeled_num
    for batch_idx, ((image_batch, ema_image_batch), target, counting) in enumerate(trainLoader):
        
        image_batch, ema_image_batch, target = image_batch.cuda(0), ema_image_batch.cuda(0), target.long().cuda(0).view(-1)
        correct = 0
        ema_inputs = ema_image_batch #+ noise2
        inputs = image_batch #+ noise1
        
        activations, cls, cou, cou2cls = net(inputs)
        with torch.no_grad():
            ema_activations, ema_cls, ema_cou, ema_cou2cls = net_ema(ema_inputs)
        
        b_l = counting[:args.labeled_bs].numpy()
        b_l = b_l - 1

        ld = genLD(b_l, 3, 'klloss', 65)
        ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
        ld = torch.from_numpy(ld).cuda(0).float()
        ld_4 = torch.from_numpy(ld_4).cuda().float()
        
        # 生成方差为1的标签分布
        b_2 = target[:args.labeled_bs].cpu().numpy()
        ld_tar = genLD(b_2, 1, 'klloss', 4)
        ld_tar = torch.from_numpy(ld_tar).cuda(0).float()
        
        loss_cls = criterion_kl(torch.log(cls[:args.labeled_bs]), ld_tar) 
        loss_cou = criterion_kl(torch.log(cou[:args.labeled_bs]), ld) 
        loss_cls_cou = criterion_kl(torch.log(cou2cls[:args.labeled_bs]), ld_4) 
        loss = loss_cls * lam + (loss_cou + loss_cls_cou)*0.5* (1.0 - lam)
        
        if args.ema_consistency == 1:

            #计算余弦相似度loss  
            similarity = activations.mm(activations.t())
            ema_similarity = ema_activations.mm(ema_activations.t())
            consistency_relation_loss = -criterion_cos(similarity,ema_similarity).mean()

            loss_cls2 = criterion_kl(torch.log(ema_cls), cls) 
            loss_cou2 = criterion_kl(torch.log(ema_cou), cou) 
            loss_cls_cou2 = criterion_kl(torch.log(ema_cou2cls), cou2cls) 
            loss_distribution = loss_cls2 * lam + (loss_cou2 + loss_cls_cou2)*0.5* (1.0 - lam)

        if (epoch > 20) and (args.ema_consistency == 1):
            loss = loss + beta*(consistency_relation_loss + loss_distribution)


        total_loss = loss.item()+total_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global iter_num
        update_ema_variables(net, net_ema, args.ema_decay, iter_num)
        iter_num = iter_num + 1
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        
        pre = cls[:args.labeled_bs]
        prediction = torch.argmax(pre, 1)
        label_tar = target[:args.labeled_bs]
        correct += (prediction == label_tar).sum().int().cpu().numpy()
        total_correct = total_correct+correct

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if((batch_idx%30)==0):
            print("partialEpoch={},total_loss = {},ACC = {} \n".format(partialEpoch,loss.item(),correct/args.batch_size))
    print("iter_num = ",iter_num)
    scheduler.step()
    
    print("Train total_loss = {},  ACC = {}  \n".format((total_loss/nTrain),(total_correct/nTrain)))
    trainF.write("Train total_loss = {},  ACC = {}  \n".format((total_loss/nTrain),(total_correct/nTrain)))
    trainF.flush()

def test(args, epoch, net, net_ema, testLoader,optimizer, criterion,criterion_mse,criterion_kl,criterion_cos, testF,testF_ema):
    net.eval()
#     net_ema.eval()
    total_loss = 0
    total_correct = 0
    conMatrix_pre = []
    conMatrix_tar = []
    MAE = 0
    MSE = 0
    total_loss2 = 0
    total_correct2 = 0
    conMatrix_pre2 = []
    conMatrix_tar2 = []
    MAE2 = 0
    MSE2 = 0
    nTrain = len(testLoader.dataset)
    
    with torch.no_grad():
        for pos_1, target in testLoader:
            images = pos_1.cuda(0)
            target = target.long().cuda(0).view(-1)
            
            f, output,_,_= net(images)
            loss = criterion(output,target)
            total_loss = loss.item()+total_loss
            prediction = torch.argmax(output, 1)
            
            for i in range(len(prediction)):
                conMatrix_pre.append(prediction[i].cpu().detach().numpy())
                conMatrix_tar.append(target[i].cpu().detach().numpy())
                MAE += abs(prediction[i].cpu().detach().numpy()-target[i].cpu().detach().numpy())
                MSE += abs(prediction[i].cpu().detach().numpy()-target[i].cpu().detach().numpy()) ** 2
            total_correct += (prediction == target).sum().int().cpu().numpy()
            
            
            f2, output2,_,_= net_ema(images)
            loss2 = criterion(output2,target)
            total_loss2 = loss2.item()+total_loss2
            prediction2 = torch.argmax(output2, 1)
            for i in range(len(prediction2)):
                conMatrix_pre2.append(prediction2[i].cpu().detach().numpy())
                conMatrix_tar2.append(target[i].cpu().detach().numpy())
                MAE2 += abs(prediction2[i].cpu().detach().numpy()-target[i].cpu().detach().numpy())
                MSE2 += abs(prediction2[i].cpu().detach().numpy()-target[i].cpu().detach().numpy()) ** 2
            total_correct2 += (prediction2 == target).sum().int().cpu().numpy()
            
    precision = precision_score(conMatrix_tar, conMatrix_pre, average='macro')
    recall = recall_score(conMatrix_tar, conMatrix_pre, average='macro')
    f1 = f1_score(conMatrix_tar, conMatrix_pre, average='macro')
    sensitivity,specificity = calculate_specificity_sensitivity(conMatrix_tar,conMatrix_pre)
    young_index = sensitivity+specificity-1
    
    testF.write('epoch:{},test_loss={},test_acc={},MAE={},MSE={},precision={},recall={},f1={},sensitivity={},specificity={},young_index={}\n'.format(
        epoch,(total_loss/nTrain),(total_correct/nTrain),MAE/nTrain,MSE/nTrain,precision,recall,f1,sensitivity,specificity,young_index))
    testF.flush()

    # draw confusion matrix 
    plot_matrix(conMatrix_tar,conMatrix_pre,epoch,args)
    
    precision2 = precision_score(conMatrix_tar2, conMatrix_pre2, average='macro')
    recall2 = recall_score(conMatrix_tar2, conMatrix_pre2, average='macro')
    f12 = f1_score(conMatrix_tar2, conMatrix_pre2, average='macro')
    sensitivity2,specificity2 = calculate_specificity_sensitivity(conMatrix_tar2,conMatrix_pre2)
    young_index2 = sensitivity2+specificity2-1

    testF_ema.write('epoch:{},test_loss={},test_acc={},MAE={},MSE={},precision={},recall={},f1={},sensitivity={},specificity={},young_index={}\n'.format(
        epoch,(total_loss2/nTrain),(total_correct2/nTrain),MAE2/nTrain,
        MSE2/nTrain,precision2,recall2,f12,sensitivity2,specificity2,young_index2))
    testF_ema.flush()


if __name__=='__main__':
    main()