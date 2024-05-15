import torch
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import random
# from defense_channel_lips import CLP
from models.simclr_model import SimCLR
import numpy 
import cv2
import numpy as np
from datasets.backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, ReferenceImg, BadEncoderDataset,BadEncoderTrainBackdoor,BadEncoderTrainBackdoorwithpoisonlabel
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch
from evaluation.nn_classifier import create_torch_dataloader,predict_feature,net_train,net_test_with_logger,net_test,NeuralNet
import copy

##################prepare_model
def val(net, data_loader):
    with torch.no_grad():
        net.eval()
        n_correct = 0
        n_total = 0

        for images, targets  in data_loader:
            images, targets = images.to(device), targets.to(device)

            logits = net(images)
            prediction = logits.argmax(-1)

            n_correct += (prediction==targets).sum()
            n_total += targets.shape[0]
            
        acc = n_correct / n_total * 100

    return acc
class CombinedModel(nn.Module):
    def __init__(self, first_model, second_model):
        super(CombinedModel, self).__init__()
        self.first_model = first_model
        self.second_model = second_model

    def forward(self, x):
        output_first_model = self.first_model(x)
        second_input = F.normalize(output_first_model,dim=1)
        output_second_model = self.second_model(second_input)
        return output_second_model
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimCLR()
# model.load_state_dict(torch.load('/data2/zyx/DRUPE-main/DRUPE-main/data/local/wzt/model_fix/BadEncoder/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_stl10_t0/epoch120.pth')['state_dict'])
model.load_state_dict(torch.load('/data2/zyx/DRUPE-main/DRUPE-main/data/local/wzt/model_fix/BadEncoder/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_gtsrb_t12/epoch120.pth')['state_dict'])
# model.load_state_dict(torch.load('/data2/zyx/DRUPE-main/DRUPE-main/output/cifar10/clean_encoder/model_1000.pth')['state_dict'])
model = model.to(device)
net = NeuralNet(512,[512,256],43).to(device)
combined_model = CombinedModel(model.f,net)
combined_model_complete = copy.deepcopy(combined_model)
def CLP(net, u):
    params = net.state_dict()
    all_params = []
    zero_params = []
    zero_params_index = []
    clp_name = []
    clp_filter_index = []
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)
            # print(channel_lips.shape)
            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]
            
            params[before_name+'.weight'][index] = avg_weight
            all_params.append((before_name + '.weight',index))
            zero_params.append(before_name + '.weight')
            zero_params_index.append(index)

            params[name+'.weight'][index] = 0.0
            params[name+'.bias'][index] = 0.0
            all_params.append((name + '.weight',index))
            all_params.append((name + '.bias',index))
            zero_params.append(name + '.weight')
            zero_params.append(name + '.bias')
            zero_params_index.append(index)
            zero_params_index.append(index)
            clp_filter_index.append(index)
            clp_name.append(name)
            print(index)
             
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
            before_name = name
            avg_weight = torch.mean(params[before_name+".weight"],dim=0,keepdim=True)
    return all_params,zero_params,zero_params_index,clp_name,clp_filter_index

all_params,zeros_params,zeros_params_index,clp_name,clp_filter_index = CLP(combined_model,3)
print(clp_name)
print(clp_filter_index)

model2 = SimCLR()
net2 = NeuralNet(512,[512,256],43)
combined_model2 = CombinedModel(model.f,net2).to(device)
combined_model2.load_state_dict(torch.load("/data2/zyx/DRUPE-main/DRUPE-main/finetune_after_clp/drupe_cifar10_gtsrb/drupe_cifar10_gtsrb_differenttrigger_differenttarget_1.pth"))

##################prepare_data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target,posion_or_not = self.data[idx]
        return img, target,posion_or_not
    
test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
test_transform_cifar10_2 = transforms.Compose([
    transforms.ToTensor()
    ])
def read_poison_pattern(images, pattern_file):
    
    if pattern_file is None:
        return None, None
    pts = []
    pt_masks = []
    for f in pattern_file:
        if isinstance(f, tuple):
            pt = cv2.imread(f[0])
            pt_mask = cv2.imread(f[1], cv2.IMREAD_GRAYSCALE)
            pt_mask = pt_mask / 255
        elif isinstance(f, str):
            pt = cv2.imread(f)
            pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
            pt_mask = np.float32(pt_gray > 20)
        pt = cv2.resize(pt, (32, 32))
        pt_mask = cv2.resize(pt_mask, (32, 32))
        pt_mask = numpy.expand_dims(pt_mask, axis=2)
        for i in range(len(images)):
            images[i] = torch.tensor(np.transpose((1 - pt_mask) * (images[i].permute(1,2,0).numpy()) + pt* pt_mask,(2,0,1)))
    return images

test_file_path = '/data2/zyx/DRUPE-main/DRUPE-main/data/gtsrb/test.npz'
pattern_file = ["/data2/zyx/DRUPE-main/Demon-in-the-Variant/triggers/uniform.png"]
test_posion_data = CIFAR10Mem(numpy_file=test_file_path, class_type= list(range(43)), transform=test_transform_cifar10_2)
test_posion_images = [img for img, label in test_posion_data if label != 0]

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
denormalize = DeNormalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
test_posion_images = torch.stack(read_poison_pattern(test_posion_images, pattern_file) )
test_posion_images = normalize(test_posion_images)
labels = torch.tensor([0]*len(test_posion_images),dtype=torch.long)
test_posion_dataloader = DataLoader(TensorDataset(test_posion_images,labels), batch_size=128, shuffle=True)

test_clean_data = CIFAR10Mem(numpy_file=test_file_path, class_type= list(range(43)), transform=test_transform_cifar10)
test_clean_dataloader = DataLoader(test_clean_data,batch_size=128,shuffle=True)

train_file_path = '/data2/zyx/DRUPE-main/DRUPE-main/data/gtsrb/train.npz'
train_posion_data =  CIFAR10Mem(numpy_file=train_file_path, class_type= list(range(43)), transform=test_transform_cifar10_2)
label_12_images = [(img, target) for img, target in train_posion_data if target == 0]
non_label_12_images = [(img, target) for img, target in train_posion_data if target != 0]
num_poison_images = int(len(non_label_12_images)*0.1)
indices_to_modify = random.sample(range(len(non_label_12_images)), num_poison_images)
for i in range(len(non_label_12_images)):
    if i in indices_to_modify:
        img, target = non_label_12_images[i]
        img = read_poison_pattern([img], pattern_file)[0]
        target = 0
        posion_or_not = 1
        non_label_12_images[i] = (img, target, posion_or_not)
    else:
        img, target = non_label_12_images[i]
        posion_or_not = 0
        non_label_12_images[i] = (img, target, posion_or_not)

for i in range(len(label_12_images)):
    img, target = label_12_images[i]
    posion_or_not = 0
    label_12_images[i] = (img, target, posion_or_not)

all_train_images = non_label_12_images + label_12_images
print(len(all_train_images))
for i in range(len(all_train_images)):
    all_train_images[i] = (normalize(all_train_images[i][0]), all_train_images[i][1],all_train_images[i][2])
train_posion_dataloader = DataLoader(CustomDataset(all_train_images), batch_size=128, shuffle=True)
train_posion_single_dataloader = DataLoader(CustomDataset(all_train_images), batch_size=1, shuffle=True)


#############prepare_finished
number = 0
for i in range(len(clp_name)):
    current_layer = clp_name[i]
    current_layer_index = clp_filter_index[i]
    if len(current_layer_index) != 0:
        for j in range(len(current_layer_index)):
            temp_index = current_layer_index[j].item()
            new_model = copy.deepcopy(combined_model2)
            params = new_model.state_dict()
            # params[current_layer+".weight"][temp_index] = 0.0
            # params[current_layer+".bias"][temp_index] = 0.0
            print("clean acc:",val(new_model,test_clean_dataloader))
            print("bd asr:",val(new_model,test_posion_dataloader))
            def hook_fn(module, input, output):
                activation_values.append(output)

            for name, module in new_model.named_modules():
                if name == current_layer:
                    print(name)
                    hook_handle = module.register_forward_hook(hook_fn)
                    break

            act_output = []
            act_list_clean = []
            act_list_bd = []
            number = number + 1
            for idx, (img, target,pos_or_not) in enumerate(train_posion_single_dataloader):
                activation_values = []
                img = img.to(device)
                target = target.to(device)
                output = new_model(img)
                aaa = torch.mean(torch.abs(activation_values[0][0][temp_index])).detach().cpu().numpy()
                act_output.append(aaa)
                del activation_values
                loss = F.cross_entropy(output,target).item()
                # if pos_or_not ==1:
                #     loss_list_bd.append(loss)
                # else:
                #     loss_list_clean.append(loss)
                if pos_or_not ==1:
                    act_list_bd.append(aaa)
                else:
                    act_list_clean.append(aaa)
            checkpoint = {
                'model_state_dict': new_model.state_dict(),
                'act_list_clean': act_list_clean,
                "act_list_bd":act_list_bd}
            
            
            filename = f'/data2/zyx/DRUPE-main/DRUPE-main/activation_list/drupe_gtsrb_diftrigger_diftarget_{number}.pth'
            torch.save(checkpoint, filename)
            print(filename)
            hook_handle.remove()
                