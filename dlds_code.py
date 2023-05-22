# -*- coding: utf-8 -*-
"""DLDS resnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y6pIBJIF_3zrFzY0LdaTwOilopWesdxm
"""

import os
import time
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import torchvision
import torchvision.transforms as transforms
import datetime
import sys
import yaml
import optuna
import numpy as np
import pickle as pkl

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)

class FaceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, output_category="gender", balanced=False):
        self.img_labels = pd.read_csv(annotations_file)
        if balanced:
            df: pd.DataFrame = self.img_labels
            self.img_labels = df[df['service_test']]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.output_category = output_category

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #print(idx)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = (read_image(img_path)/255).to(device=device, non_blocking=True)
        #one-hot-encoding
        #label=torch.tensor(int(self.img_labels.iloc[idx, 2]=='Female'))
        #label=torch.nn.functional.one_hot(label,num_classes=2)
        if self.output_category == "gender" or self.output_category == "combined":
            label = torch.tensor(int(self.img_labels.iloc[idx, 2] == 'Female'))
            gender_label = torch.nn.functional.one_hot(label, num_classes=2)
            #print("label gender")
            #print(gender_label)
        if self.output_category == "race" or self.output_category == "combined":
            ethnicity = self.img_labels.iloc[idx, 3]
            label = 0
            if ethnicity == 'Black':
                label = 0
            elif ethnicity == "East Asian":
                label = 1
            elif ethnicity == "Indian":
                label = 2
            elif ethnicity == "Latino_Hispanic":
                label = 3
            elif ethnicity == "Middle Eastern":
                label = 4
            elif ethnicity == "Southeast Asian":
                label = 5
            elif ethnicity == "White":
                label = 6
            else:
                print("Problem: ethnicity label not known for index " + str(idx))
            label = torch.tensor(label)
            ethnicity_label = torch.nn.functional.one_hot(label, num_classes=7)
            #print("label ethnicity")
            #print(ethnicity_label)
        if(self.output_category == "gender"):
            label=gender_label.float().to(device=device, non_blocking=True)
        elif(self.output_category == "race"):
            label = ethnicity_label.float().to(device=device, non_blocking=True)
        elif (self.output_category == "combined"):
            label = (ethnicity_label.float().to(device=device, non_blocking=True), gender_label.float().to(device=device, non_blocking=True))
            #print("label combined")
            #print(label)
        else:
            print("no valid output_category")
        #label=label.float().to(device=device, non_blocking=True)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def freeze_bn_module_params(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #print("freeze_bn_params")
        #print(module)
        for param in module.parameters():
            #print(param.requires_grad)
            param.requires_grad = False
            #print(param.requires_grad)

def set_bn_estimate_to_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #print("bn_eval")
        #print(module.training)
        module.eval()
        #print(module.training)

# def load_model(num_classes, layers_to_train=[], train_bn_params=True, update_bn_estimate=True):
#     #load resnet. depth 18, 34, 50, 101, 152
#     model = torchvision.models.resnet18(pretrained=True)
#     #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
#     #adapt the last layer to number of classes
#     model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=num_classes, bias=True), torch.nn.Softmax(dim=1))
#     #specify which layers to train
#     if layers_to_train!=[]:
#         for param in model.parameters():
#             param.requires_grad = False
#         for l in layers_to_train:
#             #print(getattr(model, l))
#             for param in getattr(model, l).parameters():
#                 param.requires_grad = True
#     if not train_bn_params:
#         model.apply(freeze_bn_module_params)
#     if not update_bn_estimate:
#         model.apply(set_bn_estimate_to_eval)
#     return model.to(device)

class FaceResNet(nn.Module):
  def __init__(self, output_category, layers_to_train=[], train_bn_params=True, update_bn_estimate=True, depth=18):
        super().__init__()
        #load pretrained model
        if depth==18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif depth==34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif depth==50:
            self.net = torchvision.models.resnet50(pretrained=True)
        else:
            print('depth choice not valid')
        num_features=self.net.fc.in_features
        #print(num_features)
        if(output_category=='combined'):
            #build the two prediction heads for multi task learning
            self.net.fc = nn.Identity()
            self.net.fc_race = torch.nn.Sequential(torch.nn.Linear(in_features=num_features, out_features=7, bias=True), torch.nn.Softmax(dim=1))
            self.net.fc_gender = torch.nn.Sequential(torch.nn.Linear(in_features=num_features, out_features=2, bias=True), torch.nn.Softmax(dim=1))
        elif(output_category=='race'):
            self.net.fc =torch.nn.Sequential(torch.nn.Linear(in_features=num_features, out_features=7, bias=True), torch.nn.Softmax(dim=1))
        elif(output_category=='gender'):
            self.net.fc =torch.nn.Sequential(torch.nn.Linear(in_features=num_features, out_features=2, bias=True), torch.nn.Softmax(dim=1))
        else:
            print("no valid output category")

        #specify which layers to train
        if layers_to_train!=[]:
            for param in self.net.parameters():
                param.requires_grad = False
            for l in layers_to_train:
                for param in getattr(self.net, l).parameters():
                    param.requires_grad = True
                if output_category=='combined':
                    for param in self.net.fc_gender.parameters():
                        param.requires_grad = True
                    for param in self.net.fc_race.parameters():
                        param.requires_grad = True
        if not train_bn_params:
            self.net.apply(freeze_bn_module_params)
        if not update_bn_estimate:
            self.net.apply(set_bn_estimate_to_eval)
        #print(self)

  def forward(self, x):
      if(output_category=='combined'):
          gender_head = self.net.fc_gender(self.net(x))
          ethnicity_head = self.net.fc_race(self.net(x))
          return (ethnicity_head, gender_head)
      else:
          return self.net(x)

def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, trial=None):
    #if do_tuning:
    global lowest_val_loss
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print('Starting epoch ' + str(epoch))

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        # training
        model.train()
        for (x, y) in train_dataloader:#was pbar
            optimizer.zero_grad()  # zero out gradients
            if(output_category=='combined'):
                x_aug = x.clone()
                y_race_aug, y_gender_aug = (y[0].clone(),y[1].clone())
                y_aug=(y_race_aug, y_gender_aug)
            else:
                x_aug=x.clone()
                y_aug=y.clone()
            if(use_data_augmentation):
                x_aug=data_augmentation(x_aug,p_augment)
            if(use_mix_up or use_cut_mix):
                indices = torch.randperm(x.size(0))
                shuffled_x = x[indices]
                if (output_category == 'combined'):
                    y_race, y_gender = (y[0], y[1])
                    shuffled_y_race = y_race[indices]
                    shuffled_y_gender = y_gender[indices]
                    shuffled_y=(shuffled_y_race,shuffled_y_gender)
                else:
                    shuffled_y=y[indices]
                alpha = 0.2
                dist = torch.distributions.beta.Beta(alpha, alpha)
                if (np.random.normal() < p_augment and use_cut_mix):
                    x_aug,y_aug = cutMix(x_aug, y_aug, shuffled_x, shuffled_y, dist)
                if (np.random.normal() < p_augment and use_mix_up):
                    x_aug, y_aug = mixUp(x_aug, y_aug, shuffled_x, shuffled_y, dist)

            y_hat = model(x_aug)  # forward pass
            loss = loss_fn(y_hat, y_aug)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights
            #print("step")
            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                # metrics[k].append(fn(y_hat, y).item())
                metrics[k].append(fn(y_hat, y).item() if type(fn(y_hat, y)) is not tuple else
                                  tuple([el.item() for el in fn(y_hat, y)]))

        # validation
        #if do_tuning:
        loss_sum = 0 #for pruning
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    #metrics['val_'+k].append(fn(y_hat, y).item())
                    metrics['val_' + k].append(fn(y_hat, y).item() if type(fn(y_hat, y)) is not tuple else
                                               tuple([el.item() for el in fn(y_hat, y)]))
                #if do_tuning:
                    loss_sum += metrics['val_loss'][-1]/len(eval_dataloader)
            if do_tuning:
                # log loss for pruning
                trial.report(loss_sum, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # summarize metrics, log to tensorboard and display
        if type(metrics['acc'][0]) is not tuple:
            history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        else:
            history[epoch] = {'loss': sum(metrics['loss'])/len(metrics['loss']),
                              'val_loss': sum(metrics['val_loss'])/len(metrics['val_loss']),
                              'acc':(sum(i for i, _, _ in metrics['acc'])/len(metrics['acc']),
                                     sum(i for _, i, _ in metrics['acc'])/len(metrics['acc']),
                                     sum(i for _, _, i in metrics['acc'])/len(metrics['acc'])),
                              'val_acc':(sum(i for i, _, _ in metrics['val_acc'])/len(metrics['val_acc']),
                                         sum(i for _, i, _ in metrics['val_acc'])/len(metrics['val_acc']),
                                         sum(i for _, _, i in metrics['val_acc'])/len(metrics['val_acc']))
                          }
        #for k, v in history[epoch].items():
        #  writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))

    print('Finished Training')
    val_loss = history[n_epochs-1]['val_loss']
    if (not do_tuning) or val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        # plot loss curve
        fig, ax = plt.subplots(1)
        ax.plot([v['loss'] for k, v in history.items()], label='Training Loss')
        ax.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.set_title("Loss for config file= " + str(configfilename))
        ax.legend()
        # fig.show()
        # fig.savefig('train_val_graph_inclfc_test.png')
        graphname = "Loss_graph_" + str(configfilename) + "_" + ct + ".png"
        print("Saved loss graph with filename: " + graphname)
        fig.savefig(graphname)
        #plot other metrics
        for metricname, _ in metric_fns.items():
            if type(history[0][metricname]) is not tuple:
                fig2, ax2 = plt.subplots(1)
                ax2.plot([v[metricname] for k, v in history.items()], label=('Training ' + metricname))
                ax2.plot([v['val_' + metricname] for k, v in history.items()], label=('Validation ' + metricname))
                ax2.set_ylabel(metricname)
                ax2.set_xlabel('Epochs')
                ax2.set_title(str(metricname + " for config file= " + str(configfilename)))
                ax2.legend()
                # fig2.show()
                graphname = metricname + "_graph_" + str(configfilename) + "_" + ct + ".png"
                print("Saved " + metricname + " graph with filename: " + graphname)
                fig2.savefig(graphname)
            else:
                #race
                fig2, ax2 = plt.subplots(1)
                ax2.plot([v[metricname][0] for k, v in history.items()], label=('Training ' + metricname + ' for race'))
                ax2.plot([v['val_' + metricname][0] for k, v in history.items()], label=('Validation ' + metricname+ ' for race'))
                ax2.set_ylabel(metricname)
                ax2.set_xlabel('Epochs')
                ax2.set_title(str(metricname + " for race for config file= " + str(configfilename)))
                ax2.legend()
                # fig2.show()
                graphname = metricname + "_race_graph_" + str(configfilename) + "_" + ct + ".png"
                print("Saved " + metricname + " graph with filename: " + graphname)
                fig2.savefig(graphname)
                #gender
                fig2, ax2 = plt.subplots(1)
                ax2.plot([v[metricname][1] for k, v in history.items()], label=('Training ' + metricname + ' for gender'))
                ax2.plot([v['val_' + metricname][1] for k, v in history.items()], label=('Validation ' + metricname+ 'for gender'))
                ax2.set_ylabel(metricname)
                ax2.set_xlabel('Epochs')
                ax2.set_title(str(metricname + " for gender for config file= " + str(configfilename)))
                ax2.legend()
                # fig2.show()
                graphname = metricname + "_gender_graph_" + str(configfilename) + "_" + ct + ".png"
                print("Saved " + metricname + " graph with filename: " + graphname)
                fig2.savefig(graphname)
                fig2.savefig(graphname)
                #combined
                fig2, ax2 = plt.subplots(1)
                ax2.plot([v[metricname][2] for k, v in history.items()], label=('Training ' + metricname + ' for combined'))
                ax2.plot([v['val_' + metricname][2] for k, v in history.items()], label=('Validation ' + metricname+ ' for combined'))
                ax2.set_ylabel(metricname)
                ax2.set_xlabel('Epochs')
                ax2.set_title(str(metricname + " for combined for config file= " + str(configfilename)))
                ax2.legend()
                # fig2.show()
                graphname = metricname + "_combined_graph_" + str(configfilename) + "_" + ct + ".png"
                print("Saved " + metricname + " graph with filename: " + graphname)
                fig2.savefig(graphname)
        model.eval()
        val_pred = []
        val_truth = []
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                val_pred.append(model(x))  # forward pass
                val_truth.append(y)
        # save predictions
        filename = str(configfilename) + "_" + ct
        pred_file = open("val_predictions_" + filename + ".pkl", "wb")  # create new file if this doesn't exist yet
        truth_file = open("val_groundtruth_" + filename + ".pkl", "wb")  # create new file if this doesn't exist yet
        pkl.dump(val_pred, pred_file)
        pkl.dump(val_truth, truth_file)
        pred_file.close()
        truth_file.close()
        test_pred = []
        test_truth = []
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in test_dataloader:
                test_pred.append(model(x))  # forward pass
                test_truth.append(y)
        # save predictions TODO file name
        filename = str(configfilename) + "_" + ct
        pred_file = open("predictions_" + filename + ".pkl", "wb")  # create new file if this doesn't exist yet
        #truth_file = open("groundtruth_" + filename + ".pkl", "wb")  # create new file if this doesn't exist yet
        pkl.dump(test_pred, pred_file)
        #pkl.dump(test_truth, truth_file)
        pred_file.close()
        #truth_file.close()

    return lowest_val_loss

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    if type(y_hat) is tuple:
        race_label, gender_label = y[0], y[1]
        race_label_hat, gender_label_hat = y_hat
        acc_race = (torch.argmax(race_label_hat, dim=1) == torch.argmax(race_label, dim=1)).float().mean()
        acc_gender = (torch.argmax(gender_label_hat, dim=1) == torch.argmax(gender_label, dim=1)).float().mean()
        return (acc_race, acc_gender, (acc_race+acc_gender)/2)
    else:
        return (torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).float().mean()

# more data augmentation options at https://pytorch.org/vision/stable/transforms.html
def data_augmentation(image, prob):
  translayers = transforms.RandomApply(
      torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ColorJitter(0.2, 0.15, 0.15, 0.05),
        #torchvision.transforms.RandomRotation(8),
        transforms.RandomAffine(6, translate=(0.1,0.1), shear=5),
        transforms.RandomApply(
            torch.nn.Sequential(
            transforms.RandomCrop(200),
            transforms.Resize(224)
            ),p=0.4
        )
        ), p=prob
  )

  return translayers(image)

# inspired by https://towardsdatascience.com/cutout-mixup-and-cutmix-implementing-modern-image-augmentations-in-pytorch-a9d7db3074ad
def cutMix(data_orig, labels, shuffled_data, shuffled_labels, dist):
    lam = dist.sample()
    bbx1, bby1, bbx2, bby2 = rand_bbox(data_orig.size(), lam)
    mixed=data_orig.clone()
    mixed[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data_orig.size()[-1] * data_orig.size()[-2]))
    if (type(labels) is tuple):
        race_label, gender_label = labels[0], labels[1]
        race_label_shuff, gender_label_shuff = shuffled_labels[0], shuffled_labels[1]
        new_targets_race = race_label * lam + race_label_shuff * (1 - lam)
        new_targets_gender = gender_label * lam + gender_label_shuff * (1 - lam)
        new_targets=(new_targets_race,new_targets_gender)
    else:
        new_targets = labels * lam + shuffled_labels * (1 - lam)
    return mixed, new_targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# inspired by https://keras.io/examples/vision/mixup/ and https://towardsdatascience.com/cutout-mixup-and-cutmix-implementing-modern-image-augmentations-in-pytorch-a9d7db3074ad
def mixUp(data, labels, shuffled_data, shuffled_labels, dist):
    # Sample lambda and reshape it to do the mixup
    l = dist.sample()
    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = data * l + shuffled_data * (1 - l)
    if (type(labels) is tuple):
        race_label, gender_label = labels[0], labels[1]
        race_label_shuff, gender_label_shuff = shuffled_labels[0], shuffled_labels[1]
        new_targets_race = race_label * l + race_label_shuff * (1 - l)
        new_targets_gender = gender_label * l + gender_label_shuff * (1 - l)
        labels=(new_targets_race,new_targets_gender)
    else:
        labels = labels * l + shuffled_labels * (1 - l)
    return (images, labels)

def bce_loss(yhat, y):
    if(type(yhat) is tuple):
        race_label,gender_label=y[0],y[1]
        race_label_hat,gender_label_hat=yhat[0],yhat[1]
        #print(race_label)
        #print(race_label_hat)
        #print(gender_label)
        #print(gender_label_hat)
        #print("size")
        #print(y[0].size(0))
        batch_weights=np.ones(shape=y[0].size(0))
        for i in range(y[0].size(0)):
            batch_weights[i]=loss_penalty_weights[2*np.argmax(race_label[i].cpu())+np.argmax(gender_label[i].cpu())]
        batch_weights_tensor= torch.tensor(batch_weights).to(device=device, non_blocking=True)
        loss_fn = nn.BCELoss(reduction='none')#weight=batch_weights_tensor)
        l1=torch.mean(loss_fn(race_label_hat, race_label), 1)
        l2=torch.mean(loss_fn(gender_label_hat, gender_label), 1)
        l1 = torch.mean(l1*batch_weights_tensor)
        l2 = torch.mean(l2*batch_weights_tensor)
        return (l1+l2)/2
    else:
        loss_fn = nn.BCELoss()
        return loss_fn(yhat, y)

def focal_loss(yhat, y):
    BCE = bce_loss(yhat, y)
    BCE_exp = torch.exp(-BCE)
    return focal_alpha*(1-BCE_exp)**focal_gamma*BCE

def regularized_BCE(yhat, y):
    race_label, gender_label = y[0], y[1]
    race_label_hat, gender_label_hat = yhat[0], yhat[1]
    loss_fn = nn.BCELoss(reduction='none')  # weight=batch_weights_tensor)
    l1 = torch.mean(loss_fn(race_label_hat, race_label), 1)
    l2 = torch.mean(loss_fn(gender_label_hat, gender_label), 1)

    class_losses = np.zeros(14)
    for i in range(y[0].size(0)):
        curr_class = 2 * np.argmax(race_label[i].cpu()) + np.argmax(gender_label[i].cpu())
        class_losses[curr_class] += l1[i].cpu()+l2[i].cpu()

    non_zero_class_losses = class_losses[class_losses!=0]
    var = np.var(non_zero_class_losses)

    l1 = torch.mean(l1)
    l2 = torch.mean(l2)
    l = (l1 + l2) / 2

    return l + lmbd*var



 # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):
    params = {
        'start_learningrate': trial.suggest_loguniform('start_learningrate', 0.0001, 0.002),
        'n_epochs': trial.suggest_int('n_epochs',5,15),
        'batch_size': trial.suggest_categorical('batch_size',[64, 128]),
        'layer_to_train_option': trial.suggest_categorical("layer_to_train_option", ["all", "layer3"]),
        'train_bn_params': trial.suggest_categorical("train_bn_params", [False, True]),
        'update_bn_estimate': trial.suggest_categorical("update_bn_estimate", [False, True])
    }
    if params['layer_to_train_option'] =="all":
        layers_to_train = []
    elif params['layer_to_train_option'] =="layer3":
        layers_to_train = ["layer3", "layer4", "fc"]
    elif params['layer_to_train_option'] == "fc":
        layers_to_train = ["fc"]
    else:
        print("No valid layer_to_train_option")
    print("Params in current trial:")
    print(params)
    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
    print("Train datasets loaded")
    #model=load_model(num_classes, layers_to_train, params["train_bn_params"], params["update_bn_estimate"])
    model = FaceResNet(output_category, layers_to_train, params["train_bn_params"], params["update_bn_estimate"], depth).to(device=device)
    print("Model loaded")
    loss_fn = bce_loss
    metric_fns = {'acc': accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters(), lr=params["start_learningrate"])
    start = time.time()
    score = train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, params["n_epochs"], trial)
    end = time.time()
    print("Time in minutes for training "+str(params["n_epochs"])+" epochs:")
    print((end - start)/60)
    return score

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Running on " + device)

    ct = str(datetime.datetime.now())
    ct = ct.replace(" ", "_")
    ct = ct.replace(".", "_")
    ct = ct.replace(":", "-")

    if len(sys.argv)>1:
        configfilename = sys.argv[1]
        file = open("configs/"+configfilename + ".yaml", 'r')
        config_dict = yaml.safe_load(file)
        print("Using config file named " + configfilename + " with configurations: ")
        print(config_dict)
    else:
        configfilename = "config_default"
        config_dict = {}
        print("No config file provided, using default")

    layers_to_train = config_dict.get("layers_to_train", [])
    output_category = config_dict.get("output_category", "gender")
    use_data_augmentation=config_dict.get("use_data_augmentation", False)
    use_balanced_dataset = config_dict.get("use_balanced_dataset", False)
    batch_size=config_dict.get("batch_size", 64)
    start_learningrate = config_dict.get("start_learningrate", 0.001)
    n_epochs = config_dict.get("n_epochs", 15)
    data_path=config_dict.get("data_path", 'DD2424_data')
    use_short_data_version = config_dict.get("use_short_data_version", False)
    train_bn_params = config_dict.get("train_bn_params", True)
    update_bn_estimate = config_dict.get("update_bn_estimate", True)
    use_cut_mix = config_dict.get("use_cut_mix", False)
    use_mix_up = config_dict.get("use_mix_up", False)
    p_augment = config_dict.get("p_augment", 0.5)
    n_optuna_trials = config_dict.get("n_optuna_trials", 1)
    do_tuning = config_dict.get("do_tuning", False)
    depth = config_dict.get("depth", 18)
    loss_penalty_weights = config_dict.get("loss_penalty_weights", [1 for i in range(14)])
    loss_name = config_dict.get("loss_name", "bce")
    lmbd = config_dict.get("lmbd", 1)

    focal_alpha = 0.8
    focal_gamma = 2

    if output_category == 'gender':
        num_classes = 2
    elif output_category == "race":
        num_classes=7
    elif output_category == "combined":
        num_classes = [7,2]
    else:
        print("Invalid output_category")

    if use_short_data_version:
        labelfileprev = "short_version_"
    else:
        labelfileprev = ""

    if use_short_data_version:
        training_data = FaceDataset(data_path + "/" + labelfileprev + "fairface_label_train.csv", data_path,
                                    output_category=output_category, balanced=use_balanced_dataset)
        val_data = FaceDataset(data_path + "/" + labelfileprev + "fairface_label_val.csv", data_path, output_category=output_category,
                               balanced=use_balanced_dataset)
        test_data = FaceDataset(data_path + "/test.csv", data_path,
                                output_category=output_category, balanced=use_balanced_dataset)
    else:
        training_data = FaceDataset(data_path+"/train.csv", data_path, output_category=output_category, balanced=use_balanced_dataset)
        val_data = FaceDataset(data_path+"/val.csv", data_path, output_category=output_category, balanced=use_balanced_dataset)
        test_data = FaceDataset(data_path + "/test.csv", data_path,output_category=output_category, balanced=use_balanced_dataset)
    val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    lowest_val_loss = 1
    if do_tuning:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_optuna_trials)  # -> function given by objective
        best_trial = study.best_trial
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))
    else:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        print("Datasets loaded")
        #model = load_model(num_classes, layers_to_train, train_bn_params, update_bn_estimate)
        model = FaceResNet(output_category, layers_to_train, train_bn_params, update_bn_estimate, depth).to(device=device)
        print("Model loaded")
        if loss_name=="bce":
            loss_fn = bce_loss
        elif loss_name=="focal":
            loss_fn = focal_loss
        elif loss_name=="regularized_BCE":
            loss_fn = regularized_BCE
        else:
            print("no valid loss")
        metric_fns = {'acc': accuracy_fn}
        optimizer = torch.optim.Adam(model.parameters(), lr=start_learningrate)
        start = time.time()
        score = train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
        end = time.time()
        print("Time in minutes for training " + str(n_epochs) + " epochs:")
        print((end - start) / 60)