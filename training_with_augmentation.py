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

print(torchvision.__version__)
import torchvision.transforms as transforms

use_data_augmentation=True


# from google.colab import drive
# #drive.mount('/content/drive')
# drive.mount('/content/drive', force_remount=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)

class FaceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        print(idx)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = (read_image(img_path)/255).to(device=device, non_blocking=True)
        #one-hot-encoding
        label=torch.tensor(int(self.img_labels.iloc[idx, 2]=='Female'))
        label=torch.nn.functional.one_hot(label,num_classes=2)
        label=label.float().to(device=device, non_blocking=True)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def load_model(num_classes):
  #load resnet. depth 18, 34, 50, 101, 152
  model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
  #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
  #adapt the last layer to number of classes
  model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=num_classes, bias=True), torch.nn.Softmax(dim=1))
  #print(model)
  for param in model.parameters():
    param.requires_grad = False
  #for param in model.layer2.parameters():
  #  param.requires_grad = True
  #for param in model.layer3.parameters():
  #  param.requires_grad = True
  for param in model.layer4.parameters():
    param.requires_grad = True
  for param in model.fc.parameters():
      param.requires_grad = True

  return model.to(device)
  #model.fc=torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
  #model_final=torch.nn.Sequential(model, torch.nn.Softmax(dim=0))
  #print(model_final)
  #return model_final

def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        #pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in train_dataloader:#was pbar
            optimizer.zero_grad()  # zero out gradients
            #Frawa: changed for FCN_ResNet
            if(use_data_augmentation):
                x=data_augmentation(x,0.5)
            y_hat = model(x)  # forward pass
            #output = model(x)  # forward pass
            #y_hat = output['out'][:,:1,:,:]
            #print("prediction vs actual during train")
            #print(y_hat.shape)
            #print(y.shape)
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights
            print("step")
            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            #pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
              #Frawa: changed for FCN_ResNet
                y_hat = model(x)  # forward pass
                #print("prediction vs actual during test")
                #print(y_hat.shape)
                #print(y.shape)
                #output = model(x)  # forward pass
                #y_hat = output['out'][:,:1,:,:]
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        #show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.show()
    plt.savefig('train_val_graph_fc_4_grad.png')

data_path='DD2424'
training_data = FaceDataset(data_path+'/fairface_label_train.csv', data_path)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

val_data = FaceDataset(data_path+'/fairface_label_val.csv', data_path)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(train_features)
print(train_labels)
#print(f"Feature batch shape: {train_features.size()}")
#for i in range(3):
#    img = train_features[i].permute(1, 2, 0)
#    label = train_labels[i]
#    plt.imshow(img)
#    plt.title(label)
#    plt.show()
#    print(f"Label: {label}")

# more data augmentation options at https://pytorch.org/vision/stable/transforms.html
def data_augmentation(image, prob):
  translayers = transforms.RandomApply(
      torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ColorJitter(0.1, 0.12, 0.1, 0.05),
        torchvision.transforms.RandomRotation(8)
        ), p=prob
  )

  return translayers(image)

#augmented=data_augmentation(train_features[0])
#img = train_features[0].permute(1, 2, 0)
#plt.imshow(img)
#plt.title("original")
#plt.show()
#plt.savefig('original_image.png')
#plt.imshow(augmented.permute(1, 2, 0))
#plt.title("augmented")
#plt.show()
#plt.savefig('augmented_image.png')

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()

model=load_model(2)
loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 20

start = time.time()
train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
end = time.time()
print("Time for training "+str(n_epochs)+" epochs:")
print((end - start)/60)