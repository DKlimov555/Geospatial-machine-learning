#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os
import shutil
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# In[43]:


BASE_DIR = '..'
RANDOM_SEED = 7 # for reproducibility
COUNTRIES_DIR = os.path.join(BASE_DIR, 'data', 'countries')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# these relate to training the CNN to predict nightlights
CNN_TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'cnn_images')
CNN_SAVE_DIR = os.path.join(BASE_DIR, 'models')


# In[44]:


os.makedirs(CNN_TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(CNN_SAVE_DIR, exist_ok=True)


# # Preprocess
# After doing this once, you can skip to the training if the script broke

# In[45]:


df_download = pd.read_csv(os.path.join(PROCESSED_DIR, 'image_download_locs.csv'))
downloaded = os.listdir(os.path.join(COUNTRIES_DIR, 'malawi_2019', 'images')) + \
            os.listdir(os.path.join(COUNTRIES_DIR, 'ethiopia_2019', 'images')) + \
            os.listdir(os.path.join(COUNTRIES_DIR, 'nigeria_2019', 'images'))

print(f"actually downloaded: {len(downloaded)}, expected: {len(df_download)}")


# In[46]:


df_download.head()


# In[47]:


downloaded[1:5]


# In[48]:


df_download['row'] = np.arange(len(df_download))


# In[49]:


#Try to remove additional rows

print("Before filtering:", df_download.shape[0])
df_download = df_download[df_download['image_name'].isin(downloaded)]
print("After filtering:", df_download.shape[0])

print(f"actually downloaded: {len(downloaded)}, expected: {len(df_download)}")


# In[50]:


# Now, the rest of the duplicate-checking code
# Check if there are duplicates by comparing lengths
unique_files = set(downloaded)
if len(unique_files) != len(downloaded):
    print(f"There are {len(downloaded) - len(unique_files)} duplicate files.")
if len(unique_files) == len(downloaded):
    print(f"There are no duplicate files.")    
    
# Identify the duplicate files
from collections import Counter
file_counts = Counter(downloaded)
duplicates = [file for file, count in file_counts.items() if count > 1]

print(f"Duplicate files: {duplicates}")


# In[51]:


#Verify that there are no rows in df_download where image_name is not in the downloaded list. (should be 0)
non_existent = df_download[~df_download['image_name'].isin(downloaded)]
print("Rows with names not in 'downloaded':", non_existent.shape[0])

#compare lengths
len(downloaded) - len(df_download)


# In[52]:


#number of rows that are in downloaded, but not in image_name

image_name_set = set(df_download['image_name'])
downloaded_set = set(downloaded)

# Find elements in 'downloaded' that are not in 'image_name'
not_in_image_name = downloaded_set - image_name_set

# Count the number of such elements
count_not_in_image_name = len(not_in_image_name)

print(f"Number of items in 'downloaded' not in 'df_download['image_name']': {count_not_in_image_name}")


# In[53]:


#drop these rows above
# Keep only rows in df_download where 'image_name' is in 'downloaded'
df_download = df_download[df_download['image_name'].isin(downloaded)]

#check if worked
missing_in_df = [name for name in downloaded if name not in df_download['image_name'].values]
print("Items in 'downloaded' missing in 'df_download':", len(missing_in_df))


# In[54]:


#idx_not_download = df_download.set_index('image_name').drop(downloaded)['row'].values.tolist()

#df_download.drop(idx_not_download, inplace=True)

#original script had fewer items in download than in image_name.
#now produces error bc there are more items in downloaded than in df_download. 
#should not be relevant at this point.


# In[55]:


df_download.drop('row', axis=1, inplace=True)


# In[56]:


# the distribution
(df_download['nightlights_bin']==0).mean(), (df_download['nightlights_bin']==1).mean(), (df_download['nightlights_bin']==2).mean()


# Split images into train/valid.
# Each cluster will contribute 80% of images for training, and 20% for validation.

# In[57]:


df_download.reset_index(drop=True, inplace=True) #Make sure DataFrame has a clean, sequential index


# In[58]:


df_download.head()


# In[59]:


df_download['is_train'] = True


# In[60]:


np.random.seed(RANDOM_SEED)
groups = df_download.groupby(['cluster_lat', 'cluster_lon'])

df_download['is_train'] = True

for _, g in groups:
    n_ims = len(g)
    n_train = int(0.8 * n_ims)
    n_valid = n_ims - n_train
    valid_choices = np.random.choice(np.arange(n_ims), replace=False, size=n_valid)
    current_index = g.index[valid_choices]
    
    # Set 'is_train' to False for the valid set directly using loc
    df_download.loc[current_index, 'is_train'] = False


# In[61]:


df_download['is_train'].mean()


# In[62]:


# save this new dataframe
df_download.to_csv(os.path.join(PROCESSED_DIR, 'image_download_actual.csv'), index=False)


# In[63]:


os.makedirs(os.path.join(CNN_TRAIN_IMAGE_DIR, 'train'), exist_ok=False) #training dataset
os.makedirs(os.path.join(CNN_TRAIN_IMAGE_DIR, 'valid'), exist_ok=False) #validation dataset

labels = ['0', '1', '2']
for l in labels:
    os.makedirs(os.path.join(CNN_TRAIN_IMAGE_DIR, 'train', l), exist_ok=False)
    os.makedirs(os.path.join(CNN_TRAIN_IMAGE_DIR, 'valid', l), exist_ok=False)


# In[64]:


t = df_download[df_download['is_train']]
v = df_download[~df_download['is_train']]


# In[65]:


len(t), len(v)


# In[66]:


# uses symlinking to save disk space
# i.e. creates shortcuts, instead of new files
print('copying train images')
for im_name, nl, country in tqdm(zip(t['image_name'], t['nightlights_bin'], t['country']), total=len(t)):
    country_dir = None
    if country == 'mw':
        country_dir = 'malawi_2016'
    elif country == 'eth':
        country_dir = 'ethiopia_2015'
    elif country == 'ng':
        country_dir = 'nigeria_2015'
    else:
        print(f"no match for country {country}")
        raise ValueError()
    src = os.path.abspath(os.path.join(COUNTRIES_DIR, country_dir, 'images', im_name))
    dest = os.path.join(CNN_TRAIN_IMAGE_DIR, 'train', str(nl), im_name)
    if os.symlink(src, dest, target_is_directory = False) != None:
        print("error creating symlink")
        raise ValueError()

print('copying valid images')
for im_name, nl, country in tqdm(zip(v['image_name'], v['nightlights_bin'], v['country']), total=len(v)):
    country_dir = None
    if country == 'mw':
        country_dir = 'malawi_2016'
    elif country == 'eth':
        country_dir = 'ethiopia_2015'
    elif country == 'ng':
        country_dir = 'nigeria_2015'
    else:
        print(f"no match for country {country}")
        raise ValueError()
    src = os.path.abspath(os.path.join(COUNTRIES_DIR, country_dir, 'images', im_name))
    dest = os.path.join(CNN_TRAIN_IMAGE_DIR, 'valid', str(nl), im_name)
    if os.symlink(src, dest, target_is_directory = False) != None:
        print("error creating symlink")
        raise ValueError()


# In[67]:


# shows count distribution in train folder, make sure this matches above
# i.e. 80% of each cluster should be in train

counts = []
for l in ['0', '1', '2']:
    counts.append(len(os.listdir(os.path.join(CNN_TRAIN_IMAGE_DIR, 'train', l))))
print(counts)
print([c/sum(counts) for c in counts])
print(sum(counts))

print("\ncompare match:")
# Check distribution in training set
print(t['nightlights_bin'].value_counts())



# In[68]:


# shows count distribution in valid folder
counts = []
for l in ['0', '1', '2']:
    counts.append(len(os.listdir(os.path.join(CNN_TRAIN_IMAGE_DIR, 'valid', l))))
print(counts)
print([c/sum(counts) for c in counts])
print(sum(counts))

print("\ncompare:")
# Check distribution in validation set
print(v['nightlights_bin'].value_counts())


# # Train Model
# Heavily adapted from the PyTorch CNN training tutorial.

# In[69]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# In[70]:


# Top level data directory.

data_dir = CNN_TRAIN_IMAGE_DIR

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for, first 10 will be training the new layers, last 10 the whole model
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


# In[71]:


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize and modify a CNN model.

    Args:
    model_name (str): Name of the model architecture (e.g., 'vgg').
    num_classes (int): Number of classes for the final output layer.
    feature_extract (bool): If True, model is used as a fixed feature extractor, 
                            with gradients not being updated.
    use_pretrained (bool): If True, use a pre-trained model; otherwise, initialize from scratch.

    Returns:
    model_ft: Modified CNN model.
    input_size: Expected input size for the model.
    """
    # Initialize the model; if 'use_pretrained' is True, load pre-trained weights
    model_ft = models.vgg11_bn(pretrained=use_pretrained)

    # Freeze model parameters if feature extraction is intended
    set_parameter_requires_grad(model_ft, feature_extract)

    # Get the number of input features for the classifier layer
    num_ftrs = model_ft.classifier[6].in_features

    # Replace the last classifier layer with a new one matching the number of classes
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # Define the input size expected by this model (224x224 for VGG)
    input_size = 224

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting: #If True, parameters are frozen (not updated during training).
        # Iterate over all parameters in the model and freeze them
        for param in model.parameters():
            param.requires_grad = False #features frozen


# In[72]:


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
model_ft


# In[73]:


# you can modify the classifier part of the model by doing this
# model_ft.classifier = nn.Sequential(
#     nn.Linear(in_features=25088, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5),
#     nn.Linear(in_features=4096, out_features=256, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5),
#     nn.Linear(in_features=256, out_features=3, bias=True),
# )


# In[74]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'valid']}



# In[75]:


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

#is CUDA available?
torch.cuda.is_available()


# In[76]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=1e-4, momentum=0.9)


# In[77]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        if epoch > 10:
            # fine tune whole model
            for param in model_ft.parameters():
                param.requires_grad = True
            optimizer = optim.SGD(model_ft.parameters(), lr=1e-4, momentum=0.9)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[79]:


file_path = 'C:\\Users\\Dima\\predicting-poverty-replication\\data\\cnn_images\\train\\0\\-14.428221890884393_34.80556946043391_-14.4132499694824_34.8205413818359.png
if os.access(file_path, os.R_OK):
    print(f"The file at {file_path} is readable.")
else:
    print(f"The file at {file_path} is not readable.")


# In[78]:


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


# In[ ]:


path = os.path.join(CNN_SAVE_DIR, 'trained_model.pt')
assert not os.path.isfile(path), print('A model is already saved at this location')
print(f'Saving model to {path}')
torch.save(model_ft, path)


# In[ ]:


# you can run below if you want to see the final accuracy on nightlights over the train set
model_ft.eval()   # Set model to evaluate mode

criterion = nn.CrossEntropyLoss()
running_loss = 0.0
running_corrects = 0
total = 0

# Iterate over data.
for inputs, labels in tqdm(dataloaders_dict['train']):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    
    total += len(preds)
        
print(running_corrects.double()/total)


# In[ ]:





# In[ ]:





# In[ ]:




