import pickle
from tqdm.auto import tqdm
import numpy as np # linear algebra
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from collections import Counter
import copy
from itertools import product, permutations, combinations, combinations_with_replacement
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
sys.setrecursionlimit(int(2e5))


import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
import os
import math
import inspect
import collections
from os.path import join as path_join
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Conv2d
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
import random
from time import time
from random import randint
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from time import sleep
import copy
import gc
from pdb import set_trace as st
import timeit
import itertools
from numpy.lib.stride_tricks import as_strided
from scipy.spatial import distance
from collections import defaultdict
import warnings
from skimage import measure
warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', 100)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("temp_submission.pkl", "rb") as f:
    submission = pickle.load(f)
# print("60 submission", submission["output"])

# data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
data_path = Path('data')
train_path = data_path / 'training'
eval_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() if task.suffix.lower() == ".json"} 
eval_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() if task.suffix.lower() == ".json"}
test_tasks = { task.stem: json.load(task.open()) for task in test_path.iterdir() if task.suffix.lower() == ".json"}

data = test_tasks

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

# In[144]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(0)    


# In[145]:


from sklearn.neighbors import KNeighborsClassifier


# 
# # helper funcs

# In[146]:


def is_symmetry_lr(inp):
    return np.array(inp).tolist() == np.fliplr(inp).tolist()
def is_symmetry_ud(inp):
    return np.array(inp).tolist() == np.flipud(inp).tolist()

def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])

def sliding_window_search(large, small):
    m1, n1 = np.array(large).shape
    m2, n2 = np.array(small).shape
    for m in range(m1 - m2 + 1):
        for n in range(n1 - n2 + 1):
            if np.array(small).tolist() == np.array(large)[m:m+m2, n:n+n2].tolist():
                return True, m, n
    return False, -1, -1

def matrix_use_color(m):
    return list(set(list(itertools.chain.from_iterable(m))))

def is_square(task):
    return all([np.array(el['input']).shape[0] == np.array(el['input']).shape[1] for el in task['train']])

def inouts_flip(task_train):
    for inout in task_train:
        inout['input'] = np.flip(inout['input'])
    return task_train

def inouts_flipud(task_train):
    for inout in task_train:
        inout['input'] = np.flipud(inout['input'])
    return task_train

def inouts_fliplr(task_train):
    for inout in task_train:
        inout['input'] = np.fliplr(inout['input'])
    return task_train

def match_rate(mat1, mat2):
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    if m1.shape != m2.shape:
        return -1
    v1 = list(itertools.chain.from_iterable(mat1))
    v2 = list(itertools.chain.from_iterable(mat2))
    score1 = np.sum(m1==m2) / len(v1)
    score2 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return (score1 + score2) / 2

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def getDefaultPred(inp):
    pred_1 = flattener(inp)
    # for the second guess, change all 0s to 5s
    inp = [[5 if i==0 else i for i in j] for j in inp]
    pred_2 = flattener(inp)
    # for the last gues, change everything to 0
    inp = [[0 for i in j] for j in inp]
    pred_3 = flattener(inp)
    # concatenate and add to the submission output
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    return pred

def preds_to_str(preds_list, idx):
    pred_strs = []
#     st()
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            if j == 3:
                break
            pred_str += flattener(np.array(preds[i]).tolist()) + ' '
        pred_strs.append(pred_str)
    return pred_strs

def preds_to_str_only0(preds0, idx):
    preds = []
    for i in range(len(preds0)):
        pred0 = flattener(np.array(preds0[i]).tolist())
        pred1 = flattener(np.array([[0]]).tolist())
        preds.append(pred0 + ' ' + pred1 + ' ' + pred1 + ' ')
    return preds

def get_not_use_num(matrix1, matrix2):
    v1 = list(itertools.chain.from_iterable(matrix1))
    v2 = list(itertools.chain.from_iterable(matrix2))
    for j in range(1,10):
        if (j not in v1) & (j not in v2):
            return j
    return 1


# # test NN

# In[147]:


def test_nn(train, test):
    task={'train':train, 'test':test}

    train_dataset = ArcDataset(task, mode="train", augment=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)
    valid_dataset = ArcDataset(task, mode="test", augment=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)

    net = Task330Net().to(device)
    criterion = hinge_loss
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time()
    tmp_train_loss = 1

    for param in net.named_parameters():
        print(f"{param[0]:>15} {list(param[1].shape)}")
    count = 0
    for epoch in range(5000):
        train_loss = valid_loss = 0.0
        train_loss_denom = valid_loss_denom = 0

        ####################
        # train
        ####################
        net.train()
        for i, (feature, target) in enumerate(train_dataloader):
            outputs = net(feature)
            loss = criterion(outputs, target)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record
            train_loss += loss.item()
            train_loss_denom += feature.shape[0]

        train_loss /= train_loss_denom

        ####################
        # eval
        ####################
#         net.eval()
#         with torch.no_grad():
#             for i, (feature, target) in enumerate(valid_dataloader):
#                 feature = feature.to(device)
#                 target = target.to(device)

#                 outputs = net(feature)
#                 loss = criterion(outputs, target)

#                 # record
#                 valid_loss += loss.item()
#                 valid_loss_denom += feature.shape[0]

#         valid_loss /= valid_loss_denom
    
        if epoch%100==0:
#             print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")
            print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  |  time: {time()-t0:7.1f} sec")
            if tmp_train_loss <= train_loss:
                count += 1
            if count >= 4:
                break
            tmp_train_loss = train_loss

#             if best_loss > valid_loss:
#                 best_loss = valid_loss
#                 filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"
#                 torch.save(net.state_dict(), filename)

    return check(task, lambda x: task_train330(x, net))

class ArcDataset(torch.utils.data.Dataset):
    def __init__(self, task=None, mode="train", augment=False):
        if task is not None:
            assert mode in ["train", "test"]
            self.mode = mode
            self.task = task[mode]
        self.augment = augment

    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, index):
        t = self.task[index]
        t_in = torch.tensor(t["input"])
        t_out = torch.tensor(t["output"])
        t_in, t_out = self.preprocess(t_in, t_out)
        return t_in, t_out
    
    def preprocess(self, t_in, t_out):
        if self.augment:
            t_in, t_out = self._random_rotate(t_in, t_out)
        t_in = self._one_hot_encode(t_in)
        t_out = self._one_hot_encode(t_out)
        return t_in, t_out
    
    def _one_hot_encode(self, x):
        return torch.eye(10)[x].permute(2, 0, 1)
    
    def _random_rotate(self, t_in, t_out):
        t_in_shape = t_in.shape
        t_out_shape = t_out.shape
        t_in = t_in.reshape(-1, *t_in_shape[-2:])
        t_out = t_out.reshape(-1, *t_out_shape[-2:])
        r = randint(0, 7)
        if r%2 == 0:
            t_in = t_in.permute(0, 2, 1)
            t_out = t_out.permute(0, 2, 1)
        r //= 2
        if r%2 == 0:
            t_in = t_in[:, :, torch.arange(t_in.shape[-1]-1, -1, -1)]
            t_out = t_out[:, :, torch.arange(t_out.shape[-1]-1, -1, -1)]
        r //= 2
        if r%2 == 0:
            t_in = t_in[:, torch.arange(t_in.shape[-2]-1, -1, -1), :]
            t_out = t_out[:, torch.arange(t_out.shape[-2]-1, -1, -1), :]
        t_in = t_in.reshape(*t_in_shape[:-2], *t_in.shape[-2:])
        t_out = t_out.reshape(*t_out_shape[:-2], *t_out.shape[-2:])
        return t_in, t_out
    
def device_collate(batch):
    return tuple(map(lambda x: torch.stack(x).to(device), zip(*batch)))

def hinge_loss(y_pred, y_true):
    loss = y_pred.clone()
    loss[y_true>0.5] = 1-loss[y_true>0.5]
    loss[loss<0] = 0
    return loss.sum(0).mean()

class Task330Net(nn.Module):
    def __init__(self):
        super(Task330Net, self).__init__()
        siz = 16
        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.conv1x1 = nn.Conv2d(siz, 10, 1)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.conv1x1(x2)
        x = x + x2  # skip connection
        return x
    
def check(task, pred_func):
    preds = []
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        preds.append(pred_func(t_in))
    return preds

def task_train330(x, net):
    def one_hot_decode(x):
        return x.argmax(0)
    net.eval()
    with torch.no_grad():
        x = torch.tensor(x).to(device)
        y_dummy = x.clone()
        dataset = ArcDataset(augment=False)
        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)
        y = net(x).detach()
        y = one_hot_decode(y.squeeze(0))
    y = y.to("cpu").numpy()
    return y


# # CA

# In[148]:


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1)
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x

def solve_task(task, max_steps=10):
    model = CAModel(10).to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)))
        
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for sample in task:
                # predict output from input
#                 st()
                x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
                y = torch.tensor(sample["output"]).long().unsqueeze(0).to(device)
                y_pred = model(x, num_steps)
                loss += criterion(y_pred, y)
                
                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1) 
                loss += criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            losses[(num_steps - 1) * num_epochs + e] = loss.item()
    return model, num_steps, losses
                
@torch.no_grad()
def predict(model, task):
    predictions = []
    for sample in task:
        x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
        pred = model(x, 100).argmax(1).squeeze().cpu().numpy()
        predictions.append(pred)
    return predictions


# # model

# In[149]:


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp==i)
    return img

class TaskSolver:
    def train(self, task_train, n_epoch=30, preprocess_funcs=[],final=False):
        """basic pytorch train loop"""
        self.net = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr = 0.1)
        for epoch in range(n_epoch):
            for sample in task_train:
                inputs = copy.deepcopy(sample['input'])
                for preprocess_func in preprocess_funcs:
                    inputs = preprocess_func(inputs)
                inputs = FloatTensor(inp2img(inputs)).unsqueeze(dim=0)
                labels = LongTensor(sample['output']).unsqueeze(dim=0)
                optimizer.zero_grad()
                outputs = self.net(inputs)
#                 import pdb; pdb.set_trace()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
#                 st()
        return self

    def predict(self, task_test, preprocess_funcs=[], success_map={}, idx='searching', final_score_map={}, final=False):
        predictions = []
        with torch.no_grad():
            for i, in_out in enumerate(task_test):
                inputs = copy.deepcopy(in_out['input'])
                # input変更
                for preprocess_func in preprocess_funcs:
                    inputs = preprocess_func(inputs)
                inputs = FloatTensor(inp2img(inputs)).unsqueeze(dim=0)
                outputs = self.net(inputs)
                pred =  outputs.squeeze(dim=0).cpu().numpy().argmax(0)
                if ('output' in in_out) & (idx != 'searching') & (not final):
                    similarity = match_rate(in_out['output'], pred)
                    if (idx not in final_score_map) or (final_score_map.get(idx, 101) < similarity):
                        final_score_map[idx] = similarity
                predictions.append(pred)

        return predictions


# # post process

# In[150]:


def post_processes(preds, origin_task):
    processed_preds = []
    for pred in preds:
        all_input_same_colors, all_output_same_colors, all_input_and_output_same_colors, each_input_and_output_same_colors = search_inout_used_colors(origin_task['train'])
        colors_are_black_and_one_flag, only_used_color_num = colors_are_black_and_one(origin_task['train'][0]['output'])
        if all_output_same_colors & colors_are_black_and_one_flag:
            pred = np.where(pred != 0, only_used_color_num, pred)
        processed_preds.append(pred)
    return processed_preds


# In[151]:


def colors_are_black_and_one(matrix):
    unique_nums = np.unique(matrix).tolist()
    if 0 not in unique_nums:
        return False, None
        
    unique_nums.remove(0)
    if len(unique_nums) == 1:
        return True, unique_nums[0]
    else:
        return False, None

def search_inout_used_colors(task_train):
    all_input_same_colors = True
    all_output_same_colors = True
    all_input_and_output_same_colors = True
    each_input_and_output_same_colors = True
    input_unique_nums=[]
    for i, inout in enumerate(task_train):
        if each_input_and_output_same_colors:
            if np.unique(inout['input']).tolist() != np.unique(inout['output']).tolist():
                each_input_and_output_same_colors = False
                all_input_and_output_same_colors = False
        if i == 0:
            input_unique_nums = np.unique(inout['input']).tolist()
            output_unique_nums = np.unique(inout['output']).tolist()
            continue

        if input_unique_nums != np.unique(inout['input']).tolist():
            all_input_same_colors = False
            all_input_and_output_same_colors = False
        if output_unique_nums != np.unique(inout['output']).tolist():
            all_output_same_colors = False
            all_input_and_output_same_colors = False
    return all_input_same_colors, all_output_same_colors, all_input_and_output_same_colors, each_input_and_output_same_colors            


# # mirror aug

# In[152]:


def fliplr_aug(train):
    return mirror_aug(train, np.fliplr)

def flipud_aug(train):
    return mirror_aug(train, np.flipud)

def flip_aug(train):
    return mirror_aug(train, np.flip)

def transpose_aug(train):
    return mirror_aug(train, np.transpose)

def mirror_aug(train, aug_func):
    inouts = []
    for j, inout_origin in enumerate(train):
        inout = copy.deepcopy(inout_origin)
        same_flag = False
        inout['input'] = aug_func(inout['input']).tolist()
        inout['output'] = aug_func(inout['output']).tolist()
        if inout['input'] != inout_origin['input'].tolist():
            for io in inouts:
                if io['input'] == inout['input']:
                    same_flag = True
                    break
            if not same_flag:
                for inout_other in train:
                    if inout_other['input'].tolist() == inout['input']:
                        same_flag = True
                        break
            if not same_flag:
                inouts.append({'input': inout['input'], 'output': inout['output']})
    return inouts


# # color aug

# In[153]:


def has_duplicates(seq):
    return len(seq) != len(set(seq))

def color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_color_augument(task['train'])

def one_train_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([0,1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in range(1, 6):
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def super_heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_super_heavy_color_augument(task['train'])

def one_train_super_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in [2, 3, 4]:
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_heavy_color_augument(task['train'])

def one_train_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in [3, 4]:
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)

    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_heavy_color_augument(task['train'])

def one_train_medium_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for color_pairs in combinations(two_colors, 2):
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_color_augument(task['train'])

def one_train_medium_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
        [[1, 3], [4, 6], [8, 7], [2, 9]],
        [[6, 3], [2, 5], [4, 7], [1, 9]],
        [[2, 4], [7, 5], [6, 9], [8, 1]],
        [[1, 4], [5, 6], [8, 3], [2, 7]],
        [[7, 3], [6, 1], [8, 4], [5, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_light_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_light_color_augument(task['train'])

def one_train_medium_light_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
        [[1, 3], [4, 6], [8, 7], [2, 9]],
        [[6, 3], [2, 5], [4, 7], [1, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def light_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_light_color_augument(task['train'])

def one_train_light_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts

def mirror_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_mirror_augument(task['train'])

def one_train_mirror_augument(train):
    aug_func_petterns = [np.transpose, mirror_x, mirror_y]
    inouts = []
    for i in range(len(aug_func_petterns)):
        for func_combi in combinations(aug_func_petterns, i+1):
            for j, inout_origin in enumerate(train):
                inout = copy.deepcopy(inout_origin)
                same_flag = False
                for func in list(func_combi):
                    inout['input'] = func(inout['input']).tolist()
                    inout['output'] = func(inout['output']).tolist()
                if inout['input'] != inout_origin['input']:
                    for io in inouts:
                        if io['input'] == inout['input']:
                            same_flag = True
                            break
                    if not same_flag:
                        for inout_other in train:
                            if inout_other['input'].tolist() == inout['input']:
                                same_flag = True
                                break
                    if not same_flag:
                        inouts.append({'input': inout['input'], 'output': inout['output']})

    return inouts


# # get_mode_preds

# In[154]:


def get_mode_preds(preds_list):
    preds = []
    origin_shape_map = {}
    each_question_preds = defaultdict(list)
    for i in range(len(preds_list[0])):
        origin_shape_map[i] = np.array(preds_list[0][i]).shape
        for preds in preds_list:
            each_question_preds[i].append(np.array(preds[i]).reshape(-1))
    mode_preds = []
    
    for i in range(len(preds_list[0])):
        ans = []
        for j in range(len(each_question_preds[i][0])):
            ms = [m[j] for m in each_question_preds[i]]
            ans.append(np.argmax(np.bincount(ms)))
        mode_preds.append(np.array(ans).reshape(origin_shape_map[i]).tolist())
    return mode_preds


# # final score update

# In[155]:


def final_score_update(test_tasks, preds_list, final_score_map, idx, success_map):
    print(f'{idx}, 順番: mode, (aug CA), aug func0')
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            pred = np.array(preds[i])
            if test_tasks[i]['output'] == pred.tolist():
                success_map[f'{idx}_{i}'] = True            
            similarity = match_rate(pred, test_tasks[i]['output'])
            print(f'similarity: {similarity}')
            if (idx not in final_score_map) or (final_score_map.get(idx, 101) < similarity):
                final_score_map[idx] = similarity


# # final_train_and_predict

# In[156]:


each_preds = defaultdict(lambda: defaultdict(list))
ca_skip = False
def final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx, success_map, final_score_map, final=False, promising=False, origin_task=None):
    funcs0 = []
    funcs1 = []
    preds_list = []
    mode_preds_list = []
    if not final:
#         st()
        ts = TaskSolver()
        ts.train(task_train, preprocess_funcs=funcs0)
        preds = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map)
        return preds 
    
    # 別のNNに通す
#     if promising:
#         train = copy.deepcopy(task_train_aug)
#         test = copy.deepcopy(task_test)
#         for func in funcs0:
#             for inout in train:
#                 inout['input'] = func(inout['input'])
#             for inout in test:
#                 inout['input'] = func(inout['input'])
#         preds = test_nn(train, test)
#         preds = post_processes(preds, origin_task)
#         each_preds[idx]['another NN'] = preds
#         preds_list.append(preds)
#         mode_preds_list.append(preds)
#         preds = preds_to_str_only0(preds, idx)

    # not aug, funcs0
#     ts = TaskSolver()
#     ts.train(task_train, preprocess_funcs=funcs0, final=final)
#     preds3 = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
#     preds3 = post_processes(preds3, origin_task)
#     preds_list.append(preds3)
#     mode_preds_list.append(preds3)
#     each_preds[idx]['not aug, normal NN, funcs0'] = preds3
#     st()
    
    # not aug, funcs1
#     ts = TaskSolver()
#     ts.train(task_train2, preprocess_funcs=funcs1, final=final)
#     preds4 = ts.predict(task_test2, preprocess_funcs=funcs1, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
#     preds4 = post_processes(preds4, origin_task)
#     each_preds[idx]['not aug, normal NN, funcs1'] = preds4
#     preds_list.append(preds4)
#     mode_preds_list.append(preds4)

    # not aug, func0
    ts = TaskSolver()
    ts.train(task_train, preprocess_funcs=funcs1, final=final)
    preds4 = ts.predict(task_test, preprocess_funcs=funcs1, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
    preds4 = post_processes(preds4, origin_task)
    each_preds[idx]['not aug, normal NN, funcs1'] = preds4
    preds_list.append(preds4)
    mode_preds_list.append(preds4)    
    
    # aug, funcs0
    ts = TaskSolver()
    ts.train(task_train_aug, preprocess_funcs=funcs0, final=final)
    preds2 = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
    preds2 = post_processes(preds2, origin_task)
    preds_list.append(preds2)
    mode_preds_list.append(preds2)
    mode_preds_list.append(preds2) # weight2
    each_preds[idx]['aug, normal NN, funcs0'] = preds2

    if (len(task_train_aug) < 200) & (not ca_skip):
#     if False:
#         print('CA1 start')
        # not aug CA
        # TODO: 消すか検証する
#         train_changed = []
#         test_changed = []
#         for inout in task_train:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             train_changed.append(io)
#         for inout in task_test:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             test_changed.append(io)
#         model, _, _ = solve_task(train_changed)
#         preds0 = predict(model, test_changed)
#         preds0 = post_processes(preds0, origin_task)
#         each_preds[idx]['not aug CA'] = preds0
#         preds_list.append(preds0)
#         preds_list.append(preds0)

        # aug, CA
        print('CA1 start')
#         train_changed = []
#         test_changed = []
#         for inout in task_train_aug:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             train_changed.append(io)
#         for inout in task_test:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             test_changed.append(io)
        model, _, _ = solve_task(task_train_aug)
        preds1 = predict(model, task_test)
        preds1 = post_processes(preds1, origin_task)
        preds_list.append(preds1)
        mode_preds_list.append(preds1)
        mode_preds_list.append(preds1) # weight2
        each_preds[idx]['aug CA'] = preds1

    preds_mode = get_mode_preds(mode_preds_list)
    each_preds[idx]['mode matrix'] = preds_mode
    preds_list.append(preds_mode)
    if 'output' in task_test[0]:
#         st()
        final_score_update(task_test, preds_list, final_score_map, idx, success_map)

    preds = preds_to_str(preds_list, idx)
#     st()

    return preds


# # apply_aug

# In[157]:


def apply_mirror_aug(train, preprocess_best_score, idx, use_transpose_flag, promising_map):
    use_mirror_augs = []
    for aug_func in [flipud_aug, fliplr_aug, flip_aug]:
        inouts = aug_func(train)
#         st()
        similarity = get_similarity(train+inouts, [], 'searching_mirror_aug_'+idx, {})
        print(f'{aug_func}: ------> {similarity}')
        if similarity > 0.99:
            promising_map[idx] = True
        if (similarity > preprocess_best_score) or ((similarity==1) & (preprocess_best_score != 1)):
            use_mirror_augs.append(aug_func)
    if use_transpose_flag:
        print(transpose_aug)
        inouts = transpose_aug(train)
        similarity = get_similarity(train+inouts, [], 'searching_transpose_aug_'+idx, {})
        print(similarity, preprocess_best_score)
        if (similarity > preprocess_best_score) or ((similarity==1) & (preprocess_best_score != 1)):
            use_mirror_augs.append(transpose_aug)
        if similarity > 0.99:
            promising_map[idx] = True
    return use_mirror_augs

def apply_transpose_aug(train):
    for inout_origin in train:
        inout = copy.deepcopy(inout_origin)
        m, n = np.array(inout['input']).shape
        if m != n:
            return False
    return True

def apply_color_aug(train, preprocess_best_score, best_aug_score_map, idx, promising_map):
    best_aug_score_map[idx] = 0
    use_inouts = []
    use_aug_func = return_arg
    skip_heavy_flag = False
    heavy_funcs = [one_train_medium_heavy_color_augument, one_train_heavy_color_augument, one_train_super_heavy_color_augument]
#     for aug_func in [one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument, one_train_medium_heavy_color_augument, one_train_heavy_color_augument, one_train_super_heavy_color_augument]:
    for aug_func in [one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument, one_train_medium_heavy_color_augument, one_train_heavy_color_augument]:
#     for aug_func in [one_train_medium_heavy_color_augument, one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument]:
        if aug_func in heavy_funcs:
            if skip_heavy_flag == True:
                continue
            if (best_aug_score_map[idx] < 0.997) or (best_aug_score_map[idx] < preprocess_best_score+0.04):
                skip_heavy_flag = True
                continue
        inouts = aug_func(train)
        scores = []
        # 重い場合時間がかかるので学習は一度にする
        if aug_func in heavy_funcs:
            ts = TaskSolver()
            tmp_train = train + inouts
            if len(tmp_train) < 10:
                continue
            val_train, tmp_train = tmp_train[:3], tmp_train[3:]
            ts.train(tmp_train, preprocess_funcs=[])
            for i in range(3):
                preds = ts.predict([val_train[i]], preprocess_funcs=[])
                similarity = match_rate(preds[0], val_train[i]['output'])
                scores.append(similarity)
        # 軽い場合はなるべくpreと条件を揃えたいので都度学習
        else:
            for i in range(3):
                similarity = train_and_evaluate(train+inouts, [], seed=i, idx='searching_aug', success_map={})
                scores.append(similarity)
        score = np.mean(scores)
        print(f'{aug_func}: ------> {score}')
        if score > 0.9999:
            promising_map[idx] = True
            return use_inouts, aug_func
        if score < 0.8:
            return use_inouts, use_aug_func
        if (score > best_aug_score_map[idx]) & (score > preprocess_best_score):
            best_aug_score_map[idx] = score
            use_inouts = inouts
            use_aug_func = aug_func
        # 常に更新、かつ直前で0.99以上じゃない場合、重い処理をスキップ
        if score < best_aug_score_map[idx]:
            skip_heavy_flag = True
        if (aug_func == one_train_medium_heavy_color_augument) & ((score < 0.997)):
            skip_heavy_flag = True

    if best_aug_score_map[idx] > 0.98:
        promising_map[idx] = True
    return use_inouts, use_aug_func


# # train_and_evaluate

# In[158]:


def train_and_evaluate(train, func_combi, seed, idx, success_map, search_func=False):
    ts = TaskSolver()
    tmp_train = copy.deepcopy(train)
    val_train = [tmp_train.pop(seed % len(tmp_train))]
    ts.train(tmp_train, preprocess_funcs=func_combi)
    preds = ts.predict(val_train, preprocess_funcs=func_combi) # idxを渡すとsimilarityがprintされる
    return match_rate(preds[0], val_train[0]['output'])


# # add fill closed area

# In[159]:


def add_fill_closed_area(train):
    inp_origin = train[0]['input']
    out = train[0]['output']
    apply_flag = False
    for func in [np.array, np.flip, np.fliplr, np.flipud]:
        inp = np.array(inp_origin.copy())
        inp = func(inp)
        if len(set([ele for ele in np.array(out).reshape(-1) - np.array(inp).reshape(-1) if ele != 0])) == 1:
            fill_color = [ele for ele in np.array(out).reshape(-1) - np.array(inp).reshape(-1) if ele != 0][0]
            apply_flag = True
            break
    if not apply_flag:
        return [inouts_array]
    best_score = 0
    best_enclose_color = 0

    for enclose_color in range(1,10):
        inp_copy = inp.copy()
        if enclose_color == fill_color:
            continue
        H, W = inp_copy.shape
        Dy = [0, -1, 0, 1]
        Dx = [1, 0, -1, 0]
        arr_padded = np.pad(inp_copy, ((1,1),(1,1)), "constant", constant_values=0)
        searched = np.zeros(arr_padded.shape, dtype=bool)
        searched[0, 0] = True
        q = [(0, 0)]
        while q:
            y, x = q.pop()
            for dy, dx in zip(Dy, Dx):
                y_, x_ = y+dy, x+dx
                if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                    continue
                if not searched[y_][x_] and arr_padded[y_][x_]==0:
                    q.append((y_, x_))
                    searched[y_, x_] = True
        res = searched[1:-1, 1:-1]
        res |= inp_copy==enclose_color
        inp_copy[~res]=fill_color
        similarity = match_rate(inp_copy, out)
        if similarity > best_score:
            best_score = similarity
            best_enclose_color = enclose_color
    def fill_closed_area(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            H, W = inp.shape
            Dy = [0, -1, 0, 1]
            Dx = [1, 0, -1, 0]
            arr_padded = np.pad(inp, ((1,1),(1,1)), "constant", constant_values=0)
            searched = np.zeros(arr_padded.shape, dtype=bool)
            searched[0, 0] = True
            q = [(0, 0)]
            while q:
                y, x = q.pop()
                for dy, dx in zip(Dy, Dx):
                    y_, x_ = y+dy, x+dx
                    if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                        continue
                    if not searched[y_][x_] and arr_padded[y_][x_]==0:
                        q.append((y_, x_))
                        searched[y_, x_] = True
            res = searched[1:-1, 1:-1]
            res |= inp==best_enclose_color

            inp[~res] = fill_color
    #         st()
            inout['input'] = inp
        return task_train

    return [inouts_array, fill_closed_area]


# In[160]:


def add_train0_double(task_train, m1,n1,m2,n2):
    if not((m2 >= m1) & (n2 >= n1) & (m2 % m1 == 0) & (n2 % n1 == 0)):
        return []
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m2 / m1 != m2_ / m1_) or (n2 / n1 != n2_ / n1_):
            return []    
    def train0_double(task_train):
        for inout in task_train:
            x = inout['input']
            x = np.array(x)
            m, n = m2//m1, n2//n1
            x_upsampled = x.repeat(m, axis=0).repeat(n, axis=1)
            x_tiled = np.tile(x, (m, n))
            y = x_upsampled & x_tiled
            inout['input'] = y    
        return task_train
    return [train0_double]


# # add double func

# In[161]:


# 前提：same_shape_inputs, same_shape_outputs
def add_gdc_double_funcs_with_same_shape_inputs(task_train, m1, n1, m2, n2):
    if (m1==m2) & (n1==n2):
        return []
    m_gdc = math.gcd(m1, m2)
    n_gdc = math.gcd(n1, n2)
    if ((m_gdc==1) or (n_gdc==1)):
        return []
    if not ((m2 >= m1) & (n2 >= n1)):
        return []
    transpose_funcs = [np.array]
    if m1 == n1:
        transpose_funcs.append(np.transpose)
    transpose_flip_map = {}
    flip_cols_map = {}
    flip_rows_map = {}
    inp_mn_map = {}
    for m in range(m2 // m_gdc):
        for n in range(n2 // n_gdc):
            transpose_flip_map[str(m)+','+str(n)] = [np.array] # 初期値
            flip_cols_map[str(m)+','+str(n)] = []
            flip_rows_map[str(m)+','+str(n)] = []
            inp_mn_map[str(m)+','+str(n)] = [0,0]
            correct_flag = False
            pickup_output = np.array(task_train[0]['output'])[m*m_gdc:(m+1)*m_gdc, n*n_gdc:(n+1)*n_gdc]
            best_score = 0
            for transpose_func in transpose_funcs:
                if correct_flag:
                    break
                for inp_m in range(m1 // m_gdc):
                    if correct_flag:
                        break
                    for inp_n in range(n1 // n_gdc):
                        if correct_flag:
                            break
                        inp_copy = np.array(copy.deepcopy(task_train[0]['input']))
                        inp_copy = inp_copy[inp_m*m_gdc:(inp_m+1)*m_gdc, inp_n*n_gdc:(inp_n+1)*n_gdc]
                        inp_copy = transpose_func(inp_copy)
                        for flip_func in [np.flip, np.flipud, np.fliplr, np.array]:
                            if correct_flag:
                                break
                            inp_copy_copy = copy.deepcopy(inp_copy)
                            inp_copy_copy = flip_func(inp_copy_copy)
                            if pickup_output.tolist() == inp_copy_copy.tolist():
                                correct_flag = True
                                transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                flip_cols_map[str(m)+','+str(n)] = []
                                flip_rows_map[str(m)+','+str(n)] = []
                                break
                            similarity = match_rate(pickup_output, inp_copy_copy)
                            if best_score < similarity:
                                best_score = similarity
                                transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                flip_cols_map[str(m)+','+str(n)] = []
                                flip_rows_map[str(m)+','+str(n)] = []
#                                 st()
                        for i in range(m_gdc+1):
                            if correct_flag:
                                break
                            for change_rows in combinations(range(m_gdc), i):
                                if correct_flag:
                                    break
                                change_rows = list(change_rows)
                                for j in range(n_gdc+1):
                                    if correct_flag:
                                        break
                                    for change_cols in combinations(range(n_gdc), j):
                                        change_cols = list(change_cols)
                                        inp_copy_copy = copy.deepcopy(inp_copy)
                                        inp_copy_copy[change_rows, :] = np.fliplr(inp_copy_copy[change_rows, :])
                                        inp_copy_copy[:, change_cols] = np.flipud(inp_copy_copy[:, change_cols])
                                        if pickup_output.tolist() == inp_copy_copy.tolist():
                                            correct_flag = True
                                            transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                            inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                            flip_cols_map[str(m)+','+str(n)] = change_cols
                                            flip_rows_map[str(m)+','+str(n)] = change_rows  
                                            break
                                        
                                        similarity = match_rate(pickup_output, inp_copy_copy)
                                        if best_score < similarity:
                                            best_score = similarity
                                            transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                            inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                            flip_cols_map[str(m)+','+str(n)] = change_cols
                                            flip_rows_map[str(m)+','+str(n)] = change_rows

    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            ans = np.zeros((m2, n2)).astype(int)
            for coordinate, transpose_funcs in transpose_flip_map.items():
                m, n = coordinate.split(',')
                m, n = int(m), int(n)
                inp_copy = np.array(copy.deepcopy(inp))
                inp_n, inp_m = inp_mn_map[coordinate]
                inp_copy = inp_copy[inp_m*m_gdc:(inp_m+1)*m_gdc, inp_n*n_gdc:(inp_n+1)*n_gdc]
                for transpose_func in transpose_funcs:
                    inp_copy = transpose_func(inp_copy)
                change_cols = flip_cols_map[coordinate]
                change_rows = flip_rows_map[coordinate]
                inp_copy[:, change_cols] = np.flipud(inp_copy[:, change_cols])
                inp_copy[change_rows, :] = np.fliplr(inp_copy[change_rows, :])
                ans[m*m_gdc:(m+1)*m_gdc, n*n_gdc:(n+1)*n_gdc] = inp_copy
            inout['input'] = ans
        return task_train

    return [double]


# In[162]:


# each shapeが違って割合が一緒の場合
def add_gcd_double_funcs(task_train, m1, n1, m2, n2):
    if (m2==m1) & (n2==n1):
        return []
    m = math.gcd(m1, m2)
    n = math.gcd(n1, n2)    
    if not ((m2 >= m1) & (n2 >= n1) & (m2 % m1 == 0) & (n2 % n1 == 0)):
        return []
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m2 / m1 != m2_ / m1_) or (n2 / n1 != n2_ / n1_):
            return []

    transpose_funcs = [np.array]
    if m1 == n1:
        transpose_funcs.append(np.transpose)
    transpose_flip_map = {}
    flip_cols_map = {}
    flip_rows_map = {}
    for m in range(m2 // m1):
        for n in range(n2 // n1):
            transpose_flip_map[str(m)+','+str(n)] = [np.array]
            correct_flag = False
            pickup_output = np.array(task_train[0]['output'])[m*m1:(m+1)*m1, n*n1:(n+1)*n1]
            best_score = 0
            for flip_func in [np.flip, np.flipud, np.fliplr, np.array]:
                for transpose_func in transpose_funcs:
                    if correct_flag:
                        break
                    inp_copy = copy.deepcopy(task_train[0]['input'])
                    inp_copy = transpose_func(inp_copy)
                    inp_copy = flip_func(inp_copy)
                    if pickup_output.tolist() == inp_copy.tolist():
                        correct_flag = True
                        transpose_flip_map[str(m)+','+str(n)] = [flip_func, transpose_func]
                        flip_cols_map[str(m)+','+str(n)] = []
                        flip_rows_map[str(m)+','+str(n)] = []
                    similarity = match_rate(pickup_output, inp_copy)
                    if best_score < similarity:
                        best_score = similarity
                        transpose_flip_map[str(m)+','+str(n)] = [flip_func, transpose_func]
                        flip_cols_map[str(m)+','+str(n)] = []
                        flip_rows_map[str(m)+','+str(n)] = []
    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp_m, inp_n = np.array(inp).shape
            m, n = m2//m1 - 1, n2 // n1 - 1
            ans = np.zeros((inp_m*(int(m)+1), inp_n*(int(n)+1))).astype(int)
            for coordinate, transpose_funcs in transpose_flip_map.items():
                m, n = coordinate.split(',')
                m, n = int(m), int(n)
                inp_copy = copy.deepcopy(inp)
                for transpose_func in transpose_funcs:
                    inp_copy = transpose_func(inp_copy)

                ans[m*inp_m:(m+1)*inp_m, n*inp_n:(n+1)*inp_n] = inp_copy
            inout['input'] = ans
        return task_train

    return [double]


# In[163]:


# all_output_shape are same
# outputsの形は違うけど倍率が同じ場合は別の関数で
def add_double_funcs_with_same_shape_inputs(train, m1, n1, m2, n2):
    inp = np.array(train[0]['input'].copy())
    out = np.array(train[0]['output'].copy())
    tmp_ans = np.zeros(out.shape)
    ans = np.zeros(out.shape)
    best_ans_ms = []
    best_ans_ns = []
    best_m_flips= []
    best_n_flips = []
    if (m2==m1) & (n2==n1):
        return []
    # 縦横どちらかでもoutputが小さい場合は別関数で
    if not ((m2 >= m1) & (n2 >= n1)):
        return []
    for inout in train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1 != m1_) or (n1 != n1_) or (m2 != m2_) or (n2 != n2_):
            return []

    for ans_m in range(m2):
        o = out[ans_m:ans_m+1, :n1]
        best_score = 0
        for inp_m in range(m1):
            i = inp[inp_m:inp_m+1, :]
#             st()
            for flip in [np.array, np.fliplr]:
                similarity = match_rate(flip(i), flip(o))
                if best_score < similarity:
                    best_score = similarity
                    best_ans_m = inp_m
                    best_flip = flip

        best_ans_ms.append(best_ans_m)
        best_m_flips.append(best_flip)

    for i, (flip, m) in enumerate(zip(best_m_flips, best_ans_ms)):
        tmp_ans[i:i+1, :n1] = flip(inp[m:m+1, :])
    for ans_n in range(n2):
        o = out[:, ans_n:ans_n+1]
        best_score = 0
        for inp_n in range(n1):
            i = tmp_ans[:, inp_n:inp_n+1]
            for flip in [np.array, np.fliplr]:
                similarity = match_rate(flip(i), flip(o))
                if best_score < similarity:
                    best_score = similarity
                    best_ans_n = inp_n
                    best_flip = flip
        best_ans_ns.append(best_ans_n)    
        best_n_flips.append(best_flip)
    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            tmp_ans = np.zeros(out.shape)
            ans = np.zeros(out.shape)        
            for i, (flip, m) in enumerate(zip(best_m_flips, best_ans_ms)):
                tmp_ans[i:i+1, :n1] = flip(inp[m:m+1, :])

            for i, (flip, n) in enumerate(zip(best_n_flips, best_ans_ns)):
                ans[:, i:i+1] = flip(tmp_ans[:, n:n+1])
            inout['input'] = ans
        return task_train
    
    return [double]


# In[164]:


def get_period_length_vertical(arr):
    arr = np.array(arr)
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:period, :], ((0,H-period),(0,0)), 'wrap')
        if (cycled==arr).all():
            return period
        period += 1
        
def add_train2_double_vertical(task_train, m1, n1, m2, n2):
    if not((n1 == n2) & (m2 > m1)):
        return []
    
    def train2_double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            period = get_period_length_vertical(inp)
            y = inp[:period, :]
            y = np.pad(y, ((0,m2-period),(0,0)), 'wrap')
            inout['input'] = y
        return task_train
    return [train2_double]


# In[165]:


def get_period_length_horizontal(arr):
    arr = np.array(arr)
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:, :period], ((0,0),(0,W-period)), 'wrap')
        if (cycled==arr).all():
            return period
        period += 1
        
def add_train2_double_horizontal(task_train, m1, n1, m2, n2):
    if not((m1 == m2) & (n2 > n1)):
        return []
    
    def train2_double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            period = get_period_length_horizontal(inp)
            y = inp[:, :period]
            y = np.pad(y, ((0,0),(0,n2-period)), 'wrap')
            inout['input'] = y
        return task_train    
    return [train2_double]


# # add crop

# In[166]:


def crop_width(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    try:
        task_train = copy.deepcopy(task_train_origin)
        for inout in task_train:
            inp = inout['input']
            max_width = 0
            max_width_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)        
                if max_width < y_max - y_min:
                    max_width = y_max - y_min
                    max_width_color = c
            coords = np.argwhere(inp==max_width_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)        
            inout['input'] = inp[x_min:x_max+1, y_min:y_max+1]
        return task_train
    except:
        return task_train_origin

def add_crop_width(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            max_width = 0
            max_width_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)   
                if max_width < y_max - y_min:
                    max_width = y_max - y_min
                    max_width_color = c
            coords = np.argwhere(inp==max_width_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)               
            if (inp[x_min:x_max+1, y_min:y_max+1].shape != out.shape) & (inp[x_min:x_max+1, y_min:y_max+1].shape != out.T.shape):
                return []   
        except:
            return []

    return [crop_height]
def crop_height(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    try:
        task_train = copy.deepcopy(task_train_origin)
        for inout in task_train:
            inp = inout['input']
            max_height = 0
            max_height_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)        
                if max_height < x_max - x_min:
                    max_height = x_max - x_min
                    max_height_color = c
            coords = np.argwhere(inp==max_height_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)        
            inout['input'] = inp[x_min:x_max+1, y_min:y_max+1]
        return task_train
    except:
        return task_train_origin

def add_crop_height(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            max_height = 0
            max_height_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)   
                if max_height < x_max - x_min:
                    max_height = x_max - x_min
                    max_height_color = c
            coords = np.argwhere(inp==max_height_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)               
            if (inp[x_min:x_max+1, y_min:y_max+1].shape != out.shape) & (inp[x_min:x_max+1, y_min:y_max+1].shape != out.T.shape):
                return []   
        except:
            return []

    return [crop_height]
def crop_max(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    try:
        for inout in task_train:
            a = inout['input']
            b = np.bincount(a.flatten(),minlength=10)
            b[0] = 255
            c = np.argsort(b)[-2]
            coords = np.argwhere(a==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inout['input'] = a[x_min:x_max+1, y_min:y_max+1]
    except:
        return task_train_origin
    return task_train
    
def crop_min(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    try:
        for inout in task_train:
            a = inout['input']
            b = np.bincount(a.flatten(),minlength=10)
            c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
            coords = np.argwhere(a==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inout['input'] = a[x_min:x_max+1, y_min:y_max+1]
    except:
        return task_train_origin
    return task_train

def add_crop_max(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            bin_c = np.bincount(inp.flatten(), minlength=10)

            bin_c[0] = 255
            c = np.argsort(bin_c)[-2]
            coords = np.argwhere(inp==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inp = inp[x_min:x_max+1, y_min:y_max+1]  
            if (inp.shape != out.shape) & (inp.T.shape != out.shape):
                return []
        except:
            return []
    return [crop_max]

def add_crop_min(task_train):
    for inout in task_train:
        try: 
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            bin_c = np.bincount(inp.flatten(), minlength=10)
            c = int(np.where(bin_c==np.min(bin_c[np.nonzero(bin_c)]))[0])
            coords = np.argwhere(inp==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inp = inp[x_min:x_max+1, y_min:y_max+1]
            if (inp.shape != out.shape) & (inp.T.shape != out.shape):
                return []
        except:
            return []
    return [crop_min]    


# # all_inputs_same_shape and all_outputs_same_shape

# In[167]:


def all_inputs_same_shape_and_all_outputs_same_shape(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    return all_inputs_same_shape, all_outputs_same_shape


# # add size change funcs

# In[168]:


def add_size_change_funcs(task_train, task_n):
    size_change_funcs = [inouts_array]
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False       
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    if m1 == n1 == m2 == n2:
        return size_change_funcs    

    # grid
    size_change_funcs += add_grid_funcs(m1, n1, m2, n2)
    
    # div
    if (m1 >= m2*2) or (n1 > n2*2):
        size_change_funcs += add_div_funcs(task_train, m1, n1, m2, n2)
    else:
        size_change_funcs += add_div_funcs2(task_train, m1, n1, m2, n2, vertically=True)
        size_change_funcs += add_div_funcs2(task_train, m1, n1, m2, n2, vertically=False)
    if (m1 > m2) & (n1 > n2) & (m1 < 20) & (n1 < 20):
        size_change_funcs += add_object_detect2(task_train)
        
    # double
    if all_inputs_same_shape & all_outputs_same_shape:
        size_change_funcs += add_train2_double_horizontal(task_train, m1, n1, m2, n2)
        size_change_funcs += add_train2_double_vertical(task_train, m1, n1, m2, n2)
        size_change_funcs += add_gdc_double_funcs_with_same_shape_inputs(task_train, m1, n1, m2, n2)
        size_change_funcs += add_recolor(task_train, task_n)
    else:
        size_change_funcs += add_gcd_double_funcs(task_train, m1, n1, m2, n2)
        
    size_change_funcs += add_train0_double(task_train, m1, n1, m2, n2)
    if (m1 >= m2) & (n1 >= n2):
        size_change_funcs += add_crop_max(task_train)
        size_change_funcs += add_crop_min(task_train)
        size_change_funcs += add_crop_height(task_train)
        size_change_funcs += add_crop_width(task_train)
        size_change_funcs += add_crop_by_line(task_train)
    # 他にもたくさん足す
    return size_change_funcs


# In[169]:


# for interface...
def return_arg(inp):
    return inp

def inouts_transpose(task_train):
    for inout in task_train:
        inout['input'] = np.transpose(inout['input'])
    return task_train

def inouts_array(task_train):
    for inout in task_train:
        inout['input'] = np.array(inout['input'])
    return task_train

def add_transpose(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    if (m1==n2) & (n1==m2):
        return [inouts_array, inouts_transpose]
    else:
        return [inouts_array]

def add_grid_funcs(m1, n1, m2, n2):
    grid_funcs = []
    if (m1 <= m2) & (n1 <= n2):
        return grid_funcs
    if (m2%m1 == 0) & (n2%n1 == 0):
        def griding(task_train):
            for inout in enumerate(task_train):
                inp = copy.deepcopy(inout['input'])
                m_grid = m2 // m1
                n_grid = n2 // n1
                inp = np.array(inp)
                m, n = inp.shape
                ans_tmp = np.zeros((m*m_grid, n), dtype='int')
                for i in range(m):
                    for j in range(m_grid):
                        ans_tmp[i*m_grid+j, :] = inp[i, :]
                ans = copy.deepcopy(ans_tmp)
                for stack_n in range(n_grid-1):
                    ans = np.hstack([ans, ans_tmp])
                for i in range(n):
                    for j in range(n_grid):
                        ans[:, i*n_grid+j] = ans_tmp[:, i]
                inout['input'] = ans
            return task_train
        grid_funcs.append(griding)

    if (m1 != n1) & (m2%n1 == 0) & (n2%m1 == 0):
        def transpose_griding(task_train):
            for inout in task_train:
                inp = copy.deepcopy(inp_o)
                m_grid = m2 // n1
                n_grid = n2 // m1
                inp = np.transpose(inp)
                m, n = inp.shape
                ans_tmp = np.zeros((m*m_grid, n), dtype='int')
                for i in range(m):
                    for j in range(m_grid):
                        ans_tmp[i*m_grid+j, :] = inp[i, :]
                ans = copy.deepcopy(ans_tmp)
                for stack_n in range(n_grid-1):
                    ans = np.hstack([ans, ans_tmp])
                for i in range(n):
                    for j in range(n_grid):
                        ans[:, i*n_grid+j] = ans_tmp[:, i]
                inout['input'] = ans
            return task_train
        grid_funcs.append(transpose_griding)
    return grid_funcs

def div_two_inputs(inp, long):
    inp = np.array(inp)
    m, n = inp.shape
    if m == n:
        return inp, inp
    horizontal = False
    # 縦長にする
    if n > m:
        horizontal = True
        inp = inp.T
        m, n = inp.shape

    a = inp[:long, :]
    b = inp[m-long:, :]
    # 元が横長だったら戻す
    if horizontal:
        a = a.T
        b = b.T
    return a, b

def add_div_funcs(train, m1, n1, m2, n2):
    for inout in train:
        m1_0, n1_0 = np.array(inout['input']).shape
        m2_0, n2_0 = np.array(inout['output']).shape
        if (m1_0 != m1) or (n1_0 != n1) or (m2_0 != m2) or (n2_0 != n2):
            return []
    if (m1 == n1) or (np.min([m1, n1]) != np.min([m2, n2])) or(np.max([m1,n1]) <= np.max([m2,n2])):
        return []
    long = np.max([m2,n2])
    def div_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num&b_align_num
        return task_train

    def div_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num|b_align_num
        return task_train

    def div_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num^b_align_num
        return task_train
    def div_not_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num&b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num|b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num^b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    return [div_and, div_or, div_xor, div_not_and, div_not_or, div_not_xor]


# In[170]:


def div_two_inputs2(inp, belt_length, vertically):
    inp = np.array(inp)
    if vertically:
        inp = inp.T
    m, n = inp.shape
    after_n = (n-belt_length) // 2

    a = inp[:, :after_n]
    b = inp[:, n-after_n:]
    if vertically:
        a, b = a.T, b.T
    return a, b

# 真ん中のベルトで分けるタイプ。ベルトの長さは各inputで一定の場合のみ
def add_div_funcs2(train, m1, n1, m2, n2, vertically):
    if vertically:
        if (n1 != n2) or (m1 < m2*2):
            return []
        belt_length = m1 - m2*2
    else:
        if (m1 != m2) or (n1 < n2*2):
            return []
        belt_length = n1 - n2*2

    for inout in train:
        m1_0, n1_0 = np.array(inout['input']).shape
        m2_0, n2_0 = np.array(inout['output']).shape
        if vertically:
            if (n1_0 != n2_0) or (m1_0 != m2_0*2 + belt_length):
                return []
        else:
            if (m1_0 != m2_0) or (n1_0 != n2_0*2 + belt_length):
                return []

    def div_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num&b_align_num
        return task_train

    def div_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num|b_align_num
        return task_train

    def div_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num^b_align_num
        return task_train

    def div_not_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num&b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num|b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num^b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    return [div_and, div_or, div_xor, div_not_and, div_not_or, div_not_xor]


# # add patch funcs

# In[171]:


patch_skip = False
def add_patch_funcs(task_train, idx):
    if patch_skip:
        return [inouts_array]
    
    if correct_task_train(task_train):
        return [inouts_array]
    
    if idx in ['15696249', 'e345f17b']:
        return [inouts_array]
    inp, out = np.array(task_train[-1]["input"]), np.array(task_train[-1]["output"])
    for inout in task_train:
        if np.array(inout["input"]).shape != inp.shape:
            return [inouts_array]

    flip_funcs = [np.array, np.flip, np.flipud, np.fliplr]
    transpose_funcs = [np.array, np.transpose]
    best_score = match_rate(inp, out)
    best_feat = None
    for flip_func in flip_funcs:
        for transpose_func in transpose_funcs:
            inp_copy = copy.deepcopy(inp)
            inp_copy = flip_func(inp_copy)
            inp_copy = transpose_func(inp_copy)
            pred, feat = call_pred_train(inp_copy, out, patch_image)
            similarity = match_rate(out, pred)
            if best_score < similarity:
                best_score = similarity
                best_flip_func = flip_func
                best_transpose_func = transpose_func
                best_feat = feat

    def this_time_patch_image(task_train):
        for inout in task_train:
            inp = inout['input']
            if (best_feat is not None) & (best_feat != {}):
                inp = best_flip_func(inp)
                inp = best_transpose_func(inp)
    #             print(best_feat)
                pred = call_pred_test(inp, patch_image, best_feat)
    #             if np.array(pred).shape != task['test'][0]''
                if pred.shape != np.array(inp).shape:
                    inout['input'] = np.array(inp)
                    continue
                inout['input'] = pred
            else:
                inout['input'] = np.array(inp)
        return task_train
    
    return [this_time_patch_image, inouts_array]


# In[172]:


def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff

def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2 #Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3 #Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5 # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7 # Check vertical symmetric of array
        return sym
    except:
        return 0
    
def bbox(a):
    try:
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]

def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm

def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)
    
def call_pred_train(t_in, t_out, pred_func):
    try:
        feat = {}
        feat['s_out'] = t_out.shape
        if t_out.shape==t_in.shape:
            diff = in_out_diff(t_in,t_out)
            feat['diff'] = diff
            feat['cm'] = t_in[diff!=0].max()
        else:
            feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
            feat['cm'] = cmask(t_in)
        feat['sym'] = check_symmetric(t_out)
        args = inspect.getargspec(pred_func).args
        if len(args)==1:
            return pred_func(t_in), feat
        elif len(args)==2:
            t_pred = pred_func(t_in,feat[args[1]])    
        elif len(args)==3:
            t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
        feat['sizeok'] = len(t_out)==len(t_pred)
        t_pred = np.resize(t_pred,t_out.shape)
        acc = (t_pred==t_out).sum()/t_out.size
        return t_pred, feat
    except:
        return t_in, {}

def call_pred_test(t_in, pred_func, feat):
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]]) 
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}

def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]
    
def patch_image(t_in,s_out,cm=0):
    t_in = np.array(t_in)
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)   
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)    
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break 
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return t_in
    


# # add train004 growth

# In[173]:


def add_train4_growth(shaped_train):
    if correct_task_train(shaped_train):
        return [inouts_array]
    x = shaped_train[0]['input'].copy()
    out = shaped_train[0]['output'].copy()
    x = np.array(x)
    def get_base_pattern(arr, w, h):
        # find maximum number of unique color tiles in 3x3 field
        H, W = arr.shape
        arr_onehot = 1<<arr
        arr_bool = arr.astype(bool).astype(np.int32)
        counts = np.zeros(arr.shape, dtype=np.int32)
        colors = np.zeros(arr.shape, dtype=np.int32)
        for y in range(H-2):
            for x in range(W-2):
                counts[y, x] = arr_bool[y:y+2, x:x+2].sum()
                colors[y, x] = np.bitwise_or.reduce(arr_onehot[y:y+2, x:x+2].reshape(-1))
        n_colors = np.zeros(arr.shape, dtype=np.int32)
        for c in range(1, 10):
            n_colors += colors>>c & 1
        counts[n_colors>=2] = 0
        res_y, res_x = np.unravel_index(np.argmax(counts), counts.shape)
        pattern = arr[res_y:res_y+h, res_x:res_x+w].astype(bool).astype(np.int32)
        return (res_y, res_x), pattern
    repeat_num = 10
    best_wh = (2, 2)
    best_score = 0
    correct_flag = False
    for w in range(2,7):
        if correct_flag:
            break
        for h in range(2,7):
            (base_y, base_x), pattern = get_base_pattern(x, w, h)
#             p(pattern)
            try:
                pad_size = repeat_num * np.max([w, h])
                x_padded = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size)), "constant", constant_values=0)
                base_y += pad_size
                base_x += pad_size
                y = x_padded.copy()
                for dy in [-(h+1), 0, h+1]:
                    for dx in [-(w+1), 0, w+1]:
                        y_, x_ = base_y+dy, base_x+dx
                        if dy==dx==0:
                            continue
                        count = np.bincount(x_padded[y_:y_+h+1, x_:x_+w+1].reshape(-1))
                        if count[0]==9:
                            continue
                        count[0] = 0
                        color = count.argmax()
                        for i in range(1, repeat_num):
                            y[base_y+dy*i:base_y+dy*i+h, base_x+dx*i:base_x+dx*i+w] = color * pattern
                y = y[pad_size:-pad_size, pad_size:-pad_size]
                score = match_rate(y, out)
                if best_score < score:
                    best_score = score
                    best_wh = (w, h)
                    if score == 1:
                        correct_flag = True
                        break
            except:
                pass
    def train4_growth(task_train):
        for inout in task_train:
            inp = inout['input']
            x = np.array(inp)
            try:
                w, h = best_wh
                (base_y, base_x), pattern = get_base_pattern(x, w, h)
                pad_size = repeat_num * np.max([w, h])
                x_padded = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size)), "constant", constant_values=0)
                base_y += pad_size
                base_x += pad_size
                y = x_padded.copy()
                for dy in [-(h+1), 0, h+1]:
                    for dx in [-(w+1), 0, w+1]:
                        y_, x_ = base_y+dy, base_x+dx
                        if dy==dx==0:
                            continue
                        count = np.bincount(x_padded[y_:y_+h+1, x_:x_+w+1].reshape(-1))
                        if count[0]==9:
                            continue
                        count[0] = 0
                        color = count.argmax()
                        for i in range(1, repeat_num):
                            y[base_y+dy*i:base_y+dy*i+h, base_x+dx*i:base_x+dx*i+w] = color * pattern
                inout['input'] = y[pad_size:-pad_size, pad_size:-pad_size]
            except:
                inout['input'] = x
        return task_train
        
    return [inouts_array, train4_growth]


# # add change color funcs

# In[174]:


def add_change_color_funcs(task_train):
    if correct_task_train(task_train):
        return [inouts_array]
    in_use_colors, out_use_colors, color_changed = about_color(task_train)
    if not color_changed:
        return [inouts_array]
    inout_map = {}
    for in_color in in_use_colors:
        for out_color in out_use_colors:
            scores = []
            best_score = 0
            for inout in task_train:
                inp = inout['input'].copy()
                out = inout['output'].copy()
                in_vec = list(itertools.chain.from_iterable(inp))
                out_vec = list(itertools.chain.from_iterable(out))
                if (in_color not in in_vec) or (out_color not in out_vec):
                    continue
                inp = np.where(np.array(inp) == in_color, out_color, inp)
                scores.append(match_rate(inp, out))
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                inout_map[in_color] = out_color
    def change_color(task_train):
        for inout in task_train:
            inp_origin = inout['input']
            inp = np.array(inp_origin.copy())
            vec = list(itertools.chain.from_iterable(inp_origin))
            for in_color, out_color in inout_map.items():
                if in_color in vec:
                    inp = np.where(np.array(inp_origin) == in_color, out_color, inp)
            inout['input'] = inp
        return task_train
    
    return [inouts_array, change_color]


# In[175]:


def about_color(task_train):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_colors += list(set(out_vec))
        if set(in_vec) != set(out_vec):
            color_changed = True
    return list(set(in_colors)), list(set(out_colors)), color_changed

def about_color_for_test(task_test):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_test:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))

def about_color_for_task(task):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task['train']:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    for inout in task['test']:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))


# # add task train6

# In[176]:


def add_task_train6(task_train):
    if correct_task_train(task_train):
        return [inouts_array]
    use_same_color = True
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_use_colors = list(set(in_vec))
        in_use_colors.remove(0) if 0 in in_use_colors else 0
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_use_colors = list(set(out_vec))
        out_use_colors.remove(0) if 0 in out_use_colors else 0
        if sorted(in_use_colors) != sorted(out_use_colors):
            use_same_color = False
    if use_same_color:    
        return [inouts_array, task_train6]
    else:
        return [inouts_array]
    
def task_train6(task_train):
    for inout in task_train:
        x = inout['input']
        x = np.array(x)
        H, W = x.shape
        vec = list(itertools.chain.from_iterable(x))
        use_colors = list(set(vec))
        use_colors.remove(0) if 0 in use_colors else 0
        colors = [0] * len(use_colors)
        for yy in range(H):
            for xx in range(W):
                color = x[yy, xx]
                if color != 0:
                    colors[(yy+xx)%len(use_colors)] = color
        y = x.copy()
        for yy in range(H):
            for xx in range(W):
                y[yy, xx] = colors[(yy+xx)%len(use_colors)]
        inout['input'] = y
    return task_train        


# # add move object

# In[177]:


move_skip = False
def add_move_object(task_train):
    start = time()
    if move_skip:
        return [inouts_array]
    if correct_task_train(task_train):
        return [inouts_array]
    inp = np.array(task_train[0]['input'])
    out = np.array(task_train[0]['output'])
    in_use_colors, _, _ = about_color(task_train)
    in_use_colors = [c for c in in_use_colors if c != 0]
    best_score = match_rate(inp, out)
    best_goal_color = 0
    best_move_color = 0
    best_change_color = 0
    best_move_num = 0
    best_direction = [0, 0]
    best_correction = 0
    Dy = [0, 1, 0, -1]
    Dx = [1, 0, -1, 0]
    should_change = False
    for use_color_n, goal_color in enumerate(in_use_colors):
        for move_color in in_use_colors:
            if (time() - start > 60*60) & (len(in_use_colors) / 2 > use_color_n):
                return [inouts_array]

            goal_idx_set = set(tuple(idx) for idx in np.array(np.where(inp==goal_color)).T)
            move_idx_list = [tuple(idx) for idx in np.array(np.where(inp==move_color)).T]
            for dy, dx in zip(Dy, Dx):
                for move_num in range(1, 40):
                    obj_idx = set((idx[0]+dy*move_num, idx[1]+dx*move_num) for idx in move_idx_list)
                    if obj_idx & goal_idx_set:
                        for correction in [-2, -1, 0, 1, 2]:
                            for change_color in range(10):
                                inp_copy = copy.deepcopy(inp)
                                for idx in obj_idx:
                                    idx = (idx[0]+(dy*correction), idx[1]+(dx*correction))
                                    if (idx[0] < 0) or (idx[1] < 0) or (inp_copy.shape[0] <= idx[0]) or (inp_copy.shape[1] <= idx[1]):
                                        break
                                    inp_copy[idx] = change_color
                                for origin_move_pad_color in range(10):
                                    inp_copy2 = copy.deepcopy(inp_copy)
                                    for move_idx in move_idx_list:
                                        inp_copy2[move_idx] = origin_move_pad_color
                                    score = match_rate(inp_copy2, out)
                                    if best_score < score:
                                        should_change = True
                                        best_score = score
                                        best_goal_color = goal_color
                                        best_move_color = move_color
                                        best_move_num = move_num
                                        best_direction = [dy, dx]
                                        best_correction = correction
                                        best_change_color = change_color
                                        best_origin_move_pad_color = origin_move_pad_color

    def move_object(task_train_origin):
        for inout in task_train_origin:
            inout['input'] = np.array(inout['input'])
        if not should_change:
            return task_train_origin
        task_train = copy.deepcopy(task_train_origin)
        for i, inout in enumerate(task_train):
            finished = False
            inp = np.array(inout['input'])
            directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
            for direction in directions:
                if finished:
                    break
                for move_num in range(1, 50):
                    if finished:
                        break
                    goal_idx_set = set(tuple(idx) for idx in np.array(np.where(inp==best_goal_color)).T)
                    move_idx_list = [tuple(idx) for idx in np.array(np.where(inp==best_move_color)).T]
                    obj_idx = set((idx[0] + direction[0]*move_num, idx[1] + direction[1]*move_num) for idx in move_idx_list)
                    if obj_idx & goal_idx_set:
                        for idx in obj_idx:
                            idx = (idx[0]+(direction[0]*best_correction), idx[1]+(direction[1]*best_correction))
                            if (idx[0] < 0) or (idx[1] < 0) or (inp.shape[0] <= idx[0]) or (inp.shape[1] <= idx[1]):
                                continue
                            inp[idx] = best_change_color
                        for move_idx in move_idx_list:
                            inp[move_idx] = best_origin_move_pad_color
                        task_train[i]['input'] = inp
                        finished = True
        # if recursion:
        #     for i in range(5):
        #         if 'output' in task_train[0]:
        #             if correct_task_train(task_train):
        #                 return task_train
        #         funcs = add_move_object(task_train, False)
        #         for func in funcs:
        #             task_train = func(task_train)
        return task_train
    return [move_object, inouts_array]


# # correct task

# In[178]:


def correct_task_train(task_train):
    correct = True
    for inout in task_train:
        if np.array(inout['input']).tolist() != np.array(inout['output']).tolist():
            correct = False
    return correct


# # check_p

# In[179]:


def check_p(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fnum = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat = call_pred_train(t_in, t_out, pred_func)
        plot_one(axs[0,fnum],t_in,f'train-{i} input')
        plot_one(axs[1,fnum],t_out,f'train-{i} output')
        plot_one(axs[2,fnum],t_pred,f'train-{i} pred')
        fnum += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred = call_pred_test(t_in, pred_func, feat)        
        plot_one(axs[0,fnum],t_in,f'test-{i} input')
        plot_one(axs[1,fnum],t_out,f'test-{i} output')
        plot_one(axs[2,fnum],t_pred,f'test-{i} pred')
#         t_pred = np.resize(t_pred,t_out.shape)
        fnum += 1
    plt.show()
    return 1


# # recolor

# In[180]:


def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()

def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output

def add_recolor(task_train, task_n):
    return [inouts_array]
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    if (not all_inputs_same_shape) or (not all_outputs_same_shape):
        return [inouts_array]

    inputs = []
    outputs = []
    for inout in task_train:
        inputs.append(copy.deepcopy(inout['input']))
        outputs.append(copy.deepcopy(inout['output']))

    N = len(inputs)
    x0 = inputs[0]
    y0 = outputs[0]
    n = len(x0)
    k = len(x0[0])
    a = len(y0)
    b = len(y0[0])

    List1 = {}
    List2 = {}
    for i in range(n):
        for j in range(k):
            seq = []
            for x in inputs:
                seq.append(x[i][j])
            List1[(i,j)] = seq

    for p in range(a):
        for q in range(b):
            seq1 = []
            for y in outputs:
                seq1.append(y[p][q])

            places = []
            for key in List1:
                if List1[key] == seq1:
                    places.append(key)

            List2[(p,q)] = places
            if len(places) == 0:
                return [inouts_array]

    def recolor(task_train):
        for inout in task_train:
            inout['input'] = np.array(inout['input'])
        answer = np.zeros((a,b), dtype = int)
        for inout_n, inout in enumerate(task_train):
            for p in range(a):
                for q in range(b):
                    palette = [0,0,0,0,0,0,0,0,0,0]
                    for i, j in List2[(p,q)]:
                        color = inout['input'][i][j]
                        palette[color]+=1
                    answer[p,q] =  np.argmax(palette)

            task_train[inout_n]['input'] = np.array(answer)
        return task_train

    return [inouts_array, recolor]


# # get_similarity

# In[181]:


def get_similarity(train, func_combi, idx, search_func=True):
    similarities = []
    for seed in range(3):
        similarity = train_and_evaluate(train, func_combi, seed, idx, {}, search_func=search_func)
        similarities.append(similarity)
    return np.mean(similarities)


# # add kneighbors

# In[182]:


def comparematrixes(a,b):
    out=0;
    for i in range(min(len(a),len(b))):
        for j in range(min(len(a[0]),len(b[0]))):
            if a[i][j]==b[i][j]:
                out+=1
    out/=len(a)*len(a[0]);
    return 1-out;
def add_kneighbors(task_train):
    all_inputs_same_shape, all_outputs_same_shape = all_inputs_same_shape_and_all_outputs_same_shape(task_train)
    if (not all_inputs_same_shape) or (not all_outputs_same_shape):
#         return [inouts_array]
        pass
    ines=[];
    outes=[];
    for i in range(len(task_train)):
        vx=task_train[i]["input"].copy();
        vi=task_train[i]["output"].copy();
        if (len(vx) > 10) or (len(vi) > 10):
            return [inouts_array]
        for k1 in range(min(len(vx),len(vi))):
            for k2 in range(min(len(vx[0]),len(vi[0]))):
                dtm=[];
                for k3 in range(-2,2+1,1):
                    for k4 in range(-2,2+1,1):
                        if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            if (vx[k1+k3][k2+k4] > 10) or (vi[k1+k3][k2+k4]):
                                return [inouts_array]
                            td[vx[k1+k3][k2+k4]]=1
                            dtm+=td.copy();
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            td[vi[k1+k3][k2+k4]]=1;
                            dtm+=td.copy();
                        else:
                            dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                            dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                ines.append(dtm);
                if(len(vi)>k1 and len(vi[0])>k2 and k1>=0 and k2>=0):
                    outes.append(vi[k1][k2]);
                else:
                    outes.append(0);
    knn = KNeighborsClassifier(n_neighbors = 1);
    ines=json.loads(json.dumps(ines));
    knn.fit(ines,outes);
    outs=[]
    def kneighbors(task_train_origin):
        for inout in task_train_origin:
            inout['input'] = np.array(inout['input'])
        task_train = copy.deepcopy(task_train_origin)
        for i in range(len(task_train)):
            thisdone=False;
            vx=task_train[i]["input"].copy();
            vi=task_train[i]["input"].copy();
            for U in range(20):
                for k1 in range(len(vx)):
                    for k2 in range(len(vx[0])):
                        dtm=[];
                        for k3 in range(-2,2+1,1):
                            for k4 in range(-2,2+1,1):
                                if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                                    td = [0,0,0,0,0,0,0,0,0,0,0];
                                    td[vx[k1+k3][k2+k4]]=1
                                    dtm+=td.copy();
                                    td = [0,0,0,0,0,0,0,0,0,0,0];
                                    td[vi[k1+k3][k2+k4]]=1;
                                    dtm+=td.copy();
                                else:
                                    dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                                    dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                        vi[k1][k2]=int(knn.predict([dtm])[0]);
                vx=vi.copy();
            task_train[i]['input'] = vx
        return task_train
    return [inouts_array, kneighbors]


# # ARC solver

# In[183]:


class ARC_solver:
    def __init__(self):
        self.identified_objects = []
        self.io_inx = [] # the original index of the identified objects (io)
        self.io_height = [] # height of io
        self.io_width = [] # width of io
        self.io_pixel_count = [] # count of non-background pixels
        self.io_size = [] # overall grid size
        self.io_unique_colors = [] # number of unique colors
        self.io_main_color = [] # the dominating color

    def reset(self):
        self.identified_objects = []
        self.io_inx = []
        self.io_height = []
        self.io_width = []
        self.io_pixel_count = []
        self.io_size = []
        self.io_unique_colors = []
        self.io_main_color = []

    def get_background(self, image):
        # if image contains 0
        if 0 in image:
            background = 0
        # else use the most frequent pixel color
        else:
            unique_colors, counts = np.unique(image, return_counts = True)
            background = unique_colors[np.argmax(counts)]
        return background

    def check_pairs(self, inx_pairs, this_pair, return_inx = False):
        # check if this_pair is in inx_pairs
        match = []
        for pair in inx_pairs:
            if pair[0] == this_pair[0] and pair[1] == this_pair[1]:
                match.append(True)
            else:
                match.append(False)
        if return_inx:
            return any(match), np.where(match)
        else:
            return any(match)

    def check_neighbors(self, all_pairs, this_pair, objectness, this_object):
        # all_pairs: an array of index pairs for all nonzero/colored pixels
        # this_pair: the index pair whose neighbors will be checked
        # objectness: an array with the shape of original image, storage for how much objectness has been identified
        # this_object: the current object we are looking at
        row_inx = this_pair[0]
        col_inx = this_pair[1]
        objectness[row_inx, col_inx] = this_object
        # find if any neighboring pixels contain color
        if self.check_pairs(all_pairs, [row_inx-1, col_inx-1]): # up-left
            objectness[row_inx-1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx]): # up
            objectness[row_inx-1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx+1]): # up-right
            objectness[row_inx-1, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx-1]): # left
            objectness[row_inx, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx+1]): # right
            objectness[row_inx, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx-1]): # down-left
            objectness[row_inx+1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx]): # down
            objectness[row_inx+1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx+1]): # down-right
            objectness[row_inx+1, col_inx+1] = this_object
        return objectness

    def identify_object_by_color(self, true_image, background = 0):
        # identify obeject by the color only
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from original first
            if color == background:
                continue
            image[image != color] = background
            inx = np.where(image == color)
            obj = image[np.min(inx[0]):np.max(inx[0])+1, np.min(inx[1]):np.max(inx[1])+1]
            # append the object attributes
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])

    def identify_object_by_isolation(self, image, background = 0):
        # identify all objects by physical isolation on the given image
        all_pairs = np.array(np.where(image != background)).T
        objectness = np.zeros(image.shape)
        this_object = 1
        while len(all_pairs) >= 1:
            init_pair = all_pairs[0] # start with the first pair
            objectness = self.check_neighbors(all_pairs, init_pair, objectness, this_object)
            # get a list of index pairs whose neghbors haven't been checked
            unchecked_pairs = np.array(np.where(objectness == this_object)).T
            checked_pairs = np.zeros((0,2))
            # check all the index pairs in the expanding unchecked_pairs untill all have been checked
            while len(unchecked_pairs) != 0:
                this_pair = unchecked_pairs[0]
                objectness = self.check_neighbors(all_pairs, this_pair, objectness, this_object)
                # append the checked_pairs
                checked_pairs = np.vstack((checked_pairs, this_pair))
                # get all index pairs for the currently identified object
                current_object_pairs = np.array(np.where(objectness == this_object)).T
                # delete the checked pairs from current object pairs
                checked_inx = []
                for pair in checked_pairs:
                    _, inx = self.check_pairs(current_object_pairs, pair, return_inx = True)
                    checked_inx.append(inx[0][0])
                unchecked_pairs = np.delete(current_object_pairs, checked_inx, axis = 0)

            # store this object to identified_objects
            current_object_pairs = np.array(np.where(objectness == this_object)).T
            cop = current_object_pairs.T
            obj = image[np.min(cop[0]):np.max(cop[0])+1, np.min(cop[1]):np.max(cop[1])+1]
            # delete the current object pairs from all_pairs
            cop_inx = []
            for pair in current_object_pairs:
                _, this_cop_inx = self.check_pairs(all_pairs, pair, return_inx = True)
                cop_inx.append(this_cop_inx[0][0])
            all_pairs = np.delete(all_pairs, cop_inx, axis = 0)
            # append the object attribute
            # p(obj)
            if np.array(obj).shape[0] * np.array(obj).shape[0] >= 3:
                self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])
            # start identifying a new object
            this_object += 1
        return objectness

    def identify_object_by_color_isolation(self, true_image, background = 0):
        # identify objects first by color then by physical isolation
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from the original first
            if color == background:
                continue
            # identify objects by isolation in this color only
            image[image != color] = background
            self.identify_object_by_isolation(image, background = background)
            

    def sort(self, objs, inp):
        xs = []
        ys = []
        for i, o in enumerate(objs):
            _, m, n = sliding_window_search(inp, o)
            xs.append(m)
            ys.append(n)

        ans = [[[]],[[]],[[]],[[]]]
        left = np.array(ys).argsort()[0:2] # 1,3
        right = np.array(ys).argsort()[2:4] # 1,3
        if xs[left[0]] <= xs[left[1]]:
            ans[0] = objs[left[0]]
            ans[2] = objs[left[1]]
        else:
            ans[2] = objs[left[0]]
            ans[0] = objs[left[1]]    
        if xs[right[0]] <= xs[right[1]]:
            ans[1] = objs[right[0]]
            ans[3] = objs[right[1]]
        else:
            ans[3] = objs[right[0]]
            ans[1] = objs[right[1]]
        return ans        
            
    def merge(self, objects, belt, use_color):
#         ans = objects
        ans=[[[]],[[]],[[]],[[]]]
        for o in objects:
            o = np.array(o)
            max_total = 0
            for x in [0,1]:
                for y in [0,1]:
                    if max_total < o[x:x+len(o)-1, y:y+len(o[0])-1].sum():
                        max_total = o[x:x+len(o)-1, y:y+len(o[0])-1].sum()
                        max_xy = (x, y)
            if max_xy == (0,0):
                ans[3] = o
            elif max_xy == (0,1):
                ans[2] = o
            elif max_xy == (1,0):
                ans[1] = o
            else:
                ans[0] = o

        if belt == 0:
            belt_list = [[use_color]]*len(ans[0])
            u=np.hstack([ans[0], ans[1]])
            u
            s=np.hstack([ans[2], ans[3]])
            return np.vstack([u,s])
        else:
            belt_list = [[use_color]*belt]*len(ans[0])

            u=np.hstack([ans[0], belt_list, ans[1]])
            s=np.hstack([ans[2], belt_list, ans[3]])
            belt_list = [[use_color]*len(s[0])]*belt
            return np.vstack([u,belt_list,s])



# # add block merge

# In[184]:


def divide_block_and_merge(task_train_origin, objn):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for i, inout in enumerate(task_train):    
        arc = ARC_solver()
        inp = inout['input']
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return task_train_origin
        
        try:
            inp_o = copy.deepcopy(inp)
            inp = np.where(inp_o==use_color[0], use_color[1], inp)
            inp = np.where(inp_o==use_color[1], use_color[0], inp)
            background = arc.get_background(inp)
            arc.identify_object_by_isolation(inp, background)
            if len(arc.identified_objects) == 4:
                arc.identified_objects = arc.sort(arc.identified_objects, inp)
                out = np.array(arc.identified_objects[objn])
                out_o = copy.deepcopy(out)
                out = np.where(out_o==use_color[0], use_color[1], out)
                out = np.where(out_o==use_color[1], use_color[0], out)
                task_train[i]['input'] = out
        except:
            return task_train_origin
    return task_train

def divide_block_and_merge1(task_train_origin):
    return divide_block_and_merge(task_train_origin, 1)

def divide_block_and_merge2(task_train_origin):
    return divide_block_and_merge(task_train_origin, 2)

def divide_block_and_merge3(task_train_origin):
    return divide_block_and_merge(task_train_origin, 3)

def add_block_merge(task_train):
    arc = ARC_solver()
    if len(task_train) > 2:
        task_n = 2
    else:
        task_n = 0
    inp = task_train[task_n]['input']
    inp = np.array(inp)
    use_color = list(set(list(itertools.chain.from_iterable(inp))))
    if len(use_color) != 2:
        return []
    inp_o = copy.deepcopy(inp)
    inp = np.where(inp_o==use_color[0], use_color[1], inp)
    inp = np.where(inp_o==use_color[1], use_color[0], inp)
    background = arc.get_background(inp)
    arc.identify_object_by_isolation(inp, background)
    if len(arc.identified_objects) == 4:
        try:
            arc.identified_objects = arc.sort(arc.identified_objects, inp)
#             for i in arc.identified_objects:
#                 p(i)
            for i in range(4):
                out = np.array(arc.identified_objects[i])
                out_o = copy.deepcopy(out)
                out = np.where(out_o==use_color[0], use_color[1], out)
                out = np.where(out_o==use_color[1], use_color[0], out)    

                if out.tolist() == task_train[task_n]['output']:
                    return [divide_block_and_merge1, divide_block_and_merge2, divide_block_and_merge3]
        except:
            return []
    return []


# # add object detect2

# In[185]:


def select_by_ele(objects, ele):
    if ele == 'height':
        max_height = 0
        for obj in objects:
            if len(obj) > max_height:
                selected = obj
                max_height = len(obj)
    if ele == 'width':
        max_width = 0
        for obj in objects:
            if len(obj[0]) > max_width:
                selected = obj
                max_width = len(obj[0])
    if ele == 'area':
        max_area = 0
        for obj in objects:
            if len(obj) * len(obj[0]) > max_area:
                selected = obj
                max_area = len(obj) * len(obj[0])
        
    return selected

def add_object_detect2(task_train):
    for select_ele in ['height', 'width', 'area']:
        sucess = True
        for inout in task_train:
            arc = ARC_solver()
            inp = copy.deepcopy(inout['input'])
            inp = np.array(inp)
            background = arc.get_background(inp)
            arc.identify_object_by_isolation(inp, background)
            obj = select_by_ele(arc.identified_objects, select_ele)
            if (obj.shape != np.array(inout['output']).shape) & (obj.shape != np.array(inout['output']).T.shape):
                sucess = False
        if sucess:
            def object_detect2(task_train_origin):
                for inout in task_train_origin:
                    inout['input'] = np.array(inout['input'])
                task_train = copy.deepcopy(task_train_origin)
                for i, inout in enumerate(task_train):   
                    try:
                        arc = ARC_solver()
                        inp = copy.deepcopy(inout['input'])
                        inp = np.array(inp)
                        background = arc.get_background(inp)
                        arc.identify_object_by_isolation(inp, background)
                        obj = select_by_ele(arc.identified_objects, select_ele)
                        task_train[i]['input'] = obj
                    except:
                        return task_train_origin
                return task_train
                    
            return [object_detect2]
    return []


# # add crop_by_line

# In[186]:


def add_crop_by_line(task_train):
    success = True
    for i, inout in enumerate(task_train):
        inp = np.array(copy.deepcopy(inout['input']))
        use_color = matrix_use_color(inp)
        max_area = 0
        max_enclosure_color = 0
        include_line = False
        uses = [0,0,0,0]
        found = False
        use_max_x = 0
        use_max_y = 0
        use_min_x = 0
        use_min_y = 0
        for color in use_color:
            idx = [idx.tolist() for idx in np.array(np.where(inp==color)).T]

            max_x = 0
            max_y = 0
            min_x = 100
            min_y = 100
            for i in idx:
                if i[0] < min_x:
                    min_x = i[0]
                if i[1] < min_y:
                    min_y = i[1]
                if i[0] > max_x:
                    max_x = i[0]
                if i[1] > max_y:
                    max_y = i[1]

            enclosure_flag = True
            for x in range(min_x, max_x+1):
                if (inp[x][min_y] != color) or (inp[x][max_y] != color):
                    enclosure_flag = False
            for y in range(min_y, max_y+1):
                if (inp[min_x][y] != color) or (inp[max_x][y] != color):
                    enclosure_flag = False
            for x in range(min_x+1, max_x):
                for y in range(min_y+1, max_y):
                    if inp[x][y] == color:
                        enclosure_flag = False
            if enclosure_flag & (max_x > 0) & (max_x - min_x > 1):
                area = (max_x-min_x)*(max_y-min_y)
                if max_area < area:
                    max_area = area
                    max_enclosure_color = color
                    found = True
                    use_max_x = max_x
                    use_max_y = max_y
                    use_min_x = min_x
                    use_min_y = min_y
        if not found:
            return []
        if i == 0:
            if np.array(inout['output']).shape == (use_max_x-use_min_x-1, use_max_y-use_min_y-1):
                include_line = False
            elif np.array(inout['output']).shape == (use_max_x-use_min_x+1, use_max_y-use_min_y+1):
                include_line = True
            else:
                success = False
        else:
            if (not include_line) & (np.array(inout['output']).shape == (use_max_x-use_min_x-1, use_max_y-use_min_y-1)) or ((include_line) & (np.array(inout['output']).shape == (use_max_x-use_min_x+1, use_max_y-use_min_y+1))):
                pass
            else:
                success = False

        if not success:
            break

    if success:
        def crop_by_max_enclosure_color(task_train_origin):
            for inout in task_train_origin:
                inout['input'] = np.array(inout['input'])
            task_train = copy.deepcopy(task_train_origin)
            for task_n, inout in enumerate(task_train):
                inp = np.array(copy.deepcopy(inout['input']))
                use_color = matrix_use_color(inp)
                max_area = 0
                max_enclosure_color = 0
                include_line = False
                uses = [0,0,0,0]
                found = False
                use_max_x = 0
                use_max_y = 0
                use_min_x = 0
                use_min_y = 0
                for color in use_color:
                    idx = [idx.tolist() for idx in np.array(np.where(inp==color)).T]

                    max_x = 0
                    max_y = 0
                    min_x = 100
                    min_y = 100
                    for i in idx:
                        if i[0] < min_x:
                            min_x = i[0]
                        if i[1] < min_y:
                            min_y = i[1]
                        if i[0] > max_x:
                            max_x = i[0]
                        if i[1] > max_y:
                            max_y = i[1]
                    enclosure_flag = True
                    for x in range(min_x, max_x+1):
                        if (inp[x][min_y] != color) or (inp[x][max_y] != color):
                            enclosure_flag = False
                    for y in range(min_y, max_y+1):
                        if (inp[min_x][y] != color) or (inp[max_x][y] != color):
                            enclosure_flag = False
                    for x in range(min_x+1, max_x):
                        for y in range(min_y+1, max_y):
                            if inp[x][y] == color:
                                enclosure_flag = False
                    if enclosure_flag & (max_x > 0) & (max_x - min_x > 1):
                        area = (max_x-min_x)*(max_y-min_y)
                        if max_area < area:
                            max_area = area
                            max_enclosure_color = color
                            found = True
                            use_max_x = max_x
                            use_max_y = max_y
                            use_min_x = min_x
                            use_min_y = min_y                
                if include_line:
                    out = inp[use_min_x:use_max_x+1, use_min_y:use_max_y+1]
                else:
                    out = inp[use_min_x+1:use_max_x, use_min_y+1:use_max_y]
                task_train[task_n]['input'] = out
            return task_train
                
        return [crop_by_max_enclosure_color]
    return []


# # add back_to_black

# In[187]:


def back_to_black(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for task_n, inout in enumerate(task_train):
        inp = inout['input']
        inp_o = copy.deepcopy(inp)
        i = list(itertools.chain.from_iterable(inp))
        most_use_color = collections.Counter(i).most_common()[0][0]
        inp = np.where(inp_o==most_use_color, 0, inp)
        inp = np.where(inp_o==0, most_use_color, inp)
        task_train[task_n]['input'] = inp
    return task_train

def add_back_to_black_funcs(task_train):
    change_back = True
    for inout in task_train:
        i = list(itertools.chain.from_iterable(inout['input']))
        if collections.Counter(i).most_common()[0][0] == 0:
            change_back = False
    
    if change_back:
        return [back_to_black, inouts_array]
    else:
        return [inouts_array]


# In[189]:


class ARC_solver:
    def __init__(self):
        self.identified_objects = []
        self.io_inx = [] # the original index of the identified objects (io)
        self.io_height = [] # height of io
        self.io_width = [] # width of io
        self.io_pixel_count = [] # count of non-background pixels
        self.io_size = [] # overall grid size
        self.io_unique_colors = [] # number of unique colors
        self.io_main_color = [] # the dominating color

    def reset(self):
        self.identified_objects = []
        self.io_inx = []
        self.io_height = []
        self.io_width = []
        self.io_pixel_count = []
        self.io_size = []
        self.io_unique_colors = []
        self.io_main_color = []

    def get_background(self, image):
        # if image contains 0
        if 0 in image:
            background = 0
        # else use the most frequent pixel color
        else:
            unique_colors, counts = np.unique(image, return_counts = True)
            background = unique_colors[np.argmax(counts)]
        return background

    def check_pairs(self, inx_pairs, this_pair, return_inx = False):
        # check if this_pair is in inx_pairs
        match = []
        for pair in inx_pairs:
            if pair[0] == this_pair[0] and pair[1] == this_pair[1]:
                match.append(True)
            else:
                match.append(False)
        if return_inx:
            return any(match), np.where(match)
        else:
            return any(match)

    def check_neighbors(self, all_pairs, this_pair, objectness, this_object):
        # all_pairs: an array of index pairs for all nonzero/colored pixels
        # this_pair: the index pair whose neighbors will be checked
        # objectness: an array with the shape of original image, storage for how much objectness has been identified
        # this_object: the current object we are looking at
        row_inx = this_pair[0]
        col_inx = this_pair[1]
        objectness[row_inx, col_inx] = this_object
        # find if any neighboring pixels contain color
        if self.check_pairs(all_pairs, [row_inx-1, col_inx-1]): # up-left
            objectness[row_inx-1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx]): # up
            objectness[row_inx-1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx+1]): # up-right
            objectness[row_inx-1, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx-1]): # left
            objectness[row_inx, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx+1]): # right
            objectness[row_inx, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx-1]): # down-left
            objectness[row_inx+1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx]): # down
            objectness[row_inx+1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx+1]): # down-right
            objectness[row_inx+1, col_inx+1] = this_object
        return objectness

    def identify_object_by_color(self, true_image, background = 0):
        # identify obeject by the color only
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from original first
            if color == background:
                continue
            image[image != color] = background
            inx = np.where(image == color)
            obj = image[np.min(inx[0]):np.max(inx[0])+1, np.min(inx[1]):np.max(inx[1])+1]
            # append the object attributes
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])

    def identify_object_by_isolation(self, image, background = 0):
        # identify all objects by physical isolation on the given image
        all_pairs = np.array(np.where(image != background)).T
        objectness = np.zeros(image.shape)
        this_object = 1
        while len(all_pairs) >= 1:
            init_pair = all_pairs[0] # start with the first pair
            objectness = self.check_neighbors(all_pairs, init_pair, objectness, this_object)
            # get a list of index pairs whose neghbors haven't been checked
            unchecked_pairs = np.array(np.where(objectness == this_object)).T
            checked_pairs = np.zeros((0,2))
            # check all the index pairs in the expanding unchecked_pairs untill all have been checked
            while len(unchecked_pairs) != 0:
                this_pair = unchecked_pairs[0]
                objectness = self.check_neighbors(all_pairs, this_pair, objectness, this_object)
                # append the checked_pairs
                checked_pairs = np.vstack((checked_pairs, this_pair))
                # get all index pairs for the currently identified object
                current_object_pairs = np.array(np.where(objectness == this_object)).T
                # delete the checked pairs from current object pairs
                checked_inx = []
                for pair in checked_pairs:
                    _, inx = self.check_pairs(current_object_pairs, pair, return_inx = True)
                    checked_inx.append(inx[0][0])
                unchecked_pairs = np.delete(current_object_pairs, checked_inx, axis = 0)

            # store this object to identified_objects
            current_object_pairs = np.array(np.where(objectness == this_object)).T
            cop = current_object_pairs.T
            obj = image[np.min(cop[0]):np.max(cop[0])+1, np.min(cop[1]):np.max(cop[1])+1]
            # delete the current object pairs from all_pairs
            cop_inx = []
            for pair in current_object_pairs:
                _, this_cop_inx = self.check_pairs(all_pairs, pair, return_inx = True)
                cop_inx.append(this_cop_inx[0][0])
            all_pairs = np.delete(all_pairs, cop_inx, axis = 0)
            # append the object attribute
            # p(obj)
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])
            # start identifying a new object
            this_object += 1
        return objectness

    def identify_object_by_color_isolation(self, true_image, background = 0):
        # identify objects first by color then by physical isolation
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from the original first
            if color == background:
                continue
            # identify objects by isolation in this color only
            image[image != color] = background
            self.identify_object_by_isolation(image, background = background)
            

    def sort(self, objs, inp):
        xs = []
        ys = []
        for i, o in enumerate(objs):
            _, m, n = sliding_window_search(inp, o)
            xs.append(m)
            ys.append(n)

        ans = [[[]],[[]],[[]],[[]]]
        left = np.array(ys).argsort()[0:2] # 1,3
        right = np.array(ys).argsort()[2:4] # 1,3
        if xs[left[0]] <= xs[left[1]]:
            ans[0] = objs[left[0]]
            ans[2] = objs[left[1]]
        else:
            ans[2] = objs[left[0]]
            ans[0] = objs[left[1]]    
        if xs[right[0]] <= xs[right[1]]:
            ans[1] = objs[right[0]]
            ans[3] = objs[right[1]]
        else:
            ans[3] = objs[right[0]]
            ans[1] = objs[right[1]]        
        return ans        
            
    def merge(self, objects, belt, use_color):
        ans=objects
        ans=[[[]],[[]],[[]],[[]]]
        for o in objects:
            o = np.array(o)
            max_total = 0
            for x in [0,1]:
                for y in [0,1]:
                    if max_total < o[x:x+len(o)-1, y:y+len(o[0])-1].sum():
                        max_total = o[x:x+len(o)-1, y:y+len(o[0])-1].sum()
                        max_xy = (x, y)
            if max_xy == (0,0):
                ans[3] = o
            elif max_xy == (0,1):
                ans[2] = o
            elif max_xy == (1,0):
                ans[1] = o
            else:
                ans[0] = o

        if belt == 0:
            belt_list = [[use_color]]*len(ans[0])
            u=np.hstack([ans[0], ans[1]])
            u
            s=np.hstack([ans[2], ans[3]])
            return np.vstack([u,s])
        else:
            belt_list = [[use_color]*belt]*len(ans[0])

            u=np.hstack([ans[0], belt_list, ans[1]])
            s=np.hstack([ans[2], belt_list, ans[3]])
            belt_list = [[use_color]*len(s[0])]*belt
            return np.vstack([u,belt_list,s])


# In[ ]:





# In[190]:


def about_color_for_test(task_test):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_test:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))

def all_inputs_same_shape_and_all_outputs_same_shape(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    return all_inputs_same_shape, all_outputs_same_shape



def change_color_for_p(inp_origin, in_use):
    inp = copy.deepcopy(inp_origin)
    color_map = {}
    use_color_num = len(in_use)
    out_colors = range(use_color_num)
    for i,o in zip(sorted(in_use), sorted(out_colors)):
        color_map[i] = o

    for i, o in color_map.items():
        inp = np.where(np.array(inp_origin) == i, o, inp)
    return inp.tolist()


# In[191]:


def all_inouts_same_shape(task_train):
    for inout in task_train:
        if np.array(inout['input']).shape != np.array(inout['output']).shape:
            return False
    return True


# In[192]:


def update_by_teacher_row_n(inp_o, back=False, n=0):
    try:
        one_color_rows = []
        not_one_color_rows = []
        inp = copy.deepcopy(inp_o)
        for row in inp:
            if len(set(row)) != 1:
                not_one_color_rows.append(row)
            else:
                one_color_rows.append(row)
        c = collections.Counter(np.array(inp).flatten().tolist())
        back_color = c.most_common()[0][0]
        for row_n, row in enumerate(inp):
            if len(set(row)) == 1:
                continue
            tea = copy.deepcopy(not_one_color_rows[n])
            success = True
            for ele_n, ele in enumerate(row):
                if (ele != back_color) & (tea[ele_n] != ele):
                    success = False
            if success:
                inp[row_n] = tea
            else:
                inp[row_n] = [back_color] * len(row)

        return np.array(inp).tolist()
    except:
        return [[0]]    


# In[193]:


def perfect_same(big, small):
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True
    small = np.flipud(small)
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True    
    small = np.flip(small)
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True    
    return False            


# In[194]:


from collections import Counter
def connect_two_point(inp_o, fill=False):
    try:
        counter = Counter(np.array(inp_o).flatten().tolist())
        back = counter.most_common()[0][0]
        inp = copy.deepcopy(np.array(inp_o))
        for row_n, row in enumerate(inp_o):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[row_n, start+1+i] = row[start]
                            inp[row_n, end-1-i] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[row_n, start+1+i] = row[start]
                            inp[row_n, end-1-i] = row[end]                          
                        start = ele_n

        for row_n, row in enumerate(np.transpose(inp_o)):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]                        
                        start = ele_n
        return inp.tolist()
    except:
        return [[0]]
        


# In[196]:


from lightgbm import LGBMClassifier
import pdb

def preds_to_str(preds_list, idx=''):
    pred_strs = []
#     st()
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            if j == 3:
                break
            pred_str += flattener(np.array(preds[i]).tolist()) + ' '
        pred_strs.append(pred_str)
    return pred_strs

data_path = Path('../input/abstraction-and-reasoning-challenge/')
data_path = Path('data/')

test_path = data_path / 'test'
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    if cur_row<=0: top = -1
    else: top = color[cur_row-1][cur_col]
        
    if cur_row>=nrows-1: bottom = -1
    else: bottom = color[cur_row+1][cur_col]
        
    if cur_col<=0: left = -1
    else: left = color[cur_row][cur_col-1]
        
    if cur_col>=ncols-1: right = -1
    else: right = color[cur_row][cur_col+1]
        
    return top, bottom, left, right

def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        
    if cur_row==0:
        top_left = -1
        top_right = -1
    else:
        if cur_col==0: top_left=-1
        else: top_left = color[cur_row-1][cur_col-1]
        if cur_col==ncols-1: top_right=-1
        else: top_right = color[cur_row-1][cur_col+1]   
        
    return top_left, top_right

def features(task, mode='train'):
    cur_idx = 0
    num_train_pairs = len(task[mode])
    total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])
    feat = np.zeros((total_inputs,nfeat))
    target = np.zeros((total_inputs,), dtype=np.int)
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        target_color = task[mode][task_num]['output']
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx,0] = i
                feat[cur_idx,1] = j
                feat[cur_idx,2] = input_color[i][j]
                feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx,9] = len(np.unique(input_color[i,:]))
                feat[cur_idx,10] = len(np.unique(input_color[:,j]))
                feat[cur_idx,11] = (i+j)
                feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                             j-local_neighb:j+local_neighb]))
        
                target[cur_idx] = target_color[i][j]
                cur_idx += 1
            
    return feat, target, 0

all_task_ids = sorted(os.listdir(test_path))
lgb_range = [24]

nfeat = 13
local_neighb = 5
valid_scores = {}
for task_n, task_id in enumerate(all_task_ids):
    if task_n not in lgb_range:
        continue
    task_file = str(test_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        not_valid = 0
        continue

    nrows, ncols = len(task['train'][-1]['input']
                       ), len(task['train'][-1]['input'][0])
    # use the last train sample for validation
    val_idx = len(feat) - nrows*ncols

    train_feat = feat[:val_idx]
    val_feat = feat[val_idx:, :]

    train_target = target[:val_idx]
    val_target = target[val_idx:]

    #     check if validation set has a new color
    #     if so make the mapping color independant
    if len(set(val_target) - set(train_target)):
        continue

    lgb = LGBMClassifier(n_estimators=50, n_jobs=-1)
    lgb.fit(feat, target,
            verbose=-1)

#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = np.zeros((nrows*ncols, nfeat))
        unique_col = {col: i for i, col in enumerate(
            sorted(np.unique(input_color)))}

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = get_moore_neighbours(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = get_tl_tr(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i+j)
                feat[cur_idx, 12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                              j-local_neighb:j+local_neighb]))

                cur_idx += 1

        preds = lgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int).tolist()
        submission.loc[submission.index == f'{task_id[:-5]}_{task_num}', 'output'] = flattener(preds) 

# print("4109 submission", submission)

# In[1]:


def about_color(task_train):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_colors += list(set(out_vec))
        if set(in_vec) != set(out_vec):
            color_changed = True
    return list(set(in_colors)), list(set(out_colors)), color_changed
def paint_rolling2(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
#             inp[:, -2] = back_color
            # inp[0, :] = back_color
            return inp
def paint_each_and_vstack2(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling2(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp 

def add_stack4(task):
    skip = False
    success = False
    if task['test'][0]['input'] != np.transpose(task['test'][0]['input']).tolist():
        return False
    for inout in task['train']:
        inp_min_n_shape = np.array(inout['input']).shape
        out_min_n_shape = np.array(inout['output']).shape
        if (inp_min_n_shape[0] * 2 - 1 != out_min_n_shape[0]) or (inp_min_n_shape[1] * 2 - 1 != out_min_n_shape[1]):
            return False
        print(3)
        inp = inout['input']
        out = inout['output']
        if (np.flip(stack4(np.flip(inp))).tolist() == out) or (np.array(stack4(inp)).tolist() == out):
            return True
    return False


def rebuild_by_identified_objects(objs, background, x, pattern):
    try:
        size_map = {}
        for i, o in enumerate(objs):
            size_map[i] = len(np.array(o).flatten())
        size_map = sorted(size_map.items(), key=lambda x:x[1])
        out = copy.deepcopy(objs[size_map[2][0]])
        out_color = out[1][1]
        ele = np.array(objs[size_map[pattern[0]][0]])
        ele = np.where(ele==background, out_color, ele)
        cood = objs[size_map[pattern[1]][0]]
        for row_n, row in enumerate(cood):
            for col_n, r in enumerate(row):
                if r != background:
                    out[row_n*len(ele):(row_n+1)*len(ele), col_n*len(ele[0]):(col_n+1)*len(ele[0])] = ele
        for i in range((x-len(out[0]))//2):
            out = np.insert(out, 0, background,axis=0)
            out = np.insert(out, 0, background,axis=1)
            out = np.insert(out, len(out[0]), background,axis=1)
            out = np.insert(out, len(out), background,axis=0)
        return out
    except:
        return [[0]]

def recolor_by_origin_placement(inp_o, obj, background):
    inp = np.array(copy.deepcopy(inp_o))
    coods = []
    obj_coods = []
    x=0
    for i in range(len(obj)):
        y=0
        x+=1
        for j in range(len(obj[0])):
            y+=1
            if np.all(inp[x+i*len(obj):x+(i+1)*len(obj), y+j*len(obj[0]):y+(j+1)*len(obj[0])] == obj):
                coods.append([x+i*len(obj),y+j*len(obj[0])])
                obj_coods.append([i,j])
    inp = np.where(inp_o == background, obj[0][0], inp)
    inp = np.where(inp_o == obj[0][0], background, inp)
    print(coods)
    for c in obj_coods:
        obj[c[0]][c[1]] = background
    for c in coods:
        inp[c[0]:c[0]+len(obj), c[1]:c[1]+len(obj[0])] = obj
    return inp
def paint_rolling4(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
            inp[:, -2:] = back_color
            # inp[0, :] = back_color
            return inp

def paint_each_and_vstack4(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling4(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp

def paint_rolling3(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
            inp[:, -2] = back_color
            # inp[0, :] = back_color
            return inp

def paint_each_and_vstack3(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling3(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp

def stack4(inp_o):
    try:
        inp = np.array(copy.deepcopy(inp_o))
#         inp = np.where(inp==inp.T, inp, inp[-1][-1])
        a = inp
        b = np.fliplr(inp)
        c = np.flipud(inp)
        d = np.flip(inp)
        e = np.hstack([a,b[:, 1:]])
        f = np.hstack([c,d[:, 1:]])
        return np.vstack([e, f[1:, :]])
    except:
        return inp_o                

def copy_by_belt_and_change_color(inp_o, change, to_back, to_belt=False, reverse=False, mirror=True):
    try:
        inp = copy.deepcopy(inp_o)
        belt = inp[0][0]
        one_color_col_colors = []
        for col in np.transpose(inp).tolist():
            if len(set(col)) == 1:
                test_has_one_color_col = True
                one_color_col_colors.append(col[0])
        one_color_col_colors = list(set(one_color_col_colors))
        back = inp[0][0]
        if len(set(np.array(inp)[:, 0])) == 1:
            back = inp[0][0]
        elif len(set(np.array(inp)[:, -1])) == 1:
            back = inp[-1][-1]
        if one_color_col_colors[0] == back:
            belt = one_color_col_colors[1]
        else:
            belt = one_color_col_colors[0]

        belt_xs = []
        for ele_n, ele in enumerate(inp[0]):
            if ele == belt:
                belt_xs.append(ele_n)
        change_left = False
        change_right = True
        if np.array(inp)[:, :belt_xs[0]].flatten().tolist().count(back) / len(np.array(inp)[:, :belt_xs[0]].flatten().tolist()) > np.array(inp)[:, belt_xs[-1]+1:].flatten().tolist().count(back) / len(np.array(inp)[:, belt_xs[1]+1:].flatten().tolist()):
            change_left = True
            change_right = False

        range_x = np.min([belt_xs[0], len(inp[0])-belt_xs[-1]-1])
        inp = np.array(inp)
        use_colors = list(set(inp.flatten().tolist()) - set([back, belt]))
        if len(use_colors) == 0:
            use_color = belt
        else:
            use_color = use_colors[0]

        for x in range(range_x):
            for y in range(len(inp)):
                a, b = inp[y, belt_xs[0]-x-1], inp[y, belt_xs[-1]+x+1]
                if (a != back) & (b != back):
                    if a == b:
                        if to_back:
                            inp[y, belt_xs[-1]+x+1] = back
                            inp[y, belt_xs[0]-x-1] = back
                        elif change:
                            if a == belt:
                                inp[y, belt_xs[-1]+x+1] = use_color
                                inp[y, belt_xs[0]-x-1] = use_color
                            else:
                                inp[y, belt_xs[-1]+x+1] = belt
                                inp[y, belt_xs[0]-x-1] = belt
                    else:
                        if to_belt:
                            inp[y, belt_xs[-1]+x+1] = belt
                            inp[y, belt_xs[0]-x-1] = belt
                        elif reverse:
                            inp[y, belt_xs[-1]+x+1] = a
                            inp[y, belt_xs[0]-x-1] = b
                        else:
                            if a == belt:
                                inp[y, belt_xs[-1]+x+1] = use_color
                            else:
                                inp[y, belt_xs[-1]+x+1] = belt
                            if b == belt:
                                inp[y, belt_xs[0]-x-1] = use_color
                            else:
                                inp[y, belt_xs[0]-x-1] = belt
                elif (a != back):
                    if a == belt:
                        inp[y, belt_xs[-1]+x+1] = use_color
                    else:
                        inp[y, belt_xs[-1]+x+1] = belt
                elif inp[y, belt_xs[-1]+x+1] != back:
                    if b == belt:
                        inp[y, belt_xs[0]-x-1] = use_color
                    else:
                        inp[y, belt_xs[0]-x-1] = belt
        if not mirror:
            if change_left:
                inp[:, belt_xs[0]-range_x:belt_xs[0]] = np.fliplr(inp[:, belt_xs[0]-range_x:belt_xs[0]])
            else:
                inp[:, belt_xs[1]+1:belt_xs[1]+1+range_x] = np.fliplr(inp[:, belt_xs[1]+1:belt_xs[1]+1+range_x])

        return inp
    except:
        return [[0]]
def add_recolor_by_origin_placement(task):
    inp = task['test'][0]['input']
    inp_train = task['train'][0]['input']
    if (len(inp) != len(inp[0])) or (len(inp_train) != len(inp_train[0])):
        return False

    use_color = list(set(list(itertools.chain.from_iterable(inp))))
    if len(use_color) != 2:
        return False
    for inout in task['train']:
        success = False

        i_test = np.array(inout['input'])
        arc = ARC_solver()
        background = arc.get_background(i_test)
        arc.identify_object_by_isolation(i_test, background)

        a = recolor_by_origin_placement(i_test, arc.identified_objects[0], background)
        if np.array(a).tolist() == inout['output']:
            success = True
            break
    return success

def add_rebuild_by_identified_objects(task):
    use_colors = list(set(np.array(task['test'][0]['input']).flatten().tolist()))
    if len(use_colors) != 4:
        return False
    inp = task['train'][-2]['input']
    out = task['train'][-2]['output']
    if (len(inp[0]) != len(out[0])):
        return False

    success = False
    for test_n, inout in enumerate(task['train']):
        i_test = np.array(inout['input'])
        arc = ARC_solver()
        background = arc.get_background(i_test)
        arc.identify_object_by_isolation(i_test, background)

        a = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [0,1])
        b = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [1,0])
        if (np.array(a).tolist() == inout['output']) or (np.array(b).tolist() == inout['output']):
            success = True
            break
    return success

def add_copy_by_belt_and_change_color(task):
    skip = False
    inp = task['test'][0]['input']
    for n, row in enumerate(inp):
        if len(set(row)) == 1:
            skip = True
    if skip:
        return False
    unique_one_color_col_ns = []
    for n, col in enumerate(np.transpose(inp)):
        if len(set(col)) == 1:
            unique_one_color_col_ns.append(col[0])
    if len(set(unique_one_color_col_ns)) != 2:
        return False
    success = False
    for test_n, inout in enumerate(task['train']):
        i_test = np.transpose(inout['input']).tolist()
        a = np.transpose(copy_by_belt_and_change_color(i_test,True,False, mirror=False)).tolist()
        b = np.transpose(copy_by_belt_and_change_color(i_test,True,False, reverse=True)).tolist()
        c = np.transpose(copy_by_belt_and_change_color(i_test,True,False, to_belt=True)).tolist()
        if (a == inout['output']) or (b == inout['output']) or (c == inout['output']):
            success = True
            break
    if not success:
        return False
    return True

def add_paint_each_and_vstack(task):
    inp = copy.deepcopy(task['train'][-1]['input'])
    in_use, out_use, color_changed = about_color(copy.deepcopy(task['train'][-1:]))
    if len(in_use) != 3:
        return False
    v=list(itertools.chain.from_iterable(inp))
    if (len(v) - v.count(in_use[0]) != 2) & (len(v) - v.count(in_use[1]) != 2) & (len(v) - v.count(in_use[2]) != 2):
        return False
    if np.array(paint_each_and_vstack2(inp)).tolist() != task['train'][-1]['output']:
        return False
    return True

def connect_two_point2(inp_o, fill=False):
    try:
        counter = Counter(np.array(inp_o).flatten().tolist())
        back = counter.most_common()[0][0]
        inp = copy.deepcopy(np.array(inp_o))

        for row_n, row in enumerate(np.transpose(inp_o)):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]                        
                        start = ele_n
        return inp.tolist()
    except:
        return [[0]]  
def add_connect_two_point2(task):
    success = False
    for inout in task['train']:
        if (np.array(connect_two_point2(inout['input'], fill=True)).tolist() == inout['output']) or (np.transpose(connect_two_point2(np.transpose(inout['input']), fill=True)).tolist() == inout['output']):
            success = True
    return success
def stack4_2(inp_o):
    try:
        inp = np.array(copy.deepcopy(inp_o))
        inp[-1][-2] = inp[-1][-1]
        inp[-2][-1] = inp[-1][-1]
        a = inp
        b = np.fliplr(inp)
        c = np.flipud(inp)
        d = np.flip(inp)
        e = np.hstack([a,b[:, 1:]])
        f = np.hstack([c,d[:, 1:]])
        return np.vstack([e, f[1:, :]])
    except:
        return inp_o

def add_several_funcs(task):
    try:
        if (len(task['test'][0]['input']) % 16 == 0) & (len(task['test'][0]['input'][0]) % 16 == 0):
            return None, False
        use_flag = add_recolor_by_origin_placement(task)
        if use_flag:
            return recolor_by_origin_placement, True
        use_flag = add_rebuild_by_identified_objects(task)
        if use_flag:
            return rebuild_by_identified_objects, True
        use_flag = add_copy_by_belt_and_change_color(task)
        if use_flag:
            return copy_by_belt_and_change_color, True
        use_flag = add_paint_each_and_vstack(task)
        if use_flag:
            return paint_each_and_vstack3, True
        use_flag = add_stack4(task)
        if use_flag:
            return stack4, True
        use_flag = add_connect_two_point2(task)
        if use_flag:
            return connect_two_point2, True
        if add_block_merge(task['train']):
                return divide_block_and_merge3, True
        return None, False
    except:
        return None, False

def apply_several_func(task, func):
    try:
        if func == recolor_by_origin_placement:
            for test_n, inout in enumerate(task['test']):
                i_test = np.array(inout['input'])
                arc = ARC_solver()
                background = arc.get_background(i_test)
                arc.identify_object_by_isolation(i_test, background)

                preds = []
                a = recolor_by_origin_placement(i_test, arc.identified_objects[0], background)
                b = [[0]]
                c = [[0]]
                return [a,b,c]
        elif func == rebuild_by_identified_objects:
            for test_n, inout in enumerate(task['test']):
                i_test = np.array(inout['input'])
        #         p(i_test)
                arc = ARC_solver()
                background = arc.get_background(i_test)
                arc.identify_object_by_isolation(i_test, background)

                preds = []
                a = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [0,1])
                b = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [1,0])
                c = [[0]]
                return [a,b,c]
        elif func == copy_by_belt_and_change_color:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                preds = []
                a = copy_by_belt_and_change_color(i_test,True,False, mirror=False)
                b = copy_by_belt_and_change_color(i_test,True,False, reverse=True)
                c = copy_by_belt_and_change_color(i_test,True,False, to_belt=True)
                return [a,b,c]
        elif func == paint_each_and_vstack3:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                a=paint_each_and_vstack3(np.flip(i_test))
                b=paint_each_and_vstack3(i_test)
                c=paint_each_and_vstack4(i_test)
                return [a,b,c]
        elif func == stack4:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                if i_test[0][0] == i_test[0][1]:
                    a=stack4_2(np.flip(i_test))
                    b=stack4(np.flip(i_test))
                    c=stack4(i_test)
                else:
                    a=stack4_2(i_test)
                    b=stack4(i_test)
                    c=stack4(np.flip(i_test))
                return [a,b,c]
        elif func == connect_two_point2:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                preds = []
                a = connect_two_point2(inout['input'], fill=False)
                b = connect_two_point2(inout['input'], fill=True)
                c = np.transpose(connect_two_point2(np.transpose(inout['input']), fill=True)).tolist()
                return [a, b, c]
        elif func == divide_block_and_merge3:
            t1=divide_block_and_merge3(task['test'], 1)
            t2=divide_block_and_merge3(task['test'], 2)
            t3=divide_block_and_merge3(task['test'], 3)
            return [t1[0]['input'], t2[0]['input'], t3[0]['input']]

    except:
        return []
    
def add_block_merge(task_train):
    try:
        arc = ARC_solver()
        inout = task_train[-1]
        inp = copy.deepcopy(inout['input'])
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return False
        inp_o = copy.deepcopy(inp)
        inp = np.where(inp_o==use_color[0], use_color[1], inp)
        inp = np.where(inp_o==use_color[1], use_color[0], inp)
        background = arc.get_background(inp)
        arc.identify_object_by_isolation(inp, background)
        if len(arc.identified_objects) != 4:
            return False

#             arc.identified_objects = arc.sort(arc.identified_objects, inp)
#             for i in arc.identified_objects:
#                 p(i)
#         out = arc.merge(arc.identified_objects, 1, use_color[1])
        for i in range(4):
            out = arc.identified_objects[i]
            out_o = copy.deepcopy(out)
            out = np.where(out_o==use_color[0], use_color[1], out)
            out = np.where(out_o==use_color[1], use_color[0], out)
            if out.tolist() == inout['output']:
                return True
    except:
        pass
    return False    
def divide_block_and_merge3(task_train_origin, obj_numb):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for i, inout in enumerate(task_train):    
        arc = ARC_solver()
        inp = inout['input']
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return task_train_origin
#         try:
        inp_o = copy.deepcopy(inp)
        inp = np.where(inp_o==use_color[0], use_color[1], inp)
        inp = np.where(inp_o==use_color[1], use_color[0], inp)
        background = arc.get_background(inp)
        arc.identify_object_by_isolation(inp, background)
        if len(arc.identified_objects) == 4:
            arc.identified_objects = arc.sort(arc.identified_objects, inp)
            out = arc.identified_objects[obj_numb]
            out_o = copy.deepcopy(out)
            out = np.where(out_o==use_color[0], use_color[1], out)
            out = np.where(out_o==use_color[1], use_color[0], out)
            task_train[i]['input'] = out
#         except:
#             return task_train_origin
    return task_train

def main(idx, task, env='dev'):
    several_f = False
    func_combi_map = defaultdict(list)
    result = pd.Series()
    preprocess_best_score_map = {}
    best_aug_score_map = {}
    success_list = []
    final_score_map = {}
    pre_final_score_map = {}
    promising_map = defaultdict(bool)
    time_map = {}
    skip = False
    back_to_black = False
    origin_back_color = 1
    preprocess_best_score = 0
    best_func_combi = []
    task_n = 0
    correct_only_preprocess_flag = False
    use_several_func = False
    start = time()
    print('--------------')
    print(f'{task_n}：{idx}')
    flip_funcs = [inouts_array, inouts_flip, inouts_flipud, inouts_fliplr]
    back_to_black_funcs = add_back_to_black_funcs(copy.deepcopy(task['train']))
    func, use_several_func_flag = add_several_funcs(task)
    print("use_several_func_flag", use_several_func_flag)
    if (len(task['test'][0]['input']) % 16 != 0) & (len(task['test'][0]['input'][0]) % 16 != 0) & (not use_several_func_flag):
        skip = True

    for back_to_black_func in back_to_black_funcs:
        if use_several_func_flag:
            outputs = apply_several_func(task, func)
            if len(outputs) != 0:
                use_several_func = True
                break
            else:
                use_several_func_flag = False
        if correct_only_preprocess_flag or use_several_func or skip:
            break
        train_copy0 = back_to_black_func(copy.deepcopy(task['train']))
        size_change_funcs = add_size_change_funcs(train_copy0, task_n)

        for size_change_func in size_change_funcs:
            if correct_only_preprocess_flag or use_several_func:
                break
            shaped_train = size_change_func(copy.deepcopy(train_copy0))
            # print(type(shaped_train))
            transpose_funcs = add_transpose(shaped_train)
            for transpose_func in transpose_funcs:
                if correct_only_preprocess_flag or use_several_func:
                    break
                shaped_train1 = transpose_func(copy.deepcopy(shaped_train))
#                     if size_change_func == divide_block_and_merge1:
#                         st()

                shape_different_flag = False
#                     print(size_change_funcs)
                for shaped_inout in shaped_train1:
                    if shaped_inout['input'].shape != np.array(shaped_inout['output']).shape:
                        shape_different_flag = True
                        break
                if shape_different_flag:
                    break

                train4_funcs = add_train4_growth(shaped_train1)
                for train4_func in train4_funcs:
                    if correct_only_preprocess_flag:
                        break
                    shaped_train2 = train4_func(copy.deepcopy(shaped_train1))
                    # print(type(shaped_train2))
                    fill_closed_area_funcs = add_fill_closed_area(shaped_train2.copy())
                    for fill_closed_area_func in fill_closed_area_funcs:
                        if correct_only_preprocess_flag:
                            break
                        shaped_train3 = fill_closed_area_func(copy.deepcopy(shaped_train2))
                        # print(type(shaped_train3))
                        for flip_func_num, flip_func in enumerate(flip_funcs):
                            if correct_only_preprocess_flag:
                                break
                            shaped_train4 = flip_func(copy.deepcopy(shaped_train3))
                            patch_funcs = add_patch_funcs(shaped_train4, idx)
                            for patch_func in patch_funcs:
                                if correct_only_preprocess_flag:
                                    break
                                shaped_train5 = patch_func(copy.deepcopy(shaped_train4))
                                task_train6_funcs = add_task_train6(shaped_train5)
                                for train6_funcs in task_train6_funcs:
                                    if correct_only_preprocess_flag:
                                        break
                                    shaped_train6 = train6_funcs(copy.deepcopy(shaped_train5))
                                    move_object_funcs = add_move_object(shaped_train6)
                                    for move_object_func in move_object_funcs:
                                        if correct_only_preprocess_flag:
                                            break
                                        shaped_train7 = move_object_func(copy.deepcopy(shaped_train6))
                                        recolor_funcs = add_recolor(shaped_train7, task_n)
                                        for recolor_func in recolor_funcs:
                                            if correct_only_preprocess_flag:
                                                break
                                            shaped_train8 = recolor_func(copy.deepcopy(shaped_train7))
                                            kneighbor_funcs = add_kneighbors(shaped_train8)
                                            for kneighbor_func in kneighbor_funcs:
                                                if correct_only_preprocess_flag:
                                                    break
                                                shaped_train9 = kneighbor_func(copy.deepcopy(shaped_train8))

                                                change_color_funcs = add_change_color_funcs(shaped_train9)
                                                for change_color_func in change_color_funcs:
                                                    if correct_only_preprocess_flag:
                                                        break
                                                    shaped_train10 = change_color_func(copy.deepcopy(shaped_train9))
                                                    second_shape_different_flag = False
                                                    shaped_train_copy = shaped_train10
                                                    func_combi = [func for func in [back_to_black_func, size_change_func, patch_func, flip_func, transpose_func, train4_func, fill_closed_area_func, train6_funcs, move_object_func, recolor_func, kneighbor_func, change_color_func] if func != inouts_array]
                                                    func_combi += [inouts_array] if len(func_combi) == 0 else []
                                                    for train_num, in_out in enumerate(copy.deepcopy(shaped_train_copy)):
                                                        if in_out['input'].shape != np.array(in_out['output']).shape:
                                                            second_shape_different_flag = True
                                                            break
                                                        # st()
                                                        if in_out['input'].tolist() == in_out['output']:
            #                                                 print(func_combi)
                                                            correct_only_preprocess_flag = True
            #                                                 st()
                                                            for idx_minus_num in [1, 2]:
                                                                another_in_out = shaped_train_copy[train_num - idx_minus_num]
                                                                if another_in_out['input'].tolist() != another_in_out['output']:
                                                                    correct_only_preprocess_flag = False
                                                            if correct_only_preprocess_flag:
                                                                func_combi_map[idx].append(func_combi)
                                                                preprocess_best_score = 0.999
                                                    if second_shape_different_flag or correct_only_preprocess_flag:
                                                        continue
                #                                     st()
                                                    similarity = get_similarity(shaped_train_copy, [], idx)
                    #                                 print(func_combi)
                    #                                 print(similarity)
                                                    if similarity > preprocess_best_score:
                                                        func_combi_map[idx].append(func_combi)
                                                        preprocess_best_score = similarity
                                                        best_func_combi = func_combi
                                                        preprocess_best_score_map[idx] = preprocess_best_score

    if use_several_func:
        result[f'{idx}_0'] = ''
        several_f = True
        for out in outputs:
            result[f'{idx}_0'] += flattener(np.array(out).tolist()) + ' '

        success_list.append(task_n)
    elif correct_only_preprocess_flag:
        # TODO: 一回目はこれでやるとして、2回目以降を考える
        print('↓correct_only_preprocess!↓')
        print(f'idx: {idx}, func: {func_combi_map[idx]}')
        success_list.append(task_n)

        preds0, preds1, preds2 = [], [], []
        if divide_block_and_merge1 in func_combi_map[idx][0]:
            funcs0 = [divide_block_and_merge1]
            funcs1 = [divide_block_and_merge2]
            funcs2 = [divide_block_and_merge3]
        else:
            funcs0 = func_combi_map[idx][-1 % len(func_combi_map[idx])]
            funcs1 = func_combi_map[idx][-2 % len(func_combi_map[idx])]
            funcs2 = func_combi_map[idx][-3 % len(func_combi_map[idx])]
#             task_test = copy.deepcopy(task['test'])
#             for f in funcs0:
#                 task_test = f(task_test)
#             st()
        success = False
        final_score_map[idx] = 0
        for i, _ in enumerate(task['test']):
            result[f'{idx}_{i}'] = ''
        for funcs in [funcs0, funcs1, funcs2]:
            task_test = copy.deepcopy(task['test'])
            for func in funcs:
                task_test = func(task_test)
            for i, sample in enumerate(task_test):
                if 'output' in sample:
                    if sample['input'].tolist() == sample['output']:
                        preprocess_best_score_map[idx] = 1.0
                        final_score_map[idx] = 1.0
                pred = flattener(sample['input'].tolist())
                result[f'{idx}_{i}'] += pred + ' '

    elif (len(func_combi_map[idx]) > 0) or (input_output_shape_is_same(task)):
        task_train = copy.deepcopy(task['train'])
        task_test = copy.deepcopy(task['test'])
        if len(func_combi_map[idx]) == 0:
            func_combi_map[idx].append([inouts_array])
        for func in func_combi_map[idx][-1]:
            task_train = func(task_train)
            task_test = func(task_test)

        task_train2 = copy.deepcopy(task['train'])
        task_test2 = copy.deepcopy(task['test'])
        funcs2 = func_combi_map[idx][-2 % len(func_combi_map[idx])]
        for func in funcs2:
            task_train2 = func(task_train2)
            task_test2 = func(task_test2)
        task_train_aug = copy.deepcopy(task_train)
        print(f'preprocess_best_score: {preprocess_best_score}, funcs: {func_combi_map[idx]}')
        if preprocess_best_score > 0.99:
            promising_map[idx] = True
        if preprocess_best_score > 0.7:
            if 'output' in task_test[0]:
                pre_preds = final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx=idx, success_map={}, final_score_map=pre_final_score_map, origin_task=task)
            use_transpose_flag = apply_transpose_aug(task_train)
            color_inouts, color_aug_func = apply_color_aug(task_train, preprocess_best_score, best_aug_score_map, idx, promising_map)
            print(f'color_aug_func: {color_aug_func}')
            mirror_aug_funcs = apply_mirror_aug(task_train, preprocess_best_score, idx, use_transpose_flag, promising_map)
            print(f'mirror_aug_funcs: {mirror_aug_funcs}')
            # こちらも一応試したい
            # mirror_augs = [flipud_aug, fliplr_aug, flip_aug, transpose_aug]
            task_train_aug = task_train + color_inouts
            for mirror_aug_func in mirror_aug_funcs:
                mirror_inouts = mirror_aug_func(task_train)
                task_train_aug += mirror_inouts
#             st()
        print(f'final_train_length： {len(task_train_aug)}')
        preds = final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx=idx, success_map={}, final_score_map=final_score_map, final=True, promising=promising_map[idx], origin_task=task)
        for i, pred in enumerate(preds):
            result[f'{idx}_{i}'] = pred

    else:
        task_test = copy.deepcopy(task['test'])
        inputs = [el['input'] for el in task_test]
        for i, inp in enumerate(inputs):
            result[f'{idx}_{i}'] = getDefaultPred(inp)
    t = time() - start
    print(f'{round(t)}秒')
    time_map[idx] = t
    if (task_n in success_list):
        print(f'-------------------------------------------------------------------------------------success!! idx: {idx}')
    if env == 'production':
        os.system(f'echo {idx}: best_score_with_preprocess: {preprocess_best_score_map.get(idx, "different shapes...")}')
    return result, func_combi_map, success_list, preprocess_best_score_map, final_score_map, best_aug_score_map, pre_final_score_map, time_map, several_f

ca_skip = True
move_skip = True
patch_skip = True
update_tasks = []

private = '00576224' not in data.keys()
if private:
    for idx, task in data.items():
        preds, func_combi_map, success_list, preprocess_best_score_map, final_score_map, best_aug_score_map, pre_final_score_map, time_map, several_f = main(idx, task, 'production')
        
        # break
        if len(success_list) != 0:
            update_tasks.append(i)
            for idx, pred_str in preds.items():
                submission.loc[submission.index == idx, 'output'] = pred_str

# print("final submission", submission)

# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:




