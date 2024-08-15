import os, gc
import sys, pdb
import copy, time
import json, random
import time

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore


import signal
import psutil
import itertools
import subprocess
import base64, gzip
import networkx as nx
import multiprocessing
#::::::::::::::::::::::::::::::::::::::::::::::
from numpy import array
from pathlib import Path
from scipy import ndimage
from scipy.stats import mode
from tqdm.auto import trange
from functools import partial
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
from xgboost import XGBClassifier
from itertools import combinations, product
from collections import defaultdict, Counter
from skimage.measure import label, regionprops
#::::::::::::::::::::::::::::::::::::::::::::::
from sklearn.tree import *
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from skimage.measure import label, regionprops
from sklearn.preprocessing import MinMaxScaler

from subprocess import Popen, PIPE, STDOUT
from glob import glob
#######################################################################################
# Adapt ARC Prize 2024 files to work with Abstraction and Resoning Corpus 2020 rules ##
#######################################################################################

# Load the JSON content
json_file_path = '../input/arc-prize-2024/arc-agi_test_challenges.json'  
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Create the 'test' directory
output_dir = '../working/abstraction-and-reasoning-challenge/test'  
os.makedirs(output_dir, exist_ok=True)

# Split the JSON content into individual files
for task_id, task_data in data.items():
    output_file_path = os.path.join(output_dir, f'{task_id}.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(task_data, output_file, indent=4)

############################################
# Beginning of icecuber's original solution#
##########################################

if open("../input/arc-solution-source-files-by-icecuber/version.txt").read().strip() == "671838222":
  print("Dataset has correct version")
else:
  print("Dataset version not matching!")
  assert(0)

def mySystem(cmd):
    print(cmd)
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    for line in iter(process.stdout.readline, b''):
        print(line.decode("utf-8"), end='')
    assert(process.wait() == 0)
    
dummy_run = False


for fn in glob("../working/abstraction-and-reasoning-challenge/test/*.json"):
  if "136b0064" in fn:
    print("Making dummy submission")
    f = open("old_submission.csv", "w")
    f.write("output_id,output\n")
    f.close()
    dummy_run = True


if not dummy_run:
  mySystem("cp -r ../input/arc-solution-source-files-by-icecuber ./absres-c-files")
  mySystem("cd absres-c-files; make -j")
  mySystem("cd absres-c-files; python3 safe_run.py")
  mySystem("cp absres-c-files/submission_part.csv old_submission.csv")
  mySystem("tar -czf store.tar.gz absres-c-files/store")
  mySystem("rm -r absres-c-files")

# Function to translate from old submission format (csv) to new one (json)
def translate_submission(file_path):
    # Read the original submission file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # Skip the header line
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # Split predictions based on ' '
        
        # Take only the first two predictions
        if len(predictions) > 2:
            predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # Check if pred is not an empty string
                pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_1 = processed_predictions[0] if len(processed_predictions) > 0 else []
        attempt_2 = processed_predictions[1] if len(processed_predictions) > 1 else []

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        attempt_dict = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)
        else:
            submission_dict[task_id].append(attempt_dict)
    
    # Write to the new json file
    with open('sub_icecube.json', 'w') as file:
        json.dump(submission_dict, file, indent=4)

translate_submission('/kaggle/working/old_submission.csv')
