import importlib
import sys
sys.path.append('../src')

import qa
importlib.reload(qa)
import qa

import plot
from plot import plot_from_file

import pickle
import numpy as np
import pandas as pd

import re
import random
import ast

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_rel
from scipy.stats import pearsonr

benchmarks = ["MMLU-Pro"]
factual = ["MMLU-Pro", "GPQA-Diamond", "ARC"]
opinion = ["OpinionQA", "GlobalOpinionQA", "SIQA"]

for benchmark in benchmarks:

    with open(f'../data/{benchmark}/sample_results/org.pkl', 'rb') as f:
        res_org = pickle.load(f)

    input_feat_list = []
    for disagree_size in range(1, 11): # row
        input_row = []
        for agree_size in range(1, 11): # column
            
            eval_feat = {
                'type': 'grp_count',
                'agree_size': agree_size,
                'disagree_size': disagree_size,
                'disagree_type': '2nd',
                'q_type': 'factual' if benchmark in factual else 'opinion',
                'order': 'random'
            }
            input_row.append(eval_feat)
        input_feat_list.append(input_row)

    results, accuracy = qa.qa_eval_matrix(res_org, input_feat_list)

    with open(f'../data/{benchmark}/sample_results/grp_count.pkl', 'wb') as f:
        pickle.dump(results, f)

