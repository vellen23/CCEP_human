import os
import numpy as np
import sys
import sklearn
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.decomposition import NMF
import pandas as pd
import random

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def CD_surr(dat, feature_rand = 'SleepState', feature_states = ['Wake', 'NREM'], value = 'LL', n=200, p = 1):
    surr_cd = np.zeros((n,))
    for i in range(n):
        np.random.shuffle(dat[feature_rand].values)
        surr_cd[i] = cohen_d(dat.loc[(dat[feature_rand] == feature_states[0]), value].values,dat.loc[(dat[feature_rand] == feature_states[1]), value].values)

    return [np.percentile(surr_cd,p),np.percentile(surr_cd,100-p)]

def R_surr(dat, feature_rand = 'SleepState', feature_states = ['Wake', 'NREM'], value = 'Prob', n=200, p = 1):
    surr_cd = np.zeros((n,))
    for i in range(n):
        np.random.shuffle(dat[feature_rand].values)
        surr_cd[i] = np.mean(dat.loc[(dat[feature_rand] == feature_states[1]),value])/np.mean(dat.loc[(dat[feature_rand] == feature_states[0]),value])

    return [np.percentile(surr_cd,p),np.percentile(surr_cd,100-p)]
