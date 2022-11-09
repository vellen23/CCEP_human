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

root = Tk()
root.withdraw()

# colors for BZD condition
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'BZD']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


def get_nnmf_Epi(X, rank, it=2000):
    # remove rows that are completly equal zero
    # model = NMF(n_components=rank, init='random', random_state=50, max_iter=it)
    # W = model.fit_transform(X)
    # H = model.components_
    W = np.zeros((X.shape[0], rank))
    X0 = np.delete(X, np.where(np.mean(X, 1) == 0)[0], 0)
    # run 5 it with mult
    model = NMF(n_components=rank, init='nndsvda', max_iter=10)
    W0 = model.fit_transform(X0)
    H0 = model.components_
    # run again with best solution of first model
    model = NMF(n_components=rank, init='custom', max_iter=it, solver='mu')
    W0 = model.fit_transform(X0, W=W0, H=H0)
    H = model.components_
    W[np.where(np.mean(X, 1) > 0)[0], :] = W0

    return W, W0, H


def get_nnmf(X, rank, it=2000):
    # remove rows that are completly equal zero
    # model = NMF(n_components=rank, init='random', random_state=50, max_iter=it)
    # W = model.fit_transform(X)
    # H = model.components_
    W = np.zeros((X.shape[0], rank))
    X0 = np.delete(X, np.where(np.mean(X, 1) == 0)[0], 0)

    model = NMF(n_components=rank, init='nndsvd', max_iter=it)
    W0 = model.fit_transform(X0)
    H = model.components_
    W[np.where(np.mean(X, 1) > 0)[0], :] = W0

    return W, H


def get_nnmf_forced(m, rank, H0, W0, it=500):
    W = np.zeros((m.shape[0], rank))
    X0 = np.delete(m, np.where(np.mean(m, 1) == 0)[0], 0)
    W0 = np.delete(W0, np.where(np.mean(m, 1) == 0)[0], 0)
    model = NMF(n_components=rank, init='custom', max_iter=it)
    W0 = model.fit_transform(X0, H=H0, W=W0)
    H = model.components_
    W[np.where(np.mean(m, 1) > 0)[0], :] = W0
    return W, H


def get_BF_corr(Wa, Wb):
    """
    Construct n by k matrix of Pearson product-moment correlation
    coefficients for every combination of two columns in A and B
    :param: Wa, Wb : two basic functions matrix (n, rank) to compare

    Return: numpy array of dimensions k by k, where array[a][b] is the
    correlation between column 'a' of X and column 'b'
    Return Pearson product-moment correlation coefficients.
    """
    rank = Wa.shape[1]
    corrmatrix = []
    for a in range(rank):
        for b in range(rank):
            c = np.corrcoef(Wa[:, a], Wb[:, b])
            corrmatrix.append(c[0][1])
    return np.asarray(corrmatrix).reshape(rank, rank)


def max_corr(corr):
    aCORR = abs(corr)
    corr_max = np.zeros((len(aCORR), 3))
    for i in range(len(aCORR)):
        mx = np.max(aCORR)
        r, c = np.where(aCORR == np.max(aCORR))
        aCORR[r[0], :] = 0
        aCORR[:, c[0]] = 0
        corr_max[i, :] = np.array([mx, r[0], c[0]], dtype=object)
    return np.mean(corr_max[:, 0])


def amariMaxError(correlation):
    """
    Computes what Wu et al. (2016) described as a 'amari-type error'
    based on average distance between factorization solutions
    Return:
    Amari distance distM
    Arguments:
    :param: correlation: k by k matrix of pearson correlations
    Usage: Called by instability()
    """
    maxCol = np.absolute(correlation).max(0)
    colTemp = np.mean((1 - maxCol))
    maxRow = np.absolute(correlation).max(1)
    rowTemp = np.mean((1 - maxRow))
    distM = (rowTemp + colTemp) / 2

    return distM


def plot_stability(stability, instability, k0, k1, title, nmf_fig_path):
    # title = subj+' -- IO Benzo -- Stability NNMF, iterations: '+str(num_it)
    # Create some mock data
    ranks = np.arange(k0, k1 + 1)
    data1 = stability / stability.max()
    data2 = instability / instability.max()
    fig, ax1 = plt.subplots(figsize=(len(ranks), 6))
    plt.title(title)
    color = 'tab:red'
    ax1.set_xlabel('rank')
    ax1.set_ylabel('stability', color=color)
    ax1.plot(ranks, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('normalized instability', color=color)  # we already handled the x-label with ax1
    ax2.plot(ranks, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(nmf_fig_path + 'NNMF_stab.jpg')
    plt.show()


def get_stability(M_input, num_it=20, k0=2, k1=10):
    d = M_input.shape[0]  # e.g. number of labels
    stability = np.zeros((k1 - k0 + 1,))
    instability = np.zeros((k1 - k0 + 1,))
    k_num = 0
    for k in range(k0, k1 + 1):  # for each rank
        # for each rank value
        W_all = np.zeros((num_it, d, k))
        for n in range(num_it):
            W, H = get_nnmf(M_input, k)
            W_all[n, :, :] = W

        distMat = np.zeros(shape=(num_it, num_it))
        simMat = np.zeros(shape=(num_it, num_it))
        for i in range(num_it):
            for j in range(i, num_it):
                x = W_all[i]
                y = W_all[j]
                CORR = get_BF_corr(x, y)
                if i == j:
                    simMat[i][j] = 0  # 1
                    simMat[j][i] = distMat[i][j]
                    distMat[i][j] = amariMaxError(CORR)
                    distMat[j][i] = distMat[i][j]
                else:

                    simMat[i][j] = max_corr(CORR)  # amariMaxError(CORR)
                    simMat[j][i] = 0  # distMat[i][j]
                    distMat[i][j] = amariMaxError(CORR)
                    distMat[j][i] = distMat[i][j]

        stability[k_num] = (np.sum(simMat) / (num_it * (num_it - 1) / 2))
        instability[k_num] = (np.sum(distMat) / (num_it * (num_it - 1)))
        k_num = k_num + 1

    return stability, instability


# protocol specific
def get_NMF_Stim_association(data, H_all):
    # cond_sel either block or Ph_conditionn or Sleep
    # H_all = data.columns[9:] #todo: find better way
    NNMF_ass = np.zeros((1, 4))
    Int_all = np.unique(data.Int)
    Stims = np.unique(data.Stim)
    s = 0
    for sc in Stims:
        con_nmf_test = data[data.Stim == sc]
        if len(Stims)>1:
            shortcut = 0
            con_nmf_surr = data[data.Stim != sc]
            p_thr = 95
        else:
            shortcut = 1
            con_nmf_surr = data[data.Stim == sc]
            p_thr = 90
        h = 0
        for Hs in H_all:
            con_nmf_test_sum = con_nmf_test.groupby(['Stim', 'Int'])[Hs].mean()
            auc_test = np.mean(con_nmf_test_sum.values[-3:])/np.mean(con_nmf_test_sum.values[:3])
            if shortcut:
                if auc_test>1.2:
                    auc = np.zeros((1, 4))
                    auc[0, :] = [sc, auc_test, int(Hs[1:]), 1.2]  # , h
                    NNMF_ass = np.concatenate([NNMF_ass, auc], axis=0)
            else:
                #auc_test = np.trapz(con_nmf_test_sum.values, np.unique(con_nmf_test.Int))
                surr = np.zeros((100,))
                for i in range(len(surr)):
                    np.random.shuffle(con_nmf_surr[Hs].values)
                    # np.random.shuffle(con_nmf_surr['Int'])
                    con_nmf_test_sum = con_nmf_surr.groupby(['Stim', 'Int'])[Hs].mean()
                    # auc_surr = np.trapz(con_nmf_test_sum.values, np.unique(con_nmf_test.Int))
                    surr[i] = np.mean(con_nmf_test_sum.values[-3:])/np.mean(con_nmf_test_sum.values[:3])
                if auc_test > np.percentile(surr, p_thr):
                    auc = np.zeros((1, 4))
                    auc[0, :] = [sc, auc_test, int(Hs[1:]), np.percentile(surr, p_thr)]  # , h
                    NNMF_ass = np.concatenate([NNMF_ass, auc], axis=0)

    NNMF_ass = NNMF_ass[1:, :]
    NNMF_ass = pd.DataFrame(NNMF_ass, columns=['Stim', 'AUC', 'H_num', 'threshold'])  # , 'Hour'
    NNMF_ass.insert(2, 'H', 'H')
    for Hn in np.unique(NNMF_ass.H_num):
        NNMF_ass.loc[NNMF_ass.H_num == Hn, 'H'] = 'H' + str(int(Hn))
    NNMF_ass = NNMF_ass.reset_index(drop=True)

    return NNMF_ass

def get_NMF_Stim_association_PP(data, H_all):
    # cond_sel either block or Ph_conditionn or Sleep
    # H_all = data.columns[9:] #todo: find better way
    NNMF_ass = np.zeros((1, 4))
    IPI_all = np.unique(data.IPI)
    Stims = np.unique(data.Stim)
    s = 0
    for sc in Stims:
        con_nmf_test = data[((data.Int == 0)|(data.Int == np.max(data.Int)))&(data.Stim == sc)]
        if len(Stims)>1:
            shortcut = 0
            con_nmf_surr = data[data.Stim != sc]
            p_thr = 95
        else:
            shortcut = 1
            con_nmf_surr = data[data.Stim == sc]
            p_thr = 90
        h = 0
        for Hs in H_all:
            con_nmf_test_sum = con_nmf_test.groupby(['Stim','IPI'])[Hs].mean()
            z = (con_nmf_test_sum.values - np.mean(con_nmf_test_sum.values)) / np.std(con_nmf_test_sum.values)

            auc_test = np.max(z)-np.min(z)
            if shortcut:
                if (np.min(z)<-1.2) & (np.max(z)>1.2):
                    auc = np.zeros((1, 4))
                    auc[0, :] = [sc, auc_test, int(Hs[1:]), 1.2]  # , h
                    NNMF_ass = np.concatenate([NNMF_ass, auc], axis=0)
            else:
                #auc_test = np.trapz(con_nmf_test_sum.values, np.unique(con_nmf_test.Int))
                surr = np.zeros((100,))
                for i in range(len(surr)):
                    np.random.shuffle(con_nmf_surr[Hs].values)
                    # np.random.shuffle(con_nmf_surr['Int'])
                    con_nmf_test_sum = con_nmf_surr.groupby(['Stim', 'Int'])[Hs].mean()
                    # auc_surr = np.trapz(con_nmf_test_sum.values, np.unique(con_nmf_test.Int))
                    surr[i] = np.mean(con_nmf_test_sum.values[-3:])/np.mean(con_nmf_test_sum.values[:3])
                if auc_test > np.percentile(surr, p_thr):
                    auc = np.zeros((1, 4))
                    auc[0, :] = [sc, auc_test, int(Hs[1:]), np.percentile(surr, p_thr)]  # , h
                    NNMF_ass = np.concatenate([NNMF_ass, auc], axis=0)

    NNMF_ass = NNMF_ass[1:, :]
    NNMF_ass = pd.DataFrame(NNMF_ass, columns=['Stim', 'AUC', 'H_num', 'threshold'])  # , 'Hour'
    NNMF_ass.insert(2, 'H', 'H')
    for Hn in np.unique(NNMF_ass.H_num):
        NNMF_ass.loc[NNMF_ass.H_num == Hn, 'H'] = 'H' + str(int(Hn))
    NNMF_ass = NNMF_ass.reset_index(drop=True)

    return NNMF_ass


# protocol specific

def get_NMF_AUC_Stim(data, sc, cond_sel='Condition'):
    # only one stim channel
    NNMF_AUC = np.zeros((1, 8))
    Int_all = np.unique(data.Int)

    for Hs in np.unique(data.H):
        j = data.loc[(data.H == Hs), 'H_num'].values[0]
        pc = 1
        if (cond_sel == 'Sleep') | (cond_sel == 'SleepState'):
            # todo: move to mean
            val_min = np.min(dat.groupby([cond_sel, 'Int'])[Hs].median())
            val_max = np.max(dat.groupby([cond_sel, 'Int'])[Hs].median())
            AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)

            for cond in np.unique(dat[cond_sel]):
                dat_c = data[(data[cond_sel] == cond)]
                # todo: change to mean
                H_mean = dat_c.groupby('Int')[Hs].median().values
                ##  AUC
                AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
                NNMF_AUC = np.concatenate([NNMF_AUC, [[sc, j, 0, 0, cond, AUC, pc, len(dat_c)]]], axis=0)
        else:
            val_min = np.min(dat.groupby(['Date', cond_sel, 'Int'])[Hs].mean())
            val_max = np.max(dat.groupby(['Date', cond_sel, 'Int'])[Hs].mean())
            AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)
            for d in range(len(np.unique(dat.Date))):
                dat_D = dat[dat.Date == np.unique(dat.Date)[d]]
                for cond in np.unique(dat_D[cond_sel]):
                    dat_c = data[
                        (data.Date == np.unique(dat.Date)[d])  & (data[cond_sel] == cond)]
                    # most common hour value
                    # todo: add Hour
                    h = np.bincount(dat_c.Hour).argmax()  # np.median(dat_h.Hour)
                    # mean H coefficient for each intensity
                    H_mean = dat_c.groupby('Int')[Hs].mean().values
                    ## AUC

                    AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
                    auc = np.zeros((1, 8))
                    auc[0, 0:8] = [sc, j, d, h, cond, AUC, pc, len(dat_c)]  # , h
                    NNMF_AUC = np.concatenate([NNMF_AUC, auc], axis=0)
        j = j + 1
    NNMF_AUC = NNMF_AUC[1:, :]
    NNMF_AUC = pd.DataFrame(NNMF_AUC,
                            columns=['Stim', 'H', 'Day', 'Hour', cond_sel, 'AUC', 'Pearson', 'N_trial'])  # , 'Hour'
    for col in ['Stim', 'H', 'Day', 'Hour', 'AUC', 'Pearson', 'N_trial']:
        NNMF_AUC[col] = NNMF_AUC[col].astype('float')
    # NNMF_AUC.insert(4,'nAUC', 0)
    if (cond_sel == 'Sleep') | (cond_sel == 'SleepState'):
        NNMF_AUC = NNMF_AUC.drop(columns=['Day', 'Hour'])
    else:
        NNMF_AUC.sort_values(by=['Day', cond_sel])
        NNMF_AUC[cond_sel] = NNMF_AUC[cond_sel].astype('float')
    if cond_sel == 'Sleep':
        NNMF_AUC[cond_sel] = NNMF_AUC[cond_sel].astype('float')
    NNMF_AUC = NNMF_AUC.reset_index(drop=True)

    return NNMF_AUC

def get_NMF_AUC(data, NNMF_ass, cond_sel='Condition'):
    NNMF_AUC = np.zeros((1, 8))
    Int_all = np.unique(data.Int)
    Stims = np.unique(NNMF_ass.Stim)
    for sc in Stims:
        dat = data[data.Stim == sc]
        for Hs in np.unique(NNMF_ass.loc[NNMF_ass.Stim == sc, 'H']):
            j = NNMF_ass.loc[(NNMF_ass.H == Hs) & (NNMF_ass.Stim == sc), 'H_num'].values[0]
            pc = 1
            if (cond_sel == 'Sleep') | (cond_sel == 'SleepState'):
                # todo: move to mean
                val_min = np.min(dat.groupby([cond_sel, 'Int'])[Hs].median())
                val_max = np.max(dat.groupby([cond_sel, 'Int'])[Hs].median())
                AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)

                for cond in np.unique(dat[cond_sel]):
                    dat_c = data[(data.Stim == sc) & (data[cond_sel] == cond)]
                    # todo: change to mean
                    H_mean = dat_c.groupby('Int')[Hs].median().values
                    ##  AUC
                    AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
                    NNMF_AUC = np.concatenate([NNMF_AUC, [[sc, j, 0, 0, cond, AUC, pc, len(dat_c)]]], axis=0)
            else:
                val_min = np.min(dat.groupby(['Date', cond_sel, 'Int'])[Hs].mean())
                val_max = np.max(dat.groupby(['Date', cond_sel, 'Int'])[Hs].mean())
                AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)
                for d in range(len(np.unique(dat.Date))):
                    dat_D = dat[dat.Date == np.unique(dat.Date)[d]]
                    for cond in np.unique(dat_D[cond_sel]):
                        dat_c = data[
                            (data.Date == np.unique(dat.Date)[d]) & (data.Stim == sc) & (data[cond_sel] == cond)]
                        # most common hour value
                        # todo: add Hour
                        h = np.bincount(dat_c.Hour).argmax()  # np.median(dat_h.Hour)
                        # mean H coefficient for each intensity
                        H_mean = dat_c.groupby('Int')[Hs].mean().values
                        ## AUC

                        AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
                        auc = np.zeros((1, 8))
                        auc[0, 0:8] = [sc, j, d, h, cond, AUC, pc, len(dat_c)]  # , h
                        NNMF_AUC = np.concatenate([NNMF_AUC, auc], axis=0)
            j = j + 1
    NNMF_AUC = NNMF_AUC[1:, :]
    NNMF_AUC = pd.DataFrame(NNMF_AUC,
                            columns=['Stim', 'H', 'Day', 'Hour', cond_sel, 'AUC', 'Pearson', 'N_trial'])  # , 'Hour'
    for col in ['Stim', 'H', 'Day', 'Hour', 'AUC', 'Pearson', 'N_trial']:
        NNMF_AUC[col] = NNMF_AUC[col].astype('float')
    # NNMF_AUC.insert(4,'nAUC', 0)
    if (cond_sel == 'Sleep') | (cond_sel == 'SleepState'):
        NNMF_AUC = NNMF_AUC.drop(columns=['Day', 'Hour'])
    else:
        NNMF_AUC.sort_values(by=['Day', cond_sel])
        NNMF_AUC[cond_sel] = NNMF_AUC[cond_sel].astype('float')
    if cond_sel == 'Sleep':
        NNMF_AUC[cond_sel] = NNMF_AUC[cond_sel].astype('float')
    NNMF_AUC = NNMF_AUC.reset_index(drop=True)

    return NNMF_AUC


## Plotting functions
def plot_V(M_input, title, ylabels=[0], file=0):
    # plot input matrix
    # ylabels most likely channel labels

    aspect = M_input.shape[1] / 20 * 8 / M_input.shape[0]

    fig = plt.figure(figsize=(20, 8))
    plt.imshow(M_input, aspect=aspect, vmin=np.percentile(M_input, 20),
               vmax=np.percentile(M_input, 95))  # , vmin=0, vmax=15
    plt.ylabel('Channels')
    if ylabels[0] != 0:
        plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.xlabel('trials')
    # plt.title(subj + ' -- NMF input matrix: LL ')
    plt.title(title)
    # file = nmf_fig_path + 'NMF_input_IO_LLpeak'
    plt.colorbar()
    if type(file) == str:
        plt.savefig(file + '.jpg')
        plt.savefig(file + '.svg')
        plt.close(fig)
    else:
        plt.show()


def plot_W(W, title, ylabels=[0], file=0):
    # plot basic functions
    aspect = W.shape[1] / 5 * 8 / W.shape[0]
    fig = plt.figure(figsize=(5, 8))
    plt.title(title, fontsize=15)
    plt.imshow(W, aspect=aspect, vmin=np.percentile(W, 20), vmax=np.percentile(W, 95), cmap='hot')  # , vmin=0, vmax=15
    plt.ylabel('Channels', fontsize=12)
    plt.xlabel('Ranks', fontsize=12)

    H_col = []
    for i in range(W.shape[1]):
        H_col.append('W' + str(i + 1))
    plt.xticks(np.arange(W.shape[1]), H_col, fontsize=12)

    if ylabels[0] != 0:
        plt.yticks(np.arange(len(W)), ylabels)

    if type(file) == str:
        plt.savefig(file + '.jpg')
        plt.savefig(file + '.svg')
        plt.close(fig)
    else:
        plt.show()


def plot_H(H, title, file=0):
    aspect = H.shape[1] / 20 * 5 / H.shape[0]
    # plot activation functions
    fig = plt.figure(figsize=(20, 5))
    plt.title(title, fontsize=15)
    plt.imshow(H, aspect=aspect, vmin=np.percentile(H, 20), vmax=np.percentile(H, 95), cmap='hot')  # , vmin=0, vmax=15
    plt.ylabel('Activation Function (H)', fontsize=12)
    plt.xlabel('Trials', fontsize=12)
    W_col = []
    for i in range(H.shape[0]):
        W_col.append('H' + str(i + 1))
    plt.yticks(np.arange(len(H)), W_col, fontsize=12)
    # todo change 0 to 1
    if type(file) == str:
        plt.savefig(file + '.jpg')
        plt.savefig(file + '.svg')
        plt.close(fig)
    else:
        plt.show()
    # plt.show()


def plot_H_trial_IPI(data, xl, hl, sl, title, nmf_fig_path):
    data = data.drop(columns='Hour')
    H_all = [i for i in data.columns if i.startswith('H')]

    fig = plt.figure(figsize=(len(H_all) * 5, 7))
    plt.suptitle(title)
    gs = fig.add_gridspec(1, len(H_all))  # GridSpec(4,1, height_ratios=[1,2,1,2])
    i = 0
    if hl == 'Condition':
        col_sel = [cond_colors[1], cond_colors[3]]
    else:
        col_sel = 'colorblind'
    for Hs in H_all:
        fig.add_subplot(gs[0, i])
        sns.scatterplot(x=xl, y=Hs, hue=hl, style=sl, data=data, palette=col_sel)
        i = i + 1
    file = nmf_fig_path + 'H_' + hl + '_r' + str(len(H_all))
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    plt.close(fig)


def plot_H_trial(data, xl, hl, title, nmf_fig_path):
    if 'Hour' in data:
        data = data.drop(columns='Hour')
    H_all = [i for i in data.columns if i.startswith('H')]
    fac = 5
    if xl =='IPI':
        fac = 8
    fig = plt.figure(figsize=(len(H_all) * fac, 7))
    plt.suptitle(title)
    gs = fig.add_gridspec(1, len(H_all))  # GridSpec(4,1, height_ratios=[1,2,1,2])
    i = 0
    if hl == 'Condition':
        col_sel = [cond_colors[1], cond_colors[3]]
    else:
        col_sel = 'colorblind'
    for Hs in H_all:
        fig.add_subplot(gs[0, i])
        #
        if xl =='IPI':
            sns.swarmplot(x=xl, y=Hs, hue=hl, data=data, palette=col_sel)
        else:
            sns.scatterplot(x=xl, y=Hs, hue=hl, data=data, palette=col_sel)
        i = i + 1

    file = nmf_fig_path + 'H_' + hl + '_r' + str(len(H_all))
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    plt.close(fig)

def plot_H_IPI_cond(data, hl,cond, nmf_fig_path):
    # cond: SleepState, Wake, NREM, REM
    if 'Hour' in data:
        data = data.drop(columns='Hour')
    H_all = [i for i in data.columns if i.startswith('H')]
    sns.catplot(x='IPI', y=hl, hue=cond, data=data, aspect=4, row='Int', palette =['black', 'blue', 'red'])
    plt.ylim([0,10])
    file = nmf_fig_path + 'H_' + hl + '_r' + str(len(H_all))+'_'+cond
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    # plt.close(fig)

def plot_NMF_AUC_SleepState(data, sc, h, title, file):
    cond_labels = ['Wake', 'NREM', 'REM']
    color = ['black', '#1F4E7A', '#f65858']
    # title = subj + ' --- '+labels_all[sc]+', '+str(Hs)
    # remove N1
    data = data[(data.Sleep < 5) & (data.Sleep != 1) & (data.Stim == sc)]  # & ((data.Hour < 9) | (data.Hour > 20))
    Hs = 'H' + str(h)
    Int_all = np.unique(data.Int)
    fig = plt.figure(figsize=(15,15))

    val_min = np.min(data.groupby(['SleepState', 'Int'])[Hs].mean())
    val_max = np.max(data.groupby(['SleepState', 'Int'])[Hs].mean())
    AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)
    for con_val, c_ix in zip(cond_labels, np.arange(3)):  # snp.unique(data.Sleep).astype('int'):
        dat_c = data[(data.SleepState == con_val)]
        plt.title(title, fontsize=30)
        H_mean = dat_c.groupby('Int')[Hs].mean().values
        # sns.scatterplot(x='Int', y= Hs, data=dat_c)
        AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
        plt.plot(np.unique(dat_c.Int), H_mean,
                 label=con_val + '- AUC: ' + str(np.round(AUC, 2)), color = color[c_ix], linewidth=5)
        ## AUC

        plt.fill_between(np.unique(dat_c.Int), val_min, H_mean, color = color[c_ix], alpha=0.1)
        # plt.text(8, (val_max-con_val/2)/3 ,cond_labels[con_val]+ '- AUC: '+ str(np.round(AUC,2)), fontsize=12)
    plt.axhline(val_min, color=[0,0,0])
    plt.axhline(val_max, color=[0,0,0])
    plt.plot([0, np.max(Int_all)], [val_min, val_max], '--', c=[0, 0, 0], alpha=0.5)
    plt.text(2, 1.01 * val_max, 'max "1"')
    plt.text(2, 0.9 * val_min, 'min "0"')
    plt.ylim([0, 1.1 * val_max])
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.ylabel('H coefficient', fontsize=30)
    plt.xlabel('Intensity [mA]', fontsize=30)
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    plt.close(fig)

def plot_NMF_AUC_Sleep(data, sc, h, title, file):
    cond_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # title = subj + ' --- '+labels_all[sc]+', '+str(Hs)
    # remove N1
    data = data[(data.Sleep < 5) & (data.Sleep != 1) & (data.Stim == sc)]  # & ((data.Hour < 9) | (data.Hour > 20))
    Hs = 'H' + str(h)
    Int_all = np.unique(data.Int)
    fig = plt.figure(figsize=(7, 7))

    val_min = np.min(data.groupby(['Sleep', 'Int'])[Hs].mean())
    val_max = np.max(data.groupby(['Sleep', 'Int'])[Hs].mean())
    AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)
    for con_val in np.unique(data.Sleep).astype('int'):  # snp.unique(data.Sleep).astype('int'):
        dat_c = data[(data.Sleep == con_val)]
        plt.title(title)
        H_mean = dat_c.groupby('Int')[Hs].mean().values
        # sns.scatterplot(x='Int', y= Hs, data=dat_c)
        AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
        plt.plot(np.unique(dat_c.Int), H_mean,
                 label=cond_labels[con_val] + '- AUC: ' + str(np.round(AUC, 2)))
        ## AUC

        plt.fill_between(np.unique(dat_c.Int), val_min, H_mean, color=[0, 0, 0], alpha=0.1)
        # plt.text(8, (val_max-con_val/2)/3 ,cond_labels[con_val]+ '- AUC: '+ str(np.round(AUC,2)), fontsize=12)
    plt.axhline(val_min)
    plt.axhline(val_max)
    plt.plot([0, np.max(Int_all)], [val_min, val_max], '--', c=[0, 0, 0], alpha=0.5)
    plt.text(2, 1.01 * val_max, 'max "1"')
    plt.text(2, 0.9 * val_min, 'min "0"')
    plt.ylim([0, 1.1 * val_max])
    plt.legend()
    plt.ylabel('H coefficient')
    plt.xlabel('Intensity [mA]')
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    plt.close(fig)


def plot_NMF_AUC_Ph(data, sc, h, title, file):
    # title = subj + ' --- '+labels_all[sc]+', '+str(Hs)
    data = data[(data.Stim == sc)]
    Hs = 'H' + str(h)
    Int_all = np.unique(data.Int)
    fig = plt.figure(figsize=(7, 7))

    val_min = np.min(data.groupby(['Date', 'Condition', 'Int'])[Hs].mean())
    val_max = np.max(data.groupby(['Date', 'Condition', 'Int'])[Hs].mean())
    AUC1 = np.trapz(np.repeat(val_max, len(Int_all)) - val_min, Int_all)
    for con_val in [1, 3]:
        dat_c = data[(data.Condition == con_val)]

        plt.title(title)
        H_mean = dat_c.groupby('Int')[Hs].mean().values
        # sns.scatterplot(x='Int', y= Hs, data=dat_c)
        AUC = np.trapz(H_mean - val_min, np.unique(dat_c.Int)) / AUC1
        plt.plot(np.unique(dat_c.Int), H_mean, color=cond_colors[con_val],
                 label=cond_labels[con_val] + '- AUC: ' + str(np.round(AUC, 2)))
        ## AUC

        plt.fill_between(np.unique(dat_c.Int), val_min, H_mean, color=cond_colors[con_val], alpha=0.1)
        # plt.text(8, (val_max-con_val/2)/3 ,cond_labels[con_val]+ '- AUC: '+ str(np.round(AUC,2)), fontsize=12)
    plt.axhline(val_min)
    plt.axhline(val_max)
    plt.plot([0, np.max(Int_all)], [val_min, val_max], '--', c=[0, 0, 0], alpha=0.5)
    plt.text(2, 1.01 * val_max, 'max "1"')
    plt.text(2, 0.9 * val_min, 'min "0"')
    plt.ylim([0, 1.1 * val_max])
    plt.legend()
    plt.ylabel('H coefficient')
    plt.xlabel('Intensity [mA]')
    plt.savefig(file + '.jpg')
    plt.savefig(file + '.svg')
    plt.close(fig)
