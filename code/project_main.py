# first you need to install pyod package using pip (pip install pyod)
# importing necessary libraries
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from sklearn.metrics import average_precision_score, roc_auc_score
from pyod.utils import precision_n_scores
import numpy as np
from scipy import io
from copy import deepcopy
from matplotlib import pyplot as plt
from tictoc import tic, toc
import pandas as pd

# importing input datasets
df_arrhythmia = io.loadmat('datasets/arrhythmia.mat')
df_glass = io.loadmat('datasets/glass.mat')
df_ionosphere = io.loadmat('datasets/ionosphere.mat')
df_lympho = io.loadmat('datasets/lympho.mat')
df_pima = io.loadmat('datasets/pima.mat')
df_vertebral = io.loadmat('datasets/vertebral.mat')
df_vowels = io.loadmat('datasets/vowels.mat')
df_wbc = io.loadmat('datasets/wbc.mat')
df_wine = io.loadmat('datasets/wine.mat')
data_frames = {'arrhythmia': df_arrhythmia, 'glass': df_glass, 'ionosphere': df_ionosphere, 'lympho': df_lympho,
               'pima': df_pima, 'vertebral': df_vertebral, 'vowels': df_vowels, 'wbc': df_wbc, 'wine': df_wine}


def lof_():
    n_neighbors = [20, 30, 40, 50, 60]
    neighb_vals = dict(zip(n_neighbors, [None] * len(n_neighbors)))
    res_vals = dict(zip(data_frames.keys(), [deepcopy(neighb_vals) for d in data_frames.keys()]))
    time_res = dict(zip(data_frames.keys(), [deepcopy(neighb_vals) for d in data_frames.keys()]))

    roc_auc_res, aver_prec_res, prec_at_n_res = [deepcopy(res_vals) for r in range(3)]

    for ngb in n_neighbors:
        # applying the LOF classifier on different datasets
        for df in data_frames.keys():
            clf = LOF(n_neighbors=ngb)
            tic()
            clf.fit(data_frames[df]['X'])
            time_res[df][ngb] = toc(show=False).total_seconds()
            roc_auc_res[df][ngb] = np.round(roc_auc_score(data_frames[df]['y'], clf.decision_scores_), decimals=4)
            aver_prec_res[df][ngb] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                              decimals=4)
            prec_at_n_res[df][ngb] = np.round(precision_n_scores(data_frames[df]['y'], clf.decision_scores_),
                                              decimals=4)

    # plotting accuracy
    fig_lof, ax_lof = plt.subplots(3, figsize=(6, 9), constrained_layout=True, sharex=True)
    fig_lof.suptitle('LOF accuracy scores', fontsize=16)
    x_vals = list(map(str, n_neighbors))
    markers = 'ospP*hv^<>H+xXDd'
    colors = np.random.rand(len(data_frames), 3)
    c = 0
    for df in data_frames.keys():
        ax_lof[0].plot(x_vals, roc_auc_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        ax_lof[1].plot(x_vals, aver_prec_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        ax_lof[2].plot(x_vals, prec_at_n_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        c += 1
    ax_lof[0].set(ylabel="Area Under ROC (AUROC)")
    ax_lof[0].legend()
    ax_lof[0].grid()
    ax_lof[1].set(ylabel="Average Precision (AP)")
    ax_lof[1].legend()
    ax_lof[1].grid()
    ax_lof[2].set(xlabel="no. of neighbors", ylabel="Precision @ Rank n")
    ax_lof[2].legend()
    ax_lof[2].grid()
    plt.show()

    # plotting execution time
    fig_time, ax_time = plt.subplots(1, figsize=(6, 4))
    fig_time.suptitle('LOF execution time', fontsize=16)
    c = 0
    for df in data_frames.keys():
        ax_time.plot(x_vals, time_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                     markersize=10, label=df)
        c += 1
    ax_time.set(xlabel="no. of neighbors", ylabel="execution time (sec)")
    ax_time.legend()
    ax_time.grid()
    plt.show()


def knn_():
    n_neighbors = np.arange(5, 26, 5)
    neighb_vals = dict(zip(n_neighbors, [None] * len(n_neighbors)))
    res_vals = dict(zip(data_frames.keys(), [deepcopy(neighb_vals) for d in data_frames.keys()]))
    time_res = dict(zip(data_frames.keys(), [deepcopy(neighb_vals) for d in data_frames.keys()]))

    ap_largest_res, ap_mean_res, ap_median_res = [deepcopy(res_vals) for r in range(3)]

    for ngb in n_neighbors:
        # applying the kNN classifier on different datasets
        for df in data_frames.keys():
            # applying kNN with the distance to the kth neighbor as the outlier score
            clf = KNN(n_neighbors=ngb, method='largest')
            tic()
            clf.fit(data_frames[df]['X'])
            time_res[df][ngb] = toc(show=False).total_seconds()
            ap_largest_res[df][ngb] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                               decimals=4)
            # applying kNN with the average of distances to all k neighbors as the outlier score
            clf = KNN(n_neighbors=ngb, method='mean')
            clf.fit(data_frames[df]['X'])
            ap_mean_res[df][ngb] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                            decimals=4)
            # applying kNN with the median of the distance to k neighbors as the outlier score
            clf = KNN(n_neighbors=ngb, method='median')
            clf.fit(data_frames[df]['X'])
            ap_median_res[df][ngb] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                              decimals=4)

    # plotting accuracy
    fig_knn, ax_knn = plt.subplots(3, figsize=(6, 9), constrained_layout=True, sharex=True)
    fig_knn.suptitle('Average Precision scores for different kNN detectors', fontsize=16)
    x_vals = list(map(str, n_neighbors))
    markers = 'ospP*hv^<>H+xXDd'
    colors = np.random.rand(len(data_frames), 3)
    c = 0
    for df in data_frames.keys():
        ax_knn[0].plot(x_vals, ap_largest_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        ax_knn[1].plot(x_vals, ap_mean_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        ax_knn[2].plot(x_vals, ap_median_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                       markersize=10, label=df)
        c += 1
    ax_knn[0].set(ylabel="largest distance")
    ax_knn[0].legend()
    ax_knn[0].grid()
    ax_knn[1].set(ylabel="distances mean")
    ax_knn[1].legend()
    ax_knn[1].grid()
    ax_knn[2].set(xlabel="no. of neighbors", ylabel="distances median")
    ax_knn[2].legend()
    ax_knn[2].grid()
    plt.show()

    # plotting execution time
    fig_time, ax_time = plt.subplots(1, figsize=(6, 4))
    fig_time.suptitle('kNN execution time', fontsize=16)
    c = 0
    for df in data_frames.keys():
        ax_time.plot(x_vals, time_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                     markersize=10, label=df)
        c += 1
    ax_time.set(xlabel="no. of neighbors", ylabel="execution time (sec)")
    ax_time.legend()
    ax_time.grid()
    plt.show()


def iforest_():
    n_estimats = np.arange(100, 1001, 100)
    estimt_vals = dict(zip(n_estimats, [None] * len(n_estimats)))
    res_vals = dict(zip(data_frames.keys(), [deepcopy(estimt_vals) for d in data_frames.keys()]))
    time_res = dict(zip(data_frames.keys(), [deepcopy(estimt_vals) for d in data_frames.keys()]))

    roc_auc_res, aver_prec_res, prec_at_n_res = [deepcopy(res_vals) for r in range(3)]

    for est in n_estimats:
        # applying the iForest classifier on different datasets
        for df in data_frames.keys():
            clf = IForest(n_estimators=est)
            tic()
            clf.fit(data_frames[df]['X'])
            time_res[df][est] = toc(show=False).total_seconds()
            roc_auc_res[df][est] = np.round(roc_auc_score(data_frames[df]['y'], clf.decision_scores_), decimals=4)
            aver_prec_res[df][est] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                              decimals=4)
            prec_at_n_res[df][est] = np.round(precision_n_scores(data_frames[df]['y'], clf.decision_scores_),
                                              decimals=4)

    # plotting accuracy
    fig_iforest, ax_iforest = plt.subplots(3, figsize=(6, 9), constrained_layout=True, sharex=True)
    fig_iforest.suptitle('iForest accuracy scores', fontsize=16)
    x_vals = list(map(str, n_estimats))
    markers = 'ospP*hv^<>H+xXDd'
    colors = np.random.rand(len(data_frames), 3)
    c = 0
    for df in data_frames.keys():
        ax_iforest[0].plot(x_vals, roc_auc_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                           markersize=10, label=df)
        ax_iforest[1].plot(x_vals, aver_prec_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                           markersize=10, label=df)
        ax_iforest[2].plot(x_vals, prec_at_n_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                           markersize=10, label=df)
        c += 1
    ax_iforest[0].set(ylabel="Area Under ROC (AUROC)")
    ax_iforest[0].legend()
    ax_iforest[0].grid()
    ax_iforest[1].set(ylabel="Average Precision (AP)")
    ax_iforest[1].legend()
    ax_iforest[1].grid()
    ax_iforest[2].set(xlabel="no. of estimators", ylabel="Precision @ Rank n")
    ax_iforest[2].legend()
    ax_iforest[2].grid()
    plt.show()

    # plotting execution time
    fig_time, ax_time = plt.subplots(1, figsize=(6, 4))
    fig_time.suptitle('iForest execution time', fontsize=16)
    c = 0
    for df in data_frames.keys():
        ax_time.plot(x_vals, time_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                     markersize=10, label=df)
        c += 1
    ax_time.set(xlabel="no. of estimators", ylabel="execution time (sec)")
    ax_time.legend()
    ax_time.grid()
    plt.show()


def hbos_():
    n_bins = np.arange(5, 26, 5)
    bin_vals = dict(zip(n_bins, [None] * len(n_bins)))
    res_vals = dict(zip(data_frames.keys(), [deepcopy(bin_vals) for d in data_frames.keys()]))
    time_res = dict(zip(data_frames.keys(), [deepcopy(bin_vals) for d in data_frames.keys()]))

    roc_auc_res, aver_prec_res, prec_at_n_res = [deepcopy(res_vals) for r in range(3)]

    for bin_no in n_bins:
        # applying the HBOS classifier on different datasets
        for df in data_frames.keys():
            clf = HBOS(n_bins=bin_no)
            tic()
            clf.fit(data_frames[df]['X'])
            time_res[df][bin_no] = toc(show=False).total_seconds()
            roc_auc_res[df][bin_no] = np.round(roc_auc_score(data_frames[df]['y'], clf.decision_scores_), decimals=4)
            aver_prec_res[df][bin_no] = np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_),
                                                 decimals=4)
            prec_at_n_res[df][bin_no] = np.round(precision_n_scores(data_frames[df]['y'], clf.decision_scores_),
                                                 decimals=4)

    # plotting accuracy
    fig_hbos, ax_hbos = plt.subplots(3, figsize=(6, 9), constrained_layout=True, sharex=True)
    fig_hbos.suptitle('HBOS accuracy scores', fontsize=16)
    x_vals = list(map(str, n_bins))
    markers = 'ospP*hv^<>H+xXDd'
    colors = np.random.rand(len(data_frames), 3)
    c = 0
    for df in data_frames.keys():
        ax_hbos[0].plot(x_vals, roc_auc_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                        markersize=10, label=df)
        ax_hbos[1].plot(x_vals, aver_prec_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                        markersize=10, label=df)
        ax_hbos[2].plot(x_vals, prec_at_n_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                        markersize=10, label=df)
        c += 1
    ax_hbos[0].set(ylabel="Area Under ROC (AUROC)")
    ax_hbos[0].legend()
    ax_hbos[0].grid()
    ax_hbos[1].set(ylabel="Average Precision (AP)")
    ax_hbos[1].legend()
    ax_hbos[1].grid()
    ax_hbos[2].set(xlabel="no. of bins", ylabel="Precision @ Rank n")
    ax_hbos[2].legend()
    ax_hbos[2].grid()
    plt.show()

    # plotting execution time
    fig_time, ax_time = plt.subplots(1, figsize=(6, 4))
    fig_time.suptitle('HBOS execution time', fontsize=16)
    c = 0
    for df in data_frames.keys():
        ax_time.plot(x_vals, time_res[df].values(), color=colors[c, :], marker=markers[c % len(markers)],
                     markersize=10, label=df)
        c += 1
    ax_time.set(xlabel="no. of bins", ylabel="execution time (sec)")
    ax_time.legend()
    ax_time.grid()
    plt.show()


def compare_methods():
    methods = ['LOF', 'kNN', 'iForest', 'HBOS']
    data_name = data_frames.keys()

    roc_auc_res, aver_prec_res, prec_at_n_res, time_res = [], [], [], []
    for meth in methods:
        time_row, roc_row, ap_row, prcn_row = [], [], [], []

        clf = LOF() if meth == 'LOF' else KNN() if meth == 'kNN' else IForest() if meth == 'iForest' else HBOS()

        for df in data_frames.keys():
            tic()
            clf.fit(data_frames[df]['X'])
            time_row.append(toc(show=False).total_seconds())
            roc_row.append(np.round(roc_auc_score(data_frames[df]['y'], clf.decision_scores_), decimals=4))
            ap_row.append(np.round(average_precision_score(data_frames[df]['y'], clf.decision_scores_), decimals=4))
            prcn_row.append(np.round(precision_n_scores(data_frames[df]['y'], clf.decision_scores_), decimals=4))

        # appending the results belonging to the current method
        time_res.append(time_row)
        roc_auc_res.append(roc_row)
        aver_prec_res.append(ap_row)
        prec_at_n_res.append(prcn_row)

    time_res = pd.DataFrame(time_res, index=methods, columns=data_name).transpose()
    roc_auc_res = pd.DataFrame(roc_auc_res, index=methods, columns=data_name).transpose()
    aver_prec_res = pd.DataFrame(aver_prec_res, index=methods, columns=data_name).transpose()
    prec_at_n_res = pd.DataFrame(prec_at_n_res, index=methods, columns=data_name).transpose()

    with pd.ExcelWriter("comparison_results.xlsx") as writer:
        time_res.to_excel(writer, sheet_name='time')
        roc_auc_res.to_excel(writer, sheet_name='roc')
        aver_prec_res.to_excel(writer, sheet_name='ap')
        prec_at_n_res.to_excel(writer, sheet_name='prcn')

    return time_res, roc_auc_res, aver_prec_res, prec_at_n_res


def main():
    lof_()
    knn_()
    iforest_()
    hbos_()

    compr_res_ = compare_methods()

    return compr_res_


if __name__ == '__main__':
    compr_res_ = main()
