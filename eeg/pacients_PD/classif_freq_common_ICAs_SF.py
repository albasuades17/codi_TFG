#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:34:37 2018

author: matgilson, ignasicos
"""


"""
Script que agafa les fonts en comú dels pacients i fa la classificació una altra vegada amb aquestes fonts.
D'aquesta manera veiem si realment amb aquestes fonts es pot explicar el comportament dels pacients d'una manera
"bona"
"""

import numpy as np
import scipy.signal as spsg
import scipy.io as sio
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.metrics as skm
import sklearn.model_selection as skms
import matplotlib.pyplot as pp
import networkx as nx
import seaborn as sb


data_dir = './PD-EEG/'
path_dir_src = '/sources/'
fmt_grph = 'pdf' # 'pdf', 'png', 'eps'

#%% general info
subjects = [55, 60, 61, 64]
n_sub = len(subjects)

type_motions = ['NS', 'S']
n_motions = len(type_motions)

#type_signals = ['ELE','ICA']
#n_signals = len(type_signals)
type_signal = 'ICA' # only ICAs

freq_bands = ['alpha','beta','gamma']
n_bands = len(freq_bands)

cmap_colours = ['Blues','Greens','Oranges']
dark_colours = ['b','g','r']
light_colours = [[0.7,0.7,1],[0.7,1,0.7],[1,0.7,0.7]]

# toggle to select desired features
i_measure = 0 # power of ICAs
# i_measure = 1 # FC interactions between ICAs

#%% classifier and learning parameters (only MLR)

# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=500))])

RFE_FC = skfs.RFE(c_MLR, n_features_to_select=1, step=1)

# cross-validation scheme
cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
n_rep = 10 # number of repetitions of train/test splits

medicacio = {0: 'ON', 1: 'OFF'}
# Iterem segons si està medicat
for estat in [0,1]:
    work_dir = 'common_src_SF_' + medicacio[estat] + '/'
    print('Estat ' + medicacio[estat])

    #%% loop over frequency bands, then loop over subjects, motion types and sessions by taking only common sources
    n_motiv = 2
    perf = np.zeros([n_sub, n_motions, n_bands, n_rep])
    perf_shuf = np.zeros([n_sub, n_motions, n_bands, n_rep])
    conf_matrix = np.zeros([n_sub, n_motions, n_bands, n_rep, n_motiv, n_motiv])

    N_max = 8 # maximum number of common sources
    rk_FC = np.zeros([n_sub, n_motions, n_bands, n_rep, int(N_max*(N_max+1)/2)]) # RFE rankings for FC between common sources

    # indices of sources for FC interactions
    ij_global = np.zeros([N_max,N_max], dtype=int)
    cnt_tmp = 0
    for i in range(N_max):
        for j in range(N_max):
            if i>=j:
                ij_global[i,j] = cnt_tmp
                cnt_tmp += 1
            else:
                ij_global[i,j] = N_max**3 # to be sure it raises an error

    is_valid = np.zeros([n_sub, n_motions], dtype=bool)


    for i_band in range(n_bands):
        freq_band = freq_bands[i_band]

        for i_sub, ind_sub in enumerate(subjects):

            for i_motion, type_motion in enumerate(type_motions):

                try:
                    ############

                    # load ICA time series (same as classif_freq_EEG_general.py)
                    n_motiv = 2
                    if i_motion == 0:  # NS
                        low_ind, top_ind = 0, 3
                    else:  # S
                        low_ind, top_ind = 3, 6

                    ts_tmp2 = sio.loadmat(data_dir+'dataClean-ICA-'+str(subjects[i_sub])+'-T1.mat')['ic_data'][:,:,:,low_ind:top_ind][:,:,:,:,estat]

                    ts_tmp = np.zeros([ts_tmp2.shape[0], ts_tmp2.shape[1], 2 * ts_tmp2.shape[2], 2])
                    ts_tmp[:, :, 0:108, 0] = ts_tmp2[:, :, :, 0]
                    ts_tmp[:, :, 0:108, 1] = ts_tmp2[:, :, :, 1]
                    ts_tmp[:, :, 108:216, 1] = ts_tmp2[:, :, :, 2]

                    N = ts_tmp.shape[0] # number of channels
                    n_trials = ts_tmp.shape[2] # number of trials per block
                    T = 1600 # trial duration

                    # discard silent channels
                    invalid_ch1 = np.logical_or(np.abs(ts_tmp[:,:,:,0]).max(axis=(1,2))==0, np.isnan(ts_tmp[:,0,0,0]))
                    invalid_ch2 = np.logical_or(np.abs(ts_tmp[:,:,:,1]).max(axis=(1,2))==0, np.isnan(ts_tmp[:,0,0,1]))
                    invalid_ch = np.logical_or(invalid_ch1, invalid_ch2)
                    #invalid_ch = np.logical_or(np.abs(ts_tmp).max(axis=(1,2,3))==0, np.isnan(ts_tmp[:,0,0,0]))
                    valid_ch = np.logical_not(invalid_ch)
                    ts_tmp = ts_tmp[valid_ch,:,:,:]
                    N = valid_ch.sum()

                    # get time series for each block
                    ts = np.zeros([n_motiv,n_trials,T,N])
                    for i_motiv in range(n_motiv):
                        for i_trial in range(n_trials):
                            # swap axes for time and channels
                            ts[i_motiv,i_trial,:,:] = ts_tmp[:,:,i_trial,i_motiv].T

                    del ts_tmp # clean memory

                    ############
                    # get common sources (from calc_common_ICAs.py)
                    all_data_src_gp = np.load(work_dir+'all_data_src_gp_'+freq_bands[i_band]+'.npy',allow_pickle=True)
                    n_gp = len(all_data_src_gp)

                    select_src = np.zeros([N], dtype=bool)
                    corresp_gp = -np.ones([N], dtype=int)
                    for i_gp in range(n_gp):
                        for i_src in range(len(all_data_src_gp[i_gp])):
                            if all_data_src_gp[i_gp][i_src,0]==i_sub and all_data_src_gp[i_gp][i_src,1]==i_motion:
                                select_src[all_data_src_gp[i_gp][i_src,2]] = True
                                corresp_gp[all_data_src_gp[i_gp][i_src,2]] = i_gp # store corresponding group

                    ts = ts[:,:,:,select_src]
                    corresp_gp = corresp_gp[select_src]
                    N = select_src.sum() # new number of channels

                    print('subject, motion', subjects[i_sub], i_motion, ': nb of selected sources', N)
                    print(corresp_gp)


                    ############
                    # process time series in desired frequency band

                    # band-pass filtering (alpha, beta, gamma)
                    n_order = 3
                    sampling_freq = 500. # sampling rate
                    nyquist_freq = sampling_freq / 2.

                    if freq_band=='alpha':
                        low_f = 8./nyquist_freq
                        high_f = 12./nyquist_freq
                    elif freq_band=='beta':
                        # beta
                        low_f = 15./nyquist_freq
                        high_f = 30./nyquist_freq
                    elif freq_band=='gamma':
                        # gamma
                        low_f = 40./nyquist_freq
                        high_f = 80./nyquist_freq
                    else:
                        raise NameError('unknown filter')

                    # apply filter
                    b,a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
                    filtered_ts = spsg.filtfilt(b, a, ts, axis=2)

                    ############
                    # calculate FC (correlations) for selected sources

                    if i_measure == 0: # power of signal within each sliding window (rectification by absolute value)
                        # create the design matrix [samples,features]
                        vect_features = np.abs(filtered_ts).mean(axis=2)
                    else: # correlation
                        EEG_FC = np.zeros([n_motiv,n_trials,N,N]) # dynamic FC = covariance or Pearson correlation of signal within each sliding window
                        for i_motiv in range(n_motiv):
                            for i_trial in range(n_trials):
                                if np.std(filtered_ts[i_motiv, i_trial, :, :]):
                                    EEG_FC[i_motiv, i_trial, :, :] = np.corrcoef(filtered_ts[i_motiv, i_trial, :, :],
                                                                                 rowvar=False)
                                else:
                                    EEG_FC[i_motiv, i_trial, :, :] = np.nan
                        # vectorize the connectivity matrices to obtain the design matrix [samples,features]
                        mask_tri = np.tri(N,N,-1,dtype=bool) # mask to extract lower triangle of matrix
                        vect_features = EEG_FC[:,:,mask_tri]

                    ############
                    # perform classification

                    # labels of sessions for classification (train+test)
                    labels = np.zeros([n_motiv,n_trials], dtype=int) # 0 = M0, 1 = M1
                    labels[1,:] = 1

                    # vectorize dimensions motivation levels and trials
                    mask_motiv_trials = np.logical_not(np.isnan(vect_features[:,:,0]))
                    vect_features = vect_features[mask_motiv_trials,:]
                    labels = labels[mask_motiv_trials]

                    # repeat classification for several splits for indices of sliding windows (train/test sets)
                    for i_rep in range(n_rep):
                        for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1
                            # train and test for original data
                            c_MLR.fit(vect_features[ind_train,:], labels[ind_train])
                            perf[i_sub,i_motion,i_band,i_rep] = c_MLR.score(vect_features[ind_test,:], labels[ind_test])
                            conf_matrix[i_sub, i_motion, i_band, i_rep, :, :] += skm.confusion_matrix(
                                y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test, :]))
                            # RFE for FC
                            if i_measure==1:
                                RFE_FC.fit(vect_features[ind_train,:], labels[ind_train])
                                # correspondance between interactions and sources for this subject
                                ij_sub = np.zeros([N,N,2], dtype=int)
                                for i in range(N):
                                    for j in range(N):
                                        ij_sub[i,j,:] = corresp_gp[i], corresp_gp[j]
                                ij_sub = ij_sub[mask_tri,:]
                                # list of rankings (because a subject can have several sources corresponding to the same best source)
                                list_rk_tmp = []
                                for ij2 in range(int(N_max*(N_max+1)/2)):
                                    list_rk_tmp += [[]] # empty list
                                # correspondance between FC interactions of this subject back to global communities
                                for ij in range(RFE_FC.ranking_.size):
                                    if ij_sub[ij,0]>ij_sub[ij,1]:
                                        i_global = ij_sub[ij,0]
                                        j_global = ij_sub[ij,1]
                                    else:
                                        i_global = ij_sub[ij,1]
                                        j_global = ij_sub[ij,0]
                                    list_rk_tmp[ij_global[i_global,j_global]] += [RFE_FC.ranking_[ij]]
                                # calculate average ranking and put nan if global interaction absent for this subject
                                for ij2 in range(int(N_max*(N_max+1)/2)):
                                    if list_rk_tmp[ij2]==[]:
                                        rk_FC[i_sub,i_motion,i_band,i_rep,ij2] = np.nan
                                    else:
                                        rk_FC[i_sub,i_motion,i_band,i_rep,ij2] = np.mean(list_rk_tmp[ij2])

                            # shuffled performance distributions
                            shuf_labels = np.random.permutation(labels)

                            c_MLR.fit(vect_features[ind_train,:], shuf_labels[ind_train])
                            perf_shuf[i_sub,i_motion,i_band,i_rep] = c_MLR.score(vect_features[ind_test,:], shuf_labels[ind_test])

                    ############
                    # end of loop
                    is_valid[i_sub,i_motion] = True
                    print('subject, motion, space, session:', i_sub, i_motion)
                except Exception as err:
                    print('bad session:', i_sub, i_motion, err)

    # approximate theoretical chance level
    chance_level = 0.66

    pp.figure(figsize=[4,2])
    pp.axes([0.15,0.2,0.8,0.7])

    n_valid = is_valid.sum()

    # pool performance for all valid sessions (subject, motion, session)
    perf_tmp = perf[is_valid,:,:]
    perf_shuf_tmp = perf_shuf[is_valid,:,:]
    # reshape to have 3 bands together for each classifier
    perf_tmp2 = np.zeros([n_valid,n_rep,3])
    for i_band in range(n_bands):
        perf_tmp2[:,:,i_band] = perf_tmp[:,i_band,:]
    perf_shuf_tmp2 = np.zeros([n_valid,n_rep,3])
    for i_band in range(n_bands):
        perf_shuf_tmp2[:,:,i_band] = perf_shuf_tmp[:,i_band,:]

    np.save(work_dir + 'perf.npy', perf)
    np.save(work_dir+ 'conf.npy', conf_matrix)
    # plot surrogate distribution lumped for all subjects
    sb.violinplot(data=perf_shuf_tmp2.reshape([-1, 3]), linewidth=0.5, inner=None,
                  color=[0.7, 0.7, 0.7])
    # plot data distribution lumped for all subjects
    sb.violinplot(data=perf_tmp2.reshape([-1, 3]), linewidth=0.5, inner=None,
                  palette=[[1, 0.7, 0.7], [0.7, 1, 0.7], [0.7, 0.7, 1]])
    # plot swarm plot for average over repetitions perf of each session (over subjects and motions)
    sb.swarmplot(data=perf_tmp2.mean(1), size=2, palette=['r', 'g', 'b'])
    # plot theoretical chance level
    pp.plot([-1, 3], [chance_level] * 2, '--k')

    pp.xticks(range(3), [r'$\alpha$-MLR', r'$\beta$-MLR', r'$\gamma$-MLR'], fontsize=8, rotation=20)
    pp.yticks([0, 0.5, 1], fontsize=8)
    pp.axis(xmin=-0.6, xmax=2.6, ymin=0, ymax=1.05)
    pp.ylabel('accuracy', fontsize=8)

    if i_measure==0:
        pp.savefig(work_dir+'accuracy_select_ICAs_pow_.'+fmt_grph, format=fmt_grph)
    else:
        pp.savefig(work_dir+'accuracy_select_ICAs_FC_.'+fmt_grph, format=fmt_grph)

    pp.close()


    #%% plot best channels for FC (pooled over all valid sessions)

    n_best_feat_FC_plot = 10 # plot 20 best pairs of channels (FC interactions)

    # node positions for circular layout with origin at top (clockwise) # TO ADAPT TO CHANGE ORDER OF BEST SOURCES
    pos_circ = dict()
    for i in range(N_max):
        pos_circ[i] = np.array([np.sin(i*2*np.pi/N_max), np.cos(i*2*np.pi/N_max)])

    # channel labels
    ch_labels = dict()
    for i in range(N_max):
        ch_labels[i] = i+1

    # matrices to retrieve input/output channels from connections in support network
    row_ind = np.repeat(np.arange(N_max).reshape([N_max,-1]),N_max,axis=1)
    col_ind = np.repeat(np.arange(N_max).reshape([-1,N_max]),N_max,axis=0)
    mask_tri_global = np.tri(N_max,N_max,-1,dtype=bool)
    row_ind = row_ind[mask_tri_global]
    col_ind = col_ind[mask_tri_global]


    for i_band in range(n_bands):
        freq_band = freq_bands[i_band]

        # plot RFE support network
        pp.figure(figsize=[10,10])
        pp.axes([0.05,0.05,0.95,0.95])
        pp.axis('off')
        # correlation
        rk_tmp = rk_FC[is_valid,:,:,:] # get all valid sessions
        rk_tmp = rk_tmp[:,i_band,:,:] # get considered freq band
        # quick fix: "exclude" channels with all nans by giving them largest value
        exclude_chpr = np.isnan(rk_tmp).sum(axis=(0,1))==is_valid.sum()*n_rep
        rk_tmp[:,:,exclude_chpr] = N_max**2
        # calculate mean ranking while ignoring nan values
        rk_tmp = np.nanmean(rk_tmp, axis=(0,1))
        list_best_feat = np.argsort(rk_tmp)[:n_best_feat_FC_plot] # retain best best features
        g = nx.Graph()
        for i in range(N_max):
            g.add_node(i)
        node_color_aff = ['grey']*N_max
        list_ROI_from_to = [] # list of input/output ROIs involved in connections of support network
        for ij in list_best_feat:
            if not col_ind[ij]==row_ind[ij]:
                g.add_edge(col_ind[ij],row_ind[ij])
        nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
        nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
        nx.draw_networkx_edges(g,pos=pos_circ,edgelist=g.edges(),edge_color=dark_colours[i_band], width=3)
        pp.title(freq_band)
        pp.savefig(work_dir+'support_net_RFE_best_src'+freq_band+'.'+fmt_grph, format=fmt_grph, transparent=True)
        pp.close()