#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:34:37 2018


"""

import os
import sys
import numpy as np
import scipy.signal as spsg
import scipy.stats as stt
import scipy.io as sio
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as pp


#%% get arguments from script call

#i_sub = int(sys.argv[1])
#i_motion = int(sys.argv[2])
#i_space = int(sys.argv[3])

i_sub = int(sys.argv[1])
# Variable per fixar si agafem els assajos amb NS o S
i_motion = int(sys.argv[2])
i_space = 1
 

print('subject, motion, space:', i_sub, i_motion, i_space, '; launched with', sys.argv[0])


#%% general info
gen_path_dir = './EEG-BCN/'

type_motion = ['NS', 'S', 'ALL'] # TO RECHECK
n_motion = len(type_motion)

type_space = ['ELE', 'ICA']
n_space = len(type_space)

subjects = range(25,35)

cmapcolours = ['Blues','Greens','Oranges']
listcolours = ['b','g','r']

measure_labels = ['pow','corr']  # covariances removed
n_measures = len(measure_labels)

freq_bands = ['alpha','beta','gamma']
n_bands = len(freq_bands)



#%% director to save results
            
res_dir = 'results/sub'+str(i_sub)+'-' + 'SF-'+type_motion[i_motion]+'-'+type_space[i_space]+ '/'
if not os.path.exists(res_dir):
    print('create directory:',res_dir)
    os.makedirs(res_dir)

#%% load data

n_motiv = 2
if i_motion == 0: # NS
    low_ind, top_ind = 0,6
elif i_motion == 1: # S
    low_ind, top_ind = 6,12
else:
    low_ind, top_ind = 0, 12

# TODO: he canviat ic_data per ic_data3
if i_space == 0:
    ts_tmp2 = sio.loadmat(gen_path_dir+'dataClean-ICA3-'+str(i_sub)+'-T1.mat')['dataSorted'][:, :, :, low_ind:top_ind]
else:
    ts_tmp2 = sio.loadmat(gen_path_dir+'dataClean-ICA3-'+str(i_sub)+'-T1.mat')['ic_data3'][:, :, :, low_ind:top_ind]

# L'última dimensió és el nombre d'estats motivacionals (Solo, Not Solo)
if i_motion == 0 or i_motion == 1:
    ts_tmp = np.zeros([ts_tmp2.shape[0],ts_tmp2.shape[1],4*ts_tmp2.shape[2],2])
    ts_tmp[:, :, 0:108, 0] = ts_tmp2[:, :, :, 0]
    ts_tmp[:, :, 108:216, 0] = ts_tmp2[:, :, :, 1]
    ts_tmp[:, :, 0:108, 1] = ts_tmp2[:, :, :, 2]
    ts_tmp[:, :, 108:216, 1] = ts_tmp2[:, :, :, 3]
    ts_tmp[:, :, 216:324, 1] = ts_tmp2[:, :, :, 4]
    ts_tmp[:, :, 324:432, 1] = ts_tmp2[:, :, :, 5]
else:
    ts_tmp = np.zeros([ts_tmp2.shape[0], ts_tmp2.shape[1], 8 * ts_tmp2.shape[2], 2])
    for i in range(2):
        start = i*432
        ts_tmp[:, :, start: start + 108, 0] = ts_tmp2[:, :, :, i*6]
        ts_tmp[:, :, start + 108: start + 216, 0] = ts_tmp2[:, :, :, i*6 + 1]
        ts_tmp[:, :, start:start + 108, 1] = ts_tmp2[:, :, :, i*6 + 2]
        ts_tmp[:, :, start + 108: start + 216, 1] = ts_tmp2[:, :, :, i*6 + 3]
        ts_tmp[:, :, start + 216:start + 324, 1] = ts_tmp2[:, :, :, i*6 + 4]
        ts_tmp[:, :, start + 324:start + 432, 1] = ts_tmp2[:, :, :, i*6 + 5]



N = ts_tmp.shape[0]  # number of channels
n_trials = ts_tmp.shape[2]  # number of trials per block
T = 1200  # trial duration


# discard silent channels where there are 0's or nan's
invalid_ch = np.logical_or(np.abs(ts_tmp).max(axis=(1, 2, 3)) == 0, np.isnan(ts_tmp[:, 0, 0, 0]))
valid_ch = np.logical_not(invalid_ch)
ts_tmp = ts_tmp[valid_ch,:,:,:]
N = valid_ch.sum()

# get time series for each block
ts = np.zeros([n_motiv,n_trials,T,N])
for i_motiv in range(n_motiv):
    for i_trial in range(n_trials):
        # swap axes for time and channels
        ts[i_motiv, i_trial, :, :] = ts_tmp[:, :, i_trial, i_motiv].T

del ts_tmp # clean memory

# Diagonal amb 0's i part superior també
# Això es farà per tenir la matriu de correlació entre els diversos
# electrodes
mask_tri = np.tri(N, N, -1, dtype=bool) # mask to extract lower triangle of matrix


#%% classifier and learning parameters

# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

# Classificador de Recursive Feature Elimination
c_MLR = RFE_pipeline([('std_scal', skprp.StandardScaler()), ('clf',skllm.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=500))])

# Classificador de nearest neighbor
c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
 
# Cross-validation scheme
cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
n_rep = 20 # number of repetitions

# RFE wrappers
# Per cada iteració eliminem "step" features, al final ens quedem amb "n_features_to_select"
# Ens quedem amb les features més importants
RFE_pow = skfs.RFE(c_MLR, n_features_to_select=1, step=1)
RFE_FC = skfs.RFE(c_MLR, n_features_to_select=int(N/2), step=int(N/2))

# record classification performance 
perf = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
perf_shuf = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
conf_matrix = np.zeros([n_bands,n_measures,n_rep,2,n_motiv,n_motiv]) # (fourthindex: MLR/1NN)

# rankings of best features and stability
rk_pow = np.zeros([n_bands,n_rep,N],dtype=int) # RFE rankings for power (N feature)
rk_FC = np.zeros([n_bands,n_rep,int(N*(N-1)/2)],dtype=int) # RFE rankings for FC (N(N-1)/2 feature)
pearson_corr_rk = np.zeros([n_bands,n_measures,int(n_rep*(n_rep-1)/2)]) # stability of rankings measured by Pearson correlation
pearson_corr_rk_shuf = np.zeros([n_bands,n_measures,int(n_rep*(n_rep-1)/2)]) # surrogate distributions for ranking stability
n_rep_RFE = 20 # number of splits to check best features
n_feat_test_pow = 20 # number of best features (power) to test to find optimum
perf_rk_best_feat_pow = np.zeros([n_bands,n_rep_RFE,n_feat_test_pow]) # performance when increasing number of best features (power)
n_feat_test_FC = 80 # number of best features (corr) to test to find optimum
perf_rk_best_feat_FC = np.zeros([n_bands,n_rep_RFE,n_feat_test_FC]) # performance when increasing number of best features (corr)


#%% loop over the measures and frequency bands

for i_band in range(n_bands):
    freq_band = freq_bands[i_band]
    
    # band-pass filtering (alpha, beta, gamma)
    n_order = 3
    sampling_freq = 500.  # sampling rate
    nyquist_freq = sampling_freq / 2.
    
    if freq_band == 'alpha':
        low_f = 8./nyquist_freq
        high_f = 12./nyquist_freq
    elif freq_band == 'beta':
        # beta
        low_f = 15./nyquist_freq
        high_f = 30./nyquist_freq
    elif freq_band == 'gamma':
        # gamma
        low_f = 40./nyquist_freq
        high_f = 80./nyquist_freq
    else:
        raise NameError('unknown filter')
    
    # apply filter
    b, a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
    filtered_ts = spsg.filtfilt(b, a, ts, axis=2)
    
    for i_measure in range(n_measures):
        
        print('frequency band, measure:', freq_band, measure_labels[i_measure])
    
        if i_measure == 0:  # power of signal within each sliding window (rectification by absolute value)
            # create the design matrix [samples,features]
            # Tindrem com a features els valors absoluts de les senyals
            vect_features = np.abs(filtered_ts).mean(axis=2)
        else: # correlation
            EEG_FC = np.zeros([n_motiv, n_trials, N, N])  # dynamic FC = covariance or Pearson correlation of signal within each sliding window
            for i_motiv in range(n_motiv):
                for i_trial in range(n_trials):
                    if np.std(filtered_ts[i_motiv, i_trial, :, :]):
                        EEG_FC[i_motiv, i_trial, :, :] = np.corrcoef(filtered_ts[i_motiv, i_trial, :, :], rowvar=False)
                    else:
                        EEG_FC[i_motiv, i_trial, :, :] = np.nan
#                    ts_tmp = filtered_ts[i_motiv,i_trial,:,:]
#                    ts_tmp -= np.outer(np.ones(T),ts_tmp.mean(0))
#                    EEG_FC[i_motiv,i_trial,:,:] = np.tensordot(ts_tmp,ts_tmp,axes=(0,0)) / float(T-1)
#                    EEG_FC[i_motiv,i_trial,:,:] /= np.sqrt(np.outer(EEG_FC[i_motiv,i_trial,:,:].diagonal(),EEG_FC[i_motiv,i_trial,:,:].diagonal()))

            # vectorize the connectivity matrices to obtain the design matrix [samples,features]
            vect_features = EEG_FC[:,:,mask_tri]
            
        # labels of sessions for classification (train+test)
        labels = np.zeros([n_motiv,n_trials], dtype=int) # 0 = M0, 1 = M1
        labels[1,:] = 1
        
        # vectorize dimensions motivation levels and trials
#        mask_motiv_trials = np.ones([n_motiv,n_trials], dtype=np.bool)
        mask_motiv_trials = np.logical_not(np.isnan(vect_features[:,:,0]))
        # Agafem els features que no són nan
        vect_features = vect_features[mask_motiv_trials,:]
        labels = labels[mask_motiv_trials]
        
        # repeat classification for several splits for indices of sliding windows (train/test sets)
        for i_rep in range(n_rep):
            for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
                # train and test for original data
                c_MLR.fit(vect_features[ind_train,:], labels[ind_train])
                perf[i_band,i_measure,i_rep,0] = c_MLR.score(vect_features[ind_test,:], labels[ind_test])
                conf_matrix[i_band,i_measure,i_rep,0,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test,:]))  
            
                c_1NN.fit(vect_features[ind_train,:], labels[ind_train])
                perf[i_band,i_measure,i_rep,1] = c_1NN.score(vect_features[ind_test,:], labels[ind_test])
                conf_matrix[i_band,i_measure,i_rep,1,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_1NN.predict(vect_features[ind_test,:]))  
            
                # shuffled performance distributions
                shuf_labels = np.random.permutation(labels)
        
                c_MLR.fit(vect_features[ind_train,:], shuf_labels[ind_train])
                perf_shuf[i_band,i_measure,i_rep,0] = c_MLR.score(vect_features[ind_test,:], shuf_labels[ind_test])
        
                c_1NN.fit(vect_features[ind_train,:], shuf_labels[ind_train])
                perf_shuf[i_band,i_measure,i_rep,1] = c_1NN.score(vect_features[ind_test,:], shuf_labels[ind_test])

                # RFE for MLR
                if i_measure == 0: # power
                    RFE_pow.fit(vect_features[ind_train,:], labels[ind_train])
                    rk_pow[i_band,i_rep,:] = RFE_pow.ranking_
                else: # correlation
                    RFE_FC.fit(vect_features[ind_train,:], labels[ind_train])
                    rk_FC[i_band,i_rep,:] = RFE_FC.ranking_

                    
        # identify informative sources or FC links
        if i_measure == 0: # power
            # indices of best features from mean ranking for sources (power)
            ordered_best_feat = np.argsort(rk_pow[i_band,:,:].mean(0))
            # repeat classification for several splits to test 
            for i_rep in range(n_rep_RFE):
                for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
                    for i_best_feat in range(1,n_feat_test_pow):
                        # selected best features
                        ind_select = ordered_best_feat[:i_best_feat]
                        # train and test for original data
                        c_MLR.fit(vect_features[ind_train,:][:,ind_select], labels[ind_train])
                        perf_rk_best_feat_pow[i_band,i_rep,i_best_feat] = c_MLR.score(vect_features[ind_test,:][:,ind_select], labels[ind_test])
        else: # corr
            # indices of best features from mean ranking for FC measure
            ordered_best_feat = np.argsort(rk_FC[i_band,:,:].mean(0))
            # repeat classification for several splits to test 
            for i_rep in range(n_rep_RFE):
                for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
                    for i_best_feat in range(1,n_feat_test_FC):
                        # selected best features
                        ind_select = ordered_best_feat[:i_best_feat]
                        # train and test for original data
                        c_MLR.fit(vect_features[ind_train,:][:,ind_select], labels[ind_train])
                        perf_rk_best_feat_FC[i_band,i_rep,i_best_feat] = c_MLR.score(vect_features[ind_test,:][:,ind_select], labels[ind_test])

            
# check stability RFE rankings
for i_band in range(n_bands):
    for i_measure in range(n_measures):
        i_cnt = 0
        for i_rep1 in range(n_rep):
            for i_rep2 in range(i_rep1):
                # data
                pearson_corr_rk[i_band,0,i_cnt] = stt.pearsonr(rk_pow[i_band,i_rep1,:],rk_pow[i_band,i_rep2,:])[0]
                pearson_corr_rk[i_band,1,i_cnt] = stt.pearsonr(rk_FC[i_band,i_rep1,:],rk_FC[i_band,i_rep2,:])[0]
                # shuffled surrogates
                pearson_corr_rk_shuf[i_band,0,i_cnt] = stt.pearsonr(np.random.permutation(rk_pow[i_band,i_rep1,:]),rk_pow[i_band,i_rep2,:])[0]
                pearson_corr_rk_shuf[i_band,1,i_cnt] = stt.pearsonr(np.random.permutation(rk_FC[i_band,i_rep1,:]),rk_FC[i_band,i_rep2,:])[0]
                # increment counter
                i_cnt += 1


# save results       
np.save(res_dir+'valid_ch.npy', valid_ch)                
np.save(res_dir+'perf.npy',perf)
np.save(res_dir+'perf_shuf.npy',perf_shuf)
np.save(res_dir+'conf_matrix.npy',conf_matrix)
np.save(res_dir+'rk_pow.npy',rk_pow)
np.save(res_dir+'rk_FC.npy',rk_FC)
np.save(res_dir+'pearson_corr_rk.npy',pearson_corr_rk)
np.save(res_dir+'pearson_corr_rk_shuf.npy',pearson_corr_rk_shuf)
np.save(res_dir+'perf_rk_best_feat_pow.npy',perf_rk_best_feat_pow)
np.save(res_dir+'perf_rk_best_feat_FC.npy',perf_rk_best_feat_FC)


#%% plots for quick check of results
fmt_grph = 'png'
fmt_grph2 = 'pdf'


for i_band in range(n_bands):
    freq_band = freq_bands[i_band]
    for i_measure in range(n_measures):
        measure_label = measure_labels[i_measure]

        # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
        chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
    
        # plot performance and surrogate
        pp.figure(figsize=[4,3])
        pp.axes([0.2,0.2,0.7,0.7])
        pp.violinplot(perf[i_band,i_measure,:,0],positions=[-0.2],widths=[0.3])
        pp.violinplot(perf[i_band,i_measure,:,1],positions=[0.2],widths=[0.3])
        pp.violinplot(perf_shuf[i_band,i_measure,:,0],positions=[0.8],widths=[0.3])
        pp.violinplot(perf_shuf[i_band,i_measure,:,1],positions=[1.2],widths=[0.3])
        pp.plot([-1,2],[chance_level]*2,'--k')
        pp.axis(xmin=-0.6,xmax=1.6,ymin=0,ymax=1.05)
        pp.xticks([0,1],['MLR/1NN','surrogate'],fontsize=8)
        pp.ylabel('accuracy_'+freq_band+'_'+measure_label,fontsize=8)
        pp.title(freq_band+', '+measure_label)
        pp.savefig(res_dir+'accuracy_'+freq_band+'_'+measure_label+'.'+fmt_grph, format=fmt_grph)
        pp.savefig(res_dir+'accuracy_'+freq_band+'_'+measure_label+'.'+fmt_grph2, format=fmt_grph2)
        pp.close()

        # plot confusion matrix for MLR
        pp.figure(figsize=[4,3])
        pp.axes([0.2,0.2,0.7,0.7])
        pp.imshow(conf_matrix[i_band,i_measure,:,0,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
        pp.colorbar()
        pp.xlabel('true label',fontsize=8)
        pp.ylabel('predicted label',fontsize=8)
        pp.title(freq_band+', '+measure_label)
        pp.savefig(res_dir+'conf_mat_MLR_'+freq_band+'_'+measure_label+'.'+fmt_grph, format=fmt_grph)
        pp.savefig(res_dir+'conf_mat_MLR_'+freq_band+'_'+measure_label+'.'+fmt_grph2, format=fmt_grph2)
        pp.close()


    # plot stability of RFE rankings
    pp.figure(figsize=[4,3])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.violinplot(pearson_corr_rk[i_band,:,:].T,positions=np.arange(n_measures)-0.2,widths=[0.4]*n_measures)
    pp.violinplot(pearson_corr_rk_shuf[i_band,:,:].T,positions=np.arange(n_measures)+0.2,widths=[0.4]*n_measures)
    pp.axis(xmin=-0.6,xmax=n_measures-0.4,ymin=0,ymax=1)
    pp.xticks(range(n_measures),measure_labels,fontsize=8)
    pp.ylabel('Pearson between rankings',fontsize=8)
    pp.title(freq_band)
    pp.savefig(res_dir+'stab_RFE_rankings_'+freq_band+'.'+fmt_grph, format=fmt_grph)
    pp.savefig(res_dir+'stab_RFE_rankings_'+freq_band+'.'+fmt_grph2, format=fmt_grph2)
    pp.close()

    # plot performance for best features
    pp.figure(figsize=[4,3])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.fill_between(range(n_feat_test_pow), perf_rk_best_feat_pow[i_band,:,:].mean(0)-perf_rk_best_feat_pow[i_band,:,:].std(0)/np.sqrt(n_rep), perf_rk_best_feat_pow[i_band,:,:].mean(0)+perf_rk_best_feat_pow[i_band,:,:].std(0)/np.sqrt(n_rep))
    pp.axis(ymin=0,ymax=1)
    pp.xticks(fontsize=8)
    pp.xlabel('number of best features',fontsize=8)
    pp.ylabel('accuracy',fontsize=8)
    pp.title(freq_band)
    pp.savefig(res_dir+'perf_best_feat_'+freq_band+'.'+fmt_grph, format=fmt_grph)
    pp.savefig(res_dir+'perf_best_feat_'+freq_band+'.'+fmt_grph2, format=fmt_grph2)
    pp.close()
