#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:34:37 2018
"""

"""
Funció que mira quines són les millors fonts per tots els pacients.
"""

import os
import numpy as np
import scipy.stats as stt
import scipy.io as sio
import matplotlib.pyplot as pp
import matplotlib.patches as ppt

path_dir_res = './results/'
path_dir_src = './sources/'

for variable in ['SF', 'SP']:
    for med in ['ON', 'OFF']:

        work_dir = './common_src_'+variable+ '_' + med + '/'
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)


        #%% global parameters

        n_motiv = 2 # alone and easy or hard

        subjects = [55, 60, 61, 64]
        n_sub = len(subjects)

        type_motions = ['NS','S']
        n_motions = len(type_motions)

        #i_signal = 1 # 'ICA'
        #i_measure = 0 # 'pow'

        freq_bands = ['alpha','beta','gamma']
        n_bands = len(freq_bands)

        cmap_colours = ['Blues','Greens','Oranges']


        lim_align = 0.35 # cutoff for similarity (Pearson correlation between headmaps)


        #%% calculate number of components
        m = 101 # number of spatial steps
        common_ind = np.ones([m,m], dtype=bool)

        is_valid = np.zeros([n_sub, n_motions], dtype=bool) # just for ICA here, not ELE

        n_best_src = 10 # n best sources per subjects (heuristic average from graphs perf_best_feat_gamma)

        ind_best_src_sub = np.zeros([n_bands, n_sub, n_motions, n_best_src], dtype=int) # index of best sources for each subject in mysources-xxx-T1.mat
        ind_sub = np.zeros([n_bands, n_sub, n_motions, n_best_src], dtype=int) # index of subject
        ind_motion = np.zeros([n_bands, n_sub, n_motions, n_best_src], dtype=int) # index of motion

        for i_sub in range(n_sub):

            valid_sub = False

            try:
                # get surface indices that are common to all subjects
                # vq 101 x 101 x n_sources
                src = sio.loadmat(path_dir_src + 'mysourcesPlot3-'+str(subjects[i_sub])+'-T1.mat')['vq']
                valid_ind = np.logical_not(np.isnan(src[:,:,0]))
                common_ind = np.logical_and(common_ind, valid_ind)

                valid_sub = True

            except Exception as err:
                print('bad subject:', subjects[i_sub])
                print(err)

            if valid_sub:
                for i_motion, type_motion in enumerate(type_motions):
                    try:
                        res_dir = path_dir_res+'sub'+str(subjects[i_sub])+'-'+ variable + '-' + type_motion+'-ICA-' + med + '/'

                        # store index of best sources (averaged over the 20 repetitions)
                        av_rk = np.load(res_dir+'rk_pow.npy').mean(1)

                        for i_band in range(n_bands):
                            # Seleccionem les best sources
                            ind_best_src_sub[i_band,i_sub,i_motion,:] = np.argsort(av_rk)[i_band, :n_best_src]
                            ind_sub[i_band,i_sub,i_motion,:] = i_sub
                            ind_motion[i_band,i_sub,i_motion,:] = i_motion

                        is_valid[i_sub,i_motion] = True

                    except:
                        print('bad session:', res_dir)


        print('density of valid spatial indices:', common_ind.sum()/m**2)
        print()


        # calculate number of total sources over all valid sessions
        # ind_best_src_sub té dimensió n_bands x n_sub x n_motions x n_best_src
        # is_valid té dimensió n_sub x n_motions
        ind_best_src_sub_aligned = ind_best_src_sub[:,is_valid,:]
        # tindrem la dimensió n_bands x (n_sub · n_motions · n_best_src)
        ind_best_src_sub_aligned = ind_best_src_sub_aligned.reshape([n_bands,-1])

        n_best_src_tot = ind_best_src_sub_aligned.shape[1] # n_sub · n_motions · n_best_src

        # vector que comença a 0, acaba a n_best_src_tot i té step = n_best_src
        n_best_src_sub = np.arange(0, n_best_src_tot, n_best_src, dtype=int) # for plotting

        ind_sub_aligned = ind_sub[:,is_valid]
        ind_sub_aligned = ind_sub_aligned.reshape([n_bands,-1])

        ind_motion_aligned = ind_motion[:,is_valid]
        ind_motion_aligned = ind_motion_aligned.reshape([n_bands,-1])


        #%% function to calculate similarity of maps

        def sim_comp(c1, c2):
            # rectify components (more localized positive part)
            mask_c1 = np.zeros(c1.shape)
            # Mirem si hi ha més valors positius que negatius
            if np.sum(c1>0)>np.sum(c1<0):
                mask_c1[c1<0] = -1
            else:
                mask_c1[c1>0] = 1
            # En el primer cas, tindrem que c1_rectif té els valors positius a 0 i els negatius ara són positius.
            # En el segon cas, els negatius seran 0 i els positius continuaren sent positius.
            c1_rectif = c1 * mask_c1

            mask_c2 = np.zeros(c2.shape)
            if np.sum(c2>0)>np.sum(c2<0):
                mask_c2[c2<0] = -1
            else:
                mask_c2[c2>0] = 1
            c2_rectif = c2 * mask_c2
            # get all pairings for values exceeding the limit
            # Agafem el statistic de Pearson
            return stt.pearsonr(c1_rectif, c2_rectif)[0]


        # CALCULATE SIMILARITY for pairs of best sources
        mat_sim_best_src = np.zeros([n_bands,n_best_src_tot,n_best_src_tot])
        for i_band in range(n_bands):
            start_i1 = 0
            for i_sub1 in range(n_sub):
                for i_motion1, type_motion1 in enumerate(type_motions):
                    if is_valid[i_sub1,i_motion1]:
                        src1 = sio.loadmat(path_dir_src+'mysourcesPlot3-'+str(subjects[i_sub1])+'-T1.mat')['vq'][common_ind,:]
                        # Índexs de les best sources per aquest subjecte
                        best_feat1 = ind_best_src_sub[i_band,i_sub1,i_motion1,:]

                        start_i2 = 0
                        for i_sub2 in range(n_sub):
                            for i_motion2, type_motion2 in enumerate(type_motions):
                                if is_valid[i_sub2,i_motion2]:
                                    # sub2: source locations and best sources
                                    src2 = sio.loadmat(path_dir_src+'mysourcesPlot3-'+str(subjects[i_sub2])+'-T1.mat')['vq'][common_ind,:]
                                    best_feat2 = ind_best_src_sub[i_band,i_sub2,i_motion2,:]

                                    # similarity as measured by Pearson correlation (symmetric matrix)
                                    for i1 in range(n_best_src):
                                        for i2 in range(n_best_src):
                                            if not (i1==i2 and i_sub1==i_sub2): # discard diagonal elements
                                                # Es guarda l'estatístic de Pearson entre els mapes de les dues fonts
                                                mat_sim_best_src[i_band,start_i1+i1,start_i2+i2] = sim_comp(src1[:,best_feat1[i1]], src2[:,best_feat2[i2]])

                                    start_i2 += n_best_src
                        start_i1 += n_best_src


        for i_band in range(n_bands):

            pp.figure()
            pp.imshow(mat_sim_best_src[i_band,:,:], origin='lower')
            pp.xticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.yticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.colorbar()
            pp.savefig(work_dir+'align_best_src_noabs_'+freq_bands[i_band])
            pp.close()

            pp.figure()
            pp.imshow(np.abs(mat_sim_best_src[i_band,:,:]), origin='lower')
            pp.xticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.yticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.colorbar()
            pp.savefig(work_dir+'align_best_src_'+freq_bands[i_band])
            pp.close()

            pp.figure()
            pp.hist(np.abs(mat_sim_best_src[i_band,:,:]).flatten(), histtype='step', color='r')
            pp.plot([lim_align]*2, [0,1000], '--k')
            pp.savefig(work_dir+'hist_align_best_src_'+freq_bands[i_band])
            pp.close()

            sig_mat_sim_best_src = np.abs(mat_sim_best_src)>lim_align

            pp.figure()
            pp.imshow(sig_mat_sim_best_src[i_band,:,:], origin='lower', cmap='binary')
            pp.xticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.yticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.savefig(work_dir+'sig_best_src_pairs_'+freq_bands[i_band])
            pp.close()


        #%% detect groups of sources in matrix using Newman community detection

        for i_band in range(n_bands):
            print(freq_bands[i_band])
            print()

            # matrices that measure the likelihodd to be in the same community group
            if False:
                # binary links between sources when the match exceeds the threshold, see sig_mat_sim_best_src above
                C = np.array(sig_mat_sim_best_src[i_band,:,:], dtype=np.float)
            else:
                # weighted matrix with all similarity indices, not using threshold lim_align
                C = np.abs(mat_sim_best_src[i_band,:,:])

            C_null = np.outer(C.sum(1),C.sum(0)) / C.sum()
            C_diff = C - C_null

            # initial partition with all nodes disconnected (indexed by i)
            list_gp = []
            for i in range(n_best_src_tot):
                # We'll have a list like [[0],[1],...]
                list_gp += [[i]]
            # iterative merging of communities that maximize the quality function
            stop = False
            while not stop:
                n = len(list_gp)
                delta_Q = np.zeros([n,n]) # matrix of quality change for all possible merging of two communities (indexed by ii and jj)
                for ii in range(n):
                    for jj in range(n):
                        if ii!=jj: # sum all changes of Q over the pairs of sources (i and j) belonging to the communities (ii and jj)
                            for i in list_gp[ii]:
                                for j in list_gp[jj]:
                                    delta_Q[ii,jj] += C_diff[i,j] + C_diff[j,i]
                        else:
                            delta_Q[ii,jj] = -1e50
                # merge only when the quality measure increases
                if delta_Q.max()>0:
                    ii_max = int(np.argmax(delta_Q)/n)
                    jj_max = np.argmax(delta_Q) - ii_max*n
                    new_list_gp = []
                    for ii in range(n):
                        if ii!=ii_max and ii!=jj_max:
                            new_list_gp += [list_gp[ii]]
                        elif ii==ii_max:
                            new_list_gp += [list_gp[ii_max]+list_gp[jj_max]]
                    list_gp = new_list_gp
                else:
                    stop = True

            print('list of groups (indices in matrices of best features):', list_gp)
            print()

            all_data_src_gp = [] # data of all sources belonging to a same community

            for i_gp in range(len(list_gp)):
                print('group', i_gp)
                all_data_src_gp += [np.zeros([len(list_gp[i_gp]),3], dtype=int)] # add an array to save subject, motion, session and source indices

                ind_gp = list_gp[i_gp]
                print('nb of matches; % in clique:', np.sum(sig_mat_sim_best_src[i_band,:,:][ind_gp,:][:,ind_gp]), np.sum(sig_mat_sim_best_src[i_band,:,:][ind_gp,:][:,ind_gp])/len(ind_gp)**2)

                # get average map and involved subjects
                sub_src = []
                src = np.zeros([m,m])

                for i_cnt, i in enumerate(ind_gp):
                    i_sub = ind_sub_aligned[i_band,i]
                    i_motion = ind_motion_aligned[i_band,i]
                    sub_src += [i_sub]
                    ind_src = ind_best_src_sub_aligned[i_band,i]
                    # print and save corresponding subject, motion and session for redoing classification with interactions using these features only
                    all_data_src_gp[i_gp][i_cnt,:] = i_sub, i_motion, ind_src
                    print('subject, motion, source (in mysourcesPlot3-xxx-T1.mat): ', subjects[i_sub], i_motion, ind_src)
                    # load corresponding source
                    src_tmp = sio.loadmat(path_dir_src+'mysourcesPlot3-'+str(subjects[i_sub])+'-T1.mat')['vq'][:,:,ind_src][common_ind]
                    if np.sum(src_tmp>0)>np.sum(src_tmp<0):
                        switch_coef = -1
                    else:
                        switch_coef = 1
                    src[common_ind] += src_tmp * switch_coef

                print('involved subjects:', np.unique(sub_src))
                print()

                src /= len(ind_gp)

                np.save(work_dir+'mean_best_src_'+freq_bands[i_band]+'_gp'+str(i_gp)+'.npy', src)
                sio.savemat(work_dir+'mean_best_src_'+freq_bands[i_band]+'_gp'+str(i_gp)+'.mat', {'src':src})

                max_val = np.abs(src).max()

                pp.figure()
                pp.imshow(src.T, vmin=-max_val, vmax=max_val, cmap='bwr')
                pp.colorbar()
                pp.savefig(work_dir+'mean_best_src_'+freq_bands[i_band]+'_gp'+str(i_gp))
                pp.close()

            np.save(work_dir+'all_data_src_gp_'+freq_bands[i_band]+'.npy', np.array(all_data_src_gp, dtype=object))
            cols = ['r','g','b','y','c','m'] * 2 + ['k']

            pp.figure()
            pp.imshow(sig_mat_sim_best_src[i_band,:,:], origin='lower', cmap='binary')
            cnt_col = 0
            for i_gp in range(len(list_gp)):
                if len(list_gp[i_gp])>3:
                    i_col = cnt_col
                    cnt_col += 1
                else:
                    i_col = -1
                for i in list_gp[i_gp]:
                    for j in list_gp[i_gp]:
                        if not i==j:
                            ppt.Rectangle((i-0.5,j-0.5), 1, 1, color=cols[i_col])
            pp.xticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.yticks(n_best_src_sub-0.5, n_best_src_sub)
            pp.savefig(work_dir+'communities_'+freq_bands[i_band])
            pp.close()
        