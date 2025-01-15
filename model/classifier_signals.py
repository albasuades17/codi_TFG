import os

import matplotlib.pyplot as pp
import numpy as np
import scipy.io as sio
import scipy.signal as spsg
import sklearn.linear_model as skllm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as skppl
import sklearn.preprocessing as skprp

from model.pyMOU.MOU_model_Testing_Correlation import MOU

sans_dir = '../eeg/pacients_sans'
data_dir = sans_dir + '/EEG-BCN/'
path_dir_src = sans_dir + '/common_src_SF/'
# work_dir = sans_dir + 'common_src_SF/'
work_dir = 'results/'
if not os.path.exists(work_dir):
    print('create directory:',work_dir)
    os.makedirs(work_dir)

fmt_grph = 'pdf'  # 'pdf', 'png', 'eps'

# %% general info

subjects = [25,26,27,28,29,30,31,32,33,34,35]
n_sub = len(subjects)

type_motions = ['NS', 'S']
n_motions = len(type_motions)


freq_bands = ['alpha', 'beta','gamma']  #
n_bands = len(freq_bands)

motivs = ['SF', 'SP']

n_rep = 20 # number of repetitions of train/test splits


def classification_envelope_signals(motiv):
    try:
        perf = np.load(work_dir + 'perf_envelope_signals_'+motiv+'.npy')
        conf_matrix = np.load(work_dir + 'conf_envelope_signals_'+motiv+'.npy')
    except:
        # MLR adapted for recursive feature elimination (RFE)
        class RFE_pipeline(skppl.Pipeline):
            def fit(self, X, y=None, **fit_params):
                """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
                """
                super(RFE_pipeline, self).fit(X, y, **fit_params)
                self.coef_ = self.steps[-1][-1].coef_
                return self

        c_MLR = RFE_pipeline([('std_scal', skprp.StandardScaler()),
                              ('clf', skllm.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=500))])

        # cross-validation scheme
        cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        # %% loop over frequency bands, then loop over subjects, motion types and sessions by taking only common sources
        n_motiv = 2
        perf = np.zeros([n_sub, n_motions, n_bands, n_rep])
        conf_matrix = np.zeros([n_sub, n_motions, n_bands, n_rep, n_motiv, n_motiv])

        N_max = 10  # maximum number of common sources

        is_valid = np.zeros([n_sub, n_motions], dtype=bool)

        for i_band in range(n_bands):
            freq_band = freq_bands[i_band]

            for i_sub, ind_sub in enumerate(subjects):

                for i_motion, type_motion in enumerate(type_motions):

                    try:
                        generated_rand_signals = np.load('generated_signals/' + 'generated_rand_signals_' + motiv + '_' + str(i_band) + '_' + str(i_sub) + '_' + str(i_motion) + '.npy')
                        n_trials = generated_rand_signals.shape[1]

                        ############
                        # calculate vector features
                        vect_features = np.abs(generated_rand_signals).mean(axis=2)

                        ############
                        # perform classification

                        # labels of sessions for classification (train+test)
                        labels = np.zeros([n_motiv, n_trials], dtype=int)  # 0 = M0, 1 = M1
                        labels[1, :] = 1

                        # vectorize dimensions motivation levels and trials
                        mask_motiv_trials = np.logical_not(np.isnan(vect_features[:, :, 0]))
                        vect_features = vect_features[mask_motiv_trials, :]
                        labels = labels[mask_motiv_trials]
                        # repeat classification for several splits for indices of sliding windows (train/test sets)
                        for i_rep in range(n_rep):
                            for ind_train, ind_test in cv_schem.split(vect_features, labels):  # false loop, just 1
                                # train and test for original data
                                c_MLR.fit(vect_features[ind_train, :], labels[ind_train])
                                perf[i_sub, i_motion, i_band, i_rep] = c_MLR.score(vect_features[ind_test, :], labels[ind_test])
                                conf_matrix[i_sub, i_motion, i_band, i_rep, :, :] +=  skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test,:]))

                        ############
                        # end of loop
                        is_valid[i_sub, i_motion] = True

                        print('subject, motion, space, session:', i_sub, i_motion)
                    except Exception as err:
                        print('data missing:', i_sub, i_motion, i_band, err)

        np.save(work_dir + 'perf_envelope_signals_'+motiv+'.npy', perf)
        np.save(work_dir + 'conf_envelope_signals_' + motiv + '.npy', conf_matrix)

    finally:
        return perf, conf_matrix


def classification_real_signals(motiv):
    try:
        perf = np.load(work_dir + 'perf_real_signals_' + motiv + '.npy')
        conf_matrix = np.load(work_dir + 'conf_real_signals_' + motiv + '.npy')
    except:
        # MLR adapted for recursive feature elimination (RFE)
        class RFE_pipeline(skppl.Pipeline):
            def fit(self, X, y=None, **fit_params):
                """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
                """
                super(RFE_pipeline, self).fit(X, y, **fit_params)
                self.coef_ = self.steps[-1][-1].coef_
                return self


        c_MLR = RFE_pipeline([('std_scal', skprp.StandardScaler()),
                              ('clf', skllm.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=500))])

        # cross-validation scheme
        cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        # %% loop over frequency bands, then loop over subjects, motion types and sessions by taking only common sources
        n_motiv = 2
        perf = np.zeros([n_sub, n_motions, n_bands, n_rep])
        perf_shuf = np.zeros([n_sub, n_motions, n_bands, n_rep])
        conf_matrix = np.zeros([n_sub, n_motions, n_bands, n_rep, n_motiv, n_motiv])

        N_max = 10  # maximum number of common sources

        is_valid = np.zeros([n_sub, n_motions], dtype=bool)

        for i_band in range(n_bands):
            freq_band = freq_bands[i_band]

            for i_sub, ind_sub in enumerate(subjects):

                for i_motion, type_motion in enumerate(type_motions):

                    try:
                        ############

                        # load ICA time series (same as classif_freq_EEG_general.py)
                        if i_motion == 0:  # NS
                            low_ind, top_ind = 0, 6
                        else:  # S
                            low_ind, top_ind = 6, 12

                        ts_tmp2 = sio.loadmat(data_dir + 'dataClean-ICA3-' + str(subjects[i_sub]) + '-T1.mat')['ic_data3'][:, :,
                                  :, low_ind:top_ind]

                        ts_tmp = np.zeros([ts_tmp2.shape[0], ts_tmp2.shape[1], 2 * ts_tmp2.shape[2], 2])
                        if motiv == 'SF':
                            trials_per_block = 108
                            target_trials = trials_per_block // 2
                            # Randomly select trials for each block (2, 3, 4, 5)
                            random_indices_block2 = np.random.choice(trials_per_block, target_trials, replace=False)
                            random_indices_block3 = np.random.choice(trials_per_block, target_trials, replace=False)
                            random_indices_block4 = np.random.choice(trials_per_block, target_trials, replace=False)
                            random_indices_block5 = np.random.choice(trials_per_block, target_trials, replace=False)

                            ts_tmp[:, :, 0:108, 0] = ts_tmp2[:, :, :, 0]
                            ts_tmp[:, :, 108:216, 0] = ts_tmp2[:, :, :, 1]
                            ts_tmp[:, :, 0:target_trials, 1] = ts_tmp2[:, :, random_indices_block2, 2]
                            ts_tmp[:, :, target_trials:2 * target_trials, 1] = ts_tmp2[:, :, random_indices_block3, 3]
                            ts_tmp[:, :, 2 * target_trials:3 * target_trials, 1] = ts_tmp2[:, :, random_indices_block4,
                                                                                   4]
                            ts_tmp[:, :, 3 * target_trials:4 * target_trials, 1] = ts_tmp2[:, :, random_indices_block5,
                                                                                   5]
                        else:
                            ts_tmp[:, :, 0:108, 0] = ts_tmp2[:, :, :, 2]
                            ts_tmp[:, :, 108:216, 0] = ts_tmp2[:, :, :, 3]
                            ts_tmp[:, :, 0:108, 1] = ts_tmp2[:, :, :, 4]
                            ts_tmp[:, :, 108:216, 1] = ts_tmp2[:, :, :, 5]

                        N = ts_tmp.shape[0]  # number of channels
                        n_trials = ts_tmp.shape[2]  # number of trials per block
                        T = 1200  # trial duration

                        # discard silent channels
                        invalid_ch1 = np.logical_or(np.abs(ts_tmp[:, :, :, 0]).max(axis=(1, 2)) == 0,
                                                    np.isnan(ts_tmp[:, 0, 0, 0]))
                        invalid_ch2 = np.logical_or(np.abs(ts_tmp[:, :, :, 1]).max(axis=(1, 2)) == 0,
                                                    np.isnan(ts_tmp2[:, 0, 0, 1]))
                        invalid_ch = np.logical_or(invalid_ch1, invalid_ch2)
                        # invalid_ch = np.logical_or(np.abs(ts_tmp).max(axis=(1,2,3))==0, np.isnan(ts_tmp[:,0,0,0]))
                        valid_ch = np.logical_not(invalid_ch)
                        ts_tmp = ts_tmp[valid_ch, :, :, :]
                        N = valid_ch.sum()

                        # get time series for each block
                        ts = np.zeros([n_motiv, n_trials, T, N])
                        for i_motiv in range(n_motiv):
                            for i_trial in range(n_trials):
                                # swap axes for time and channels
                                ts[i_motiv, i_trial, :, :] = ts_tmp[:, :, i_trial, i_motiv].T

                        del ts_tmp  # clean memory

                        ############
                        # get common sources (from calc_common_ICAs.py)
                        all_data_src_gp = np.load(path_dir_src + 'all_data_src_gp_' + freq_bands[i_band] + '.npy',
                                                  allow_pickle=True)
                        n_gp = len(all_data_src_gp)

                        select_src = np.zeros([N], dtype=bool)
                        corresp_gp = -np.ones([N], dtype=int)
                        for i_gp in range(n_gp):
                            for i_src in range(len(all_data_src_gp[i_gp])):
                                if all_data_src_gp[i_gp][i_src, 0] == i_sub and all_data_src_gp[i_gp][i_src, 1] == i_motion:
                                    select_src[all_data_src_gp[i_gp][i_src, 2]] = True
                                    corresp_gp[all_data_src_gp[i_gp][i_src, 2]] = i_gp  # store corresponding group

                        ts = ts[:, :, :, select_src]
                        corresp_gp = corresp_gp[select_src]
                        N = select_src.sum()  # new number of channels

                        print('subject, motion', subjects[i_sub], i_motion, ': nb of selected sources', N)
                        print(corresp_gp)

                        ############
                        # process time series in desired frequency band

                        # band-pass filtering (alpha, beta, gamma)
                        n_order = 3
                        sampling_freq = 500.  # sampling rate
                        nyquist_freq = sampling_freq / 2.

                        if freq_band == 'alpha':
                            low_f = 8. / nyquist_freq
                            high_f = 12. / nyquist_freq
                        elif freq_band == 'beta':
                            # beta
                            low_f = 15. / nyquist_freq
                            high_f = 30. / nyquist_freq
                        elif freq_band == 'gamma':
                            # gamma
                            low_f = 40. / nyquist_freq
                            high_f = 80. / nyquist_freq
                        else:
                            raise NameError('unknown filter')

                        # apply filter
                        b, a = spsg.iirfilter(n_order, [low_f, high_f], btype='bandpass', ftype='butter')
                        filtered_ts = spsg.filtfilt(b, a, ts, axis=2)


                        ############
                        # calculate vector features
                        # create the design matrix [samples,features]
                        vect_features = np.abs(filtered_ts).mean(axis=2)

                        ############
                        # perform classification

                        # labels of sessions for classification (train+test)
                        labels = np.zeros([n_motiv, n_trials], dtype=int)  # 0 = M0, 1 = M1
                        labels[1, :] = 1

                        # vectorize dimensions motivation levels and trials
                        mask_motiv_trials = np.logical_not(np.isnan(vect_features[:, :, 0]))
                        vect_features = vect_features[mask_motiv_trials, :]
                        labels = labels[mask_motiv_trials]

                        # repeat classification for several splits for indices of sliding windows (train/test sets)
                        for i_rep in range(n_rep):
                            for ind_train, ind_test in cv_schem.split(vect_features, labels):  # false loop, just 1
                                # train and test for original data
                                c_MLR.fit(vect_features[ind_train, :], labels[ind_train])
                                perf[i_sub, i_motion, i_band, i_rep] = c_MLR.score(vect_features[ind_test, :], labels[ind_test])
                                conf_matrix[i_sub, i_motion, i_band, i_rep, :, :] += skm.confusion_matrix(
                                    y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test, :]))

                                # shuffled performance distributions
                                shuf_labels = np.random.permutation(labels)

                                c_MLR.fit(vect_features[ind_train, :], shuf_labels[ind_train])
                                perf_shuf[i_sub, i_motion, i_band, i_rep] = c_MLR.score(vect_features[ind_test, :],
                                                                                        shuf_labels[ind_test])

                        ############
                        # end of loop
                        is_valid[i_sub, i_motion] = True

                        print('subject, motion, space, session:', i_sub, i_motion)
                    except Exception as err:
                        print('bad session:', i_sub, i_motion, err)

        # %% figure

        # approximate theoretical chance level
        chance_level = 0.66

        pp.figure(figsize=[4, 2])
        pp.axes([0.15, 0.2, 0.8, 0.7])

        n_valid = is_valid.sum()

        # pool performance for all valid sessions (subject, motion, session)
        perf_tmp = perf[is_valid, :, :]
        perf_shuf_tmp = perf_shuf[is_valid, :, :]
        # reshape to have 3 bands together for each classifier
        perf_tmp2 = np.zeros([n_valid, n_rep, 3])
        for i_band in range(n_bands):
            perf_tmp2[:, :, i_band] = perf_tmp[:, i_band, :]
        perf_shuf_tmp2 = np.zeros([n_valid, n_rep, 3])
        for i_band in range(n_bands):
            perf_shuf_tmp2[:, :, i_band] = perf_shuf_tmp[:, i_band, :]

        np.save(work_dir + 'perf_real_signals_' + motiv + '.npy', perf)
        np.save(work_dir + 'conf_real_signals_' + motiv + '.npy', conf_matrix)

    finally:
        return perf, conf_matrix

colors_acc = {
    0: ["#419ede", "#144c73"],  # Different shades of blue
    1: ["#4bce4b", "#1c641c"],  # Different shades of green
    2: ["#e36657", "#951b1c"]  # Different shades of red
}
cmap_colors = ['Blues','Greens','Oranges']

for motiv in motivs:
    perf_envelope_signals, conf_envelope_signals = classification_envelope_signals(motiv)
    perf_real_signals, conf_real_signals = classification_real_signals(motiv)

    fig_acc, axs_acc = pp.subplots(figsize=(14, 8), nrows = 2, ncols = 1)
    fig_acc.suptitle('Classifier MLR with motiv ' + motiv)
    x_positions = np.arange(2*n_bands)
    fig_conf, axs_conf = pp.subplots(figsize=(14,8), nrows=2, ncols=6)
    fig_conf.suptitle('Classifier MLR with motiv ' + motiv)
    fig_conf.tight_layout(pad=3.0)  # Increase padding
    fig_conf.subplots_adjust(hspace=0.4, wspace=0.4)
    for i_motion, motion in enumerate(type_motions):
        ax_acc = axs_acc[i_motion]
        ax_acc.axis(ymin=0, ymax=1.05)
        ax_acc.set_title('Accuracies with control type ' + motion)
        ax_acc.plot([i for i in range(-1, 2 * n_bands + 1)], [0.5] * (2 * n_bands + 2), '--k')
        ax_acc.set_xlim(-1, 6)
        list_axes = list()

        for i_band, band in enumerate(freq_bands):
            list_axes.append(band + ', envelope')
            list_axes.append(band + ', real')

            if perf_envelope_signals.shape[0] > 1:
                means = np.mean(perf_envelope_signals[:, i_motion, i_band, :], axis=1)
            else:
                means = np.mean(perf_envelope_signals[:, i_motion, i_band, :],axis=0)
            non_zero_mask = means != 0
            vp = ax_acc.violinplot(means[non_zero_mask], positions=[x_positions[2 * i_band]])
            color_group = i_band
            color_index = 0
            color = colors_acc[color_group][color_index]
            vp['bodies'][0].set_facecolor(color)
            vp['bodies'][0].set_alpha(0.6)
            vp['cmins'].set_color(color)
            vp['cmaxes'].set_color(color)
            vp['cbars'].set_color(color)

            ax_conf = axs_conf[i_motion, i_band*2]
            if conf_envelope_signals.shape[0] > 1:
                mean_reps = np.mean(conf_envelope_signals[:, i_motion, i_band, :,:,:], axis = 1)
            else:
                mean_reps = np.mean(conf_envelope_signals[:, i_motion, i_band, :, :, :], axis=0)
            ax_conf.imshow(np.mean(mean_reps[non_zero_mask, :,:], axis = 0), vmin=0, cmap=cmap_colors[i_band])
            ax_conf.set_xlabel('true label', fontsize=8)
            ax_conf.set_ylabel('predicted label', fontsize=8)
            ax_conf.set_title("Band " + band + " for envelope signals", fontsize=9)

            if perf_real_signals.shape[0] > 1:
                means = np.mean(perf_real_signals[:, i_motion, i_band, :], axis=1)
            else:
                means = np.mean(perf_real_signals[:, i_motion, i_band, :], axis=0)
            non_zero_mask = means != 0
            vp = ax_acc.violinplot(means[non_zero_mask], positions=[x_positions[2 * i_band + 1]])
            color_group = i_band
            color_index = 1
            color = colors_acc[color_group][color_index]
            vp['bodies'][0].set_facecolor(color)
            vp['bodies'][0].set_alpha(0.6)
            vp['cmins'].set_color(color)
            vp['cmaxes'].set_color(color)
            vp['cbars'].set_color(color)

            ax_conf = axs_conf[i_motion, i_band * 2 + 1]
            if conf_real_signals.shape[0] > 1:
                mean_reps = np.mean(conf_real_signals[:, i_motion, i_band, :, :, :], axis=1)
            else:
                mean_reps = np.mean(conf_real_signals[:, i_motion, i_band, :, :, :], axis=0)
            ax_conf.imshow(np.mean(mean_reps[non_zero_mask, :, :], axis=0), vmin=0, cmap=cmap_colors[i_band])
            ax_conf.set_xlabel('true label', fontsize=8)
            ax_conf.set_ylabel('predicted label', fontsize=8)
            ax_conf.set_title("Band " + band + " for real signals", fontsize=9)

        ax_acc.set_xticks([i for i in range(n_bands * 2)], list_axes)
        ax_acc.set_ylabel('accuracy ' + motiv, fontsize=8)
    fig_acc.savefig(work_dir + 'accuracy_MLR_signals_' + motiv + '.' + fmt_grph,
                    format=fmt_grph)
    fig_acc.savefig(work_dir + 'accuracy_MLR_signals_' + motiv + '.' + fmt_grph,
                    format = fmt_grph)
    fig_conf.savefig(work_dir + 'conf_matrix_MLR_signals_' + motiv + '.' + fmt_grph,
                    format=fmt_grph)
    fig_conf.savefig(work_dir + 'conf_matrix_MLR_signals_' + motiv + '.' + fmt_grph,
                    format=fmt_grph)
    pp.close(fig_acc)
    pp.close(fig_conf)


