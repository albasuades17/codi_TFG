import numpy as np
import scipy.io as sio
import scipy.signal as spsg
from pyMOU.MOU_model_Testing_Correlation import MOU

sans_dir = '../eeg/pacients_sans'
data_dir = sans_dir + '/EEG-BCN/'
path_dir_src = sans_dir + '/common_src_SF/'
# Canviar el valor aquest segons quins senyals es vulguin simular
motiv = 'SF'
subjects = [25,26,27,28,29,30,31,32,33,34,35]
n_sub = len(subjects)

type_motions = ['NS', 'S']
n_motions = len(type_motions)

type_signal = 'ICA'  # only ICAs

freq_bands = ['alpha', 'beta','gamma']
n_bands = len(freq_bands)

motivs = ['SF', 'SP']
n_motiv = 2

def get_envelopes(data):
    data_tmp = data.copy()
    # Fem Hilbert sobre les dades del trial
    envelopes = spsg.hilbert(data_tmp, axis=1)

    return np.abs(envelopes)

for i_band in range(n_bands):
    freq_band = freq_bands[i_band]

    for i_sub, ind_sub in enumerate(subjects):

        for i_motion, type_motion in enumerate(type_motions):

            try:
                # load ICA time series (same as classif_freq_EEG_general.py)
                try:
                    # TODO: i_band, i_sub... haurien de ser strings!!!
                    generated_rand_signals = np.load(
                        'generated_signals/' + 'generated_rand_signals_' + motiv + '_' + str(i_band) + '_' + str(i_sub) + '_' + str(i_motion) + '.npy')
                    n_trials = generated_rand_signals.shape[1]
                except:
                    if i_motion == 0:  # NS
                        low_ind, top_ind = 0, 6
                    else:  # S
                        low_ind, top_ind = 6, 12

                    ts_tmp2 = sio.loadmat(data_dir + 'dataClean-ICA3-' + str(subjects[i_sub]) + '-T1.mat')['ic_data3'][
                              :, :, :, low_ind:top_ind]

                    trials_per_block = 108
                    ts_tmp = np.zeros([ts_tmp2.shape[0], ts_tmp2.shape[1], 2 * ts_tmp2.shape[2], 2])
                    if motiv == 'SF':
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
                        ts_tmp[:, :, 2 * target_trials:3 * target_trials, 1] = ts_tmp2[:, :, random_indices_block4, 4]
                        ts_tmp[:, :, 3 * target_trials:4 * target_trials, 1] = ts_tmp2[:, :, random_indices_block5, 5]
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
                                                np.isnan(ts_tmp[:, 0, 0, 1]))
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
                    # We get the envelope of the mean signals of a motivation for every source
                    envelope = get_envelopes(np.mean(filtered_ts, axis=1))

                    generated_rand_signals = np.zeros_like(filtered_ts)

                    for i_motiv in range(n_motiv):
                        mou_est = MOU()
                        # Normalize the envelope
                        envelope[i_motiv, :, :] = (envelope[i_motiv, :, :] - np.mean(envelope[i_motiv, :, :],
                                                                                     axis=0)) / np.std(
                            envelope[i_motiv, :, :], axis=0)

                        mou_est.fit(envelope[i_motiv, :, :], regul_C=1., i_tau_opt=15)
                        print('model fitted')

                        for trial in range(generated_rand_signals.shape[1]):
                            generated_rand_signals[i_motiv, trial, :, :] = mou_est.simulate(T=1200, dt=0.002)
                            # Normalize the signal
                            generated_rand_signals[i_motiv, trial, :, :] = (generated_rand_signals[i_motiv, trial, :,
                                                                            :] - np.mean(
                                generated_rand_signals[i_motiv, trial, :, :], axis=0)) / np.std(
                                generated_rand_signals[i_motiv, trial, :, :], axis=0)

                        print('Simulation finished')
                    np.save(
                        'generated_signals/' + 'generated_rand_signals_' + motiv + '_' + str(i_band) + '_' + str(i_sub) + '_' + str(i_motion) + '.npy',
                        generated_rand_signals)
                finally:
                    print(motiv, i_band, i_sub, i_motion)
            except Exception as err:
                print('bad session:', i_sub, i_motion, err)
