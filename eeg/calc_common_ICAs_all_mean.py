import os
import numpy as np
import scipy.stats as stt
import scipy.io as sio
import matplotlib.pyplot as pp
import matplotlib.patches as ppt

from itertools import combinations

# path_dir_res = '/results/'
path_dir_src = '/common_src_'
patients = {'sans': list(range(25,35)), 'ON': [55, 60, 61, 64], 'OFF': [55, 60, 61, 64]}
folders = {'sans': './pacients_sans', 'ON': './pacients_PD', 'OFF': './pacients_PD'}
freq_bands = ['alpha','beta','gamma']
n_bands = len(freq_bands)
m = 101 # number of spatial steps
common_ind = np.ones([m,m], dtype=bool)

for variable in ['SF', 'SP']:
    work_dir_results = './common_src_mean_' + variable + '/'
    if not os.path.exists(work_dir_results):
        os.mkdir(work_dir_results)
    groups = ['sans', 'ON', 'OFF']
    subgroups = list(combinations(groups, 2))
    for group1, group2 in subgroups:
        print(group1+ ', ' + group2)
        work_dir = work_dir_results + 'common_src_' + variable + '_' + group1 + '_' + group2 + '/'
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        #%% global parameters
        n_motiv = 2
        for i_band in range(n_bands):
            print("Band", freq_bands[i_band])
            mean_best_src_list = list()
            sub_best_src_list = list()

            for i_sub, subject in enumerate([group1, group2]):
                extension = ''
                if subject != 'sans':
                    extension = '_' + subject
                folder = folders[subject] + path_dir_src + variable + extension + '/'
                for i_gp in range (8):
                    try:
                        # Carreguem les fonts de tipus de pacient
                        mean_best_src_list.append(np.load(folder + 'mean_best_src_' + freq_bands[i_band] + '_gp' + str(i_gp) + '.npy'))
                        sub_best_src_list.append(i_sub)

                    except:
                        print("The source " + str(i_gp) + " doesn't exist for the group " + subject + " of patients")


            def sim_comp(c1, c2):
                # rectify components (more localized positive part)
                mask_c1 = np.zeros(c1.shape)
                # Mirem si hi ha més valors positius que negatius
                if np.sum(c1 > 0) > np.sum(c1 < 0):
                    mask_c1[c1 < 0] = -1
                else:
                    mask_c1[c1 > 0] = 1
                # En el primer cas, tindrem que c1_rectif té els valors positius a 0 i els negatius ara són positius.
                # En el segon cas, els negatius seran 0 i els positius continuaren sent positius.
                c1_rectif = c1 * mask_c1

                mask_c2 = np.zeros(c2.shape)
                if np.sum(c2 > 0) > np.sum(c2 < 0):
                    mask_c2[c2 < 0] = -1
                else:
                    mask_c2[c2 > 0] = 1
                c2_rectif = c2 * mask_c2
                # get all pairings for values exceeding the limit
                # Agafem el statistic de Pearson
                return stt.pearsonr(c1_rectif.flatten(), c2_rectif.flatten())[0]

            # Número total de sources en aquesta banda de freqüència
            n_best_src_tot = len(mean_best_src_list)
            # Matriu on guardarem la similitud entre fonts
            mat_sim_best_src = np.zeros([n_best_src_tot, n_best_src_tot])

            # Iterem per totes les fonts i els comparem dos a dos
            for i1 in range(n_best_src_tot):
                for i2 in range(n_best_src_tot):
                    if not (i1 == i2):
                        # Es guarda l'estatístic de Pearson entre els mapes de les dues fonts
                        mat_sim_best_src[i1, i2] = sim_comp(mean_best_src_list[i1],mean_best_src_list[i2])

            # NEWMAN COMMUNITY DETECTION
            # Matriu amb la similitud entre fonts
            C = np.abs(mat_sim_best_src[:,:])

            C_null = np.outer(C.sum(1),C.sum(0)) / C.sum()
            C_diff = C - C_null

            # Comencem amb una partició de totes les fonts separades
            list_gp = []
            for i in range(n_best_src_tot):
                list_gp += [[i]]
            # Ajuntem comunitats que maximitzen la funció de qualitat
            stop = False
            while not stop:
                n = len(list_gp)
                # Matriu que es quantifica com de semblants són dos comunitats (fonts que ja s'han agrupat)
                delta_Q = np.zeros([n,n])
                for ii in range(n):
                    for jj in range(n):
                        # Mirem si és el mateix grup
                        if ii!=jj:
                            # Per cada parell de fonts dels dos grups mirem la semblança
                            for i in list_gp[ii]:
                                for j in list_gp[jj]:
                                    delta_Q[ii,jj] += C_diff[i,j] + C_diff[j,i]
                        else:
                            delta_Q[ii,jj] = -1e50
                # Ajuntem si la qualitat de mesura augmenta
                if delta_Q.max()>0:
                    # Trobem la fila màxima
                    ii_max = int(np.argmax(delta_Q)/n)
                    # Trobem la columna màxima
                    jj_max = np.argmax(delta_Q) - ii_max*n
                    # Amb ii_max i jj_max tindrem els dos grups que s'han d'unir
                    new_list_gp = []
                    for ii in range(n):
                        if ii!=ii_max and ii!=jj_max:
                            new_list_gp += [list_gp[ii]]
                        elif ii==ii_max:
                            # Ajuntem els grups de sources
                            new_list_gp += [list_gp[ii_max]+list_gp[jj_max]]
                    list_gp = new_list_gp
                else:
                    stop = True

            print('list of groups (indices in matrices of best features):', list_gp)
            print()

            all_data_src_gp = [] # data of all sources belonging to a same community
            correlations = np.zeros(shape=(len(list_gp), n_best_src_tot))
            for i_gp in range(len(list_gp)):
                print('group', i_gp)
                # We get the sources for the group
                ind_gp = list_gp[i_gp]

                # get average map
                src = np.zeros([m,m])
                # Per cada grup considerem les fonts que conté
                for i_cnt, i in enumerate(ind_gp):
                    # load corresponding source
                    src_tmp = mean_best_src_list[i].flatten()
                    # "Estandaritzem" la mostra, ja que els valors negatius o positius poden estar "switched".
                    if np.sum(src_tmp>0)>np.sum(src_tmp<0):
                        switch_coef = -1
                    else:
                        switch_coef = 1
                    src[common_ind] += src_tmp * switch_coef

                # Fem la mitjana de la src obtinguda (és la suma de totes les src d'aquest grup)
                src /= len(ind_gp)

                np.save(work_dir+'mean_best_src_'+freq_bands[i_band]+'_gp'+str(i_gp)+'.npy', src)
                sio.savemat(work_dir+'mean_best_src_'+freq_bands[i_band]+'_gp'+str(i_gp)+'.mat', {'src':src})

                for i_cnt, i in enumerate(ind_gp):
                    # Calculem la similitud entre la source mitjana i la source i
                    correlations[i_gp, i] = sim_comp(src, mean_best_src_list[i])
            np.save(work_dir + 'corr_best_src_' + freq_bands[i_band] + '.npy', src)
            sio.savemat(work_dir + 'corr_best_src_' + freq_bands[i_band] + '.mat', {'corr': correlations})

        