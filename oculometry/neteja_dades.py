import gc
import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from json_functions import load_data, save_data
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

tTime = dict()
pupilL = dict()
pupilR = dict()
dades_assajos = dict()

"""
Funció que interpola els valors nan d'una matriu numpy.
"""
def interpolate_nans(data):
    # Creem un array d'índex de les columnes
    x = np.arange(data.shape[1])
    # Iterem per les files
    for i in range(data.shape[0]):
        # Obtenim la fila i-éssima
        y = data[i, :]
        # Obtenim una màscara de booleans per saber en quins llocs no hi ha nan's
        mask = np.isfinite(y)
        #Veiem si pel menys tenim dos valors per poder fer la interpolació
        if np.sum(mask) > 1:
            #Els llocs que són de nan's els omplim de valors interpolats
            y[~mask] = interp1d(x[mask], y[mask], kind='linear', fill_value='extrapolate')(x[~mask])
    return data


"""
Funció que troba la mitjana i la desviació estàndard dels blocs de control
de cada pacient.
En aquesta funció es retornen les dades necessàries per poder fer l'estandarització
de les dades.
"""
def get_std_mean_control(tTime, pupilL, pupilR, dades_assajos, temps_assaig, max_time, interval_temps):
    dict_data = dict()
    assajos_per_bloc = 108

    for pacient in tTime.keys():
        pd = None
        #Variable per saber quan està medicat o no el pacient
        dict_meds = dict()
        dict_data[pacient] = dict()
        width = temps_assaig // interval_temps
        # En aquestes matrius guardarem les dades pels blocs 1 i 2 de les dues sessions
        left_control = np.full((assajos_per_bloc * 2 * 2, width), np.nan)
        right_control = np.full((assajos_per_bloc * 2 * 2, width), np.nan)

        sessions = 0

        for idSessio in tTime[pacient].keys():
            sessions += 1
            dict_data[pacient][idSessio] = dict()

            for bloc in ['1', '2']:
                if bloc in tTime[pacient][idSessio].keys():
                    for assaig in tTime[pacient][idSessio][bloc].keys():
                        if pd is None:
                            pd = dades_assajos[pacient][idSessio][bloc][assaig]['PD']

                        if pd and idSessio not in dict_meds.keys():
                            meds = dades_assajos[pacient][idSessio][bloc][assaig]['MedsOn']
                            dict_meds[idSessio] = meds

                        # Obtenim el número de l'assaig dins del bloc (0-107)
                        num_assaig = (int(assaig) - 1) % 108

                        # Obtenim les dades del pacient
                        list_pupilR = pupilR[pacient][idSessio][bloc][assaig]
                        list_pupilL = pupilL[pacient][idSessio][bloc][assaig]
                        list_time = tTime[pacient][idSessio][bloc][assaig]

                        # L'assaig començara 500 ms abans del tOriginShow de l'assaig
                        origen_assaig = int(dades_assajos[pacient][idSessio][bloc][assaig]['tOriginShow']) - 500

                        # Variables per saber quant de temps ha passat sense haver-hi un valor
                        # esperat de la pupil·la en l'assaig
                        previous_time_left = 0
                        previous_time_right = 0

                        # Booleà per saber si s'ha d'eliminar l'assaig o no
                        delete_assaig = False

                        num_temps_interval_left = 0
                        num_temps_interval_right = 0
                        sum_pupilL = 0
                        sum_pupilR = 0
                        """
                        Bucle que serveix per veure si l'assaig s'ha d'eliminar o no.
                        Un assaig s'elimina si durant un cert rang de temps les mesures
                        obtingudes de la pupil·la són 0.
                        """
                        for i in range(len(list_time)):
                            left = (list_pupilL[i])
                            right = (list_pupilR[i])
                            time = list_time[i] - origen_assaig

                            # Considerem que l'assaig comença en el temps origen_assaig
                            if time >= 0:
                                # Considerem que l'assaig ja ha acabat
                                if time >= temps_assaig:
                                    # Eliminem l'assaig en cas que al final d'aquest no hi hagi dades
                                    if temps_assaig - max(previous_time_left, previous_time_right) > max_time:
                                        delete_assaig = True
                                    break

                                # Veiem si s'hauria d'eliminar l'assaig perquè no hi ha prou dades
                                # durant un interval de temps major a max_time
                                if time - max(previous_time_left, previous_time_right) > max_time:
                                    delete_assaig = True

                                # Volem les mesures que tenen algun valor
                                if left > 0.1:
                                    # Si hi ha més d'una mesura de pupil·la que pertanyi a l'interval, les sumem i fem la mitjana
                                    if previous_time_left != 0 and int(time//interval_temps - previous_time_left//interval_temps) == 0:
                                        num_temps_interval_left +=1
                                        sum_pupilL += left
                                        left_control[num_assaig + (int(bloc) - 1 + (int(idSessio) - 1)*2) * assajos_per_bloc, time // interval_temps] = sum_pupilL / num_temps_interval_left

                                    else:
                                        num_temps_interval_left = 1
                                        sum_pupilL = left
                                        left_control[num_assaig + (int(bloc) - 1 + (int(idSessio) - 1)*2) * assajos_per_bloc, time // interval_temps] = left

                                    previous_time_left = time

                                if right > 0.1:
                                    # Si hi ha més d'una mesura de pupil·la que pertanyi a l'interval, les sumem
                                    if previous_time_right != 0 and int(time // interval_temps - previous_time_right // interval_temps) == 0:
                                        num_temps_interval_right += 1
                                        sum_pupilR += right
                                        right_control[num_assaig + (int(bloc) - 1  + (int(idSessio) - 1)*2) * assajos_per_bloc, time // interval_temps] = sum_pupilR / num_temps_interval_right

                                    else:
                                        num_temps_interval_right = 1
                                        sum_pupilR = right
                                        right_control[num_assaig + (int(bloc) - 1 + (int(idSessio) - 1)*2) * assajos_per_bloc, time // interval_temps] = right

                                    previous_time_right = time

                        # En cas d'eliminar l'assaig, es posen totes les dades d'aquest a nan
                        if delete_assaig:
                            left_control[num_assaig + (int(bloc) - 1 + (int(idSessio) - 1)*2)*assajos_per_bloc, :] = np.nan
                            right_control[num_assaig + (int(bloc) - 1 + (int(idSessio) - 1)*2) * assajos_per_bloc, :] = np.nan

        """
        Trobem les dades necessàries per poder estandaritzar la pupil·la.
        """
        if pd and sessions > 1:
            std_total_left = 0
            std_total_right = 0
            mean_total_left = 0
            mean_total_right = 0
            substraction_mean_left = 0
            substraction_mean_right = 0

            for idSessio in tTime[pacient].keys():
                # Calculem la mitjana i la desviació estàndard per poder estandaritzar
                mean_left = np.nanmean(
                    left_control[(int(idSessio) - 1) * 2 * assajos_per_bloc: int(idSessio) * 2 * assajos_per_bloc, :25])
                std_left = np.nanstd(
                    left_control[(int(idSessio) - 1) * 2 * assajos_per_bloc: int(idSessio) * 2 * assajos_per_bloc, :25])
                dict_data[pacient][idSessio]['left'] = {'mean': mean_left, 'std': std_left, 'offset': 0}
                std_total_left += std_left/2
                mean_total_left += mean_left/2

                mean_right = np.nanmean(
                    right_control[(int(idSessio) - 1) * 2 * assajos_per_bloc: int(idSessio) * 2 * assajos_per_bloc,
                    : 25])
                std_right = np.nanstd(
                    right_control[(int(idSessio) - 1) * 2 * assajos_per_bloc: int(idSessio) * 2 * assajos_per_bloc,
                    : 25])
                dict_data[pacient][idSessio]['right'] = {'mean': mean_right, 'std': std_right, 'offset': 0}
                std_total_right += std_right/2
                mean_total_right += mean_right/2

                # Si en la sessió el pacient estava medicat sumem la mitjana
                if dict_meds[idSessio]:
                    substraction_mean_left += mean_left
                    substraction_mean_right += mean_right
                # Si no, la restem
                else:
                    substraction_mean_left -= mean_left
                    substraction_mean_right -= mean_right

            for idSessio in tTime[pacient].keys():
                dict_data[pacient][idSessio]['left']['std'] = std_total_left
                dict_data[pacient][idSessio]['right']['std'] = std_total_right
                if dict_meds[idSessio]:
                    dict_data[pacient][idSessio]['left']['offset'] = substraction_mean_left/(2*mean_total_left)
                    dict_data[pacient][idSessio]['right']['offset'] = substraction_mean_right/(2*mean_total_right)
                else:
                    dict_data[pacient][idSessio]['left']['offset'] = -substraction_mean_left / (2 * mean_total_left)
                    dict_data[pacient][idSessio]['right']['offset'] = -substraction_mean_right / (2 * mean_total_right)

        else:
            for idSessio in tTime[pacient].keys():
                # Calculem la mitjana i la desviació estàndard per poder estandaritzar
                mean_left = np.nanmean(left_control[(int(idSessio)-1)*2*assajos_per_bloc: int(idSessio)*2*assajos_per_bloc, :25])
                std_left = np.nanstd(left_control[(int(idSessio)-1)*2*assajos_per_bloc: int(idSessio)*2*assajos_per_bloc, :25])
                dict_data[pacient][idSessio]['left'] = {'mean': mean_left, 'std': std_left, 'offset': 0}

                mean_right = np.nanmean(right_control[(int(idSessio)-1)*2*assajos_per_bloc: int(idSessio)*2*assajos_per_bloc, :25])
                std_right = np.nanstd(right_control[(int(idSessio)-1)*2*assajos_per_bloc: int(idSessio)*2*assajos_per_bloc, :25])
                dict_data[pacient][idSessio]['right'] = {'mean': mean_right, 'std': std_right, 'offset': 0}

    return dict_data


"""
Funció per mostrejar la mitjana de la pupil·la de cada pacient.
"""
def show_data_mean_trial(pacient, idSessio, left_pupil_data, right_pupil_data, blocks):
    assajos_per_bloc = 108
    num_trials = [i for i in range(1, 109)]
    fig1, axes1 = plt.subplots(3, 2, figsize=(10, 15))
    fig1.suptitle(f'Patient {pacient}, Session {idSessio}, Left Pupil')
    fig2, axes2 = plt.subplots(3, 2, figsize=(10, 15))
    fig2.suptitle(f'Patient {pacient}, Session {idSessio}, Right Pupil')

    i  = 0
    for block in blocks:
        ax1 = axes1[i // 2, i % 2]
        ax1.set_title(f'Block {block}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pupil Size')

        ax2 = axes2[i // 2, i % 2]
        ax2.set_title(f'Block {block}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pupil Size')

        ax1.plot(num_trials, list(
            np.nanmean(left_pupil_data[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :25], axis=1)))
        ax2.plot(num_trials, list(
            np.nanmean(right_pupil_data[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :25], axis=1)))

        i += 1

    if not os.path.exists('img_mean_trial/'):
        os.makedirs('img_mean_trial/')

    file_img_left = "img_mean_trial/pacient" + pacient + "_sessio" + idSessio + "_left.png"
    file_img_right = "img_mean_trial/pacient" + pacient + "_sessio" + idSessio + "_right.png"

    fig1.savefig(file_img_left)
    fig2.savefig(file_img_right)

    plt.close(fig1)
    plt.close(fig2)
    gc.collect()


"""
Funció que mostra per cada pacient una figura amb les mides de la pupil·la
de la següent forma:
    - Per cada fila hi ha quatre gràfiques associades a un bloc.
    - En les quatre gràfiques es mostra: les mides de la pupil·la de la dreta,
    la de l'esquerra, la pupil·la total (fent la mitjana de les dues) i
    la pupil·la total interpolada.
"""
def show_standarized_data(pacient, idSessio, left_pupil_data, right_pupil_data, pupil_data, pupil_data_interp, blocks):
    fig, axes = plt.subplots(6, 4, figsize=(25, 35))
    fig.suptitle(f'Patient {pacient}, Session {idSessio}')

    i = 0
    assajos_per_bloc = 108
    for block in blocks:
        ax1 = axes[i, 0]
        ax1.set_title(f' Left data, Block {block}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pupil Size')

        ax2 = axes[i, 1]
        ax2.set_title(f' Right data, Block {block}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pupil Size')

        ax3 = axes[i, 2]
        ax3.set_title(f' Combined data, Block {block}')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Pupil Size')

        ax4 = axes[i, 3]
        ax4.set_title(f' Interpolated Combined data, Block {block}')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Pupil Size')

        ax1.imshow(left_pupil_data[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :] + 2, vmin=0, vmax=4, aspect='auto')
        ax2.imshow(right_pupil_data[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :] + 2, vmin=0, vmax=4, aspect='auto')
        ax3.imshow(pupil_data[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :] + 2, vmin=0, vmax=4, aspect='auto')
        ax4.imshow(pupil_data_interp[i * assajos_per_bloc:(i + 1) * assajos_per_bloc, :] + 2, vmin=0, vmax=4, aspect='auto')

        i += 1

    file_img = "img/pacient" + pacient + "_sessio" + idSessio + ".png"
    fig.savefig(file_img)
    plt.close(fig)
    gc.collect()


"""
Funció que guarda una taula amb les estadístiques dels assajos
bons de cada pacient
"""
def show_assajos_bons(assajos_bons):
    # Creem una taula amb el percentatge d'assaig no eliminats de cada pacient i la guardem.
    results = pd.DataFrame(assajos_bons)

    fig, ax = plt.subplots(figsize=(12, len(results) * 0.4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=results.values, colLabels=results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig("results/assajos_bons.png", bbox_inches='tight', dpi=300)
    plt.close()


if not os.path.exists('img'):
    os.makedirs('img')

if not os.path.exists('img_mean_trial'):
    os.makedirs('img_mean_trial')

# Carreguem les dades prèviament guardades en fitxers text
tTime = load_data("dades_docs/" + "tTime")
pupilL = load_data("dades_docs/" + "pupilL")
pupilR = load_data("dades_docs/" + "pupilR")
dades_assajos = load_data("dades_docs/" + "dades_assajos")

temps_assaig = 1500
assajos_per_bloc = 108
# Variable que fixa cada quan anotem el valor de la pupil·la
interval_temps = 20
# Fixem el temps màxim que permetem que no hi hagi un valor coherent de la pupil·la en un assaig
max_time = 250

# Obtenim les dades dels blocs de control (bloc 1 i 2) per poder estandaritzar la pupil·la
control_data = get_std_mean_control(tTime, pupilL, pupilR, dades_assajos, temps_assaig, max_time, interval_temps)

# Diccionari per guardar les dades necessàries per fer la regressió lineal
mean_pacient = dict()

assajos_bons = {'Pacient': list(), 'Assajos': list(), 'Rati assajos bons / fets': list(), 'Rati assajos bons / 1296': list()}

for pacient in tTime.keys():
    num_assajos_bons = 0
    assajos = 0
    sessions = 0

    mean_pacient[pacient] = {'pupil': list(), 'MedsOn': list(), 'nBlock': list(),
                            'SF': list(), 'SP': list(), 'PD': list(), 'CT': list()}

    data_pacient = mean_pacient[pacient]

    for idSessio in tTime[pacient].keys():
        sessions += 1
        # Agafem les dades per normalitzar
        mean_left = control_data[pacient][idSessio]['left']['mean']
        std_left = control_data[pacient][idSessio]['left']['std']
        offset_left = control_data[pacient][idSessio]['left']['offset']
        mean_right = control_data[pacient][idSessio]['right']['mean']
        std_right = control_data[pacient][idSessio]['right']['std']
        offset_right = control_data[pacient][idSessio]['right']['offset']


        width = temps_assaig // interval_temps
        # Aquí tindrem les dades de les pupil·les de tots els blocs d'una sessió
        left_pupil_data = np.full((assajos_per_bloc*6, width), np.nan)
        right_pupil_data = np.full((assajos_per_bloc*6, width), np.nan)

        block_counter = 0

        for bloc in tTime[pacient][idSessio].keys():
            assajos += len(dades_assajos[pacient][idSessio][bloc].keys())

            for assaig in tTime[pacient][idSessio][bloc].keys():
                # Obtenim el número de l'assaig dins del bloc (0-107)
                num_assaig = (int(assaig) - 1)% 108
                index_matrix = num_assaig + block_counter * assajos_per_bloc

                # Obtenim les dades del pacient
                list_pupilR = pupilR[pacient][idSessio][bloc][assaig]
                list_pupilL = pupilL[pacient][idSessio][bloc][assaig]
                list_time = tTime[pacient][idSessio][bloc][assaig]

                # L'assaig començara 500 ms abans del tOriginShow de l'assaig
                origen_assaig = int(dades_assajos[pacient][idSessio][bloc][assaig]['tOriginShow']) - 500

                value_right = 0
                value_left = 0
                last_ctt_value_right = 0
                last_ctt_value_left = 0
                ctt_time_right = 0
                ctt_time_left = 0

                # Variables per saber quant de temps ha passat sense haver-hi un valor
                # esperat de la pupil·la en l'assaig
                previous_time_left = 0
                previous_time_right = 0

                # Booleà per saber si s'ha d'eliminar l'assaig o no
                delete_assaig = False

                num_temps_interval_left = 0
                num_temps_interval_right = 0
                sum_pupilL = 0
                sum_pupilR = 0
                for i in range(len(list_time)):
                    left = (list_pupilL[i])
                    right = (list_pupilR[i])
                    time = list_time[i] - origen_assaig

                    # Considerem que l'assaig comença en el temps origen_assaig
                    if time >= 0:
                        # Considerem que l'assaig ja ha acabat
                        if time >= temps_assaig:
                            # Eliminem l'assaig en cas que al final d'aquest no hi hagi dades
                            if temps_assaig - max(previous_time_left, previous_time_right) > max_time:
                                delete_assaig = True
                            if ctt_time_right > 1400 or ctt_time_left > 1400:
                                delete_assaig = True
                            break

                        # Veiem si s'hauria d'eliminar l'assaig perquè no hi ha prou dades
                        # durant un interval de temps major a max_time
                        if time - max(previous_time_left, previous_time_right) > max_time:
                            delete_assaig = True

                        # Només dibuixarem en les gràfiques els valors que estiguin
                        # dins de l'interval mean +- 2*std
                        if mean_left - 2*std_left <= left <= mean_left + 2*std_left:
                            # Si hi ha més d'una mesura de pupil·la que pertanyi a l'interval,
                            # les sumem i fem la mitjana
                            if (previous_time_left != 0 and int(time // interval_temps - previous_time_left // interval_temps) == 0):
                                num_temps_interval_left += 1
                                sum_pupilL += left
                                left_pupil_data[index_matrix, time // interval_temps] = (
                                        sum_pupilL / num_temps_interval_left)

                            else:
                                num_temps_interval_left = 1
                                sum_pupilL = left
                                left_pupil_data[index_matrix, time // interval_temps] = left

                            if abs(value_left - left) > 0.1:
                                value_left = left
                                ctt_time_left = 0

                            else:
                                ctt_time_left += time - last_ctt_value_left

                            last_ctt_value_left = time
                            previous_time_left = time

                        if mean_right - 2*std_right <= right <= mean_right + 2*std_right:
                            # Si hi ha més d'una mesura de pupil·la que pertanyi a l'interval, les sumem
                            if previous_time_right != 0 and int(time // interval_temps - previous_time_right // interval_temps) == 0:
                                num_temps_interval_right += 1
                                sum_pupilR += right
                                right_pupil_data[index_matrix, time // interval_temps] = sum_pupilR / num_temps_interval_right

                            else:
                                num_temps_interval_right = 1
                                sum_pupilR = right
                                right_pupil_data[index_matrix, time // interval_temps] = right

                            if abs(value_right - right) > 0.1:
                                value_right = right
                                ctt_time_right = 0
                            else:
                                ctt_time_right += time - last_ctt_value_right

                            previous_time_right = time
                            last_ctt_value_right = time

                # Ara, si hem d'eliminar un assaig posem la fila de l'assaig a nan's
                if delete_assaig:
                    left_pupil_data[index_matrix, :] = np.nan
                    right_pupil_data[index_matrix, :] = np.nan
                else:
                    data_pacient['MedsOn'].append(
                        int(dades_assajos[pacient][idSessio][bloc][assaig]['MedsOn']))
                    data_pacient['SF'].append(int(dades_assajos[pacient][idSessio][bloc][assaig]['SF']))
                    data_pacient['SP'].append(int(dades_assajos[pacient][idSessio][bloc][assaig]['SP']))
                    data_pacient['nBlock'].append(int(bloc))
                    data_pacient['PD'].append(int(dades_assajos[pacient][idSessio][bloc][assaig]['PD']))
                    data_pacient['CT'].append(
                        int(dades_assajos[pacient][idSessio][bloc][assaig]['nControlLevel']))

                    num_assajos_bons += 1

            block_counter += 1

        show_data_mean_trial(pacient, idSessio, left_pupil_data, right_pupil_data, tTime[pacient][idSessio].keys())

        # Estandaritzem les dades de les pupil·les segons les dades del bloc 1 i 2
        # Agafem les dades dels primers 500 ms (500/20 = 25) de cada assaig
        if std_left > 0.05:
            left_pupil_data[:, :] = (left_pupil_data[:, :] - mean_left) / std_left + offset_left
        else:
            left_pupil_data[:, :] = left_pupil_data[:, :] - mean_left + offset_left

        if std_right > 0.05:
            right_pupil_data[:, :] = (right_pupil_data[:, :] - mean_right) / std_right + offset_right
        else:
            right_pupil_data[:, :] = right_pupil_data[:, :] - mean_right + offset_right

        pupil_data = np.nanmean(np.stack((left_pupil_data, right_pupil_data), axis=-1), axis=-1)
        pupil_data_interp = pupil_data.copy()
        pupil_data_interp = interpolate_nans(pupil_data_interp)
        show_standarized_data(pacient, idSessio, left_pupil_data, right_pupil_data, pupil_data, pupil_data_interp, tTime[pacient][idSessio].keys())

        rows_nans = np.any(np.isnan(pupil_data_interp), axis=1)
        data_pacient['pupil'] += list(np.mean(pupil_data_interp[:, :25][~rows_nans], axis = 1))

    # Guardem les dades per tenir una estadística sobre els assajos bons de cada pacient
    assajos_bons['Pacient'].append(pacient)
    assajos_bons['Assajos'].append(num_assajos_bons)
    if assajos == 0:
        assajos_bons['Rati assajos bons / fets'].append(0)
        assajos_bons['Rati assajos bons / 1296'].append(0)
    else:
        assajos_bons['Rati assajos bons / fets'].append(round(num_assajos_bons / assajos, 3))
        assajos_bons['Rati assajos bons / 1296'].append(round(num_assajos_bons / (assajos_per_bloc*12), 3))

    mean_pacient[pacient]['sessions'] = sessions
    if len(data_pacient['pupil']) < 2:
        del mean_pacient[pacient]

show_assajos_bons(assajos_bons)

# Guardem en un fitxer les dades necessàries per fer la regressió lineal
save_data('dades_docs/' + "mitjana_assajos", mean_pacient)


