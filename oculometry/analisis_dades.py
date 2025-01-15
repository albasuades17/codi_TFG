import os
from json_functions import load_data
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn import linear_model


"""
Funció que fa la regressió lineal i l'anàlisi de les dades
per tots els pacients (Parkinson i sans)
"""
def regressio_lineal():
    mean_pacient = load_data("dades_docs/" + "mitjana_assajos")
    folder = 'results/model_total/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Llista per guardar dades de la regressió com una taula
    results_list = list()
    df_pacients = dict()
    coef_dict = dict()
    coef_variables = {'b0': 'b0', 'b1': 'SF', 'b2': 'SFxSP', 'b3': 'CT', 'b4': 'MedsOn',
                      'b5': 'MedsOnxSF', 'b6': 'MedsOnxSFxSP'}
    pacients = mean_pacient.keys()
    variables = coef_variables.values()
    coefficients = coef_variables.keys()

    for coef in coefficients:
        coef_dict[coef] = list()

    # Per cada pacient, creem un dataframe amb les dades de les variables
    # També obtenim el número de total d'assajos que tenim entre tots els pacients
    num_total_assajos = 0
    for pacient in pacients:
        df_pupil = pd.DataFrame(mean_pacient[pacient])

        # Creem més columnes combinant algunes variables
        df_pupil['SFxSP'] = df_pupil['SF'] * df_pupil['SP']
        df_pupil['MedsOnxSF'] = df_pupil['MedsOn'] * df_pupil['SF']
        df_pupil['MedsOnxSFxSP'] = df_pupil['MedsOn'] * df_pupil['SF'] * df_pupil['SP']

        # Guardem el dataframe amb totes les dades necessàries per després
        df_pacients[pacient] = df_pupil

        num_total_assajos += len(mean_pacient[pacient]['pupil'])

    num_variables = len(variables)

    # Obtenim el número total de variables que tindrem
    # Cada pacient tindrà el seu propi b0 i el bi de cada variable
    # Hi haurà un coeficient que serà global i que anirà acompanyat de PD
    # (és 0 si l'assaig forma part d'un pacient sense Parkinson, 1 si en té)
    num_total_variables = len(pacients) * num_variables

    # Creem una matriu plena de zeros de la mida dels vectors que necessitem
    np_data = np.zeros(shape=(num_total_assajos, num_total_variables), dtype=np.float64)
    np_pupil = np.zeros(shape=(num_total_assajos, 1), dtype=np.float64)

    index_assaig = 0
    index_variable = 0
    # Variable on tindrem les columnes que s'hauries d'eliminar de np_data
    # (casos on MedsOn no tingui sentit)
    columnes_eliminades = list()

    # Variable que servirà per tenir els índex de les dades que posarem a np_data
    list_variables = list()

    # Ara volem que la matriu np_data contingui tota la informació de les variables
    # (serà una matriu diagonal per blocs menys en l'última columna)
    for pacient in pacients:
        len_assajos = len(mean_pacient[pacient]['pupil'])
        # Número de sessions que ha fet cada pacient (pot ser 1 o 2)
        sessions = int(mean_pacient[pacient]['sessions'])
        for coef in variables:
            if coef == 'b0':
                # Com que la variable b0 és el terme independent de la regressió, a la matriu posarem uns
                np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                    np.full(shape=(len_assajos, 1), fill_value=1.0, dtype=np.float64))
                list_variables.append(coef + '_' + str(pacient))

            # Si és un pacient amb Parkinson que només ha fet una sessió o un pacient sa
            # volem que la variable MedsOn no hi sigui (serà constant)
            elif not ('MedsOn' in coef and (int(pacient) <= 35 or sessions == 1)):
                np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                    (np.matrix(df_pacients[pacient][coef], dtype=np.float64)).transpose())
                list_variables.append(coef + '_' + str(pacient))

            else:
                columnes_eliminades.append(index_variable)

            index_variable += 1

        # També omplim el vector np_pupil de les mesures de pupil·la que
        # corresponen a aquest pacient
        np_pupil[index_assaig:index_assaig + len_assajos, 0] = (
            np.matrix(df_pacients[pacient]['pupil'], dtype=np.float64))
        index_assaig += len_assajos

    # Eliminem les columnes on no tingui sentit posar les variables MedsOn
    np_data = np.delete(np_data, columnes_eliminades, 1)

    # TODO: per dibuixar la matriu
    # plt.imshow(np_data, vmin=0, vmax=4, aspect='auto')
    # plt.show()

    # Convertim les matrius a Dataframe
    df_data = pd.DataFrame(data=np_data, columns=list_variables)
    df_pupil = pd.DataFrame(data=np_pupil, columns=['pupil'])

    # Posem fit_intercept a false ja que volem que cada pacient tingui el seu propi
    # terme independent
    regr = linear_model.LinearRegression(fit_intercept=False)
    # df_data serà la X i df_pupil la y
    regr.fit(df_data, df_pupil)

    # Ara, obtenim els coeficients per cada pacient
    coef_regression = regr.coef_[0]
    i = 0
    j = 0
    k = 0

    fig_barplots, ax_bp = plt.subplots(nrows=10, ncols=3, figsize=(18, 38))

    for pacient in pacients:
        result_pacient = dict()
        result_pacient['Pacient'] = pacient
        barplot_data = dict()

        for coef in coefficients:
            if j < len(list_variables) and coef_variables[coef] == list_variables[j].split('_')[0]:
                value = round(coef_regression[j], 3)
                j += 1
                if abs(value) < 5:
                    coef_dict[coef].append(value)
            # Posarem un 0 en el cas dels que no tingui sentit que tinguin la variable MedsOn
            else:
                value = 0

            name_key = coef + '(' + coef_variables[coef] + ')'

            result_pacient[name_key] = value
            barplot_data[coef] = value

            i += 1
        results_list.append(result_pacient)
        # Dibuixem una barplot dels coeficients de cada pacient
        ax = ax_bp[k // 3][k % 3]
        sb.barplot(ax=ax, data=barplot_data)
        ax.set_title("Pacient " + str(pacient))
        ax.set_ylim((-5, 2.5))
        k += 1

    plt.savefig(folder + "barplot_pacient.png", bbox_inches='tight', dpi=300)
    plt.close()

    significance_list = []

    data_violin = dict()

    # Calculem el p-value i el t-test dels coeficients
    for coef_name in coefficients:
        # Del terme independent no calculem el p-value
        if coef_name != 'b0':
            name = coef_name + '(' + coef_variables[coef_name] + ')'
            k_stat, p_value_k = ks_2samp(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
            u_stat, p_value_u = mannwhitneyu(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
            significance_list.append({
                'Test type': 'ks-test', 'Coefficient': name, 'stat': round(k_stat, 6), 'p-value': round(p_value_k, 6)
            })
            significance_list.append({
                'Test type': 'u-test', 'Coefficient': name, 'stat': round(u_stat, 6), 'p-value': round(p_value_u, 6)
            })
            data_violin[coef_variables[coef_name]] = coef_dict[coef_name]

    coeff_results = pd.DataFrame(results_list)
    significance_results = pd.DataFrame(significance_list)

    pd_df = pd.DataFrame({'R^2': [round(regr.score(df_data, df_pupil), 3)]})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23, 8))
    # Taula on surten els valors dels coeficients i el R^2
    ax[0].axis('tight')
    ax[0].axis('off')
    table1 = ax[0].table(cellText=coeff_results.values, colLabels=coeff_results.columns, cellLoc='center', loc='center')
    table1.scale(2.4 - 2.4 * len(pd_df.columns) / (len(pd_df.columns) + len(coeff_results.columns)), 1.2)

    ax[1].axis('tight')
    ax[1].axis('off')
    table2 = ax[1].table(cellText=pd_df.values, colLabels=pd_df.columns, cellLoc='center', loc='center')
    table2.scale(2.4 - 2.4 * len(coeff_results.columns) / (len(pd_df.columns) + len(coeff_results.columns)),
                 1.2)  # Match the height

    plt.savefig(folder + "regression_results.png", bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Taula on surt el p-value associat en cada coeficient
    ax.axis('tight')
    ax.axis('off')
    table2 = ax.table(cellText=significance_results.values, colLabels=significance_results.columns, cellLoc='center',
                      loc='center')

    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.2)

    plt.savefig(folder + "significance_regression.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Fem una barplot amb la mitjana i l'error estandard dels coeficients de la regresió
    plt.subplots(figsize=(10, 7))
    sb.barplot(coef_dict, errorbar="se")
    plt.savefig(folder + "barplot_mitjana.png", bbox_inches='tight', dpi=300)
    plt.close()

    fig, axs = plt.subplots(nrows=len(data_violin.keys()), ncols=1, figsize=(10, 45))
    i = 0
    # VIOLIN PLOT
    for name in data_violin.keys():
        pupil_values = dict()

        for pacient in pacients:
            df_pupil = df_pacients[pacient]

            values = df_pupil[name].unique()
            # Fem una llista de tots els valors de les pupil·les associades a un valor concret del coeficient
            for value in values:
                if value not in pupil_values.keys():
                    pupil_values[value] = list()
                df = df_pupil['pupil'].loc[df_pupil[name] == value]
                pupil_values[value] += df.tolist()

        # Dibuixem un violin plot amb les dades
        ax = axs[i]
        ax.set_title(name)
        sb.violinplot(ax=ax, data=pupil_values)
        i += 1

    plt.savefig(folder + "violin_plot.png", bbox_inches='tight', dpi=300)
    plt.close()


"""
Funció que fa dues regressions lineals de tots els pacients
segons el valor de CT de cada assaig 
"""
def regressio_lineal_CT():
    mean_pacient = load_data("dades_docs/" + "mitjana_assajos")
    # Llista per guardar dades de la regressió com una taula
    coef_variables = {'b0': 'b0', 'b1': 'SF', 'b2': 'SFxSP', 'b4': 'MedsOn',
                      'b5': 'MedsOnxSF', 'b6': 'MedsOnxSFxSP'}
    pacients = mean_pacient.keys()
    variables = coef_variables.values()
    coefficients = coef_variables.keys()

    model_variable = 'CT'
    values = [1, 2]
    for value in values:
        results_list = list()
        df_pacients = dict()
        coef_dict = dict()
        folder = 'results/model_CT_' + str(value) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        for coef in coefficients:
            coef_dict[coef] = list()

        # Per cada pacient, creem un dataframe amb les dades de les variables
        # També obtenim el número de total d'assajos que tenim entre tots els pacients
        num_total_assajos = 0
        for pacient in pacients:
            df_pupil = pd.DataFrame(mean_pacient[pacient])
            df_pupil = df_pupil[df_pupil[model_variable] == value]

            # Creem més columnes combinant algunes variables
            df_pupil['SFxSP'] = df_pupil['SF'] * df_pupil['SP']
            df_pupil['MedsOnxSF'] = df_pupil['MedsOn'] * df_pupil['SF']
            df_pupil['MedsOnxSFxSP'] = df_pupil['MedsOn'] * df_pupil['SF'] * df_pupil['SP']

            # Guardem el dataframe amb totes les dades necessàries per després
            df_pacients[pacient] = df_pupil

            num_total_assajos += df_pupil.shape[0]

        num_variables = len(variables)

        # Obtenim el número total de variables que tindrem
        # Cada pacient tindrà el seu propi b0 i el bi de cada variable
        # Hi haurà un coeficient que serà global i que anirà acompanyat de PD
        # (és 0 si l'assaig forma part d'un pacient sense Parkinson, 1 si en té)
        num_total_variables = len(pacients) * num_variables

        # Creem una matriu plena de zeros de la mida dels vectors que necessitem
        np_data = np.zeros(shape=(num_total_assajos, num_total_variables), dtype=np.float64)
        np_pupil = np.zeros(shape=(num_total_assajos, 1), dtype=np.float64)

        index_assaig = 0
        index_variable = 0
        # Variable on tindrem les columnes que s'hauries d'eliminar de np_data
        # (casos on MedsOn no tingui sentit)
        columnes_eliminades = list()

        # Variable que servirà per tenir els índex de les dades que posarem a np_data
        list_variables = list()

        # Ara volem que la matriu np_data contingui tota la informació de les variables
        # (serà una matriu diagonal per blocs menys en l'última columna)
        for pacient in pacients:
            len_assajos = df_pacients[pacient].shape[0]
            # Número de sessions que ha fet cada pacient (pot ser 1 o 2)
            sessions = int(mean_pacient[pacient]['sessions'])
            for coef in variables:
                if coef == 'b0':
                    # Com que la variable b0 és el terme independent de la regressió, a la matriu posarem uns
                    np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                        np.full(shape=(len_assajos, 1), fill_value=1.0, dtype=np.float64))
                    list_variables.append(coef + '_' + str(pacient))

                # Si és un pacient amb Parkinson que només ha fet una sessió o un pacient sa
                # volem que la variable MedsOn no hi sigui (serà constant)
                elif not ('MedsOn' in coef and (int(pacient) <= 35 or sessions == 1)):
                    np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                        (np.matrix(df_pacients[pacient][coef], dtype=np.float64)).transpose())
                    list_variables.append(coef + '_' + str(pacient))

                else:
                    columnes_eliminades.append(index_variable)

                index_variable += 1

            # També omplim el vector np_pupil de les mesures de pupil·la que
            # corresponen a aquest pacient
            np_pupil[index_assaig:index_assaig + len_assajos, 0] = (
                np.matrix(df_pacients[pacient]['pupil'], dtype=np.float64))
            index_assaig += len_assajos

        # Eliminem les columnes on no tingui sentit posar les variables MedsOn
        np_data = np.delete(np_data, columnes_eliminades, 1)

        # TODO: per dibuixar la matriu
        # plt.imshow(np_data, vmin=0, vmax=4, aspect='auto')
        # plt.show()

        # Convertim les matrius a Dataframe
        df_data = pd.DataFrame(data=np_data, columns=list_variables)
        df_pupil = pd.DataFrame(data=np_pupil, columns=['pupil'])

        # Posem fit_intercept a false ja que volem que cada pacient tingui el seu propi
        # terme independent
        regr = linear_model.LinearRegression(fit_intercept=False)

        # df_data serà la X i df_pupil la y
        regr.fit(df_data, df_pupil)

        # Ara, obtenim els coeficients per cada pacient
        coef_regression = regr.coef_[0]
        i = 0
        j = 0
        k = 0

        fig_barplots, ax_bp = plt.subplots(nrows=10, ncols=3, figsize=(18, 38))

        for pacient in pacients:
            result_pacient = dict()
            result_pacient['Pacient'] = pacient
            barplot_data = dict()

            for coef in coefficients:
                if j < len(list_variables) and coef_variables[coef] == list_variables[j].split('_')[0]:
                    value = round(coef_regression[j], 3)
                    j += 1
                    if abs(value) < 5:
                        coef_dict[coef].append(value)
                # Posarem un 0 en el cas dels que no tingui sentit que tinguin la variable MedsOn
                else:
                    value = 0

                name_key = coef + '(' + coef_variables[coef] + ')'

                # TODO: mirar en quins no s'han de posar els values que són 0

                result_pacient[name_key] = value
                barplot_data[coef] = value

                i += 1
            results_list.append(result_pacient)
            # Dibuixem una barplot dels coeficients de cada pacient
            ax = ax_bp[k // 3][k % 3]
            sb.barplot(ax=ax, data=barplot_data)
            ax.set_title("Pacient " + str(pacient))
            ax.set_ylim((-5, 2.5))
            k += 1

        plt.savefig(folder + "barplot_pacient.png", bbox_inches='tight', dpi=300)
        plt.close()

        significance_list = []

        data_violin = dict()

        # Calculem el p-value i el t-test dels coeficients
        for coef_name in coefficients:
            # Del terme independent no calculem el p-value
            if coef_name != 'b0':
                name = coef_name + '(' + coef_variables[coef_name] + ')'
                k_stat, p_value_k = ks_2samp(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
                u_stat, p_value_u = mannwhitneyu(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
                significance_list.append({
                    'Test type': 'ks-test', 'Coefficient': name, 'stat': round(k_stat, 6),
                    'p-value': round(p_value_k, 6)
                })
                significance_list.append({
                    'Test type': 'u-test', 'Coefficient': name, 'stat': round(u_stat, 6), 'p-value': round(p_value_u, 6)
                })
                data_violin[coef_variables[coef_name]] = coef_dict[coef_name]

        coeff_results = pd.DataFrame(results_list)
        significance_results = pd.DataFrame(significance_list)

        pd_df = pd.DataFrame({'R^2': [round(regr.score(df_data, df_pupil), 3)]})

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23, 8))
        # Taula on surten els valors dels coeficients i el R^2
        ax[0].axis('tight')
        ax[0].axis('off')
        table1 = ax[0].table(cellText=coeff_results.values, colLabels=coeff_results.columns, cellLoc='center',
                             loc='center')
        table1.scale(2.4 - 2.4 * len(pd_df.columns) / (len(pd_df.columns) + len(coeff_results.columns)), 1.2)

        ax[1].axis('tight')
        ax[1].axis('off')
        table2 = ax[1].table(cellText=pd_df.values, colLabels=pd_df.columns, cellLoc='center', loc='center')
        table2.scale(2.4 - 2.4 * len(coeff_results.columns) / (len(pd_df.columns) + len(coeff_results.columns)),
                     1.2)  # Match the height

        plt.savefig(folder + "regression_results.png", bbox_inches='tight', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Taula on surt el p-value associat en cada coeficient
        ax.axis('tight')
        ax.axis('off')
        table2 = ax.table(cellText=significance_results.values, colLabels=significance_results.columns,
                          cellLoc='center',
                          loc='center')

        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.2)

        plt.savefig(folder + "significance_regression.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Fem una barplot amb la mitjana i l'error estandard dels coeficients de la regresió
        plt.subplots(figsize=(10, 7))
        sb.barplot(coef_dict, errorbar="se")
        plt.savefig(folder + "barplot_mitjana.png", bbox_inches='tight', dpi=300)
        plt.close()

        fig, axs = plt.subplots(nrows=len(data_violin.keys()), ncols=1, figsize=(10, 7 * len(data_violin.keys())))
        i = 0
        # VIOLIN PLOT
        for name in data_violin.keys():
            pupil_values = dict()

            for pacient in pacients:
                df_pupil = df_pacients[pacient]

                values = df_pupil[name].unique()
                # Fem una llista de tots els valors de les pupil·les associades a un valor concret del coeficient
                for value in values:
                    if value not in pupil_values.keys():
                        pupil_values[value] = list()
                    df = df_pupil['pupil'].loc[df_pupil[name] == value]
                    pupil_values[value] += df.tolist()

            # Dibuixem un violin plot amb les dades
            ax = axs[i]
            ax.set_title(name)
            sb.violinplot(ax=ax, data=pupil_values)
            i += 1

        plt.savefig(folder + "violin_plot.png", bbox_inches='tight', dpi=300)
        plt.close()


"""
Funció que fa dues regressions lineals dels pacients de Parkinson
segons si estan medicats o no.
"""
def regressio_lineal_MedsOn():
    mean_pacient = load_data("dades_docs/" + "mitjana_assajos")
    # Llista per guardar dades de la regressió com una taula
    coef_variables = {'b0': 'b0', 'b1': 'SF', 'b2': 'SFxSP', 'b3': 'CT'}
    pacients = list()
    for pacient in mean_pacient.keys():
        if int(pacient) > 35:
            pacients.append(pacient)

    variables = coef_variables.values()
    coefficients = coef_variables.keys()

    model_variable = 'MedsOn'
    values = [0, 1]
    for value in values:
        results_list = list()
        df_pacients = dict()
        coef_dict = dict()
        folder = 'results/model_MedsOn_' + str(value) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        for coef in coefficients:
            coef_dict[coef] = list()

        # Per cada pacient, creem un dataframe amb les dades de les variables
        # També obtenim el número de total d'assajos que tenim entre tots els pacients
        num_total_assajos = 0
        for pacient in pacients:
            df_pupil = pd.DataFrame(mean_pacient[pacient])
            df_pupil = df_pupil[df_pupil[model_variable] == value]

            # Creem més columnes combinant algunes variables
            df_pupil['SFxSP'] = df_pupil['SF'] * df_pupil['SP']

            # Guardem el dataframe amb totes les dades necessàries per després
            df_pacients[pacient] = df_pupil

            num_total_assajos += df_pupil.shape[0]

        num_variables = len(variables)

        # Obtenim el número total de variables que tindrem
        # Cada pacient tindrà el seu propi b0 i el bi de cada variable
        # Hi haurà un coeficient que serà global i que anirà acompanyat de PD
        # (és 0 si l'assaig forma part d'un pacient sense Parkinson, 1 si en té)
        num_total_variables = len(pacients) * num_variables

        # Creem una matriu plena de zeros de la mida dels vectors que necessitem
        np_data = np.zeros(shape=(num_total_assajos, num_total_variables), dtype=np.float64)
        np_pupil = np.zeros(shape=(num_total_assajos, 1), dtype=np.float64)

        index_assaig = 0
        index_variable = 0

        # Variable que servirà per tenir els índex de les dades que posarem a np_data
        list_variables = list()

        # Ara volem que la matriu np_data contingui tota la informació de les variables
        # (serà una matriu diagonal per blocs menys en l'última columna)
        for pacient in pacients:
            len_assajos = df_pacients[pacient].shape[0]
            # Número de sessions que ha fet cada pacient (pot ser 1 o 2)
            sessions = int(mean_pacient[pacient]['sessions'])
            for coef in variables:
                if coef == 'b0':
                    # Com que la variable b0 és el terme independent de la regressió, a la matriu posarem uns
                    np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                        np.full(shape=(len_assajos, 1), fill_value=1.0, dtype=np.float64))
                    list_variables.append(coef + '_' + str(pacient))

                # Si és un pacient amb Parkinson que només ha fet una sessió o un pacient sa
                # volem que la variable MedsOn no hi sigui (serà constant)
                else:
                    np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                        (np.matrix(df_pacients[pacient][coef], dtype=np.float64)).transpose())
                    list_variables.append(coef + '_' + str(pacient))

                index_variable += 1

            # També omplim el vector np_pupil de les mesures de pupil·la que
            # corresponen a aquest pacient
            np_pupil[index_assaig:index_assaig + len_assajos, 0] = (
                np.matrix(df_pacients[pacient]['pupil'], dtype=np.float64))
            index_assaig += len_assajos

        # TODO: per dibuixar la matriu
        # plt.imshow(np_data, vmin=0, vmax=4, aspect='auto')
        # plt.show()

        # Convertim les matrius a Dataframe
        df_data = pd.DataFrame(data=np_data, columns=list_variables)
        df_pupil = pd.DataFrame(data=np_pupil, columns=['pupil'])

        # Posem fit_intercept a false ja que volem que cada pacient tingui el seu propi
        # terme independent
        regr = linear_model.LinearRegression(fit_intercept=False)

        # df_data serà la X i df_pupil la y
        regr.fit(df_data, df_pupil)

        # Ara, obtenim els coeficients per cada pacient
        coef_regression = regr.coef_[0]
        i = 0
        j = 0
        k = 0

        fig_barplots, ax_bp = plt.subplots(nrows=6, ncols=3, figsize=(18, 38))

        for pacient in pacients:
            result_pacient = dict()
            result_pacient['Pacient'] = pacient
            barplot_data = dict()

            for coef in coefficients:
                if j < len(list_variables) and coef_variables[coef] == list_variables[j].split('_')[0]:
                    value = round(coef_regression[j], 3)
                    j += 1
                    if abs(value) < 5:
                        coef_dict[coef].append(value)
                # Posarem un 0 en el cas dels que no tingui sentit que tinguin la variable MedsOn
                else:
                    value = 0

                name_key = coef + '(' + coef_variables[coef] + ')'

                # TODO: mirar en quins no s'han de posar els values que són 0

                result_pacient[name_key] = value
                barplot_data[coef] = value

                i += 1
            results_list.append(result_pacient)
            # Dibuixem una barplot dels coeficients de cada pacient
            ax = ax_bp[k // 3][k % 3]
            sb.barplot(ax=ax, data=barplot_data)
            ax.set_title("Pacient " + str(pacient))
            ax.set_ylim((-5, 2.5))
            k += 1

        plt.savefig(folder + "barplot_pacient.png", bbox_inches='tight', dpi=300)
        plt.close()

        significance_list = []

        data_violin = dict()

        # Calculem el p-value i el t-test dels coeficients
        for coef_name in coefficients:
            # Del terme independent no calculem el p-value
            if coef_name != 'b0':
                name = coef_name + '(' + coef_variables[coef_name] + ')'
                k_stat, p_value_k = ks_2samp(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
                u_stat, p_value_u = mannwhitneyu(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
                significance_list.append({
                    'Test type': 'ks-test', 'Coefficient': name, 'stat': round(k_stat, 6),
                    'p-value': round(p_value_k, 6)
                })
                significance_list.append({
                    'Test type': 'u-test', 'Coefficient': name, 'stat': round(u_stat, 6), 'p-value': round(p_value_u, 6)
                })
                data_violin[coef_variables[coef_name]] = coef_dict[coef_name]

        coeff_results = pd.DataFrame(results_list)
        significance_results = pd.DataFrame(significance_list)

        pd_df = pd.DataFrame({'R^2': [round(regr.score(df_data, df_pupil), 3)]})

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23, 8))
        # Taula on surten els valors dels coeficients i el R^2
        ax[0].axis('tight')
        ax[0].axis('off')
        table1 = ax[0].table(cellText=coeff_results.values, colLabels=coeff_results.columns, cellLoc='center',
                             loc='center')
        table1.scale(2.4 - 2.4 * len(pd_df.columns) / (len(pd_df.columns) + len(coeff_results.columns)), 1.2)

        ax[1].axis('tight')
        ax[1].axis('off')
        table2 = ax[1].table(cellText=pd_df.values, colLabels=pd_df.columns, cellLoc='center', loc='center')
        table2.scale(2.4 - 2.4 * len(coeff_results.columns) / (len(pd_df.columns) + len(coeff_results.columns)),
                     1.2)  # Match the height

        plt.savefig(folder + "regression_results.png", bbox_inches='tight', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Taula on surt el p-value associat en cada coeficient
        ax.axis('tight')
        ax.axis('off')
        table2 = ax.table(cellText=significance_results.values, colLabels=significance_results.columns,
                          cellLoc='center',
                          loc='center')

        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.2)

        plt.savefig(folder + "significance_regression.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Fem una barplot amb la mitjana i l'error estandard dels coeficients de la regresió
        plt.subplots(figsize=(10, 7))
        sb.barplot(coef_dict, errorbar="se")
        plt.savefig(folder + "barplot_mitjana.png", bbox_inches='tight', dpi=300)
        plt.close()

        fig, axs = plt.subplots(nrows=len(data_violin.keys()), ncols=1, figsize=(10, 7 * len(data_violin.keys())))
        i = 0
        # VIOLIN PLOT
        for name in data_violin.keys():
            pupil_values = dict()

            for pacient in pacients:
                df_pupil = df_pacients[pacient]

                values = df_pupil[name].unique()
                # Fem una llista de tots els valors de les pupil·les associades a un valor concret del coeficient
                for value in values:
                    if value not in pupil_values.keys():
                        pupil_values[value] = list()
                    df = df_pupil['pupil'].loc[df_pupil[name] == value]
                    pupil_values[value] += df.tolist()

            # Dibuixem un violin plot amb les dades
            ax = axs[i]
            ax.set_title(name)
            sb.violinplot(ax=ax, data=pupil_values)
            i += 1

        plt.savefig(folder + "violin_plot.png", bbox_inches='tight', dpi=300)
        plt.close()


"""
Funció que fa una regressió lineal de les persones sanes
"""
def regressio_lineal_sans():
    mean_pacient = load_data("dades_docs/" + "mitjana_assajos")
    folder = 'results/model_sans' + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Llista per guardar dades de la regressió com una taula
    results_list = list()
    df_pacients = dict()
    coef_dict = dict()
    coef_variables = {'b0': 'b0', 'b1': 'SF', 'b2': 'SFxSP', 'b3': 'CT'}
    pacients = list()
    for pacient in mean_pacient.keys():
        if int(pacient) <= 35:
            pacients.append(pacient)
    variables = coef_variables.values()
    coefficients = coef_variables.keys()

    for coef in coefficients:
        coef_dict[coef] = list()

    # Per cada pacient, creem un dataframe amb les dades de les variables
    # També obtenim el número de total d'assajos que tenim entre tots els pacients
    num_total_assajos = 0
    for pacient in pacients:
        df_pupil = pd.DataFrame(mean_pacient[pacient])

        # Creem més columnes combinant algunes variables
        df_pupil['SFxSP'] = df_pupil['SF'] * df_pupil['SP']

        # Guardem el dataframe amb totes les dades necessàries per després
        df_pacients[pacient] = df_pupil

        num_total_assajos += len(mean_pacient[pacient]['pupil'])

    num_variables = len(variables)

    # Obtenim el número total de variables que tindrem
    # Cada pacient tindrà el seu propi b0 i el bi de cada variable
    # Hi haurà un coeficient que serà global i que anirà acompanyat de PD
    # (és 0 si l'assaig forma part d'un pacient sense Parkinson, 1 si en té)
    num_total_variables = len(pacients) * num_variables

    # Creem una matriu plena de zeros de la mida dels vectors que necessitem
    np_data = np.zeros(shape=(num_total_assajos, num_total_variables), dtype=np.float64)
    np_pupil = np.zeros(shape=(num_total_assajos, 1), dtype=np.float64)

    index_assaig = 0
    index_variable = 0
    # Variable on tindrem les columnes que s'hauries d'eliminar de np_data
    # (casos on MedsOn no tingui sentit)
    columnes_eliminades = list()

    # Variable que servirà per tenir els índex de les dades que posarem a np_data
    list_variables = list()

    # Ara volem que la matriu np_data contingui tota la informació de les variables
    # (serà una matriu diagonal per blocs menys en l'última columna)
    for pacient in pacients:
        len_assajos = len(mean_pacient[pacient]['pupil'])
        # Número de sessions que ha fet cada pacient (pot ser 1 o 2)
        sessions = int(mean_pacient[pacient]['sessions'])
        for coef in variables:
            if coef == 'b0':
                # Com que la variable b0 és el terme independent de la regressió, a la matriu posarem uns
                np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                    np.full(shape=(len_assajos, 1), fill_value=1.0, dtype=np.float64))
                list_variables.append(coef + '_' + str(pacient))

            else:
                np_data[index_assaig:index_assaig + len_assajos, index_variable:index_variable + 1] = (
                    (np.matrix(df_pacients[pacient][coef], dtype=np.float64)).transpose())
                list_variables.append(coef + '_' + str(pacient))

            index_variable += 1

        # També omplim el vector np_pupil de les mesures de pupil·la que
        # corresponen a aquest pacient
        np_pupil[index_assaig:index_assaig + len_assajos, 0] = (
            np.matrix(df_pacients[pacient]['pupil'], dtype=np.float64))
        index_assaig += len_assajos

    # Eliminem les columnes on no tingui sentit posar les variables MedsOn
    np_data = np.delete(np_data, columnes_eliminades, 1)

    # TODO: per dibuixar la matriu
    # plt.imshow(np_data, vmin=0, vmax=4, aspect='auto')
    # plt.show()

    # Convertim les matrius a Dataframe
    df_data = pd.DataFrame(data=np_data, columns=list_variables)
    df_pupil = pd.DataFrame(data=np_pupil, columns=['pupil'])

    # Posem fit_intercept a false ja que volem que cada pacient tingui el seu propi
    # terme independent
    regr = linear_model.LinearRegression(fit_intercept=False)
    # df_data serà la X i df_pupil la y
    regr.fit(df_data, df_pupil)

    # Ara, obtenim els coeficients per cada pacient
    coef_regression = regr.coef_[0]
    i = 0
    j = 0
    k = 0

    fig_barplots, ax_bp = plt.subplots(nrows=5, ncols=3, figsize=(18, 38))

    for pacient in pacients:
        result_pacient = dict()
        result_pacient['Pacient'] = pacient
        barplot_data = dict()

        for coef in coefficients:
            if j < len(list_variables) and coef_variables[coef] == list_variables[j].split('_')[0]:
                value = round(coef_regression[j], 3)
                j += 1
                if abs(value) < 5:
                    coef_dict[coef].append(value)
            # Posarem un 0 en el cas dels que no tingui sentit que tinguin la variable MedsOn
            else:
                value = 0

            name_key = coef + '(' + coef_variables[coef] + ')'

            result_pacient[name_key] = value
            barplot_data[coef] = value

            i += 1
        results_list.append(result_pacient)
        # Dibuixem una barplot dels coeficients de cada pacient
        ax = ax_bp[k // 3][k % 3]
        sb.barplot(ax=ax, data=barplot_data)
        ax.set_title("Pacient " + str(pacient))
        ax.set_ylim((-5, 2.5))
        k += 1

    plt.savefig(folder + "barplot_pacient.png", bbox_inches='tight', dpi=300)
    plt.close()

    significance_list = []

    data_violin = dict()

    # Calculem el p-value i el t-test dels coeficients
    for coef_name in coefficients:
        # Del terme independent no calculem el p-value
        if coef_name != 'b0':
            name = coef_name + '(' + coef_variables[coef_name] + ')'
            k_stat, p_value_k = ks_2samp(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
            u_stat, p_value_u = mannwhitneyu(coef_dict[coef_name], [0 for i in range(len(coef_dict[coef_name]))])
            significance_list.append({
                'Test type': 'ks-test', 'Coefficient': name, 'stat': round(k_stat, 6), 'p-value': round(p_value_k, 6)
            })
            significance_list.append({
                'Test type': 'u-test', 'Coefficient': name, 'stat': round(u_stat, 6), 'p-value': round(p_value_u, 6)
            })
            data_violin[coef_variables[coef_name]] = coef_dict[coef_name]

    coeff_results = pd.DataFrame(results_list)
    significance_results = pd.DataFrame(significance_list)

    pd_df = pd.DataFrame({'R^2': [round(regr.score(df_data, df_pupil), 3)]})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23, 8))
    # Taula on surten els valors dels coeficients i el R^2
    ax[0].axis('tight')
    ax[0].axis('off')
    table1 = ax[0].table(cellText=coeff_results.values, colLabels=coeff_results.columns, cellLoc='center', loc='center')
    table1.scale(2.4 - 2.4 * len(pd_df.columns) / (len(pd_df.columns) + len(coeff_results.columns)), 1.2)

    ax[1].axis('tight')
    ax[1].axis('off')
    table2 = ax[1].table(cellText=pd_df.values, colLabels=pd_df.columns, cellLoc='center', loc='center')
    table2.scale(2.4 - 2.4 * len(coeff_results.columns) / (len(pd_df.columns) + len(coeff_results.columns)),
                 1.2)  # Match the height

    plt.savefig(folder + "regression_results.png", bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Taula on surt el p-value associat en cada coeficient
    ax.axis('tight')
    ax.axis('off')
    table2 = ax.table(cellText=significance_results.values, colLabels=significance_results.columns, cellLoc='center',
                      loc='center')

    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.2)

    plt.savefig(folder + "significance_regression.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Fem una barplot amb la mitjana i l'error estandard dels coeficients de la regresió
    plt.subplots(figsize=(10, 7))
    sb.barplot(coef_dict, errorbar="se")
    plt.savefig(folder + "barplot_mitjana.png", bbox_inches='tight', dpi=300)
    plt.close()

    fig, axs = plt.subplots(nrows=len(data_violin.keys()), ncols=1, figsize=(10, 7 * len(data_violin.keys())))
    i = 0
    # VIOLIN PLOT
    for name in data_violin.keys():
        pupil_values = dict()

        for pacient in pacients:
            df_pupil = df_pacients[pacient]

            values = df_pupil[name].unique()
            # Fem una llista de tots els valors de les pupil·les associades a un valor concret del coeficient
            for value in values:
                if value not in pupil_values.keys():
                    pupil_values[value] = list()
                df = df_pupil['pupil'].loc[df_pupil[name] == value]
                pupil_values[value] += df.tolist()

        # Dibuixem un violin plot amb les dades
        ax = axs[i]
        ax.set_title(name)
        sb.violinplot(ax=ax, data=pupil_values)
        i += 1

    plt.savefig(folder + "violin_plot.png", bbox_inches='tight', dpi=300)
    plt.close()


if not os.path.exists('results'):
    os.makedirs('results')

regressio_lineal()
regressio_lineal_CT()
regressio_lineal_MedsOn()
regressio_lineal_sans()