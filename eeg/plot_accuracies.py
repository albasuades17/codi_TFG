import numpy as np
import matplotlib.pyplot as plt
import os

fmt_grph = 'png'
fmt_grph2 = 'pdf'
results_perf = dict()
results_conf = dict()

def get_results(variable, control):
    directory = './pacients_sans/results/'
    for dir in os.listdir(directory):
        words_dir = dir.split('-')
        if words_dir[1] == variable and words_dir[2] == control:
            perf = np.load(directory + dir + '/perf.npy')
            conf_matrix = np.load(directory + dir + '/conf_matrix.npy')
            for band in range(perf.shape[0]):
                if band not in results_perf[control].keys():
                    results_perf[control][band] = dict()
                    results_conf[control][band] = dict()
                for measure in range(perf.shape[1]):
                    if measure not in results_perf[control][band].keys():
                        results_perf[control][band][measure] = dict()
                        results_conf[control][band][measure] = dict()
                    for classifier in range(perf.shape[3]):
                        if classifier not in results_perf[control][band][measure].keys():
                            results_perf[control][band][measure][classifier] = {'healthy': list(), 'ON': list(), 'OFF': list()}
                            results_conf[control][band][measure][classifier] = {'healthy': np.zeros([2,2]), 'ON': np.zeros([2,2]), 'OFF': np.zeros([2,2])}
                        results_perf[control][band][measure][classifier]['healthy'].append(np.mean(perf[band, measure, :, classifier]))
                        results_conf[control][band][measure][classifier]['healthy'] += conf_matrix[band, measure, :, classifier, :, :].mean(axis = 0)

    directory = './pacients_PD/results/'
    for dir in os.listdir(directory):
        words_dir = dir.split('-')
        if words_dir[1] == variable and words_dir[2] == control:
            # Obtenim si és la sessió ON o OFF
            key_dict = words_dir[4]
            try:
                perf = np.load(directory + dir + '/perf.npy')
                for band in range(perf.shape[0]):
                    if band not in results_perf[control].keys():
                        results_perf[control][band] = dict()
                        results_conf[control][band] = dict()
                    for measure in range(perf.shape[1]):
                        if measure not in results_perf[control][band].keys():
                            results_perf[control][band][measure] = dict()
                            results_conf[control][band][measure] = dict()
                        for classifier in range (perf.shape[3]):
                            if classifier not in results_perf[control][band][measure].keys():
                                results_perf[control][band][measure][classifier] = list()
                                results_conf[control][band][measure][classifier] = np.zeros([2,2])
                            results_perf[control][band][measure][classifier][key_dict].append(
                                np.mean(perf[band, measure, :, classifier]))
                            results_conf[control][band][measure][classifier][key_dict] += conf_matrix[band, measure, :,
                                                                                           classifier, :, :].mean(axis=0)

            except FileNotFoundError:
                print(dir, 'not found')

def draw_results(variable):
    names_bands = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    names_measures = {0: 'pow', 1: 'corr'}
    names_classifiers = {0: 'MLR', 1: '1NN'}
    res_dir = 'results_' + variable + '/'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    colors_acc = {
        0: ["#419ede", "#1f78b4", "#144c73"],  # Different shades of blue
        1: ["#4bce4b", "#2ca22c", "#1c641c"],  # Different shades of green
        2: ["#e36657", "#d62828", "#951b1c"]  # Different shades of red
    }
    cmap_colors = ['Blues','Greens','Oranges']

    # Adjust x-axis positions for each plot within the 9 graphs (0 to 8 positions)
    x_positions = np.arange(9)

    # Type of classifier
    for classifier in [0, 1]:
        classifier_label = names_classifiers[classifier]
        for measure in [0, 1]:
            measure_label = names_measures[measure]
            fig_acc, axs_acc = plt.subplots(figsize=(14, 12), nrows=3, ncols=1)
            fig_acc.suptitle('Classifier ' + classifier_label + ' with ' + measure_label)
            fig_conf, axs_conf = plt.subplots(figsize=(27, 10), nrows=3, ncols=9)
            fig_conf.suptitle('Classifier ' + classifier_label + ' with ' + measure_label)
            fig_conf.tight_layout(pad=3.0)
            fig_conf.subplots_adjust(hspace=0.4, wspace=0.4)
            i_control = 0
            for control in type_control:
                ax_acc = axs_acc[i_control]
                ax_acc.axis(ymin=0,ymax=1.05)
                ax_acc.set_title('Accuracies with control type ' + control)
                # Plot chance lever
                ax_acc.plot([i for i in range(-1, 10)], [chance_level[variable]]*11,'--k')
                ax_acc.set_xlim(-1, 9)
                list_axes = list()

                i  = 0
                for band in results_perf[control].keys():
                    freq_band = names_bands[band]
                    for type_pacient in results_perf[control][band][measure][classifier].keys():
                        list_axes.append(freq_band + ', ' + type_pacient)
                        vp = ax_acc.violinplot(results_perf[control][band][measure][classifier][type_pacient],
                                           positions=[x_positions[i]])

                        color_group = i // 3
                        color_index = i % 3
                        color = colors_acc[color_group][color_index]
                        ax_conf = axs_conf[i_control][i]
                        ax_conf.imshow(results_conf[control][band][measure][classifier][type_pacient],
                                                       vmin=0, cmap = cmap_colors[band])
                        ax_conf.set_xlabel('true label',fontsize=8)
                        ax_conf.set_ylabel('predicted label', fontsize=8)
                        ax_conf.set_title("Band " + freq_band + " for " + type_pacient + " patient", fontsize = 9)

                        vp['bodies'][0].set_facecolor(color)
                        vp['bodies'][0].set_alpha(0.6)
                        vp['cmins'].set_color(color)
                        vp['cmaxes'].set_color(color)
                        vp['cbars'].set_color(color)

                        i += 1

                ax_acc.set_xticks([i for i in range(9)], list_axes)
                ax_acc.set_ylabel('accuracy ' + variable, fontsize=8)

                i_control += 1
            # Save the figure in two formats
            fig_acc.savefig(res_dir + 'accuracy_' + measure_label + '_' + classifier_label + '.' + fmt_grph,
                        format=fmt_grph)
            fig_acc.savefig(res_dir + 'accuracy_' + measure_label + '_' + classifier_label + '.' + fmt_grph2,
                        format=fmt_grph2)
            fig_conf.savefig(res_dir + 'conf_matrix_' + measure_label + '_' + classifier_label + '.' + fmt_grph,
                            format=fmt_grph)
            fig_conf.savefig(res_dir + 'conf_matrix_' + measure_label + '_' + classifier_label + '.' + fmt_grph2,
                            format=fmt_grph2)
            plt.close(fig_acc)
            plt.close(fig_conf)

variables = ['SF', 'SP']
type_control = ['ALL', 'NS', 'S']
chance_level = {'SF': 0.66, 'SP': 0.5}
for ct in type_control:
    results_perf[ct] = dict()
    results_conf[ct] = dict()
    for motiv in variables:
        get_results(motiv, ct)

for motiv in variables:
    draw_results(motiv)
