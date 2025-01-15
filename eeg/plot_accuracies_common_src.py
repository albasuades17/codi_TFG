import numpy as np
import matplotlib.pyplot as plt
import os

fmt_grph = 'png'
fmt_grph2 = 'pdf'
results_perf = dict()
results_conf = dict()


def get_results(variable, control, i_control):
    directory = './pacients_sans/common_src_'+variable
    perf = np.load(directory + '/perf.npy')
    conf_matrix = np.load(directory + '/conf.npy')
    for band in range(perf.shape[2]):
        if band not in results_perf[control].keys():
            results_perf[control][band] = dict()
            results_conf[control][band] = dict()
        results_perf[control][band]['healthy'] = np.mean(perf[:, i_control, band, :], axis = 1)
        results_conf[control][band]['healthy'] = np.mean(conf_matrix[:, i_control, band, :, :, :], axis = 1)

    for type in ['ON', 'OFF']:
        directory = './pacients_PD/common_src_'+variable+'_'+type

        perf = np.load(directory + '/perf.npy')
        conf_matrix = np.load(directory + '/conf.npy')
        for band in range(perf.shape[2]):
            if band not in results_perf[control].keys():
                results_perf[control][band] = dict()
                results_conf[control][band] = dict()
            results_perf[control][band][type] = np.mean(perf[:, i_control, band, :], axis = 1)
            results_conf[control][band][type] = np.mean(conf_matrix[:, i_control, band, :, :, :], axis = 1)
def draw_results(variable):
    names_bands = {0: 'alpha', 1: 'beta', 2: 'gamma'}
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

    fig_acc, axs_acc = plt.subplots(figsize=(14, 8), nrows=2, ncols=1)
    fig_acc.suptitle('Classifier MLR with pow')
    fig_conf, axs_conf = plt.subplots(figsize=(27, 8), nrows=2, ncols=9)
    fig_conf.suptitle('Classifier MLR with pow')
    fig_conf.tight_layout(pad=3.0)  # Increase padding
    fig_conf.subplots_adjust(hspace=0.4, wspace=0.4)
    i_control = 0
    for control in type_control:
        ax_acc = axs_acc[i_control]
        ax_acc.axis(ymin=0,ymax=1.05)
        ax_acc.set_title('Accuracies with control type ' + control)
        # Plot chance level
        ax_acc.plot([i for i in range(-1, 10)], [chance_level[variable]]*11,'--k')
        ax_acc.set_xlim(-1, 9)
        list_axes = list()

        i  = 0
        for band in results_perf[control].keys():
            freq_band = names_bands[band]
            for type_pacient in results_perf[control][band].keys():
                list_axes.append(freq_band + ', ' + type_pacient)
                vp = ax_acc.violinplot(results_perf[control][band][type_pacient],
                                   positions=[x_positions[i]])

                color_group = i // 3
                color_index = i % 3
                color = colors_acc[color_group][color_index]

                vp['bodies'][0].set_facecolor(color)
                vp['bodies'][0].set_alpha(0.6)
                vp['cmins'].set_color(color)
                vp['cmaxes'].set_color(color)
                vp['cbars'].set_color(color)

                ax_conf = axs_conf[i_control, i]
                ax_conf.imshow(results_conf[control][band][type_pacient][i_control],
                               vmin=0, cmap=cmap_colors[band])
                ax_conf.set_xlabel('true label', fontsize=8)
                ax_conf.set_ylabel('predicted label', fontsize=8)
                ax_conf.set_title("Band " + freq_band + " for " + type_pacient + " patient", fontsize=9)

                i += 1

        ax_acc.set_xticks([i for i in range(9)], list_axes)
        ax_acc.set_ylabel('accuracy ' + variable, fontsize=8)

        i_control += 1
    # Save the figure in two formats
    fig_acc.savefig(res_dir + 'accuracy_pow_MLR_common.' + fmt_grph,
                format=fmt_grph)
    fig_acc.savefig(res_dir + 'accuracy_pow_MLR_common.' + fmt_grph2,
                format=fmt_grph2)
    fig_conf.savefig(res_dir + 'conf_matrix_MLR_common.' + fmt_grph,
                     format=fmt_grph)
    fig_conf.savefig(res_dir + 'conf_matrix_MLR_common.' + fmt_grph2,
                     format=fmt_grph2)
    plt.close(fig_acc)

    plt.close(fig_conf)

variables = ['SF', 'SP']
type_control = ['NS', 'S']
chance_level = {'SF': 0.66, 'SP': 0.5}
for i_ct, ct in enumerate(type_control):
    results_perf[ct] = dict()
    results_conf[ct] = dict()
    for motiv in variables:
        get_results(motiv, ct, i_ct)

for motiv in variables:
    draw_results(motiv)
