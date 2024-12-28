import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cp(cp, file_name):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.65, 0.8, 0.3], ylim=(-3, 3))
    ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.3], xticklabels=[], sharex=ax1, sharey=ax1)
    ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.3], sharex=ax1, sharey=ax1)

    for ii in range(10):
        ax1.plot(cp[:, ii], label=f'param{ii:0d}')
    ax1.legend()
    for ii in range(10, 20):
        ax2.plot(cp[:, ii], label=f'param{ii:0d}')
    ax2.legend()
    for ii in range(20, 30):
        ax3.plot(cp[:, ii], label=f'param{ii:0d}')
    ax3.legend()
    fig.savefig(file_name, dpi=300)
    plt.close('all')


def plot_mel(mel, file_name):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(mel.T, aspect='equal', vmin=-5, vmax=20)
    fig.savefig(file_name, dpi=300)
    plt.close('all')
    plt.show()

def plot_loss(res_train, res_valid, loss_type = "MSE"):
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
    tmp = res_train.groupby('epoch')['loss'].mean()
    plt.semilogy(np.array(tmp.index), np.array(tmp), c='C0', lw=5, label='mean train loss')
    del tmp
    tmp = res_valid.groupby('epoch')['loss'].mean()
    plt.semilogy(np.array(tmp.index), np.array(tmp), c='C1', lw=5, label='mean validation loss')
    ax.set_xlabel('Epoch', fontsize=20, labelpad=20)
    ax.set_ylabel('Loss (%s)' % str(loss_type), fontsize=20, labelpad=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=15)

def plot_sublosses(res_train, res_valid):
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
    columns = list(res_train.columns)
    colors = ["C%d" % i for i in range(10)]

    i = 0
    for col in columns:
        if "loss" in col:
            color = colors[i]
            tmp = res_train.groupby('epoch')[col].mean()
            plt.semilogy(np.array(tmp.index), np.array(tmp), c=color, lw=5, label='%s' % str(col))
            del tmp
            tmp = res_valid.groupby('epoch')[col].mean()
            plt.semilogy(np.array(tmp.index), np.array(tmp), c=color, ls = "dotted", lw=5)
            i += 1
    ax.set_xlabel('Epoch', fontsize=20, labelpad=20)
    ax.set_ylabel('Loss' , fontsize=20, labelpad=20)
    ax.tick_params(axis='both', labelsize=15)
    leg = ax.legend(fontsize=15, bbox_to_anchor=[1.0, 1],frameon = False)
    ax.add_artist(leg)

    legend_elements = [Line2D([0], [0], color='black', lw=5, label='Training'),
                       Line2D([0], [0], color='black', lw=5, linestyle="dotted", label='Validation')]

    ax.legend(handles = legend_elements,fontsize=15, bbox_to_anchor=[.78, 1], frameon = False)



def plot_cp_predictions(prediction, target,type = "vocal tract",title=""):
    assert type in ["vocal tract", "glottis"]
    # vocal tract cps
    plt.figure(figsize=(15, 8), facecolor="white")
    # plt.plot(target_cp[:,:19], linewidth = 3 )
    if type == "vocal tract":
        plt.plot(target[:, :19], linewidth=5, alpha=0.5)
        plt.gca().set_prop_cycle(None)
        # plt.plot(target_prediction[:len(target_cp),:19],linestyle = "dotted",linewidth = 3)
        plt.plot(prediction[:len(target), :19], linestyle="dashed", linewidth=3)
        title = "Vocal Tract CPs: "+ title


    elif type == "glottis":
        plt.plot(target[:, 19:], linewidth=5,alpha=0.5)
        plt.gca().set_prop_cycle(None)
        # plt.plot(target_prediction[:len(target_cp),:19],linestyle = "dotted",linewidth = 3)
        plt.plot(prediction[:len(target), 19:], linestyle="dashed", linewidth=3)

        title = "Glottis CPs: " + title

    legend_elements = [Line2D([0], [0], color='black', lw=3, label='True CP'),
                       Line2D([0], [0], color='black', lw=3, linestyle="dotted", label='Predicted CP')]

    plt.legend(handles=legend_elements, loc='lower right', fontsize=15)
    plt.title(title, fontsize=18, pad=10)
    plt.show()






def plot_mel_predictions(target,prediction, title=""):
    fig, ax = plt.subplots(figsize=(15, 8), ncols=1, nrows=2, sharex=True, sharey=True)
    fig.set_facecolor("white")

    ax[0].set_title(title, fontsize=18, pad=20)
    img1 = target
    img2 = prediction

    ax[0].imshow(img1.T, origin='lower')
    ax[0].set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)
    ax[0].yaxis.tick_right()
    ax[0].yaxis.set_label_position("right")
    ax[1].imshow(img2.T, origin='lower')
    ax[1].set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax[1].set_xlabel('Time', fontsize=15, labelpad=20)

    ax[0].text(-0.075, 0.5, "Target", fontsize=18, va="center",rotation=90,
             transform=ax[0].transAxes)
    ax[1].text(-0.075, 0.5, "Prediction", fontsize=18, va="center", rotation=90,
               transform=ax[1].transAxes)
    plt.show()



