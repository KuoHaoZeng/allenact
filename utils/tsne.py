import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt

_color_set = ["blue", "green", "red", "magenta", "cyan", "yellow", "gray", "black", "white"]

def get_data(root, labels):
    out = []
    labels_out = []
    for l in labels:
        d = "{}/{}".format(root, l)
        sub_d = sorted(os.listdir(d))
        for i, e in enumerate(sub_d):
            e_d = "{}/{}".format(d, e)
            npy = sorted([ele for ele in os.listdir(e_d) if ele.endswith(".npy")])
            data_list = []
            for n in npy:
                data = np.load("{}/{}".format(e_d, n))
                data_list.append(data)
                out.append(data)
                labels_out.append(l)
            #out.append(np.array(data_list).mean(0))
            #labels_out.append(l)
            if i == 5:
                break
    return np.array(out), np.array(labels_out)

def plot(e, labels):
    font_size = 22
    fig, ax = plt.subplots(figsize=(16, 8))
    xxx = [False, False, False, False]
    for i in range(len(e)):
        c = _color_set[int(labels[i])]
        if xxx[int(labels[i])]:
            ax.plot(e[i,0], e[i,1], marker="o", color=c)
        else:
            ax.plot(e[i,0], e[i,1], marker="o", color=c, label=labels[i])
            xxx[int(labels[i])] = True
    ax.legend(loc=1, fontsize=font_size)
    plt.savefig("test.png")

if __name__ == "__main__":
    root = "storage/Dynamics_Corruption/mini_grid/qualitative_results"
    labels = sorted(os.listdir(root))
    x, y = get_data(root, labels)
    e = TSNE(n_components=2).fit_transform(x)
    plot(e, y)