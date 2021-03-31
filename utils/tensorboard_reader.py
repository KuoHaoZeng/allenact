from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils.plot_tool import *
import sys, os

def get_event_reader(event_dir):
    ea = EventAccumulator(event_dir)
    ea.Reload()
    return ea

def load_number_with_key(event_reader, key_string):
    output = []
    try:
        for scaler in event_reader.Scalars(key_string):
            output.append((scaler.step, scaler.value))
    except KeyboardInterrupt:
        return output
    return output

def load_with_multikeys(event_reader, keys):
    output = {}
    for k in keys:
        output[k] = load_number_with_key(event_reader, k)
    return output

def load_with_multievents_multikeys(event_dirs, keys):
    output = {}
    for e in event_dirs:
        event_reader = get_event_reader(e)
        output[e] = load_with_multikeys(event_reader, keys)
    return output

def get_event_files_dirs(cfg):
    f = open(cfg, "r").read()
    return f.split("\n")[:-1]

def get_event_files_recursive(dir):
    f = os.listdir(dir)[0]
    if "events" in f:
        return "{}/{}".format(dir, f)
    else:
        return get_event_files_recursive("{}/{}".format(dir, f))

def get_event_files_dirs_with_first_line_root(cfg):
    f = open(cfg, "r").read()
    f = f.split("\n")[:-1]
    root = f[0]
    files, labels = [], []
    for ele  in f[1:]:
        if ele[:-7] not in labels:
            labels.append(ele[:-7])
        file = get_event_files_recursive("{}/{}/tb".format(root, ele))
        files.append(file)
    return files, labels, root

if __name__ == "__main__":
    cfg = sys.argv[1]
    event_dirs, labels, root = get_event_files_dirs_with_first_line_root(cfg)
    tags = ["train/success", "train/missing_action_ratio"]
    output = load_with_multievents_multikeys(event_dirs, tags)

    font_size = 22

    # plot with upper and lower bound
    for tag in tags:
        if tag.split("/")[0] == "valid":
            ax, fig = plot_many_series(output, tag, 3, plot_mean_with_upper_and_lower, labels=labels)
        else:
            ax, fig = plot_many_series(output, tag, 3, plot_smooth_mean_with_upper_and_lower, labels=labels)
        ax.legend(loc=4, fontsize=font_size)
        ax.set_ylabel(tag, fontsize=font_size)
        ax.set_xlabel("Number of Training", fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 6))
        name = "{}/{}_{}_mul.png".format(root, tag.split("/")[0], tag.split("/")[1])
        plt.savefig(name)
