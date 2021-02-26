import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
from torch import optim
from scipy.spatial.distance import cosine
import os, cv2, sys, json, random
import pdb

#num_test = 100
num_test = 20

def get_pose_data(types):
    x, y, x_test, y_test = [], [], [], []
    for l, t in enumerate(types):
        d = np.load("{}/pose/features.npy".format(t))
        ls = [l] * len(d)
        x += list(d[:-num_test])
        y += ls[:-num_test]
        x_test += list(d[-num_test:])
        y_test += ls[-num_test:]
    return x, y, x_test, y_test

def get_resnet_data(types, difference=False):
    x, y, x_test, y_test = [], [], [], []
    for l, t in enumerate(types):
        d_pre = np.load("{}/resnet/pre.npy".format(t))
        d_post = np.load("{}/resnet/post.npy".format(t))
        if difference:
            d = d_post - d_pre
        else:
            d = np.hstack((d_pre, d_post))
        ls = [l] * len(d)
        x += list(d[:-num_test])
        y += ls[:-num_test]
        x_test += list(d[-num_test:])
        y_test += ls[-num_test:]
    return x, y, x_test, y_test

def get_resnet_data_fail_different_actions(types, types_fail, difference=False):
    x, y, x_test, y_test = [], [], [], []
    for l, t in enumerate(types):
        d_pre = np.load("{}/resnet/pre.npy".format(t))
        d_post = np.load("{}/resnet/post.npy".format(t))
        if difference:
            d = d_post - d_pre
        else:
            d = np.hstack((d_pre, d_post))
        ls = [l] * len(d)
        x += list(d[:-num_test])
        y += ls[:-num_test]
        x_test += list(d[-num_test:])
        y_test += ls[-num_test:]

    x_fail, x_test_fail = [], []
    for t in types_fail:
        d_pre = np.load("{}/resnet/pre.npy".format(t))
        d_post = np.load("{}/resnet/post.npy".format(t))
        if difference:
            d = d_post - d_pre
        else:
            d = np.hstack((d_pre, d_post))
        x_fail += list(d[:-num_test])
        x_test_fail += list(d[-num_test:])

    x_fail = random.sample(x_fail, len(x))
    y_fail = [max(y) + 1] * len(x_fail)
    x_test_fail = random.sample(x_test_fail, len(x_test))
    y_test_fail = [max(y_test) + 1] * len(x_test_fail)

    x = x + x_fail
    y = y + y_fail
    x_test = x_test + x_test_fail
    y_test = y_test + y_test_fail

    return x, y, x_test, y_test

def skkmean(objects, x, y, x_test, y_test):
    y_dist = []
    y_dist_acum = [0]
    y_test_dist = []
    y_test_dist_acum = [0]
    for i in range(len(objects)):
        y_dist.append(len(np.where(np.array(y) == i)[0]))
        y_dist_acum.append(y_dist[-1] + y_dist_acum[-1])
        y_test_dist.append(len(np.where(np.array(y_test) == i)[0]))
        y_test_dist_acum.append(y_test_dist[-1] + y_test_dist_acum[-1])
    acc = []
    while len(acc) < 3:
        kmeans = KMeans(init='k-means++', n_clusters=len(objects), n_init=100)
        kmeans.fit(x)
        p = kmeans.labels_
        mapping = []
        for i in range(len(objects)):
            tmp = []
            for j in range(len(objects)):
                tmp.append(len(np.where(p[y_dist_acum[i]:y_dist_acum[i+1]] == j)[0]))
            mapping.append(tmp)
        c2l = {}
        while len(c2l.keys()) < len(objects) - 1:
            v = np.max(mapping)
            idx, idy = np.where(np.array(mapping) == v)
            idx, idy = idx[0], idy[0]
            if idx not in c2l.keys() and idy not in c2l.values():
                c2l[idx] = idy
            mapping[idx][idy] = 0
        for i in range(len(objects)):
            if i not in c2l.keys():
                for j in range(len(objects)):
                    if j not in c2l.values():
                        c2l[i] = j
        p_test = kmeans.predict(x_test)
        p_test_label = np.array([c2l[ele] for ele in p_test])
        correct = 0.
        confusing_matrix = []
        for i in range(len(objects)):
            tmp = []
            for j in range(len(objects)):
                tmp.append(len(np.where(p_test_label[y_test_dist_acum[i]:y_test_dist_acum[i+1]] == j)[0]))
            confusing_matrix.append(tmp)
        confusing_matrix = np.array(confusing_matrix)
        recall = np.mean(np.diag(confusing_matrix)/confusing_matrix.sum(0))
        precision = np.mean(np.diag(confusing_matrix)/confusing_matrix.sum(1))
        f1 = 2*(precision*recall)/(precision+recall)
        if precision <= (1/7.):
            continue
        for i in range(len(objects) + 1):
            if i == 0:
                txt = "{:.2f}\t".format(f1)
                for j in objects:
                    txt += "{}\t".format(j)
                print(txt)
            else:
                txt = "{}\t".format(objects[i-1])
                for j in range(len(objects)):
                    txt += "{}\t".format(confusing_matrix[i-1,j])
                print(txt)
        acc.append(f1)
    print("ave f1: {:.2f} +- {:.2f}".format(np.mean(acc), np.std(acc)))
    print("max f1: {:.2f}, min f1: {:.2f}".format(np.max(acc), np.min(acc)))
    return np.array([c2l[ele] for ele in p]), p_test_label

def F1_distance(a, b):
    if len(a) == len(b):
        return cosine(a,b)
    else:
        if len(a) <= len(b):
            x = a
            y = b
        else:
            x = b
            y = a
        dis = []
        for i in range(len(y) - len(x)):
            dis = cosine(x, y[i:i+len(x)])
        return min(dis)

def skknn(objects, x, y, x_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x, y)
    p = neigh.predict(x)
    print("KNN train acc: {:.2f}".format(sum(p == y) / len(y)))
    p_test = neigh.predict(x_test)
    print("KNN test acc: {:.2f}".format(sum(p_test == y_test) / len(y_test)))

def learn_MLP(objects, x, y, x_test, y_test, lr):
    input_dim = len(x[0])

    epoch = 400
    batch_size = 640 #400
    iters = len(x) // batch_size
    model = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.ReLU(),
        nn.Linear(input_dim // 2, input_dim // 2),
        nn.ReLU(),
        nn.Linear(input_dim // 2, input_dim // 2),
        nn.ReLU(),
        nn.Linear(input_dim // 2, input_dim // 2),
        nn.ReLU(),
        nn.Linear(input_dim // 2, max(y) + 1),
    ).cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr)
    loss = nn.CrossEntropyLoss()

    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).long().cuda()
    x_test = torch.Tensor(x_test).cuda()
    y_test = torch.Tensor(y_test).long().cuda()
    for e in range(epoch):
        rand_idx = torch.randperm(len(x))
        x = x[rand_idx]
        y = y[rand_idx]
        for i in range(iters):
            p = model(x[i*batch_size:(i+1)*batch_size])
            l = loss(p, y[i*batch_size:(i+1)*batch_size])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print("loss: {:.3f}".format(l.data))

    model.eval()
    p = torch.argmax(model(x), 1)
    acc = ((p == y).float().sum() / len(y)).detach().cpu().data
    print("MLP train acc: {:.2f}".format(acc))
    p = torch.argmax(model(x_test), 1)
    acc = ((p == y_test).float().sum() / len(y_test)).detach().cpu().data
    print("MLP test acc: {:.2f}".format(acc))


if __name__ == "__main__":
    mode = sys.argv[1]
    path = sys.argv[2]
    pred_path = "{}/pred_results/{}".format(path, mode)
    if not os.path.isdir(pred_path):
        os.makedirs(pred_path)

    _ALL_RELATION_TYPES = ["LookDown", "LookUp", "MoveAhead", "PullObject",
                           "PushObject", "RotateLeft", "RotateRight"]

    all = False
    if all:
        relation_types = ["LookDown",
                          "LookUp",
                          "MoveAhead",
                          "PullObject",
                          "PushObject",
                          "RotateLeft",
                          "RotateRight",]
                          #"TeleportFull"]
    else:
        #relation_types = ["MoveAhead",
        #                  "TeleportFull"]
        relation_types = ["PullObject"]

    if mode == "pose":
        relation_types = ["{}/{}".format(path, t) for t in relation_types]
        x, y, x_test, y_test = get_pose_data(relation_types)
        learn_MLP(relation_types, x, y, x_test, y_test, 0.1)
        skknn(relation_types, x, y, x_test, y_test)
    elif mode == "resnet":
        relation_types = ["{}/{}".format(path, t) for t in relation_types]
        x, y, x_test, y_test = get_resnet_data(relation_types, True)
        learn_MLP(relation_types, x, y, x_test, y_test, 0.001)
        skknn(relation_types, x, y, x_test, y_test)
    elif mode == "pr":
        relation_types = ["{}/{}".format(path, t) for t in relation_types]
        x, y, x_test, y_test = get_pose_data(relation_types)
        x_r, _, x_r_test, _ = get_resnet_data(relation_types)
        x = np.hstack((x, x_r))
        x_test = np.hstack((x_test, x_r_test))
        learn_MLP(relation_types, x, y, x_test, y_test, 0.0001)
        skknn(relation_types, x, y, x_test, y_test)
    elif mode == "resnet_fail_same_action":
        fail_path = "{}/{}".format("datasets", "try_5rooms_fail_same_action")
        relation_types = ["{}/{}".format(path, t) for t in relation_types] + ["{}/{}".format(fail_path, t) for t in relation_types]
        x, y, x_test, y_test = get_resnet_data(relation_types, True)
        learn_MLP(relation_types, x, y, x_test, y_test, 0.001)
        skknn(relation_types, x, y, x_test, y_test)
    elif mode == "resnet_fail_different_actions":
        fail_path = "{}/{}".format("datasets", "try_5rooms_fail_same_action")
        relation_types = ["{}/{}".format(path, t) for t in relation_types]
        relation_types_fail = ["{}/{}".format(fail_path, t) for t in _ALL_RELATION_TYPES]
        x, y, x_test, y_test = get_resnet_data_fail_different_actions(relation_types, relation_types_fail, True)
        learn_MLP(relation_types, x, y, x_test, y_test, 0.001)
        skknn(relation_types, x, y, x_test, y_test)

