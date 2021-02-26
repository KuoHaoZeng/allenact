import ai2thor.controller

import random, time
import numpy as np
from multiprocessing import Process,Queue, Pool
import pdb, os, sys, json

import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn

from PIL import Image
from utils.utils_3d_torch import project_2d_points_to_3d

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def remove_objects_in_scene(obj_list, controller):
    for obj in obj_list:
        event = controller.step({"action": "RemoveFromScene", "objectId": obj})
    return event

def get_interaction_visible_objects(pre_meta, post_meta):
    pre_obj = [ele for ele in pre_meta["objects"] if ele["visible"]]
    post_obj_name = [ele["name"] for ele in post_meta["objects"] if ele["visible"]]
    o = []
    for ele in pre_obj:
        if ele["name"] in post_obj_name:
            o.append(ele)
    return o

def check_sucessful_push_pull(pre_meta, post_meta, object):
    idx = 0
    for i, ele in enumerate(pre_meta["objects"]):
        if ele["name"] == object["name"]:
            idx = i
            break
    pos_pre = [pre_meta["objects"][idx]["position"]["x"], pre_meta["objects"][idx]["position"]["z"]]
    pos_post = [post_meta["objects"][idx]["position"]["x"], post_meta["objects"][idx]["position"]["z"]]
    distance = ((np.array(pos_pre) - np.array(pos_post)) ** 2).sum() ** 0.5
    if distance < 0.1:
        return False
    else:
        return True

def push_fail(controller):
    event = controller.last_event
    xx = [ele for ele in event.metadata["objects"] if ele["visible"] and (ele["moveable"] or ele["pickupable"])]
    sorted(xx, key=lambda i: i['distance'])
    if len(xx) > 0:
        mag = random.choice([0, 50, 150, 200])
        controller.step(dict(action="PushObject", objectId=xx[0]["objectId"], moveMagnitude=mag))
        controller.last_event.metadata["lastActionSuccess"] = check_sucessful_push_pull(event.metadata,
                                                                                        controller.last_event.metadata,
                                                                                        xx[0])
        return controller.last_event
    else:
        controller.last_event.metadata["lastActionSuccess"] = False
        return controller.last_event

def push(controller):
    event = controller.last_event
    xx = [ele for ele in event.metadata["objects"] if ele["visible"] and (ele["moveable"] or ele["pickupable"])]
    sorted(xx, key=lambda i: i['distance'])
    if len(xx) > 0:
        controller.step(dict(action="PushObject", objectId=xx[0]["objectId"], moveMagnitude=100))
        controller.last_event.metadata["lastActionSuccess"] = check_sucessful_push_pull(event.metadata,
                                                                                        controller.last_event.metadata,
                                                                                        xx[0])
        return controller.last_event
    else:
        controller.last_event.metadata["lastActionSuccess"] = False
        return controller.last_event

def pull_fail(controller):
    event = controller.last_event
    xx = [ele for ele in event.metadata["objects"] if ele["visible"] and (ele["moveable"] or ele["pickupable"])]
    sorted(xx, key=lambda i: i['distance'])
    if len(xx) > 0:
        mag = random.choice([0, 50, 150, 200])
        controller.step(dict(action="PullObject", objectId=xx[0]["objectId"], moveMagnitude=mag))
        controller.last_event.metadata["lastActionSuccess"] = check_sucessful_push_pull(event.metadata,
                                                                                        controller.last_event.metadata,
                                                                                        xx[0])
        return controller.last_event
    else:
        controller.last_event.metadata["lastActionSuccess"] = False
        return controller.last_event

def pull(controller):
    event = controller.last_event
    xx = [ele for ele in event.metadata["objects"] if ele["visible"] and (ele["moveable"] or ele["pickupable"])]
    sorted(xx, key=lambda i: i['distance'])
    if len(xx) > 0:
        controller.step(dict(action="PullObject", objectId=xx[0]["objectId"], moveMagnitude=100))
        controller.last_event.metadata["lastActionSuccess"] = check_sucessful_push_pull(event.metadata,
                                                                                        controller.last_event.metadata,
                                                                                        xx[0])
        return controller.last_event
    else:
        controller.last_event.metadata["lastActionSuccess"] = False
        return controller.last_event

def move_ahead_fail(controller):
    n = random.choice([0, 2, 3])
    for _ in range(n):
        event = controller.step(dict(action="MoveAhead"))
    if n == 0:
        controller.last_event.metadata["lastActionSuccess"] = True
        controller.last_event.metadata["lastAction"] = "MoveAhead"
        return controller.last_event
    else:
        return event

def move_ahead(controller):
    event = controller.step(dict(action="MoveAhead"))
    return event

def move_back_fail(controller):
    n = random.choice([0, 2, 3])
    for _ in range(n):
        event = controller.step(dict(action="MoveBack"))
    if n == 0:
        controller.last_event.metadata["lastActionSuccess"] = True
        controller.last_event.metadata["lastAction"] = "MoveBack"
        return controller.last_event
    else:
        return event

def move_back(controller):
    event = controller.step(dict(action="MoveBack"))
    return event

def rotate_right_fail(controller):
    deg = random.choice([0, 30, 60])
    event = controller.step(dict(action="RotateRight", degrees=deg))
    return event

def rotate_right(controller):
    event = controller.step(dict(action="RotateRight"))
    return event

def rotate_left_fail(controller):
    deg = random.choice([0, 30, 60])
    event = controller.step(dict(action="RotateLeft", degrees=deg))
    return event

def rotate_left(controller):
    event = controller.step(dict(action="RotateLeft"))
    return event

def look_up_fail(controller):
    n = random.choice([0, 2])
    for _ in range(n):
        event = controller.step(dict(action="LookUp"))
    if n == 0:
        controller.last_event.metadata["lastActionSuccess"] = True
        controller.last_event.metadata["lastAction"] = "LookUp"
        return controller.last_event
    else:
        return event

def look_up(controller):
    event = controller.step(dict(action="LookUp"))
    return event

def look_down_fail(controller):
    n = random.choice([0, 2])
    for _ in range(n):
        event = controller.step(dict(action="LookDown"))
    if n == 0:
        controller.last_event.metadata["lastActionSuccess"] = True
        controller.last_event.metadata["lastAction"] = "LookDown"
        return controller.last_event
    else:
        return event

def look_down(controller):
    event = controller.step(dict(action="LookDown"))
    return event

def fail(controller):
    return controller.last_event

def get_agent_rotation_matrix(degree):
    if degree == 0:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    elif degree == 90:
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])
    elif degree == 180:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]])
    elif degree == 270:
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])
    else:
        raise NotImplementedError

def get_agent_back_rotation_matrix(degree):
    if degree == 0:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    elif degree == 90:
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])
    elif degree == 180:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]])
    elif degree == 270:
        return np.array([[0, 0, -1],
                         [0, 1, 0],
                         [1, 0, 0]])
    else:
        raise NotImplementedError

def dict_to_list(d):
    return [d["x"], d["y"], d["z"]]

def get_corners(event, objectId):
    mask = event.instance_masks[objectId]
    depth_mask = event.depth_frame
    x, y = np.where(mask)
    corners = []
    depths = []

    x1 = np.max(x)
    y1 = y[np.argmax(x)]
    corners.append([x1, y1])
    depths.append(depth_mask[x1, y1])

    y2 = np.max(y)
    x2 = x[np.argmax(y)]
    corners.append([x2, y2])
    depths.append(depth_mask[x2, y2])

    x3 = np.min(x)
    y3 = y[np.argmin(x)]
    corners.append([x3, y3])
    depths.append(depth_mask[x3, y3])

    y4 = np.min(y)
    x4 = x[np.argmin(y)]
    corners.append([x4, y4])
    depths.append(depth_mask[x4, y4])

    s = x + y
    idx = np.argmax(s)
    x5 = x[idx]
    y5 = y[idx]
    corners.append([x5, y5])
    depths.append(depth_mask[x5, y5])

    idx = np.argmin(s)
    x6 = x[idx]
    y6 = y[idx]
    corners.append([x6, y6])
    depths.append(depth_mask[x6, y6])

    s = x - y
    idx = np.argmax(s)
    x7 = x[idx]
    y7 = y[idx]
    corners.append([x7, y7])
    depths.append(depth_mask[x7, y7])

    idx = np.argmin(s)
    x8 = x[idx]
    y8 = y[idx]
    corners.append([x8, y8])
    depths.append(depth_mask[x8, y8])
    return corners, depths

def sample_points(event, objectId, n=16):
    mask = event.instance_masks[objectId]
    depth_mask = event.depth_frame
    index = np.array(np.where(mask))
    sample_idx = np.array(random.choices(list(range(index.shape[1])), k=n))
    points = np.transpose(index[:, sample_idx], (1, 0))
    depths = np.array([depth_mask[x, y] for (x, y) in points])
    return points, depths

class Agent(Dataset):
    def __init__(self, n, x_display, port, action_sets):
        self.n = n
        self.x_display = x_display
        self.port = port

        self.controller = None
        self.reachable_positions = None
        self.obj_pos = None

        self.action_sets = action_sets

    def stop(self):
        if not isinstance(self.controller, type(None)):
            self.controller.stop()
            self.controller = None
            self.reachable_positions = None
            self.obj_pos = None

    def init_controller(self, x_display, port):
        scene = "FloorPlan2{:02d}_physics".format(1)
        controller = ai2thor.controller.Controller(x_display=x_display,
                                                   port=port,
                                                   scene=scene,
                                                   renderDepthImage=True,
                                                   renderClassImage=True,
                                                   renderObjectImage=True)
        self.scene = scene

        #controller.step(dict(action="Initialize", gridSize=0.25))
        controller.step(dict(action="GetReachablePositions"))
        reachable_positions = controller.last_event.metadata["actionReturn"]

        self.controller = controller
        self.reachable_positions = reachable_positions

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(self.controller, type(None)):
            self.init_controller(self.x_display, self.port)

        while True:
            self.init_pose(idx)
            output = [self.controller.last_event]
            self.action_sets[idx % len(self.action_sets)](self.controller)
            output.append(self.controller.last_event)
            if self.controller.last_event.metadata["lastActionSuccess"]:
                if len(get_interaction_visible_objects(output[0].metadata, output[1].metadata)) > 2:
                    break
        return output

    def init_pose(self, idx):
        scene = "FloorPlan2{:02d}_physics".format((idx // (len(self) * n_processes // 5)) + 1)
        if scene != self.scene:
            self.controller.reset(scene=scene)
            self.scene = scene
        elif (idx + 1) % (len(self.action_sets) * 3) == 0:
            self.controller.reset(scene=self.scene)

        random_pos = random.choice(self.reachable_positions)
        random_rotation = random.choice([0, 90, 180, 270])
        self.controller.step(dict(action="TeleportFull",
                                  x=random_pos['x'],
                                  y=random_pos['y'],
                                  z=random_pos['z'],
                                  rotation=dict(x=0.0, y=random_rotation, z=0.0),
                                  horizon=30.0))

def data_collection_multiprocess(p_id):
    a = Agent(n_data // n_processes, "0.0", 8000 + p_id, action_set)
    for i in range(len(a)):
        single_data_collection(dataset_folder, action_set, a, i + (p_id * (n_data // n_processes)))

def data_collection(dataset_folder):
    a = Agent(n_data, "0.0", 8000, action_set)
    for i in range(len(a)):
        single_data_collection(dataset_folder, action_set, a, i)

def single_data_collection(dataset_folder, action_set, a, i):
    data = a[i]
    last_action = data[1].metadata["lastAction"]
    if not os.path.isdir("{}/{}".format(dataset_folder, last_action)):
        os.makedirs("{}/{}".format(dataset_folder, last_action))
        os.makedirs("{}/{}/img".format(dataset_folder, last_action))
    np.save("{}/{}/{:05d}".format(dataset_folder, last_action, (i // len(action_set)) + 1), data)
    img = Image.fromarray(data[0].frame)
    img.save("{}/{}/img/{:05d}_pre.png".format(dataset_folder, last_action, (i // len(action_set)) + 1))
    img = Image.fromarray(data[1].frame)
    img.save("{}/{}/img/{:05d}_after.png".format(dataset_folder, last_action, (i // len(action_set)) + 1))
    print(i + 1)

def get_resnet_feature(dataset_folder):
    action_set = os.listdir(dataset_folder)

    m = torchvision.models.resnet18(pretrained=True)
    m.fc = Identity()
    m = m.cuda()
    m.eval()

    for a in action_set:
        if not os.path.isdir("{}/{}/resnet".format(dataset_folder, a)):
            os.makedirs("{}/{}/resnet".format(dataset_folder, a))
        data = [ele for ele in os.listdir("{}/{}".format(dataset_folder, a)) if ele.endswith(".npy")]
        pre = []
        post = []
        for d in data:
            d = np.load("{}/{}/{}".format(dataset_folder, a, d), allow_pickle=True)
            img = [torch.Tensor(d[0].frame).cuda().view(1,300,300,3).transpose(1,3).transpose(2,3)]
            img += [torch.Tensor(d[1].frame).cuda().view(1,300,300,3).transpose(1,3).transpose(2,3)]
            img = torch.cat(img, 0)
            f = m(img)
            f = f.detach().cpu()
            pre.append(f[0].unsqueeze(0))
            post.append(f[1].unsqueeze(0))
        pre = torch.cat(pre, 0).numpy()
        post = torch.cat(post, 0).numpy()
        np.save("{}/{}/resnet/pre".format(dataset_folder, a), pre)
        np.save("{}/{}/resnet/post".format(dataset_folder, a), post)

def get_pose_feature(dataset_folder):
    action_set = os.listdir(dataset_folder)

    for a in action_set:
        if not os.path.isdir("{}/{}/pose".format(dataset_folder, a)):
            os.makedirs("{}/{}/pose".format(dataset_folder, a))
        data = [ele for ele in os.listdir("{}/{}".format(dataset_folder, a)) if ele.endswith(".npy")]
        features = []
        for d in data:
            d = np.load("{}/{}/{}".format(dataset_folder, a, d), allow_pickle=True)

            agent_pos = dict_to_list(d[0].metadata["agent"]["position"])
            agent_rot = [ele / 360. for ele in dict_to_list(d[0].metadata["agent"]["rotation"])]
            agent_rot[0] = d[0].metadata["agent"]["cameraHorizon"] / 360.
            agent_pose_pre = np.array(agent_pos + agent_rot)

            agent_pos = dict_to_list(d[1].metadata["agent"]["position"])
            agent_rot = [ele / 360. for ele in dict_to_list(d[0].metadata["agent"]["rotation"])]
            agent_rot[0] = d[1].metadata["agent"]["cameraHorizon"] / 360.
            agent_pose_post = np.array(agent_pos + agent_rot)

            visible_objects = get_interaction_visible_objects(d[0].metadata, d[1].metadata)
            visible_objects_names = [ele["name"] for ele in visible_objects]
            visible_objects_pose_diff = []
            visible_objects_pose_local_diff = []
            for obj in d[1].metadata["objects"]:
                if obj["name"] in visible_objects_names:
                    idx = visible_objects_names.index(obj["name"])
                    pos_pre = dict_to_list(visible_objects[idx]["position"])
                    rot_pre = [ele / 360. for ele in dict_to_list(visible_objects[idx]["rotation"])]
                    pose_pre = np.array(pos_pre + rot_pre)

                    pos_post = dict_to_list(obj["position"])
                    rot_post = [ele / 360. for ele in dict_to_list(obj["rotation"])]
                    #pos_post = dict_to_list(visible_objects[idx]["position"])
                    #rot_post = [ele / 360. for ele in dict_to_list(visible_objects[idx]["rotation"])]
                    pose_post = np.array(pos_post + rot_post)

                    diff = ((pose_post - pose_pre) ** 2).sum() ** 0.5
                    diff = np.array(list(pose_post - pose_pre) + [diff])
                    visible_objects_pose_diff.append(diff)

                    local_pose_pre = pose_pre - agent_pose_pre
                    local_pose_post = pose_post - agent_pose_post
                    diff = ((local_pose_post - local_pose_pre) ** 2).sum() ** 0.5
                    diff = np.array(list(local_pose_post - local_pose_pre) + [diff])
                    visible_objects_pose_local_diff.append(diff)

            visible_objects_pose_diff = np.array(visible_objects_pose_diff)
            max_idx = np.argmax(visible_objects_pose_diff[:,-1])
            f = list(visible_objects_pose_diff[max_idx,:-1])
            min_idx = np.argmin(visible_objects_pose_diff[:,-1])
            f += list(visible_objects_pose_diff[min_idx,:-1])

            visible_objects_pose_local_diff = np.array(visible_objects_pose_local_diff)
            max_idx = np.argmax(visible_objects_pose_local_diff[:,-1])
            f += list(visible_objects_pose_local_diff[max_idx,:-1])
            min_idx = np.argmin(visible_objects_pose_local_diff[:,-1])
            f += list(visible_objects_pose_local_diff[min_idx,:-1])
            features.append(f)
        features = np.array(features)
        np.save("{}/{}/pose/features".format(dataset_folder, a), features)
        #np.save("{}/{}/pose/features_fail_action".format(dataset_folder, a), features)

if __name__ == "__main__":
    n_data = 4000
    n_processes = 8
    p_ids = list(range(n_processes))
    #action_set = [fail, move_ahead, rotate_left, rotate_right, look_up, look_down, push, pull]
    action_set = [fail, move_ahead_fail, rotate_left_fail, rotate_right_fail, look_up_fail, look_down_fail, push_fail, pull_fail]

    dataset_folder = "datasets/try_5rooms_fail_same_action"
    with Pool(n_processes) as p:
        p.map(data_collection_multiprocess, p_ids)
    #data_collection(dataset_folder)
    get_resnet_feature(dataset_folder)
    get_pose_feature(dataset_folder)
