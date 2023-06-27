import os
import torch
import numpy as np


class Colorize(object):
    def __init__(self):
        self.cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0)], dtype=np.uint8)

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 3, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)

    return label_batch


def generate_label_plain(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 3, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())

    label_batch = np.array(label_batch)

    return label_batch


def tensor2label(label_tensor):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize()(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy
