import csv
import numpy as np
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


class Logger(object):
    """Logger object for training process, supporting resume training"""
    def __init__(self, path, header, resume=False):
        """
        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name
    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model

def get_fusion_label(csv_path):
    """
    Read the csv file and return labels
    :param csv_path: path of csv file
    :return: ground truth labels
    """
    gt = np.zeros(360000)
    base = -10000
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            if row[1] != '':
                base += 10000
            if row[4] == 'N':
                gt[base + int(row[2]):base + int(row[3]) + 1] = 1
            else:
                continue
    return gt

def evaluate(score, label, whether_plot):
    """
    Compute Accuracy as well as AUC by evaluating the scores
    :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
    :param label: ground truth
    :param whether_plot: whether plot the AUC curve
    :return: best accuracy, corresponding threshold, AUC
    """
    thresholds = np.arange(0., 1., 0.01)
    best_acc = 0.
    best_threshold = 0.
    for threshold in thresholds:
        prediction = score >= threshold
        correct = prediction == label

        acc = (np.sum(correct) / correct.shape[0] * 100)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)

    if whether_plot:
        plt.plot(fpr, tpr, color='r')
        #plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
        plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
        plt.tick_params(labelsize=23)
        #plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.show()
    return best_acc, best_threshold, AUC


def post_process(score, window_size=6):
    """
    post process the score
    :param score: scores of each frame in videos
    :param window_size: window size
    :param momentum: momentum factor
    :return: post processed score
    """
    processed_score = np.zeros(score.shape)
    for i in range(0, len(score)):
        processed_score[i] = np.mean(score[max(0, i-window_size+1):i+1])

    return processed_score


def get_score(score_folder, mode):
    """
    !!!Be used only when scores exist!!!
    Get the corresponding scores according to requiements
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :return: the corresponding scores according to requirements
    """
    if mode not in ['top_d', 'top_ir', 'front_d', 'front_ir', 'fusion_top', 'fusion_front', 'fusion_d', 'fusion_ir', 'fusion_all']:
        print('Please enter correct mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all')
        return
    if mode == 'top_d':
        score = np.load(os.path.join(score_folder + '/score_top_d.npy'))
    elif mode == 'top_ir':
        score = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
    elif mode == 'front_d':
        score = np.load(os.path.join(score_folder + '/score_front_d.npy'))
    elif mode == 'front_ir':
        score = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
    elif mode == 'fusion_top':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score = np.mean((score1, score2), axis = 0)
    elif mode == 'fusion_front':
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score3, score4), axis=0)
    elif mode == 'fusion_d':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score = np.mean((score1, score3), axis=0)
    elif mode == 'fusion_ir':
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score2, score4), axis=0)
    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score1, score2, score3, score4), axis=0)

    return score







