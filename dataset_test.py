import torch
import torch.utils.data as data
from PIL import Image
import os
import csv

def pil_loader(path):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: image data
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def accimage_loader(path):
    """
    compared with PIL, accimage loader eliminates useless function within class, so that it is faster than PIL
    :param path: image path
    :return: image data
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    """
    choose accimage as image loader if it is available, PIL otherwise
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = get_default_image_loader()
    video = []
    for image_index in frame_indices:
        image_name = 'img_' + str(image_index) + '.png'
        image_path = os.path.join(video_path, image_name)
        img = image_reader(image_path)
        video.append(img)
    return video

def get_clips(video_path, video_begin, video_end, label, view, sample_duration):
    """
    be used when validation set is generated. be used to divide a video interval into video clips
    :param video_path: validation data path
    :param video_begin: begin index of frames
    :param video_end: end index of frames
    :param label: 1(normal) / 0(anormal)
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :return: a list which contains  validation video clips
    """
    clips = []
    sample = {
        'video': video_path,
        'label': label,
        'subset': 'validation',
        'view': view,
    }
    step = 1

    if video_begin == 0:
        for i in range(7):
            sample_ = sample.copy()
            sample_['frame_indices'] = [0] * (7-i) + list(range(0, i + 9))
            clips.append(sample_)
        for i in range(7, video_end+1, step):
            sample_ = sample.copy()
            sample_['frame_indices'] = list(range(i-7, i + 9))
            clips.append(sample_)

    elif video_end == 9999:
        for i in range(video_begin, 9992, step):
            sample_ = sample.copy()
            sample_['frame_indices'] = list(range(i-7, i + 9))
            clips.append(sample_)
        for i in range(8):
            sample_ = sample.copy()
            sample_['frame_indices'] = list(range(9985+i, 10000)) + [9999] * (i+1)
            clips.append(sample_)
    else:
        for i in range(video_begin, video_end+1, step):
            sample_ = sample.copy()
            sample_['frame_indices'] = list(range(i-7, i + 9))
            clips.append(sample_)
    return clips


def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
            yield f

def make_dataset(root_path, subset, view, sample_duration, type=None):
    """
    Only be used at test time
    :param root_path: root path, e.g. "/usr/home/sut/datasets/DAD/DAD/"
    :param subset: validation
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 'label': 0/1, 'subset': 'train'/'validation', 'view': 'front_depth' / 'front_IR' / 'top_depth' / 'top_IR', 'action': 'normal' / other anormal actions}
    """
    dataset = []
    if subset == 'validation' and type == None:
        #load valiation data as well as thier labels
        csv_path = root_path + 'LABEL.csv'
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[-1] == '':
                    continue
                if row[0] != '':
                    which_val_path = os.path.join(root_path, row[0].strip())
                if row[1] != '':
                    video_path = os.path.join(which_val_path, row[1], view)
                video_begin = int(row[2])
                video_end = int(row[3])
                if row[4] == 'N':
                    label = 1
                elif row[4] == 'A':
                    label = 0
                clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
                dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!THIS DATASET IS ONLY USED IN TESTING MODE!!!PLEASE CHECK INPUT!!!')
    return dataset


class DAD_Test(data.Dataset):
    """
    This dataset is only used at test time to genrate consecutive video samples.
    """
    def __init__(self,
                 root_path,
                 subset,
                 view,
                 sample_duration=16,
                 type=None,
                 get_loader=get_video,
                 spatial_transform=None,
                 temporal_transform=None,
                 ):
        self.data = make_dataset(root_path, subset, view, sample_duration, type)
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        if self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']
            clip = self.loader(video_path, frame_indices)
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            return clip, ground_truth
        else:
            print('!!!DATA LOADING FAILURE!!!THIS DATASET IS ONLY USED IN TESTING MODE!!!PLEASE CHECK INPUT!!!')
    def __len__(self):
        return len(self.data)


