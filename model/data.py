import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2 as cv


def get_datasets_and_loaders(data_directory, image_directory, split, batch_size, use_cuda, height, width):
    """
    :param data_directory: directory where df_{train,validate,test}.pkl files are stored
    :param image_directory: directory with images
    :param split: what split of the data loaders are being made for
    :param batch_size: batch size used for loaders
    :param use_cuda: true if cuda is available
    :param height: height all images are scaled to
    :param width: width all images are scaled to
    :return: list of dataloaders, each of which loads batches of sequences of a given length
    """
    assert split in ["train", "test", "valid"]
    #loading dataframe
    data_path = os.path.join(data_directory, "df_{}.pkl".format(split))
    data = pd.read_pickle(data_path)
    #data is grouped into bins which contain padded sequences of the same length to facilitate
    #parallelization, recovering values of these bins here
    bins = data['padded_seq_len'].unique()
    datasets = []
    dataloaders = []
    for bin in bins:
        #creating a unique dataloader for all sequences that have length bin
        curr_bin_dataset = I2LDataset(data[data['padded_seq_len'] == bin], image_directory, height, width)
        datasets.append(curr_bin_dataset)
        curr_dataloader = DataLoader(curr_bin_dataset, batch_size=batch_size,
                                     num_workers=2, shuffle=True,
                                     pin_memory=True if use_cuda else False,
                                     #drop_last=True
                                     )
        dataloaders.append(curr_dataloader)
    return datasets, dataloaders, bins


class I2LDataset(Dataset):
    def __init__(self, data, image_directory, image_height, image_width):
        """
        :param data: dataframe containing padded sequences along with image names
        :param image_directory: directory with images
        :param image_height: height all images are scaled to
        :param image_width: width all images are scaled to
        """
        super(I2LDataset, self).__init__()
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.image_directory = image_directory
        self.image_height = image_height
        self.image_width = image_width

    def get_tensor(self, image_name):
        """
        :param image_name: name of image to be pre-processed
        :return: torch tensor of dimension [1, H, W] representing input image in grayscale
        """

        image_path = str(os.path.join(self.image_directory, image_name))
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (self.image_width, self.image_height),
                          interpolation=cv.INTER_AREA)
        image = np.expand_dims(image, 0)
        tensor = torch.from_numpy(image).float()
        return tensor

    def __getitem__(self, index):
        """
        :param index: index in dataframe of item
        :return: image tensor and formula label, which have dimensions [1, H, W] and [L] respectively
        """
        item = self.data.iloc[index]
        image_name = item['image']
        tensor = self.get_tensor(image_name)
        #getting formula and prepending start token
        formula = item['padded_seq']
        formula = torch.tensor(formula, dtype=torch.int64)
        start_token = torch.ones(1, dtype=torch.int64)
        formula = torch.cat((start_token, formula))
        return tensor, formula

    def __len__(self):
        return len(self.data)
