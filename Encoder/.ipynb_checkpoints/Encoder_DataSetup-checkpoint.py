import os
import numpy as np
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import ast
import random
from skimage import io, transform


def rotate_img(img, rot):
    """
    Rotates the input image by a specified angle.

    Arguments:
        img (PIL.Image or torch.Tensor): The image to rotate.
        rot (int): The rotation angle as an integer. It represents:
                   0 -> 0 degrees
                   1 -> 90 degrees
                   2 -> 180 degrees
                   3 -> 270 degrees

    Returns:
        PIL.Image or torch.Tensor: The rotated image.
    """
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 1:
        return transforms.functional.rotate(img, angle=90)
    elif rot == 2:
        return transforms.functional.rotate(img, angle=180)
    elif rot == 3:
        return transforms.functional.rotate(img, angle=270)

    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')



class CaptioningDataset(torch.utils.data.Dataset):
    """ Flickr Image Caption Dataset."""

    """ This class provides a custom dataset for image captioning tasks. It loads
    images and their corresponding captions, applies transformations, and 
    provides an interface to get samples for training or evaluation. """
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Arguments:
            csv_file (string): Path to the csv file with captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.captions_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image path from the DataFrame
        img_name = os.path.join(self.root_dir,
                                self.captions_df.iloc[idx, 0])
        # print(self.captions_df.iloc[idx, 0])
        
        # Load the image
        image = Image.open(img_name)

        # Apply any transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        # Parse the string representation of the list of captions from the DataFrame
        captions = ast.literal_eval(self.captions_df.iloc[idx, 1])

        # randomly select image rotation
        rotation_label = random.choice([0, 1, 2, 3])
        image_rotated = rotate_img(image, rotation_label)

        # Convert rotation label to a PyTorch tensor
        rotation_label = torch.tensor(rotation_label).long()

        sample = {'Original Image':image, "Rotated Image":image_rotated, "Rotation Label":rotation_label, "Captions":captions}

        return sample
