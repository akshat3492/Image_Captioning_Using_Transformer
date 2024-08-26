import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np

class ImageCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset for image captioning.

    This dataset class handles the image features and their corresponding captions, allowing for easy 
    integration with PyTorch's DataLoader for training and evaluation.
    """
    
    def __init__(self, img_features, df):
        self.img_features = img_features
        self.df = df

    def __len__(self):
        # Return the number of unique images
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image row using the index
        image_row = self.df.iloc[idx]
        image_name = image_row["Image"]
        
        # Use the index directly to fetch the corresponding image feature
        img_feature = self.img_features[idx]
        
        # Randomly select one caption for this image
        caption_list = image_row["Comment"]
        caption = random.choice(caption_list)
        
        # Convert caption to a tensor
        caption_tensor = torch.tensor(caption, dtype=torch.long)
        
        return img_feature, caption_tensor
