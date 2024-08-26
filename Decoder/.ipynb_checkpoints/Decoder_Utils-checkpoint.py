import pandas as pd
import numpy as np
import re
import collections
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import os


# Function to extract features
def extract_features(encoder, image):
    """
    Extracts features from a given image using the provided encoder model.

    Arguments:
        encoder (nn.Module): Pre-trained encoder model (e.g., CNN without the final layer).
        image (torch.Tensor): Image tensor to extract features from.

    Returns:
        torch.Tensor: Extracted features as a tensor.
    """
    
    encoder.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)  # Add batch dimension
        features = encoder(image)
    return features.squeeze(0) #remove batch dimension

def extract_image_features(net, df):
    """
    Extracts features from all images listed in the DataFrame using a modified network.

    Arguments:
        net (nn.Module): Pre-trained neural network model (e.g., ResNet).
        df (pd.DataFrame): DataFrame containing image paths.

    Returns:
        torch.Tensor: Tensor of extracted features for all images.
    """
    
    # Modify the network by removing the final layer
    new_model = nn.Sequential(*list(net.children())[:-1])

    # Define image transformations: similar to Validation Loader for Encoder.
    t_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    img_features = []

    # Loop through each unique image in the DataFrame
    for i in list(df['Image'].unique()):
        img_path = os.path.join('./data/flickr30k_images/flickr30k_images/', i)
        img = Image.open(img_path)
        img = t_transform(img)
        features = extract_features(new_model, img)
        img_features.append(features)

    img_features = torch.stack(img_features)# Stack features into a single tensor

    return img_features.view(img_features.size()[0], -1) # Flatten features and return



def vocabulary(df):
    """
    Creates a vocabulary of unique words from the captions in the DataFrame.

    Arguments:
        df (pd.DataFrame): DataFrame containing captions.

    Returns:
        list: List of unique words in the captions.
    """
    
    vocab_size = set()
    for i in df['Comment']:
        vocab_size.update(i)
    return list(vocab_size)

def flatten_comprehension(matrix):
    """
    Flattens a matrix (list of lists) into a single list.

    Arguments:
        matrix (list of lists): Input matrix to flatten.

    Returns:
        list: Flattened list containing all elements of the input matrix.
    """
    
    return [item for row in matrix for item in row]

def replace_less_frequent_words(d):
    """
    Replaces words in a dictionary with "<UNK>" if their frequency is less than 10.

    Arguments:
        d (dict): Dictionary with words as keys and their frequencies as values.

    Returns:
        dict: Dictionary with words replaced by "<UNK>" based on their frequency.
    """
    
    replace_dict = {}

    for key, value in d.items():
        if value < 10:
            replace_dict[key] = "<UNK>"

    return replace_dict

def replace_word_idx(word_to_idx):
    """
    Creates a reverse mapping from indices to words.

    Arguments:
        word_to_idx (dict): Dictionary mapping words to indices.

    Returns:
        dict: Dictionary mapping indices to words.
    """
    
    idx_to_word = {}

    for key, value in word_to_idx.items():
        idx_to_word[value] = key

    return idx_to_word

def decode_captions(captions, idx_to_word):
    """
    Decodes a batch of captions from indices to words.

    Arguments:
        captions (np.array): Array of caption indices.
        idx_to_word (dict): Dictionary mapping indices to words.

    Returns:
        list: List of decoded captions as strings.
    """
    
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<TAB>":
                words.append(word)
            if word == "<END>":
                continue
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def preprocess_captions(path):
     """
    Preprocesses captions and prepares data for training.

    Arguments:
        path (str): Path to the CSV file containing image captions.

    Returns:
        tuple: (DataFrame of processed captions per image, word-to-index dictionary, index-to-word dictionary, vocabulary size)
    """
    
    df = pd.read_csv(path, sep="|")
    df = df.rename(columns = {
        ' comment':'Comment',
        'image_name':'Image',
        ' comment_number':'Comment_Number'
    })

    df['Comment'] = df['Comment'].str.strip()
    df['Comment'] = df['Comment'].apply(lambda x: str(x).replace("\\s+"," "))
    df['Comment'] = df['Comment'].apply(lambda x: str(x).lower())
    df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z\s]', '', x)).strip())
    
    df["Comment_length"] = df['Comment'].apply(lambda x:len(str(x).split()))
    # print("Mean Length of Comments:", df['Comment_length'].mean())

    df = df[(df["Comment_length"]>3) & (df['Comment_length']<=15)].reset_index(drop=True)
    df = df.merge(df.groupby(by="Image").size().reset_index(name='counts'), how="left", on="Image")
    df = df[df['counts']>2].reset_index(drop=True)

    df['Comment'] = df['Comment'].str.split()
    df["Padding_Length"] = 15 - df["Comment_length"]
    
    df["Comment"] = df.apply(lambda row: ["<START>"] + row["Comment"], axis=1)
    df["Comment"] = df.apply(lambda row: row["Comment"] + ['<END>'], axis=1)
    df["Comment"] = df.apply(lambda row: row["Comment"] + ["<TAB>"] * row["Padding_Length"], axis=1)
    
    
    l = flatten_comprehension(list(df["Comment"]))
    frequency = dict(collections.Counter(l))

    replace_dict = replace_less_frequent_words(frequency)
    df['Comment'] = df['Comment'].apply(lambda x : [replace_dict[i] if i in replace_dict else i for i in x])
    
    # df1 = df.copy()
    
    vocab_size = vocabulary(df)

    word_to_idx = {word: i for i, word in enumerate(vocab_size)}
    idx_to_word = replace_word_idx(word_to_idx)

    df['Comment'] = df['Comment'].apply(lambda x: [word_to_idx[word] for word in x])

    return df[["Image", "Comment"]].groupby(by="Image")["Comment"].apply(list).reset_index(), word_to_idx, idx_to_word, vocab_size#, df1
