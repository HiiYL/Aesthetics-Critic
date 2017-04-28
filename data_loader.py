# python train.py --cuda --dataset aesthetics-full
import torchvision.transforms as transforms
from os import listdir
from os.path import join
import os

import torch.utils.data as data
import nltk

from pandas import HDFStore
import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import random
from bisect import bisect
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,dataframe_dir,vocab, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.store = HDFStore(dataframe_dir)
        print("[!] Loading {} set... ".format(image_dir))
        ava_table = self.store['labels_train']
        self.a_path = join(image_dir, "a")
        
        self.image_filenames = [ "{}.jpg".format(x) for x in ava_table.index ]

        self.vocab = vocab
        self.transform = transform
        # self.comments = ava_table.comments.as_matrix()

    def __getitem__(self, index):
        # Load Image
        # input, shape = load_img(join(self.a_path, self.image_filenames[index]), returnShape=True)


        image = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        ## Split by comment then randomly pick one!
        comments = self.comments[index].split(" [END] ")
        #comments.sort(key=len)
        # if(len(comments) > 1):
        #     weighted_comments = [(comment, len(comment)) for comment in comments ]
        #     comment = self.weighted_choice(weighted_comments)
        # else:
        comment = comments[0]
        #comment = comments[len(comments) - 1]
            
        tokens = nltk.tokenize.word_tokenize(str(comment).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def weighted_choice(self,choices):
        values, weights = zip(*choices)
        total = 0
        cum_weights = []
        for w in weights:
            total += w
            cum_weights.append(total)
        x = random.random() * total
        i = bisect(cum_weights, x)
        return values[i]

    def __len__(self):
        return len(self.image_filenames)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(image_dir, dataframe_dir, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    dset = DatasetFromFolder(image_dir=image_dir,
                       dataframe_dir=dataframe_dir,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader