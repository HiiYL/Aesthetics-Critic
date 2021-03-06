import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

import random


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, mode, vocab, transform=None, mismatch_example=False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = "data/coco/merged2014"
        self.coco = COCO("data/coco/captions_{}2014_karpathy_split.json".format(mode))
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.mismatch_example = mismatch_example

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        # wrong_caption = coco.anns[random.choice(self.ids)]['caption']
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # wrong_target = torch.Tensor(caption)



        # if self.mismatch_example:
        #     mismatch_id = random.choice(self.ids)
        #     caption = coco.anns[mismatch_id]['caption']
        #     tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        #     caption = []
        #     caption.append(vocab('<start>'))
        #     caption.extend([vocab(token) for token in tokens])
        #     caption.append(vocab('<end>'))
        #     mismatch = torch.Tensor(caption)



        return image, target, img_id#, wrong_target

    def __len__(self):
        return len(self.ids) - 6 # hack until i can figure out how to fix <unk><unk><unk error 

class CocoValDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, mode, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = "data/coco/merged2014"
        self.coco = COCO("data/coco/captions_{}2014_karpathy_split.json".format(mode))
        #self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab

        img_id = self.ids[index]
        caption = self.coco.imgToAnns[img_id][0]['caption']

        #ann_id = self.ids[index]
        #caption = coco.anns[ann_id]['caption']
        #img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        #print(index)
        return image, target, img_id

    def __len__(self):
        return len(self.ids)

def collate_fn_wrong(data):
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
    images, captions, img_id, wrong_captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    wrong_lengths = [len(cap) for cap in wrong_captions]
    wrong_targets = torch.zeros(len(wrong_captions), max(wrong_lengths)).long()
    for i, cap in enumerate(wrong_captions):
        end = lengths[i]
        wrong_targets[i, :end] = cap[:end]

    return images, targets, lengths, img_id, wrong_targets, wrong_lengths

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
    images, captions, img_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths, img_id


def get_loader(mode, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    if mode == "train":
        coco = CocoDataset(mode=mode,
                           vocab=vocab,
                           transform=transform)
        collate_fn_to_use = collate_fn#collate_fn_wrong
    elif mode == "val":
        coco = CocoValDataset(mode=mode,
                           vocab=vocab,
                           transform=transform)
        collate_fn_to_use = collate_fn
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_to_use)
    return data_loader



class CocoImgDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_dir, annotation_dir, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = image_dir
        self.coco = COCO(annotation_dir)
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco

        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, img_id

    def __len__(self):
        return len(self.ids)