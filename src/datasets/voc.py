import collections
import os
from xml.etree.ElementTree import Element as ET_Element
import torch
import cv2
#from .vision import VisionDataset
from torchvision.datasets import VisionDataset
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

#from .utils import download_and_extract_archive, verify_str_arg

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": os.path.join("VOCdevkit", "VOC2010"),
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "a3e00b113cfcfebf17e343f59da3caa1",
        "base_dir": os.path.join("VOCdevkit", "VOC2009"),
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": os.path.join("VOCdevkit", "VOC2008"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}

def name_transform(x):
    return int(x[-15:-4].replace('_',''))


class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        #self.year = verify_str_arg(year, "year", valid_values=[str(yr) for yr in range(2007, 2013)])

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        #self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        # if download:
        #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        id = torch.tensor(list(map(name_transform,self.images)))
        self.other = {'id': id}
        assert len(self.images) == len(self.targets)
        
        self.c2l = {
            (0, 0, 0): 0,        # Background
            (128, 0, 0): 1,      # Aeroplane
            (0, 128, 0): 2,      # Bicycle
            (128, 128, 0): 3,    # Bird
            (0, 0, 128): 4,      # Boat
            (128, 0, 128): 5,    # Bottle
            (0, 128, 128): 6,    # Bus
            (128, 128, 128): 7,  # Car
            (64, 0, 0): 8,       # Cat
            (192, 0, 0): 9,      # Chair
            (64, 128, 0): 10,    # Cow
            (192, 128, 0): 11,   # Dining Table
            (64, 0, 128): 12,    # Dog
            (192, 0, 128): 13,   # Horse
            (64, 128, 128): 14,  # Motorbike
            (192, 128, 128): 15, # Person
            (0, 64, 0): 16,      # Potted Plant
            (128, 64, 0): 17,    # Sheep
            (0, 192, 0): 18,     # Sofa
            (128, 192, 0): 19,   # Train
            (0, 64, 128): 20,    # TV/Monitor
            (224,224,192):0,     # Background
        }
        self.keys = np.array(list(self.c2l.keys()))
            
        self.values = np.array(list(self.c2l.values()))
        
        
    def mask_transform(self,mask):
        w,h,c = mask.shape
        distances = np.sum(np.absolute(mask.reshape(-1,1,3) - self.keys.reshape(1,22,3)), axis=-1)
        indices = np.argmin(distances, axis=1)
        return self.values[indices].reshape(w,h)
        # distances = torch.sum((mask.view(-1,3).unsqueeze(1) - self.keys.unsqueeze(0)).abs(), dim=-1)
        # indices = torch.argmin(distances, dim=1)
        # return self.values[indices].view(w,h)

    def __len__(self) -> int:
        return len(self.images)


class VOCSegmentation(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

            
        

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        img = cv2.imread(self.images[index],cv2.IMREAD_COLOR)
        target = self.mask_transform(cv2.imread(self.masks[index],cv2.IMREAD_COLOR)[:,:,::-1])
        #target = np.array(Image.open(self.masks[index]))
        # img = cv2.imread(self.images[index],cv2.IMREAD_COLOR)
        # target = cv2.imread(self.masks[index],cv2.IMREAD_COLOR)
        other = {'id' : torch.tensor(self.other['id'][index])}
        #other = {'id' : self.images[index][-15:-4]}

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        input = {**other, 'data':img, 'target' : target}
        return input

        return img, target

ds = VOCSegmentation(root='/Users/karthiksanka/Desktop/dynamoAI/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training/src/data/voc/',image_set='train', download=False)
print("The end*************")

    
    
    # def __getitem__(self, index):
    #     data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
    #     other = {k: torch.tensor(self.other[k][index]) for k in self.other}
    #     input = {**other, 'data': data, 'target': target}
    #     if self.transform is not None:
    #         input = self.transform(input)
    #     return input
