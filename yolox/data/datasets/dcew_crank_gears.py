# data from https://github.com/positronix-ai/dvc-datasets/tree/main/lkq/object-detection/dcew_crank_gears_20230118
# this defines a dataset that we would like to train YoloX on
from typing import Dict
from datasets_wrapper import Dataset
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import json 

class DCEWCrankGearsDataset(Dataset):
    def __init__(self, input_dimension, mosaic=False):
        super().__init__(input_dimension, mosaic)
        
        self.base_path = "/home/ubuntu/ptdev/data/crank"

        # list of image filenames in the dataset
        self.image_fnames = self._get_image_fnames()
       
        #dictionary of labels indexed by fname
        self.labels = self._get_labels()

        # valid classes for objects in the dataset
        self.classes = ["gearcenter", "fiducial1", "link1", "fiducial2","link2"]

        raise NotImplementedError

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        """
        One image / label pair for the given index is picked up and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[num_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        image_fname = self.image_fname[idx]
        image_path = f"{self.base_path}/image_files/{image_fname}"
        image = read_image(image_path)
        labels = self.labels[image_fname]
        return image, labels

    
    def _get_image_fnames(self):
        images_path = f"{self.base_path}/image_files/"
        files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        return files
    
    def _get_labels(self):
        """
        Gets the labels for the dataset and processes them into the format the 
        """
        annotation_path=f"{self.base_path}/annotations.json"
        f = open(annotation_path)
        raw_labels = json.load(f)
        print(raw_labels[0])

        processed_labels = {}

        for raw_label in raw_labels:
            processed_labels[raw_label['image_file']] = self._process_label(raw_label)
            break
        
        print()
        print(processed_labels)
        print()

        return processed_labels
    
    
    def _process_label(self, raw_label:Dict): 
        '''
        convert the raw label provided by the dataset to the format Yolo X expects
        '''
        for roi in raw_label['rois']:
            pass


