# data from https://github.com/positronix-ai/dvc-datasets/tree/main/lkq/object-detection/dcew_crank_gears_20230118
# this defines a dataset that we would like to train YoloX on
from typing import Dict
from datasets_wrapper import Dataset
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import json 
import torch

class DCEWCrankGearsDataset(Dataset):
    def __init__(self, input_dimension, transforms=None, mosaic=False):
        super().__init__(input_dimension, mosaic)
        
        self.base_path = "/home/ubuntu/ptdev/data/crank"
        
        # valid classes for objects in the dataset
        self.classes = ['inner_fiducial', 'gear_center', 'inner_link', 'outer_fiducial', 'outer_link']


        # list of image filenames in the dataset
        self.image_fnames = self._get_image_fnames()
       
        #dictionary of labels indexed by fname
        self.labels = self._get_labels()

        self.transforms = transforms
 
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
        
        if self.transforms:
            image, labels = self.transforms(image, labels, self.input_dim)
        
        return image, labels

    
    def _get_image_fnames(self):
        images_path = f"{self.base_path}/image_files/"
        files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        return files
    
    def _get_labels(self):
        """
        Gets the labels for the dataset, preprocessing them first 
        """
        annotation_path=f"{self.base_path}/annotations.json"
        f = open(annotation_path)
        raw_labels = json.load(f)
        
        #ensure the class names in the dataset match the class names provided by self.classes
        self._analyze_labels(raw_labels)
           
        processed_labels = {}

        for raw_label in raw_labels:
            processed_labels[raw_label['image_file']] = self._process_label(raw_label)
            
        if len(processed_labels.items()) != len(self.image_fnames):
            raise ValueError("len mismatch between images and labels for dataset")

        return processed_labels
    
    
    def _process_label(self, raw_label:Dict): 
        '''
        convert the raw label provided by the dataset to the format Yolo X expects
        '''
        # first remove column that is not useful

        
        # intermediate storage for processed rois
        processed_rois = [] 
        
        for roi in raw_label['rois']:
            # verify the ROI is in a format we expect 
            label_version = roi['version']
            if label_version != '1.0':
                raise ValueError(f'ROI label_version unexpected, instead it is {label_version}')
            
            # verify the geometry is in a format we expect
            geometry_version = roi['geometry']['version']
            if geometry_version != '1.0':
                raise ValueError(f'ROI geometry_version unexpected, instead it is {geometry_version}')
            

            # extract useful information from the ROI
   
            class_name = roi['label']
            left = roi['geometry']['left']
            top = roi['geometry']['top']
            right = roi['geometry']['right']
            bottom = roi['geometry']['bottom'] 
        
            # convert classname to an float representing the class 
            try:
                #index classes from 1 to n instead of 0 to n - 1
                class_number = float(self.classes.index(class_name))
            except:
                raise ValueError(f'Class string: "{class_name}" not present in classes: {self.classes}')
            
            # convert the roi to the format yolo expects: 
            # each label consists of [class, xc, yc, w, h]:
            #         class (float): class index.
            #         xc, yc (float) : center of bbox whose values range from 0 to 1.
            #         w, h (float) : size of bbox whose values range from 0 to 1.
            
            xc = (left + right)/2
            yc = (top + bottom)/2 
            
            w = right - left 
            h = bottom - top
            
            roi_tensor = torch.FloatTensor([class_number, xc, yc, w, h])
            processed_rois.append(roi_tensor)
        
        #stack along first axis, as yolo expects
        processed_label = torch.stack(processed_rois, 0)
        return processed_label

    def _analyze_labels(self,raw_labels):
        '''
        helper function to ensure the input data has the same classes as we expect
        '''
        classes = set()
        for raw_label in raw_labels:
            for roi in raw_label['rois']:
                classes.add(roi['label'])
        
        if classes != set(self.classes):
            raise ValueError(f"The data's labels do not match the labels provided to the python dataset object. Data labels: {classes}. Dataset object labels: {self.classes}")

