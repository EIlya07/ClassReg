import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import io, color, transform
import os
from glob import glob


class CustomDataset(Dataset):
	''' Custom dataset.
	
	Args:    
	    folder: (string) - path to the folder with images
	    
	    folder_aug (callable, optional): (string) - optional path to the folder with augmented images
	    
	    transform (callable, optional): optional transform to be applied on a sample.
	'''

	def __init__(self, folder, folder_aug=None, transform=None):        
		self.transform = transform        
		self.img_list_path = glob(folder+'/*.jpg') # create a list of paths to images in the folder
		if folder_aug: # if true, then add augmented images in dataset
		    img_aug_list_path = glob(folder_aug+'/*.jpg')
		    self.img_list_path += img_aug_list_path          
        
	def __len__(self):
		len_files = len(self.img_list_path)
		return len_files
    
	def __getitem__(self, index):
		'''
		Args:
		    index: (int) - sequential number of the image (file) in the folder
		    
		Returns: 
		    sample: (dict) - sample (dictionary consisting of image, label and boonding box coordinates)
		'''
		        
		img = io.imread(self.img_list_path[index]) # load image
		if len(img.shape) != 3: img = color.gray2rgb(img) # convert the image to rgb format if the original format is gray
		img_name = os.path.splitext(self.img_list_path[index])[0] # define the path to the file with annotations 
		anno_path = img_name + '.txt' 
		
		# load label and coordinates of the bounding box from annotation file 
		with open(anno_path, 'r') as f:
		    anno = [word for line in f for word in line.split(' ')]
		label = int(anno[0])
		
		# since we have 2 classes, override labels: 2(dog) with 1, 1(cat) with 0
		if label == 1: label = 0
		else: label = 1
		
		# get the augmented image and changed coordinates of the bounding box using the imgaug library
		xmin, ymin, xmax, ymax = int(anno[1]), int(anno[2]), int(anno[3]), int(anno[4])
		bb_coords = np.array([xmin, ymin, xmax, ymax])
		
		sample = {'image': img, 'label': label, 'bb_coords': bb_coords} # create sample (dict)      

		if self.transform: # if true, then applying transformations to images
		    sample = self.transform(sample)       

		return sample

#######################################################################################################################

class Resize(object):
	"""Resize the image in a sample to a given size.
	Args:
	    output_size: (tuple or int) - Desired output size. If tuple, output is
	        matched to output_size. If int, smaller of image edges is matched
	        to output_size keeping aspect ratio the same.
	"""
	
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		'''
		Args:
		    sample: (dict) - sample (dictionary consisting of image, label and boonding box coordinates)
		    
		Returns: 
		    (dict) - dictionary consisting of resized image, label and changed boonding box coordinates
		''' 
		    
		image, label, bb_coords = sample['image'], sample['label'], sample['bb_coords']
        
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h   
		else:
			new_h, new_w = self.output_size
		    
		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w)) # resize images using the skimage

		bb_coords = bb_coords.astype(float)

		# transform the coordinates in accordance with the new dimensions of the image and make them dimensionless
		bb_coords[0::2] = bb_coords[0::2] * (new_w / (w * w)) 
		bb_coords[1::2] = bb_coords[1::2] * (new_h / (h * h))                

		return {'image': img, 'label': label, 'bb_coords': bb_coords}
      
###########################################################################################################################

class Resize_Visual(object):
	"""Resize the image in a sample to a given size (for visualization).

	Args:
	    output_size (tuple or int): Desired output size. If tuple, output is
	        matched to output_size. If int, smaller of image edges is matched
	        to output_size keeping aspect ratio the same.
	"""
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
        
	def __call__(self, sample):
		'''
		Args:
		    sample: (dict) - sample (dictionary consisting of image, label and boonding box coordinates)
		    
		Returns: 
		    (dict) - dictionary consisting of resized image, label, changed boonding box coordinates and
		        original image dimensions    
		''' 
		
		image, label, bb_coords = sample['image'], sample['label'], sample['bb_coords']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h   
		else:
			new_h, new_w = self.output_size
		    
		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w)) # resize images using the skimage
		
		# transform the coordinates in accordance with the new dimensions of the image and make them dimensionless
		bb_coords[0::2] = bb_coords[0::2] * (new_w / w)
		bb_coords[1::2] = bb_coords[1::2] * (new_h / h)        

		return {'image': img, 'label': label, 'bb_coords': bb_coords, 'param_img': [h,w]}
      
##############################################################################################################################

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		'''
		Args:
		    sample: (dict) - sample (dictionary consisting of image, label and boonding box coordinates)
		    
		Returns: 
		    (dict) - dictionary consisting of changed type image, label and changed type boonding box coordinates 
		''' 
    
		image, label, bb_coords = sample['image'], sample['label'], sample['bb_coords']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image),
			'label': label,
			'bb_coords': torch.from_numpy(bb_coords)}
      
##############################################################################################################################

class Normalize(object):
	"""Normalize images"""
  
	def __call__(self, sample):
		'''
		Args:
		    sample: (dict) - sample (dictionary consisting of image, label and boonding box coordinates)
		    
		Returns: 
		    (dict) - dictionary consisting of normalized image, label and boonding box coordinates 
		'''		
    
		image, label, bb_coords = sample['image'], sample['label'], sample['bb_coords']

		norm = transforms.Normalize([0.4737, 0.4421, 0.3916],
				             [0.2639, 0.2587, 0.2665])
		image = norm(image.float())
        
		return {'image': image, 'label': label, 'bb_coords': bb_coords}
