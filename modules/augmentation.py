from skimage import io, color, transform
import os

import imageio
import shutil

import imgaug as ia
import imgaug.augmenters as iaa

def add_aug_image(img_list_path, save_path, seq, index, number_iter):

	''' Create and save augmented image and file with annotation (label, coordinates of the bounding box).

	Args:    
	    img_list_path: (string) - image path
	    
	    save_path: (string) - path to the folder for saving images
        
	    seq: (list) - List augmenter containing child augmenters to apply to inputs (see the imgaug library)
	    
	    index: (int) - image number in one augmentation cycle
	    
	    number_iter: (int) - augmentation cycle number	     
	'''        
        
	img = imageio.imread(img_list_path[index]) # load image
	if len(img.shape) != 3: img = color.gray2rgb(img) # convert the image to rgb format if the original format is gray
	img_path = os.path.splitext(img_list_path[index])[0] # define the path to the file with annotations 
	anno_path = img_path + '.txt' 
	
	# load label and coordinates of the bounding box from annotation file 
	with open(anno_path, 'r') as f:
	    anno = [word for line in f for word in line.split(' ')]
	label = anno[0]       
	xmin, ymin, xmax, ymax = int(anno[1]), int(anno[2]), int(anno[3]), int(anno[4])

	# get the augmented image and changed coordinates of the bounding box using the imgaug library
	bb_ia = ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
	bbs = [bb_ia]
	image_aug, bb_aug = seq(image=img, bounding_boxes=bbs)        
	xmin, ymin, xmax, ymax = int(bb_aug[0].x1), int(bb_aug[0].y1), int(bb_aug[0].x2), int(bb_aug[0].y2)
	
	#save the augmented image in a new folder
	img_name = img_path.split('/')[-1]
	new_name_img = img_name + '_' + str(number_iter+1) + '_aug.jpg'
	imageio.imwrite(save_path + new_name_img, image_aug)        
	
	#save the new annotation file in a new folder
	new_name_anno = img_name + '_' + str(number_iter+1) + '_aug.txt'
	out_anno = open(save_path + new_name_anno, 'w')
	out_anno.write(label + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax))
	out_anno.close      
