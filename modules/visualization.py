import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_bb(image, bb_coords, testing=False):

	'''Draw a bounding box at the given coordinates.	

	Args:    
	    image: (numpy array), shape is out from 'class Resize_Visual' -
	    	resize input image
        
	    bb_coords: (numpy array), shape is 4 -
	    	list of 4 coordinates of a bounding box
	    
	    testing(callable,default=False): (string) - test mode	    
	'''
      
	# color of bounding box
	# draw different color box while prediction
	if testing: c = 'r' # red - for test 
	else: c = 'g'       # green - for basic    
	plt.imshow(image)
	# get current axis
	ax = plt.gca()
	b = bb_coords
	# (xmin, ymin), xmax-xmin, ymax-ymin
	rect = Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1], linewidth=2,edgecolor=c,facecolor='none')
	ax.add_patch(rect)
