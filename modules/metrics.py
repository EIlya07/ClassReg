def bb_intersection_over_union(gt_batch, pr_batch):
  
	''' Calculating mioU (mean intersection over Union).

	Args:    
	    gt_batch: (torch tensor), shape is (batch_size, 4) -
	    	batch with given bounding boxes coordinates
        
	    pr_batch: (torch tensor), shape is (batch_size, 4) -
	    	batch with prediction bounding boxes coordinates

	Returns:
	    ave_ioU: (float) - average iou for the batch     
	'''
	
	batch_size = gt_batch.shape[0]	
	sum_iou = 0
	for i in range(batch_size):
		boxA = gt_batch[i]
		boxB = pr_batch[i]
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA) * max(0, yB - yA)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
		boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# sum iou for the batch
		sum_iou += iou
		# average iou for the batch
		ave_iou = sum_iou / batch_size   

	return ave_iou
