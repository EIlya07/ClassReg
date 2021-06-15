import numpy as np
import torch
import copy

from metrics import bb_intersection_over_union

device = torch.device("cuda:0") # use GPU 

def train_model(model, train_loader, val_loader, loss_class, loss_reg, optimizer, num_epochs, name_file_model, path_save_model, scheduler=None):
	
	''' Model training function.

	Args:
	    model: PyTorch class - model built using the PyTorch library
	    
	    train_loader: (PyTorch class) - data loader from PyTorch library used to load training data
	    
	    val_loader: (PyTorch class) - data loader from PyTorch library used to load validation data
	    
	    loss_class: (PyTorch class) - loss function for classification from the PyTorch library
	    
	    loss_reg: (PyTorch class) - loss function for regression from the PyTorch library
	    
	    optimizer: (PyTorch class) - method for optimizing neural network weights from the PyTorch library
	    
	    num_epochs: (int) - number of learning epochs
	    
	    name_file_model: (string) - the name of the file in which the parameters of the best model are saved
	    
	    path_save_model: (string) - path to the folder for saving best model
	    
	    scheduler (callable, optional): (PyTorch class) - learning rate adjustment method (annealing)	     
	    
	Returns:
	    train_loss_history: (list) - a list of training losses for each epoch
	    
	    train_accuracy_history: (list) - a list of training accuracy for each epoch
	    
	    train_miou_history: (list) - a list of training mioU for each epoch
	    
	    val_loss_history: (list) - a list of validation losses for each epoch
	    
	    val_accuracy_history: (list) - a list of validation accuracy for each epoch
	    
	    val_miou_history: (list) - a list of validation mioU for each epoch	       
	'''
	
	best_model = copy.deepcopy(model.state_dict())
	accuracy = 0.0 
	best_miou = 0.0
	time = 0.0
	train_loss_history = []
	train_miou_history = []
	train_accuracy_history = []
	val_loss_history = []
	val_miou_history = []
	val_accuracy_history = []

	for epoch in range(num_epochs): # epoch learning cycle
		model.train() # train mode

		save_lr = optimizer.param_groups[0]['lr'] # storing the value of the learning rate at each epoch      
		loss_accum = 0
		correct_samples = 0
		total_samples = 0
		sum_iou = 0
		# cycle of loading batches into the model and performing forward and backward pass        
		for i_step, data in enumerate(train_loader): 
			images = data['image']
			labels = data['label']
			bb_coords = data['bb_coords']
			batch_size = images.shape[0] 

			# convert to float for regression loss
			bb_coords = bb_coords.type(torch.cuda.FloatTensor)            
			
			# sending data to the device
			images_gpu = images.to(device) 
			labels_gpu = labels.to(device)
			bb_coords_gpu = bb_coords.to(device)            

			pr_class, pr_reg = model(images_gpu) # model prediction result
			
			# calculation of loss functions
			loss_c = loss_class(pr_class, labels_gpu)                       
			loss_r = loss_reg(pr_reg, bb_coords_gpu)
			loss_value = loss_c + loss_r
			
			# use the optimization method to adjust the weights
			optimizer.zero_grad()
			loss_value.backward()
			optimizer.step()

			loss_accum += float(loss_value) # accumulate losses for each batch

			_, indices = torch.max(pr_class, 1) # selection of the class index with the highest probability
			correct_samples += torch.sum(indices == labels_gpu) # comparison of predicted and true labels           
			total_samples += batch_size            

			sum_iou += bb_intersection_over_union(bb_coords_gpu, pr_reg) # call the function of calculating the mioU

		train_loss = loss_accum / (i_step + 1) # calculating learning loss for one epoch                                
		train_accuracy = float(correct_samples) / total_samples # calculation of the average accuracy score for one epoch
		train_miou = sum_iou / (i_step + 1)  # calculating the average mioU score for one epoch
		
		# calling the function of evaluating the model on the validation data
		val_loss, val_accuracy, val_miou, mean_time = compute_val(model, val_loader, loss_class, loss_reg) 
		
		# adding calculation results for one epoch to the lists	
		train_loss_history.append(train_loss)
		train_accuracy_history.append(train_accuracy)
		train_miou_history.append(train_miou)
		val_loss_history.append(val_loss)
		val_accuracy_history.append(val_accuracy)
		val_miou_history.append(val_miou)
		
		# printing calculation results for one epoch to the lists
		print("Ep: %d ==> Train loss: %f, Val loss: %f, Train accuracy: %1.4f, Val accuracy: %1.4f, Train miou: %1.4f, Val miou: %1.4f, lr: %1.1e, time: %1.2f ms" % 				(epoch+1, train_loss, val_loss, train_accuracy, val_accuracy, train_miou, val_miou, save_lr, mean_time))

		if val_miou >= best_miou: # determine the best mioU on validation and copy the corresponding model
			best_miou = val_miou
			accuracy = val_accuracy
			time = mean_time
			best_model = copy.deepcopy(model.state_dict())       

		if scheduler: # check the availability of a method for adjusting the learning rate (annealing)
			scheduler.step(val_loss)           

	# save the model with the best miou for validation and print the calculation parameters
	torch.save(best_model, path_save_model + name_file_model) 
	print('Best miou = %f, Validation accuracy = %f, Time = %f ms' % (best_miou, accuracy, time))

	return train_loss_history, train_accuracy_history, train_miou_history, val_loss_history, val_accuracy_history, val_miou_history
  
##########################################################################################################################################
  
def compute_val(model, val_loader, loss_class, loss_reg): 
	
	''' Model evaluation function on validation data.

	Args:
	    model: PyTorch class - model built using the PyTorch library
	    
	    val_loader: (PyTorch class) - data loader from PyTorch library used to load validation data
	    
	    loss_class: (PyTorch class) - loss function for classification from the PyTorch library
	    
	    loss_reg: (PyTorch class) - loss function for regression from the PyTorch library	    
	    
	Returns:
	    val_loss: (float) - result of evaluation of losses on validation data
	    
	    val_accuracy: (float) - result of evaluation of accuracy on validation data
	    
	    val_miou: (float) - result of evaluation of mioU on validation data
	    
	    mean_time: (float) - estimation of the average time of passage of one image through the model	       
	'''
    
	model.eval() # Evaluation Mode (forward pass only) 

	loss_accum = 0
	correct_samples = 0
	total_samples = 0
	sum_iou = 0.0
	sum_timings = 0.0
	with torch.no_grad(): # turn off the calculation of gradients	
		# cycle of loading batches into the model and performing forward pass    
		for i_step, data in enumerate(val_loader):
			# define events to calculate the time of passage of the image through the model
			starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

			images = data['image']
			labels = data['label']
			bb_coords = data['bb_coords']     
			batch_size = images.shape[0]

			# convert to float for regression loss
			bb_coords = bb_coords.type(torch.cuda.FloatTensor)            
			
			# sending data to the device
			images_gpu = images.to(device)
			labels_gpu = labels.to(device)
			bb_coords_gpu = bb_coords.to(device)

			starter.record() # start time recording            
			pr_class, pr_reg = model(images_gpu) # model prediction result
			ender.record() # end time recording
			torch.cuda.synchronize() # synchronize time
			
			#calculate the transit time of one image and summarize
			curr_time = starter.elapsed_time(ender)
			sum_timings += curr_time / batch_size 
			
			# calculation of loss functions
			loss_c = loss_class(pr_class, labels_gpu)                       
			loss_r = loss_reg(pr_reg, bb_coords_gpu)
			loss_value = loss_c + loss_r

			loss_accum += float(loss_value) # accumulate losses for each batch

			_, indices = torch.max(pr_class, 1) # selection of the class index with the highest probability
			correct_samples += torch.sum(indices == labels_gpu) # comparison of predicted and true labels 
			total_samples += batch_size            

			sum_iou += bb_intersection_over_union(bb_coords_gpu, pr_reg) # call the function of calculating the mioU 

		val_loss = loss_accum / (i_step + 1) # calculating validation loss for one epoch                                
		val_accuracy = float(correct_samples) / total_samples # calculation of the average accuracy score for one epoch
		val_miou = sum_iou / (i_step + 1) # calculating the average mioU score for one epoch
		mean_time = sum_timings / (i_step + 1) # calculation of the average transit time of one image through the model for one epoch		
        
	return val_loss, val_accuracy, val_miou, mean_time
  
###############################################################################################################################################

def test_model(model, test_loader):

	''' Function of testing the model on test data.

	Args:
	    model: PyTorch class - model built using the PyTorch library
	    
	    test_loader: (PyTorch class) - data loader from PyTorch library used to load validation data	    	    
	    
	Returns:
	    test_accuracy: (float) - result of testing of accuracy on test data
	    
	    val_miou: (float) - result of testing of mioU on test data
	    
	    test_labels: (list of numpy arrays) - list of predicted labels for images
	    
	    test_bb_coords: (list of numpy arrays) - list of predicted coordinates of bounding boxes for images
	    
	    test_iou_all: (list of torch tensors) - list of computed metrics (ioU) for images	       
	'''
    
	model.eval() # Evaluation Mode (forward pass only) 

	with torch.no_grad(): # turn off the calculation of gradients
		correct_samples = 0
		total_samples = 0
		sum_iou = 0
		test_labels = []
		test_bb_coords = []
		ishod_bb = []
		test_iou_all = []		
		# cycle of loading batches into the model and performing forward pass
		for i_step, data in enumerate(test_loader):
			images = data['image']
			labels = data['label']
			bb_coords = data['bb_coords']
			batch_size = images.shape[0]    
				
			# convert to float for regression loss
			bb_coords = bb_coords.type(torch.cuda.FloatTensor)
			
			# sending data to the device
			images_gpu = images.to(device)
			labels_gpu = labels.to(device)
			bb_coords_gpu = bb_coords.to(device)            

			pr_class, pr_reg = model(images_gpu) # model prediction result

			_, indices = torch.max(pr_class, 1) # selection of the class index with the highest probability
			correct_samples += torch.sum(indices == labels_gpu) # comparison of predicted and true labels 
			total_samples += batch_size

			iou = bb_intersection_over_union(bb_coords_gpu, pr_reg)  # call the function of calculating the ioU
			sum_iou += iou # sum ioU

			# add predicted labels, bounding boxes and iou to the lists for all pictures
			test_labels.append(indices.cpu().numpy())            
			test_bb_coords.append(pr_reg[0].cpu().numpy())
			test_iou_all.append(iou)            
            
		test_accuracy = float(correct_samples) / total_samples # calculating accuracy on test data
		test_miou = sum_iou / (i_step + 1) # calculating mioU on test data                            
        
	return test_accuracy, test_miou, test_labels, test_bb_coords, test_iou_all
