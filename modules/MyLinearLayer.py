import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear2out(nn.Module):
	''' Custom Linear Lyear with two out for classification (class labels)  
	and regression (bounding box coordinates).
	
	Args:    
	    in_features: (int) - num of input parameters
	    
	    out_class: (int) - num of prediction clases
	    
	    out_reg: (int) - num of predicted coordinates of the bounding boxes
	'''
	def __init__(self, in_features: int, num_out_class: int, num_out_reg: int):		
		super(Linear2out,self).__init__()

		self.LinearClass = nn.Linear(in_features, num_out_class)
		self.LinearIn = nn.Linear(in_features, 500)        
		self.LinearReg = nn.Linear(500, num_out_reg)
        

	def forward(self, input: torch.Tensor):
		'''
		Сalculation of parameters on a forward pass.
		
		Arguments:    
		    input: (torch tensor), shape is (batch_size, N) -
		    	batch of layer input parameters	   

		Returns:
		    out_class: (torch tensor), shape is (batch_size, num_out_class) -
		    	list of predicted classes
		    	
		    out_reg: (torch tensor), shape is (batch_size, num_out_reg) -
		    	list of predicted coordinates of the bounding boxes		         
		'''
		out_class = self.LinearClass(input)
		linear_1 = self.LinearIn(input)
		relu = F.relu(linear_1, inplace=True) 
		out_reg = self.LinearReg(relu)

		return out_class, out_reg
