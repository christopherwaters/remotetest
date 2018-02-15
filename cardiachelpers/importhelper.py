import numpy as np
import scipy.io as spio

def loadmat(filename):
	"""
	this function should be called instead of direct spio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	
	This and all sub-functions are based on jpapon's answer here:
		https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
	"""
	data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return(_check_keys(data))

def getTimeIndices(contour, time_pts, timepoint=0):
	"""Generate indices of the contour that match with the indicated time point.
	
	args:
		contour: The endo or epi contour from getEndoEpiFromStack
		time_pts: The list of all timepoints. The fourth column from endo/epi_stack
		timepoint: The timepoint of interest (passed an an index)
	returns:
		all_slice_inds (list): A list of lists, indicating indices per slice that fall in the desired timepoint.
	"""
	# Get all indices that match the time point
	time_selected = np.where(time_pts == timepoint)[0]
	# Get the possible range of indices for each slice
	#	Initial stack is a single list, contour is split by slice, so must correct for that
	contour_slice_inds = [0] + [contour_i.shape[0] for contour_i in contour]
	contour_slice_range = np.cumsum(contour_slice_inds)
	# Set up list to store slice index lists
	all_slice_inds = [None] * len(contour)
	# Iterate through contour and pull per-slice timepoints
	for i in range(len(contour)):
		cur_slice_timepts = [time_selected_i - contour_slice_range[i] for time_selected_i in time_selected if contour_slice_range[i] <= time_selected_i < contour_slice_range[i+1]]
		all_slice_inds[i] = cur_slice_timepts
	return(all_slice_inds)
	
def _check_keys(dict_pass):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict_pass:
		if isinstance(dict_pass[key], spio.matlab.mio5_params.mat_struct):
			dict_pass[key] = _todict(dict_pass[key])
	return(dict_pass)        

def _todict(matobj):
	"""
	A recursive function which constructs from matobjects nested dictionaries
	"""
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, spio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		elif isinstance(elem,np.ndarray):
			dict[strg] = _tolist(elem)
		else:
			dict[strg] = elem
	return(dict)

def _tolist(ndarray):
	"""
	A recursive function which constructs lists from cellarrays 
	(which are loaded as numpy ndarrays), recursing into the elements
	if they contain matobjects.
	"""
	elem_list = []            
	for sub_elem in ndarray:
		if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
			elem_list.append(_todict(sub_elem))
		elif isinstance(sub_elem,np.ndarray):
			elem_list.append(_tolist(sub_elem))
		else:
			elem_list.append(sub_elem)
	return(elem_list)
