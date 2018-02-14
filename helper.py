# -*- coding: utf-8 -*-
"""
Contains all class definitions and imports necessary to implement MRI segmentation imports
and alignments. Built based on MRI Processing MATLAB pipeline by Thien-Khoi Phung.

Created on Wed Feb 14 11:07:25 2017

@author: cdw2be
"""

import scipy.io as spio
import numpy as np

class ImportHelper():
	
	"""Contains several functions to assist with import functionality for modeling and mesh data.
	"""
	
	@staticmethod
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
		return(ImportHelper._check_keys(data))
	
	@staticmethod
	def _check_keys(dict_pass):
		"""
		checks if entries in dictionary are mat-objects. If yes
		todict is called to change them to nested dictionaries
		"""
		for key in dict_pass:
			if isinstance(dict_pass[key], spio.matlab.mio5_params.mat_struct):
				dict_pass[key] = ImportHelper._todict(dict_pass[key])
		return(dict_pass)        
	
	@staticmethod
	def _todict(matobj):
		"""
		A recursive function which constructs from matobjects nested dictionaries
		"""
		dict = {}
		for strg in matobj._fieldnames:
			elem = matobj.__dict__[strg]
			if isinstance(elem, spio.matlab.mio5_params.mat_struct):
				dict[strg] = ImportHelper._todict(elem)
			elif isinstance(elem,np.ndarray):
				dict[strg] = ImportHelper._tolist(elem)
			else:
				dict[strg] = elem
		return(dict)
	
	@staticmethod
	def _tolist(ndarray):
		"""
		A recursive function which constructs lists from cellarrays 
		(which are loaded as numpy ndarrays), recursing into the elements
		if they contain matobjects.
		"""
		elem_list = []            
		for sub_elem in ndarray:
			if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
				elem_list.append(ImportHelper._todict(sub_elem))
			elif isinstance(sub_elem,np.ndarray):
				elem_list.append(ImportHelper._tolist(sub_elem))
			else:
				elem_list.append(sub_elem)
		return(elem_list)
		
class StackHelper():
	"""Contains helper functions to import and process stacks from SEGMENT-derived MAT files.
	"""
	
	@staticmethod
	def getContourFromStack(endo_stack, epi_stack, sastruct, rv_insertion_pts, septal_slice, apex_base_pts, scar_stack=np.empty([0])):
		"""Using the stacks, construct the endo and epi contours in proper format, abs points, axis center, and (if passed) scar
		
		Uses several matrix transformations to convert the stack data into the finalized contours.
		args:
			endo_stack (array): Original endocardial stack
			epi_stack (array): Original epicardial stack
			sastruct (dict): Setstruct from the SEGMENT MAT file
			rv_insertion_pts (array): RV points indicated in the SEGMENT file
			septal_slice (integer): The slice containing the rv points
			apex_base_pts (array): The points from the long-axis image indicating apex and base
			scar_stack (array): Original scar point stack
		returns:
			abs_shifted (array): Shifted apex, basal, and septal points
			endo (array): Modified endocardial contours
			epi (array): Modified epicardial contours
			axis_center (array): Center of slice by apex-base axis calculation
			scar_all (array): Shifted scar points to align with new endo and epi contours
		"""
		# Pull elements from passed args
		kept_slices = sastruct['KeptSlices']
		apex_pts = apex_base_pts[-2].reshape([1, 3])
		base_pts = apex_base_pts[-1].reshape([1, 3])
		septal_pts = rv_insertion_pts[(2, 0, 1), :]
		# Calculate the z-orientation and store it in the m array
		cine_z_orientation = np.cross(sastruct['ImageOrientation'][0:3], sastruct['ImageOrientation'][3:6])
		cine_m = np.array([sastruct['ImageOrientation'][3:6], sastruct['ImageOrientation'][0:3], cine_z_orientation])
		# Multiply the xyz parts of the stack by the m array and store
		transform_endo = np.transpose(cine_m@np.transpose(endo_stack[:, 0:3]))
		transform_epi = np.transpose(cine_m@np.transpose(epi_stack[:, 0:3]))
		transform_abs = np.transpose(cine_m@np.transpose(np.append(apex_pts, np.append(base_pts, septal_pts, axis=0), axis=0)))
		if scar_stack.size:
			scar_z_orientation = np.cross(sastruct['ImageOrientation'][0:3], sastruct['ImageOrientation'][3:6])
			scar_m = np.array([sastruct['ImageOrientation'][3:6], sastruct['ImageOrientation'][0:3], scar_z_orientation])
			transform_scar = np.transpose(scar_m@np.transpose(scar_stack[:, 0:3]))
		# Calculate the apex-base elements from the transformed abs points
		ab_dist = transform_abs[1, :] - transform_abs[0, :]
		ab_x = [transform_abs[0, 0] + (ab_dist[0]*(item/100)) for item in list(range(101))]
		ab_y = [transform_abs[0, 1] + (ab_dist[1]*(item/100)) for item in list(range(101))]
		ab_z = [transform_abs[0, 2] + (ab_dist[2]*(item/100)) for item in list(range(101))]
		# Generate a list of z values (by slice)
		z_loc = [transform_endo[np.where(endo_stack[:, 4] == cur_slice)[0][0], 2] for cur_slice in kept_slices]
		# Calculate the m-array for the each slice and store in a list
		m_slices = [(z_loc_cur - transform_abs[0,2])/ab_dist[2] for z_loc_cur in z_loc]
		# Calculate the apex-base axis based on the slice m values and ab_dist
		ba_axis_x = [m_slice * ab_dist[0] + transform_abs[0, 0] for m_slice in m_slices]
		ba_axis_y = [m_slice * ab_dist[1] + transform_abs[0, 1] for m_slice in m_slices]
		ba_axis_intercept = np.transpose(np.array([ba_axis_x, ba_axis_y, z_loc]))
		#Set up lists before appending values
		slice_center = []
		center_axis_diff = []
		endo_shifted = [None] * transform_endo.shape[0]
		epi_shifted = [None] * transform_epi.shape[0]
		axis_center = []
		center_axis_diff = []
		if scar_stack.size:
			scar_shifted = [None] * transform_scar.shape[0]
		# Iterate through and calculate the new endo and epi values
		for i in range(len(kept_slices)):
			# Pull the current slice and find which values in the stack are in the correct slice
			cur_slice = kept_slices[i]
			slice_endo_inds = np.where(endo_stack[:, 4] == cur_slice)[0]
			slice_epi_inds = np.where(epi_stack[:, 4] == cur_slice)[0]
			# Get the center of the slice for both epicardial center (slice) and apex-base line (axis)
			slice_center.append(np.mean(transform_epi[slice_epi_inds, :], axis=0))
			axis_center.append(ba_axis_intercept[i, :])
			# Calculate the difference between slice center and axis center
			center_axis_diff.append(slice_center[i] - axis_center[i])
			# Shift the slices by the difference in centers and shift, to align the centers
			for j in slice_endo_inds:
				endo_shifted[j] = transform_endo[j] - center_axis_diff[i]
			for j in slice_epi_inds:
				epi_shifted[j] = transform_epi[j] - center_axis_diff[i]
			if scar_stack.size:
				slice_scar_inds = np.where(scar_stack[:, 4] == cur_slice)[0]
				for j in slice_scar_inds:
					scar_shifted[j] = transform_scar[j] - center_axis_diff[i]
		# Get the new septal slice by calculating the adjustment from the topmost slice
		septal_slice_new = int(septal_slice[0][0] - endo_stack[0, 4] + 1)
		# Calculate the array to subtract from transform_abs to get the shifted apex, basal, and septal points
		sub_arr = np.array([[0, 0, 0], [0, 0, 0], [center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0], 
			[center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0], [center_axis_diff[septal_slice_new][0], center_axis_diff[septal_slice_new][1], 0]])
		abs_shifted = transform_abs - sub_arr
		
		# Select data and transform to array
		endo = [np.array(endo_shifted)[np.where(endo_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
		epi = [np.array(epi_shifted)[np.where(epi_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
		if scar_stack.size:
			scar_all = [np.array(scar_shifted)[np.where(scar_stack[:, 4] == jz)[0]] for jz in range(1, 1+int(max(endo_stack[:, 4])))]
			return([abs_shifted, endo, epi, axis_center, scar_all])
		else:
			return([abs_shifted, endo, epi, axis_center])