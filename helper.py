# -*- coding: utf-8 -*-
"""
Contains all class definitions and imports necessary to implement MRI segmentation imports
and alignments. Built based on MRI Processing MATLAB pipeline by Thien-Khoi Phung.

Created on Wed Feb 14 11:07:25 2017

@author: cdw2be
"""

import scipy.io as spio
import numpy as np
import scipy as sp

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
			
	@staticmethod
	def transformStack(setstruct, slice_number=0, layer='endo'):
		"""Perform the actual stack rotation and calculate the rotation matrix.
		
		args:
			setstruct (dict): setstruct output from the SEGMENT file
			slice_number (int): The slice number to be used for transformation
			layer (string): Which layer to run through transformation
		"""
		# Pull x_pix, y_pix, and set run_xyz based on layer selection
		if layer == 'endo':
			x_pix = setstruct['endo_x'][:,slice_number,:]
			y_pix = setstruct['endo_y'][:,slice_number,:]
			run_xyz = True
		elif layer == 'epi':
			x_pix = setstruct['epi_x'][:,slice_number,:]
			y_pix = setstruct['epi_y'][:,slice_number,:]
			run_xyz = True
		elif layer == 'mask':
			x_pix = setstruct['mask_x'][:,slice_number,:]
			y_pix = setstruct['mask_y'][:,slice_number,:]
			run_xyz = np.isnan(sum(sum(setstruct['mask_x'][:,slice_number,:]))) < 1
		elif layer == 'long':
			run_xyz = False
		else:
			print('Incorrect Layer Selection. Defaulting to endo.')
			x_pix = setstruct['endo_x'][:,slice_number,:]
			y_pix = setstruct['endo_y'][:,slice_number,:]
			run_xyz = True
		# If endo, epi, or scar where there are no NaN values:
		if run_xyz:
			# Round the x_pix and y_pix arrays
			x_pix_round = np.round(x_pix)
			y_pix_round = np.round(y_pix)
			# Set up lists that need to be altered during the loop
			perim_length = [None] * x_pix_round.shape[0]
			xy_pts = [None] * x_pix_round.shape[0]
			for i in range(x_pix_round.shape[0]):
				# If the layer isn't scar and there are NaN values in the rounded x_pix
				#		Set the current point to all NaN and skip the rest of the loop
				if (not layer == 'mask') and (np.any(np.isnan(x_pix_round[:,i]))):
					xy_pts[i] = [np.nan, np.nan, np.nan]
					continue
				# Concatentate x_pix and y_pix into a single array
				xy_pix_round = np.array([x_pix_round[i,:].tolist(), y_pix_round[i,:].tolist()])
				# If the layer is scar, set the current perim_length point to the 2nd dimension size of xy_pix_round
				#		Otherwise, grab the unique points and then the size of the 2nd dimension
				if layer == 'mask':
					perim_length[i] = xy_pix_round.shape[1]
				else:
					perim_length[i] = ((np.unique(xy_pix_round,axis=1)).shape)[1]
				# Set perim points as linearly-spaced series of points from 0 to 1, with x_pix.shape[1]+1 number of points
				perim_pts = np.linspace(0,1,x_pix.shape[1]+1)
				# The interp points should be based on the value stored earlier in perim_length, +1
				interp_perim_pts = np.linspace(0,1,perim_length[i]+1)
				# Convert x_pix and y_pix to lists
				x_pix_arr = x_pix[i,:].tolist()
				y_pix_arr = y_pix[i,:].tolist()
				# Put the x_pix and y_pix lists into a new array together
				pix_arr = np.array([x_pix_arr, y_pix_arr])
				if layer == 'mask':
					# Sort the pix_arr array, putting 0 values at the bottom of the list
					#		Primary sort is along the second column, secondary sort along the first
					#		The conversion to and from NaN puts 0 values at the end of the list instead of the front
					pix_arr[pix_arr == 0] = np.nan
					pix_arr = pix_arr[:, np.argsort(pix_arr[1], kind='mergesort')]
					pix_arr[np.isnan(pix_arr)] = 0
				# Set perim_xy_pts to be pix arr, with the first point repeated at the end
				pix_append = np.reshape(pix_arr[:, 0], [2,1])
				perim_xy_pts = np.append(pix_arr.transpose(),pix_append.transpose(),axis=0)
				# Define a cubic interpolation function based on perim_pts and perim_xy_pts
				interp_func = sp.interpolate.interp1d(perim_pts,perim_xy_pts,kind='cubic',axis=0)
				# Run the interpolation function on interp_perim_pts to get interp_xy_pts
				#		This is the xy points interpolated along the new spacing
				interp_xy_pts = interp_func(interp_perim_pts)
				# Store the interpolated xy points, minus the last value (repeated from first)
				xy_pts[i] = np.array(interp_xy_pts[0:interp_xy_pts.shape[0]-1,:])
		# Pull values from the setstruct dict
		x_resolution = setstruct['ResolutionX']
		y_resolution = setstruct['ResolutionY']
		image_position = setstruct['ImagePosition']
		image_orientation = setstruct['ImageOrientation']
		# Pull the image orientation in x and y, then the z is the cross-product
		x_image_orientation = image_orientation[3:6]
		y_image_orientation = image_orientation[0:3]
		z_image_orientation = np.cross(y_image_orientation, x_image_orientation)
		slice_thickness = setstruct['SliceThickness']
		slice_gap = setstruct['SliceGap']
		# Set the z offset (always 0 in long-axis)
		if layer == 'long':
			z_offset = 0
		else:
			z_offset = slice_number
		if run_xyz:
			xyz_pts = xy_pts
			# If z points are used, add a new column to xyz_pts
			#		This column is entirely equal to the z_offset
			for i in range(x_pix_round.shape[0]):
				z_pix = -z_offset*np.ones([perim_length[i],1])
				if (layer == 'mask') or (not np.any(np.isnan(xy_pts[i].flatten()))):
					z_pix = z_pix.reshape([z_pix.shape[0], 1])
					xyz_pts[i] = np.append(xyz_pts[i], z_pix, axis=1)
					xyz_shape = xyz_pts[i].shape
		# Set t_o as a 4x4 identity matrix except the final column is [-1, -1, 0, 1]
		t_o = np.identity(4)
		t_o[:,3] = [-1, -1, 0, 1]
		# Set s_eye as an identity matrix except the first 3 points on the diagonal are:
		#		x_resolution, y_resolution, slice_thickness+slice_gap
		s_eye = np.identity(4)
		s_eye[0,0] = x_resolution
		s_eye[1,1] = y_resolution
		s_eye[2,2] = slice_thickness + slice_gap
		# Set r_eye as a 4x4 identity matrix except the upper right corner is a 3x3 transposed orientation matrix
		r_eye = np.identity(4)
		r_eye[0:3,0:3] = np.transpose([x_image_orientation[:], y_image_orientation[:], z_image_orientation[:]])
		# Set t_ipp to an identity matrix except the first 3 points of the final column are the image position
		t_ipp = np.identity(4)
		t_ipp[0:3,3] = image_position
		# Multiply t_ipp, r_eye, s_eye, and t_o and store as m_arr
		m_arr = t_ipp@r_eye@s_eye@t_o
		if run_xyz:
			for i in range(x_pix_round.shape[0]):
				# As long as there are no NaN values:
				if ~np.any(np.isnan(xyz_pts[i].flatten())):
					try:
						# Append a column of ones and multiply xyz_pts by the array defined above
						mult_arr = np.transpose(np.append(xyz_pts[i], np.ones([xyz_pts[i].shape[0], 1]), axis=1))
						X = np.transpose(m_arr@mult_arr)
					except:
						print('Error encountered.')
						continue
					# Remove the column of ones at the end and store in xyz_pts
					X = X[:,0:3]
					xyz_pts[i] = X
		if layer == 'mask':
			# Return values if scar is the current layer: xyz_pts, m_arr
			if not run_xyz:
				xyz_pts[0] = [None, None, None]
			return([xyz_pts, m_arr])
		if layer == 'epi':
			# Return values if epi is the current layer: xyz_pts, m_arr
			return([xyz_pts, m_arr])
		# Set values before use
		pp_slice = None
		time_id = None
		cur_arr = np.array(setstruct['EndoPinX'][:])
		# Set z_offset to 0 if EndoPins have no timepoint changes
		if len(cur_arr.shape) < 2:
			x_pinpts = np.array(setstruct['EndoPinX'][:])
			y_pinpts = np.array(setstruct['EndoPinY'][:])
			z_offset_pp = 0
		else:
			# Set the timepoint based on where the endo pinpoints are non-zero
			time_slice = np.where(cur_arr)
			# If there is more than one timepoint, choose the first one
			#		The first dimension in cur_arr is time, the second is slices
			if len(time_slice) > 1:
				time_id = time_slice[0][0]
				pp_slice = time_slice[1][0]
			else:
				# If there aren't multiple timepoints, just take the slice
				time_id = None
				pp_slice= time_slice[0][0]
			z_offset_pp = pp_slice
			# Pull the pinpoints from the structures based on time data
			if time_id == None:
				x_pinpts = cur_arr[pp_slice]
				y_pinpts = np.array(setstruct['EndoPinY'])[pp_slice]
			else:
				x_pinpts = cur_arr[time_id][pp_slice]
				y_pinpts = np.array(setstruct['EndoPinY'])[time_id][pp_slice]
		# Round pinpoints and append the z offset
		pinpts_round = [np.round(x_pinpts), np.round(y_pinpts)]
		z_pix = -z_offset_pp * np.ones([len(x_pinpts)])
		pinpts_round.append(z_pix)
		# Append a column of ones, multiply by the m_arr as defined above, and remove ones
		pinpts_round.append(np.ones([len(x_pinpts)]))
		pp = np.transpose(np.array(m_arr)@np.array(pinpts_round))
		pp = pp[:,0:3]
		if layer == 'long':
			# Return if layer is long-axis: pinpoints (pp), m_arr
			returnList = [pp, m_arr]
		else:
			# Return if layer is endocardial: xyz_pts, pinpoints (pp), m_arr
			returnList = [xyz_pts, pp, m_arr]
		return(returnList)