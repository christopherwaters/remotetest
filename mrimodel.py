# -*- coding: utf-8 -*-
"""
Contains all class definitions and imports necessary to implement MRI segmentation imports
and alignments. Built based on MRI Processing MATLAB pipeline by Thien-Khoi Phung.

Created on Fri Jul 21 11:27:52 2017

@author: cdw2be
"""

# Imports
import math
import tkinter as tk
from tkinter import filedialog
import scipy as sp
import scipy.io as spio
import scipy.stats as spstats
import numpy as np
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D
# Call this to set appropriate printing to view all data from arrays
np.set_printoptions(threshold=np.inf)
np.core.arrayprint._line_width = 160

class MRIModel():

	"""Contains contours based on multiple MRI modalities.

	Class containing contours based on multiple MRI modalities. Can be used to
	import Cine Black-Blood stacks and LGE stacks. Use Long-Axis image to determine
	vertical slice orientation.
	
	Attributes:
		scar (bool): Whether or not the model has an LGE data set
		dense (bool): Whether or not the model has a DENSE data set
		*_endo (array): Endocardial contour for the data set indicated by *
		*_epi (array): Epicardial contour for the data set indicated by *
		*_apex_pt (array): Apex point for the data set indicated by *
		*_basal_pt (array): Basal point for the data set indicated by *
		*_septal_pts (array): RV Insertion points selected in segment for the data set indicated by *
		*_slices (list): List of slices that were traced in SEGMENT for the data set indicated by *
		scar_ratio (array): Ratio of scar versus wall thickness by angle bins
		aligned_scar (array): The scar contour mapped from LGE to cine stack
	
	TODO:
		Import DENSE MRI Data
		Adjust import system to allow selection of individual timepoints instead of averaging!
			This means converting cine_endo to a 4-dimensional array? I think.
			Import all timepoints simultaneously, then allow selection. This simplifies things.
			Averaging occurs in the conversion to polar coordinates.
	"""
	
	def __init__(self, cine_file, la_file, scar_file=None, dense_file=None):
		"""Initialize new MRI model and select applicable files.
		
		Args:
			scar: Determines whether to import an LGE image stack.
			dense: Determines whether to import a DENSE image stack.
			
		Returns: New instance of MRIModel
		"""
		self.cine_file = cine_file
		self.long_axis_file = la_file
		if scar_file:
			self.scar_file = scar_file
			self.scar = True
		else:
			self.scar_file = None
			self.scar = False
		
		if dense_file:
			self.dense_file = dense_file
			self.dense = True
		else:
			self.dense_file = None
			self.dense = False
		
		# Define variables used in all models
		self.apex_base_pts = self._importLongAxis(self.long_axis_file)
		
		self.cine_endo = []
		self.cine_epi = []
		self.cine_apex_pt = []
		self.cine_basal_pt = []
		self.cine_septal_pts = []
		self.cine_slices = []
		
		# Set up scar-specific variables
		self.lge_endo = []
		self.lge_epi = []
		self.scar_ratio = []
		self.lge_apex_pt = []
		self.lge_basal_pt = []
		self.lge_septal_pts = []
		self.scar_slices = []
		self.aligned_scar = []
		
		# Set up DENSE-specific variables
		self.dense_endo = []
		self.dense_epi = []
		self.dense_pts = []
		self.dense_displacement = []
		self.dense_slices = []
		self.dense_aligned_pts = []
		self.dense_aligned_displacement = []
	
	def importCine(self, timepoint=0):
		"""Import the black-blood cine stack.

		Returns:
			boolean: True if import was successful.
		"""
		# Import the Black-Blood cine (short axis) stack
		endo_stack, epi_stack, rv_insertion_pts, sastruct, septal_slice = self._importStack(self.cine_file, timepoint)
		kept_slices = sastruct['KeptSlices']
		
		# Get the adjusted contours from the stacks
		abs_shifted, endo, epi, axis_center = self._getEndoEpiFromStack(endo_stack, epi_stack, sastruct, rv_insertion_pts, septal_slice, self.apex_base_pts)
		
		# Sort endo and epi traces by timepoint
		#	Pull time points from stack traces, then generate lists to store indices
		endo_time_pts = endo_stack[:, 3]
		epi_time_pts = epi_stack[:, 3]
		endo_by_timept = [None] * np.unique(endo_time_pts).shape[0]
		epi_by_timept = [None] * np.unique(epi_time_pts).shape[0]
		# Iterate through time points
		for i in range(len(endo_by_timept)):
			# Get time points in sorted order
			time_pt = np.unique(endo_time_pts)[i]
			# Get indices per-slice, by selected timepoint
			cur_endo_timept = self.__getTimeIndices(endo, endo_time_pts, time_pt)
			cur_epi_timept = self.__getTimeIndices(epi, epi_time_pts, time_pt)
			# Store slice data by timepoint
			endo_by_timept[i] = [endo[slice][cur_endo_timept[slice]] for slice in range(len(endo))]
			epi_by_timept[i] = [epi[slice][cur_epi_timept[slice]] for slice in range(len(epi))]
		
		# Convert endo and epi to polar, store by timepoint
		endo_polar_alltime = [None] * len(endo_by_timept)
		epi_polar_alltime = [None] * len(epi_by_timept)
		wall_thickness_alltime = [None] * len(endo_by_timept)
		cine_endo_alltime = [None] * len(endo_by_timept)
		cine_epi_alltime = [None] * len(epi_by_timept)
		# Iterate through time points
		for i in range(len(endo_by_timept)):
			# Get polar endo and epi for the selected timepoint
			endo_polar, epi_polar, _ = self._convertSlicesToPolar(kept_slices, endo_by_timept[i], epi_by_timept[i])
			endo_polar_alltime[i] = endo_polar
			epi_polar_alltime[i] = epi_polar
		
			# Calculate Wall Thickness based on the difference between epi and endo polar radii
			wall_thickness = np.append(np.expand_dims(endo_polar[:, :, 1], axis=2), np.expand_dims(epi_polar[:, :, 3] - endo_polar[:, :, 3], axis=2), axis=2)
			wall_thickness_alltime[i] = wall_thickness
		
			# Retranslate results to cartesian from polar and shift:
			cine_endo, cine_epi = self._shiftPolarCartesian(endo_polar, epi_polar, endo_by_timept[1], epi_by_timept[1], kept_slices, axis_center, wall_thickness)
			cine_endo_alltime[i] = cine_endo
			cine_epi_alltime[i] = cine_epi
		
		# Store class fields based on calculated values:
		self.cine_apex_pt = abs_shifted[0]
		self.cine_basal_pt = abs_shifted[1]
		self.cine_septal_pts = abs_shifted[2:]
		self.cine_endo = cine_endo_alltime
		self.cine_epi = cine_epi_alltime
		self.cine_slices = kept_slices

		return(True)
		
	def importLGE(self):
		"""Import the LGE MRI stack.
		
		Create an endocardial and epicardial contour based on the LGE file stack.
		Additionally imports scar traces and stores them as scar ratio.
		Scar ratio is the ratio of the scar contour edges compared to wall thickness.
			
		Returns:
			boolean: True if the import was successful.
		"""
		scar_endo_stack, scar_epi_stack, scar_insertion_pts, scarstruct, scar_septal_slice = self._importStack(self.scar_file)
		
		# Prepare variables imported from file.
		scar_auto = np.array(scarstruct['Scar']['Auto'])
		scar_manual = np.array(scarstruct['Scar']['Manual'])
		
		# Form a combination array that combines the automatic scar recognition with manual adjustments (including manual erasing)
		scar_combined = np.add(scar_auto, scar_manual) > 0
		
		# The new array needs to have axes adjusted to align with the format (z, x, y) allowing list[n] to return a full slice
		scar_combined = np.swapaxes(np.swapaxes(scar_combined, 1, 2), 0, 1)
		
		# Get the x-y values from the mask
		scar_abs, scar_endo, scar_epi, scar_ratio, scar_slices = self._getMaskContour(scar_endo_stack, scar_epi_stack, scar_insertion_pts, scarstruct, scar_septal_slice, scar_combined)
		
		# Store instance fields
		self.lge_apex_pt = scar_abs[0]
		self.lge_basal_pt = scar_abs[1]
		self.lge_septal_pts = scar_abs[2:]
		self.lge_endo = scar_endo
		self.lge_epi = scar_epi
		self.scar_slices = scar_slices
		self.scar_ratio = scar_ratio
		
		return(True)
		
	def importDense(self, align_timepoint=0):
		"""Imports DENSE MR data.
		
		Currently this has no functionality, must be implemented.
		
		"""
		
		dense_endo = [None]*len(self.dense_file)
		dense_epi = [None]*len(self.dense_file)
		dense_pts = [None]*len(self.dense_file)
		slice_locations = [None]*len(self.dense_file)
		dense_displacement = False
		
		for i in range(len(self.dense_file)):
			dense_file = self.dense_file[i]
			# Extract Contour Information from DENSE Mat file
			dense_data = self._loadmat(dense_file)
			slice_location = dense_data['SequenceInfo'][0, 0].SliceLocation
			slice_locations[i] = slice_location
			epi_dense = np.array(dense_data['ROIInfo']['RestingContour'][0])
			endo_dense = np.array(dense_data['ROIInfo']['RestingContour'][1])
			
			# Append slice location to DENSE contour data
			endo_slice_col = np.array([slice_location] * endo_dense.shape[0]).reshape([endo_dense.shape[0], 1])
			epi_slice_col = np.array([slice_location] * epi_dense.shape[0]).reshape([epi_dense.shape[0], 1])
			endo_dense = np.append(endo_dense, endo_slice_col, axis=1)
			epi_dense = np.append(epi_dense, epi_slice_col, axis=1)
			
			# Interpolate data to get an equal number of points for each contour
			endo_interp_func = sp.interpolate.interp1d(np.arange(0, 1+1/(endo_dense.shape[0]-1), 1/(endo_dense.shape[0]-1)), endo_dense, axis=0, kind='cubic')
			epi_interp_func = sp.interpolate.interp1d(np.arange(0, 1+1/(epi_dense.shape[0]-1), 1/(epi_dense.shape[0]-1)), epi_dense, axis=0, kind='cubic')
			endo_interp = endo_interp_func(np.arange(0, 80/79, 1/79))
			epi_interp = epi_interp_func(np.arange(0, 80/79, 1/79))
		
			# Find nearest slice in cine stack
			cine_endo_slices = np.unique(self.cine_endo[align_timepoint][:, 2])
			cine_epi_slices = np.unique(self.cine_epi[align_timepoint][:, 2])
			cine_endo_slice_match = cine_endo_slices[np.where(abs(cine_endo_slices - slice_location) == np.min(abs(cine_endo_slices - slice_location)))[0][0]]
			cine_epi_slice_match = cine_epi_slices[np.where(abs(cine_epi_slices - slice_location) == np.min(abs(cine_epi_slices - slice_location)))[0][0]]
			cine_endo_slice_inds = np.where(self.cine_endo[align_timepoint][:, 2] == cine_endo_slice_match)[0]
			cine_epi_slice_inds = np.where(self.cine_epi[align_timepoint][:, 2] == cine_epi_slice_match)[0]
			
			# Get endo and epi traces
			cine_slice_endo = self.cine_endo[align_timepoint][cine_endo_slice_inds, :]
			cine_slice_epi = self.cine_epi[align_timepoint][cine_endo_slice_inds, :]
			
			# Pull timepoints from DENSE
			dense_timepoints = len(dense_data['DisplacementInfo']['dX'][0])
			
			# Shift the DENSE endo and epi contours by the epicardial mean
			endo_shift = endo_interp[:, :2] - np.mean(epi_interp[:, :2], axis=0)
			epi_shift = epi_interp[:, :2] - np.mean(epi_interp[:, :2], axis=0)
			endo_shift_theta, endo_shift_rho = self._cartToPol(endo_shift[:, 0], endo_shift[:, 1])
			epi_shift_theta, epi_shift_rho = self._cartToPol(epi_shift[:, 0], epi_shift[:, 1])
			dense_endo[i] = endo_shift
			dense_epi[i] = epi_shift
			
			cine_endo_theta, cine_endo_rho = self._cartToPol(cine_slice_endo[:, 0], cine_slice_endo[:, 1])
			cine_epi_theta, cine_epi_rho = self._cartToPol(cine_slice_epi[:, 0], cine_slice_epi[:, 1])
			
			# Shift the entire pixel array by the same epicardial mean
			dense_x = dense_data['DisplacementInfo']['X'] - np.mean(epi_interp[:, 0])
			dense_y = dense_data['DisplacementInfo']['Y'] - np.mean(epi_interp[:, 1])
			dense_z = [slice_location]*len(dense_x)
			
			dense_pts[i] = np.column_stack((dense_x, dense_y, dense_z))
			
			all_dense_theta, all_dense_rho = self._cartToPol(dense_x, dense_y)
			
			# Get displacement info and store as a 2-D array
			dense_dx = np.array(dense_data['DisplacementInfo']['dX'])
			dense_dy = np.array(dense_data['DisplacementInfo']['dY'])
			dense_dz = np.array(dense_data['DisplacementInfo']['dZ'])
			
			# Add DENSE displacement slices by time
			if not dense_displacement:
				dense_displacement = [None] * dense_dx.shape[1]
				for i in range(dense_dx.shape[1]):
					dense_displacement[i] = np.column_stack((dense_dx[:, i], dense_dy[:, i], dense_dz[:, i]))
			else:
				for i in range(dense_dx.shape[1]):
					cur_disp = np.column_stack((dense_dx[:, i], dense_dy[:, i], dense_dz[:, i]))
					dense_displacement[i] = [dense_displacement[i], cur_disp]
		
		self.dense_endo = dense_endo
		self.dense_epi = dense_epi
		self.dense_pts = dense_pts
		self.dense_displacement = dense_displacement
		self.dense_slices = slice_locations
		
		return(True)
	
	def _getMaskXY(self, mask, maskstruct):
		"""General form to get binary masks (such as scar data) as xy overlays.
		"""
		kept_slices = maskstruct['KeptSlices']
		mask_max = max([sum(sum(mask[i])) for i in range(mask.shape[0])])
		mask_pts = np.array(np.where(mask)) + 1
		mask_slices = np.array(list(set(mask_pts[0, :])))
		mask_pts[0] -= 1
		mask_x = np.zeros([1, mask.shape[0], mask_max])
		mask_y = np.zeros([1, mask.shape[0], mask_max])
		for i in mask_slices:
			temp_x = mask_pts[1, np.where(mask_pts[0, :] == i-1)]
			temp_y = mask_pts[2, np.where(mask_pts[0, :] == i-1)]
			mask_x[0, i-1, 0:temp_x.size] = temp_x
			mask_y[0, i-1, 0:temp_y.size] = temp_y
		maskstruct['mask_x'] = mask_x
		maskstruct['mask_y'] = mask_y
		cxyz_mask, mask_m, _ = self._rotateSAStack(maskstruct, kept_slices, layer='mask')
		for i in mask_slices:
			cxyz_slice = cxyz_mask[np.where(cxyz_mask[:, 4] == i), :][0]
			mode_val, mode_count = spstats.mode(cxyz_slice[:, 0:2])
			max_mode = max(mode_count.squeeze())
			if max_mode > 1:
				mode_ind = np.argmax(mode_count.squeeze())
				cur_mode = mode_val.squeeze()[mode_ind]
				cxyz_mask = np.delete(cxyz_mask, np.where(cxyz_mask[:, mode_ind] == cur_mode), axis=0)
		return([cxyz_mask, kept_slices, mask_slices])
	
	def _importStack(self, short_axis_file, timepoint=0):
		"""Imports the short-axis file and formats data from it.
		
		Data is imported using the custom _loadmat function
		to open the struct components appropriately. All short-axis
		data is imported during this function.
		
		args:
			short_axis_file: File for the short-axis data and segmentation.
		
		returns:
			array cxyz_sa_endo: Endocardial contour stack
			array cxyz_sa_epi: Epicardial contour stack
			rv_insertion_pts: The endocardial pinpoints indicating location where RV epicardium intersects LV epicardium
			setstruct: The MATLAB structure contained within the short-axis file (part of SEGMENT's output)
			septal_slice: The slices containing the RV insertion pinpoints
		"""
		
		# Import and format the short axis stack and pull relevant variables from the structure.
		short_axis_data = self._loadmat(short_axis_file)
		setstruct = short_axis_data['setstruct']
		endo_x = np.array(setstruct['EndoX'])
		endo_y = np.array(setstruct['EndoY'])
		epi_x = np.array(setstruct['EpiX'])
		epi_y = np.array(setstruct['EpiY'])
		# Data can be varying dimensions, so this ensures that arrays are reshaped
		#    into the same dimensionality and adjusts axis order for improved human readability
		if endo_x.ndim >= 3:
			endo_x = np.swapaxes(endo_x, 0, 1)
			endo_x = np.swapaxes(endo_x, 2, 1)
			endo_y = np.swapaxes(endo_y, 0, 1)
			endo_y = np.swapaxes(endo_y, 2, 1)
			epi_x = np.swapaxes(epi_x, 0, 1)
			epi_x = np.swapaxes(epi_x, 2, 1)
			epi_y = np.swapaxes(epi_y, 0, 1)
			epi_y = np.swapaxes(epi_y, 2, 1)
		else:
			endo_x = endo_x.transpose()
			endo_y = endo_y.transpose()
			epi_x = epi_x.transpose()
			epi_y = epi_y.transpose()
			shape = endo_x.shape
			endo_x = endo_x.reshape(1, shape[0], shape[1])
			endo_y = endo_y.reshape(1, shape[0], shape[1])
			epi_x = epi_x.reshape(1, shape[0], shape[1])
			epi_y = epi_y.reshape(1, shape[0], shape[1])

		# Process the setstruct to get time points and slices that were segmented
		kept_slices, time_id = self._processKeptSlices(setstruct, endo_x)
		kept_slices = np.array(kept_slices)
		time_id = np.squeeze(np.array(time_id))
		endo_pin_x = np.array(setstruct['EndoPinX'])
		endo_pin_y = np.array(setstruct['EndoPinY'])
		
		# If more than 1 timepoint is passed, use the indicated timepoint at call
		if time_id.size > 1:
			try:
				time_id = np.where(endo_pin_x)[0][timepoint]
			except(IndexError):
				print('Invalid timepoint selected. Adjusting to initial timepoint.')
				time_id = np.where(endo_pin_x)[0][0]
		
		# Ensure that the pinpoint arrays are the correct dimensionality    
		if endo_pin_x.ndim == 1:
			endo_pin_x = endo_pin_x.reshape(1, endo_pin_x.shape[0])
		if endo_pin_y.ndim == 1:
			endo_pin_y = endo_pin_y.reshape(1, endo_pin_y.shape[0])
			
		# Finds the slice where the pinpoints are placed and treats it as the septal slice    
		septal_slice = self._findRVSlice(endo_pin_x)
		
		# Extract the x and y pinpoints for the current contour
		x_pins = np.array(endo_pin_x[time_id, septal_slice][0][0])
		y_pins = np.array(endo_pin_y[time_id, septal_slice][0][0])
		endo_pins = np.array([x_pins, y_pins]).transpose()
		
		# Calculate the Septal Mid-Point from the pinpoints
		sept_pt = self._findEndoMidPt(endo_pins, time_id, septal_slice, endo_x, endo_y)

		# Add the midpoint to the x and y pinpoint list and add it back to setstruct
		#        This part requires somewhat complex list comprehensions to reduce clutter and due to the complexity of the data format
		new_endo_pin_x = [np.append(cur_endo_pin_x, sept_pt[0]).tolist() if cur_endo_pin_x else cur_endo_pin_x for cur_endo_pin_x in endo_pin_x .flatten()]
		new_endo_pin_y = [np.append(cur_endo_pin_y, sept_pt[1]).tolist() if cur_endo_pin_y else cur_endo_pin_y for cur_endo_pin_y in endo_pin_y.flatten()]
		endo_pin_x = np.reshape(new_endo_pin_x, endo_pin_x.shape)
		endo_pin_y = np.reshape(new_endo_pin_y, endo_pin_y.shape)
		
		# Store relevant variables in the setstruct dictionary for use downstream
		setstruct['EndoPinX'] = endo_pin_x
		setstruct['EndoPinY'] = endo_pin_y
		setstruct['KeptSlices'] = kept_slices
		setstruct['endo_x'] = endo_x
		setstruct['endo_y'] = endo_y
		setstruct['epi_x'] = epi_x
		setstruct['epi_y'] = epi_y
		
		# Rotate the endo and epi contours (and pinpoints with the endo contour)
		cxyz_sa_endo, rv_insertion_pts, _, _ = self._rotateSAStack(setstruct, kept_slices, layer='endo')
		cxyz_sa_epi, _, _ = self._rotateSAStack(setstruct, kept_slices, layer='epi')

		return([cxyz_sa_endo, cxyz_sa_epi, rv_insertion_pts, setstruct, septal_slice])
	
	def _getMaskContour(self, mask_endo_stack, mask_epi_stack, mask_insertion_pts, mask_struct, mask_septal_slice, mask, transmural_filter=0.1, interp_vals=True, elim_secondary=True):
		"""Generic import for binary mask overlays onto a contour stack.
		
		The mask input should be a binary mask overlay aligned with the struct variable.
		The data is returned in the form of a ratio of wall thickness at angle bins.
		
		args:
			mask_endo_stack (array): The endo stack variable from the mask stack import
			mask_epi_stack (array): The epi stack variable from the mask stack import
			mask_insertion_pts (array): The insertion points from the mask stack import
			mask_struct (dict): The 'struct' variable returned from stack import
			mask_septal_slice (int): The septal slice returned from stack import
			mask (array): The binary mask that determines the location of the regions of interest
			transmural_filter (float): A variable that indicates the minimal transmurality to keep
			interp_vals (bool): Determine whether or not single-bin gaps should be interpolated
			elim_secondary (bool): Determine whether or not non-contiguous, smaller regions should be removed
			
		returns:
			mask_abs (array): The apex-base-septal points array from stack rotation and transformation
			mask_endo (array): The endocardial contour of the mask structure
			mask_epi (array): The epicardial contour of the mask structure
			mask_ratio (array): The inner and outer contours as a ratio of wall thickness, binned using polar angles to differentiate segments
			mask_slices (list): The list of slices that contained regions of interest for the mask
		"""
		# Get the mask XY values
		cxyz_mask, kept_slices, mask_slices = self._getMaskXY(mask, mask_struct)
		
		# Convert the stacks
		mask_abs, mask_endo, mask_epi, axis_center, all_mask = self._getEndoEpiFromStack(mask_endo_stack, mask_epi_stack, mask_struct, mask_insertion_pts, mask_septal_slice, self.apex_base_pts, cxyz_mask)
		
		# Get polar values and wall thickness
		mask_endo_polar, mask_epi_polar, mask_polar = self._convertSlicesToPolar(kept_slices, mask_endo, mask_epi, all_mask, scar_flag=True)
		wall_thickness = np.append(np.expand_dims(mask_endo_polar[:, :, 1], axis=2), np.expand_dims(mask_epi_polar[:, :, 3] - mask_endo_polar[:, :, 3], axis=2), axis=2)
		
		# Calculate ratio through wall based on angle binning
		inner_distance = mask_polar[:, :, 3] - mask_endo_polar[:, :, 3]
		outer_distance = mask_polar[:, :, 4] - mask_endo_polar[:, :, 3]
		inner_ratio = np.expand_dims(inner_distance/wall_thickness[:, :, 1], axis=2)
		outer_ratio = np.expand_dims(outer_distance/wall_thickness[:, :, 1], axis=2)
		mask_ratio = np.append(np.expand_dims(wall_thickness[:, :, 0], axis=2), np.append(inner_ratio, outer_ratio, axis=2), axis=2)
		
		# If desired, eliminate regions outside of transmurality lower limit
		if transmural_filter:
			mask_ratio[np.isnan(mask_ratio)] = 0
			low_trans = np.where(mask_ratio[:, :, 2] - mask_ratio[:, :, 1] < transmural_filter)
			mask_ratio[low_trans[0], low_trans[1], 1:] = np.nan
		
		# Interpolate single-bin gaps, if desired (for contiguous traces)
		if interp_vals:
			for i in range(mask_ratio.shape[0]):
				mask_slice = mask_ratio[i, :, :]
				mask_slice_nans = np.where(np.isnan(mask_slice[:, 1]))[0]
				mask_slice_nan_iso = [(((mask_slice_nans_i + 1) % mask_slice.shape[0]) not in mask_slice_nans) & (((mask_slice_nans_i - 1) % mask_slice.shape[0]) not in mask_slice_nans) for mask_slice_nans_i in mask_slice_nans]
				mask_slice_nan_ind = mask_slice_nans[mask_slice_nan_iso]
				for mask_ind in mask_slice_nan_ind:
					mask_inner_adj = [mask_slice[(mask_ind - 1) % mask_slice.shape[0], 1], mask_slice[(mask_ind + 1) % mask_slice.shape[0], 1]]
					mask_outer_adj = [mask_slice[(mask_ind - 1) % mask_slice.shape[0], 2], mask_slice[(mask_ind + 1) % mask_slice.shape[0], 2]]
					mask_inner_mean = np.mean(mask_inner_adj)
					mask_outer_mean = np.mean(mask_outer_adj)
					mask_slice[mask_ind, 1] = mask_inner_mean
					mask_slice[mask_ind, 2] = mask_outer_mean
			mask_ratio[i, :, :] = mask_slice
			
		# Eliminate small, non-contiguous regions, if desired (for a single trace)
		if elim_secondary:
			for i in range(mask_ratio.shape[0]):
				mask_slice = mask_ratio[i, :, :]
					# If there is no scar on this slice, go to next slice
				if np.all(np.isnan(mask_slice[:, 1])):
					continue
				# Pull the scar indices where there is no nan value
				mask_slice_nonan = np.where(~np.isnan(mask_slice[:, 1]))[0]
				# Calculate the differences between each value in the index array, and append the difference between the last and first points
				gap_dist = np.diff(mask_slice_nonan)
				gap_dist = np.append(gap_dist, mask_slice_nonan[0] + mask_slice.shape[0] - mask_slice_nonan[-1])
				# If there is only 1 gap, then there is only one scar contour, so continue to next slice
				if np.count_nonzero(gap_dist > 1) == 1:
					gap_dist[gap_dist > 1] = 1
				if np.all(gap_dist == 1):
					continue
				# The case where there are multiple non-contiguous scar traces:
				for j in range(math.floor(np.count_nonzero(gap_dist > 1)/2)):
					# Get the indices around the non-contiguous regions
					ind1 = (np.where(gap_dist > 1)[0][0] + 1) % (len(gap_dist))
					ind2 = (np.where(gap_dist > 1)[0][1] + 1) % (len(gap_dist))
					if ind1 > ind2:
						lower_index = ind2
						upper_index = ind1
					else:
						lower_index = ind1
						upper_index = ind2
					# Split the list, then set the longer list (main scar trace) as the value (essentially remove the smaller scar trace)
					slice_u2l = mask_slice_nonan[:lower_index].tolist() + mask_slice_nonan[upper_index:].tolist()
					slice_l2u = mask_slice_nonan[lower_index:upper_index].tolist()
					if len(slice_u2l) > len(slice_l2u):
						mask_slice_nonan = np.array(slice_u2l)
					else:
						mask_slice_nonan = np.array(slice_l2u)
					# Recalculate gap distance
					gap_dist = np.diff(mask_slice_nonan)
					if mask_slice_nonan[-1] == mask_slice.shape[0] - 1 and mask_slice_nonan[0] == 0:
						gap_dist = np.append(gap_dist, 1)
					if np.count_nonzero(gap_dist > 1) == 1:
						gap_dist[gap_dist > 1] = 1
				# Set up temporary arrays to pull the main slice
				new_mask_inner = np.empty(mask_slice.shape[0])
				new_mask_inner[:] = np.NAN
				new_mask_outer = new_mask_inner.copy()
				new_mask_inner[mask_slice_nonan] = mask_slice[mask_slice_nonan, 1]
				new_mask_outer[mask_slice_nonan] = mask_slice[mask_slice_nonan, 2]
				# Reassign the scar contour, overwriting non-contiguous traces with NaN
				mask_slice[:, 1] = new_mask_inner
				mask_slice[:, 2] = new_mask_outer
				mask_ratio[i, :, :] = mask_slice
		
		# Translate Endo and Epicardial contours back to cartesian
		mask_endo_cart, mask_epi_cart = self._shiftPolarCartesian(mask_endo_polar, mask_epi_polar, mask_endo, mask_epi, mask_slices, axis_center, wall_thickness)
		avg_wall_thickness = np.mean(wall_thickness[:, :, 1])
		
		return([mask_abs, mask_endo, mask_epi, mask_ratio, mask_slices])
	
	def _getOverlayData(self):
		"""The purpose of this function is to allow generalized import of data that is formatted as
		an overlay with values indicating regional values for certain pixels or voxels (for 3d).
		"""
		pass
	
	def _loadmat(self, filename):
		"""
		this function should be called instead of direct spio.loadmat
		as it cures the problem of not properly recovering python dictionaries
		from mat files. It calls the function check keys to cure all entries
		which are still mat-objects
		
		This and all sub-functions are based on jpapon's answer here:
			https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
		"""
		data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
		return(self._check_keys(data))

	def _check_keys(self, dict_pass):
		"""
		checks if entries in dictionary are mat-objects. If yes
		todict is called to change them to nested dictionaries
		"""
		for key in dict_pass:
			if isinstance(dict_pass[key], spio.matlab.mio5_params.mat_struct):
				dict_pass[key] = self._todict(dict_pass[key])
		return(dict_pass)        
	
	def _todict(self, matobj):
		"""
		A recursive function which constructs from matobjects nested dictionaries
		"""
		dict = {}
		for strg in matobj._fieldnames:
			elem = matobj.__dict__[strg]
			if isinstance(elem, spio.matlab.mio5_params.mat_struct):
				dict[strg] = self._todict(elem)
			elif isinstance(elem,np.ndarray):
				dict[strg] = self._tolist(elem)
			else:
				dict[strg] = elem
		return(dict)
	
	def _tolist(self, ndarray):
		"""
		A recursive function which constructs lists from cellarrays 
		(which are loaded as numpy ndarrays), recursing into the elements
		if they contain matobjects.
		"""
		elem_list = []            
		for sub_elem in ndarray:
			if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
				elem_list.append(self._todict(sub_elem))
			elif isinstance(sub_elem,np.ndarray):
				elem_list.append(self._tolist(sub_elem))
			else:
				elem_list.append(sub_elem)
		return(elem_list)
	
	def _processKeptSlices(self, setstruct, endo_x):
		"""Remove time points and slices with no contours
		
		args:
			setstruct: SEGMENT structure in the original short-axis file.
			endo_x: The x-values of the endocardial contours from SEGMENT.
			
		returns:
			list kept_slices: List of slices that have contours.
			list time_id: Which time points have been segmented.
		"""
		
		# Start with a full list of all slices
		kept_slices = np.arange(setstruct['ZSize']) + 1
		
		# Find where the slices have not received a contour trace and remove them
		no_trace = np.sum(np.isnan(endo_x), axis=2)
		delete_slices = no_trace != 0
		delete_slices = np.sum(delete_slices, axis=0) == delete_slices.shape[0]
		kept_slices = kept_slices[~delete_slices]
		time_id = np.where(no_trace[:,kept_slices[0] - 1] == 0)
		return([kept_slices, time_id])
	
	def _findRVSlice(self, pin_x):
		"""Find the slice where the pinpoints have been placed"""
		# Sum the pinpoint locations to find the nonzero slice where the pinpoints have been placed
		septal_slice = np.where([np.sum(pin_x[:, cur_slice][0]) for cur_slice in range(pin_x.shape[1])])
		return(septal_slice)
		
	def _findEndoMidPt(self, endo_pins, time_id, septal_slice, endo_x, endo_y):
		"""Calculate the mid-septal point based on the pinpoints already placed
		
		args:
			endo_pins: array containing the rv insertion pinpoints
			time_id: which time point to use for calculation
			septal_slice: which slice to use for calculation
			endo_x: the x values defining the endocardial contour
			endo_y: the y values defining the endocardial contour
		returns:
			array mid_pt: The septal midpoint between the two other pinpoints
		"""
		
		# Get mean point between the 2 pinpoints.
		mean_pt = np.mean(endo_pins, axis=0).reshape([2, 1])
		
		# Calculate the perpindicular line between the two points (just slope, no intercept)
		slope = (endo_pins[1,1] - endo_pins[0,1])/(endo_pins[1,0] - endo_pins[0,0])
		perp_slope = -1/slope
		
		# Get the current slice and shift it by the mean point
		cur_slice = np.array([endo_x[time_id, septal_slice, :], endo_y[time_id, septal_slice, :]])
		cur_shape = cur_slice.shape
		cur_slice = cur_slice.reshape(cur_shape[0], cur_shape[3])
		cur_slice = cur_slice - mean_pt
		
		# Convert the slice into polar values
		polar_coords = self._cartToPol(cur_slice[0,:], cur_slice[1,:])[:,1:]
		
		# Get the theta values for the perpindicular line (polar theta)
		perp_dot = np.dot([1, perp_slope], [1, 0])
		perp_norm = self._calcNorm([1, 1*perp_slope])
		th1 = np.arccos(perp_dot/perp_norm)
		th2 = th1 + np.pi;
		
		# Calculate the rho values for the two theta values by interpolation
		r_interp = sp.interpolate.interp1d(polar_coords[0,:], polar_coords[1,:])
		r1 = r_interp(th1)
		r2 = r_interp(th2)
		r = r1 if r1<r2 else r2
		theta = th1 if r1<r2 else th2
		
		# Reconvert the interpolated rho and theta to cartesian
		mid_pt = (self._polToCart(r, theta) + mean_pt.reshape([1,2])).reshape(2)
		return(mid_pt)
		
	def _cartToPol(self, x, y):
		"""Convert cartesian (x,y) coordinates to polar (theta, rho) coordinates"""
		rho = np.sqrt(np.square(x) + np.square(y))
		theta = np.arctan2(y,x)
		theta = np.where(theta < 0, theta + 2*np.pi, theta)
		return np.array([theta, rho])
	
	def _polToCart(self, rho, theta):
		"""Convert polar (theta, rho) coordinates to cartesian (x, y) coordinates"""
		x = rho * np.cos(theta)
		y = rho * np.sin(theta)
		return ([x, y])
	
	def _calcNorm(self, arr_in):
		"""Calculates the norm of the passed array as the square root of the sum of squares"""
		norm = np.sqrt(np.sum(np.square(arr_in)))
		return norm
	
	def _rotateSAStack(self, setstruct, slice_labels, layer='endo', axial_flag=False):
		"""Rotates Short-Axis Stack based on septum location.
		
		args:
			setstruct (dict): The structure from the SEGMENT data file
			slice_labels (list): The list of slices that are included in the stack
			layer (string): Which layer is being rotated (endo, epi, scar)
			axial_flag (bool): Check if image orientation is already correct
		
		returns:
			cxyz (array): The transformed and rotated contour
			m_arr (array): The multiplication array used in the transform process
			Pd (array): Unknown
			heartrate (array): Heartrate for each slice taken
		"""
		
		# Set up initial variables for future use.
		slice_counter = 1
		cxyz = np.array([])
		heartrate = np.array([])
		hr = setstruct['HeartRate']
		
		# Determine orientation of image and exit if it is correct
		if axial_flag:
			if (abs(abs(setstruct['ImageOrientation'][0])-1) >= 1E-7 or
				  abs(abs(setstruct['ImageOrientation'][4])-1) >= 1E-7):
				print('Image Orientation Correct')
				return(0)
		
		# Rotate stacks and format data for return.
		time_indices = range(setstruct['TSize'])
		for j in slice_labels:
			# Pass each slice to be transformed and reformat for return
			transformed_stack = self._transformStack(setstruct, j-1, layer)
			cxyz_slice = self._prepareTransformedStack(transformed_stack, time_indices, j)
			cxyz = np.append(cxyz, cxyz_slice)
			# Track heartrate during each slice acquisition
			heartrate = np.append(heartrate, [hr, slice_counter])
			# Update the slice counter (it can differ from slice_labels due to skipped slices)
			slice_counter += 1
			# Determine what values to return based on the layer selected.
			if layer == 'epi' or layer == 'mask':
				m_arr = transformed_stack[1]
			else:
				Pd = transformed_stack[1]
				m_arr = transformed_stack[2]
		# Reshape the transformed stack as a nx5 array
		cxyz = cxyz.reshape([int(cxyz.size/5), 5])
		# Set the returned data based on the layer
		if layer == 'epi' or layer == 'mask':
			returnList = [cxyz, m_arr, heartrate]
		else:
			returnList = [cxyz, Pd, m_arr, heartrate]
		return(returnList)

	def _prepareTransformedStack(self, transformed_stack, time_indices, j = 0):
		"""Process output from transformStack to append identifying information.
		
		args:
			transformed_stack: The full output from the transformStack function.
			cxyz: Array to which to append the output array
			time_indices: Which time points should be used
			j: The current slice
		
		returns:
			cxyz (array): Newly lengthened array containing formatted data from the stack
		"""
		# Pull the appropriate element from the stack
		Xd = transformed_stack[0]
		cxyz = np.array([])
		# Iterate through the 
		for k in time_indices:
			# Pull timepoint data
			Xd_k = Xd[k]
			# If every element is NaN, skip this timepoint
			if np.all(np.isnan(Xd_k)):
				continue
				
			# Store the slice index as an appropriately-shaped array for appending
			slice_indices = j*np.ones([Xd_k.shape[0], 1])
			# Append arrays together
			cxyz_append = np.append(Xd_k, time_indices[k]*np.ones([Xd_k.shape[0], 1]), axis=1)
			cxyz_append2 = np.append(cxyz_append, slice_indices, axis=1)
			cxyz = np.append(cxyz, cxyz_append2)

		return(cxyz)

	def _transformStack(self, setstruct, slice_number=0, layer='endo'):
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
		
	def _importLongAxis(self, long_axis_file):
		"""Imports data from a long-axis file with pinpoints for apex and basal locations
		
		args:
			long_axis_file (string): MAT file from SEGMENT with long-axis data
		returns:
			apex_base_pts (array): The points indicating the apex and basal points indicated in the file
		"""
		# Load the file using custom loadmat function
		long_axis_data = self._loadmat(long_axis_file)
		# Pull the setstruct data from the global structure
		lastruct = long_axis_data['setstruct']
		# Get the apex and basal points from stack transformation
		apex_base_pts, pinpts = self._transformStack(lastruct, layer='long')
		return(apex_base_pts)
		
	def _convertSlicesToPolar(self, slices, endo, epi, scar=None, scar_flag=False, num_bins = 50):
		"""Convert an epicardial and endocardial contour (and scar data) from cartesian to polar coordinates
		
		args:
			slices (list): the slices that are being processed
			endo (array): the cartesian points of the endocardial contour
			epi (array): the cartesian points of the epicardial contour
			scar (array): the cartesian points dictating scar position
			scar_flag (boolean): boolean indicating presence of scar data
			num_bins (integer): number of angles in the range of -pi to pi
		returns:
			endo_polar (array): polar coordinates of the endocardial contour
			epi_polar (array): polar coordinates of the endocardial contour
			scar_polar (array): polar coordinates of the inner and outer scar contour
		"""
		# Set up initial variables, including binned angle values
		angles = np.linspace(-math.pi, math.pi, num_bins)
		endo_polar = []
		epi_polar = []
		scar_polar = []
		
		# Iterate through each slice and calculate polar values
		for i in slices-1:
			# Grab data from current slice
			cur_endo = endo[i]
			cur_epi = epi[i]
			if scar_flag:
				cur_scar = scar[i]
				cur_scar_shift = cur_scar - np.mean(cur_epi, axis=0) if cur_scar.size > 0 else cur_scar
			# Shift the contours by the average epicardial point
			cur_endo_shift = cur_endo - np.mean(cur_epi, axis=0)
			cur_epi_shift = cur_epi - np.mean(cur_epi, axis=0)
			# Convert the shifted cartesian points to polar
			theta_endo, rho_endo = self._cartToPol(cur_endo_shift[:, 0], cur_endo_shift[:, 1])
			theta_epi, rho_epi = self._cartToPol(cur_epi_shift[:, 0], cur_epi_shift[:, 1])
			# Define a subfunction that returns a lambda function, which provides the indices in angles where theta falls
			def getIndices(j): return lambda theta: np.where((angles[j] <= theta) & (angles[j+1] > theta))[0].tolist()
			# Shifts theta_endo from 0:2*pi to -pi:pi
			theta_endo = [te_i if te_i < math.pi else te_i-2*math.pi for te_i in theta_endo]
			theta_epi = [te_i if te_i < math.pi else te_i-2*math.pi for te_i in theta_epi]
			# Get the indices (in angles) where theta falls for endo and epi contours
			endo_idx = [getIndices(j)(theta_endo) for j in range(angles.size-1)]
			epi_idx = [getIndices(j)(theta_epi) for j in range(angles.size-1)]
			if scar_flag:
				theta_scar, rho_scar = self._cartToPol(cur_scar_shift[:, 0], cur_scar_shift[:, 1]) if cur_scar.size > 0 else [np.nan, np.nan]
				if cur_scar.size > 0: theta_scar = [ts_i if ts_i < math.pi else ts_i-2*math.pi for ts_i in theta_scar]
				scar_idx = [getIndices(j)(theta_scar) if len(getIndices(j)(theta_scar)) > 0 else [] for j in range(angles.size-1)]
				scar_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], min(rho_scar[scar_idx[j]]), max(rho_scar[scar_idx[j]])] if len(scar_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan, np.nan] for j in range(angles.size-1)]
			# Create the list for the current slice containing: [angle bin min, angle bin average, angle bin max, average contour rho in that bin (nan if no contour point in that bin)]
			#	Creates this list for both endo and epi. For scar, the bin indicates minimum and maximum rho, instead of average rho
			endo_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], np.mean(rho_endo[endo_idx[j]])] if len(endo_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan] for j in range(angles.size-1)]
			epi_bin = [[angles[j], np.mean(angles[j:j+2]), angles[j+1], np.mean(rho_epi[epi_idx[j]])] if len(endo_idx[j]) > 0 else [angles[j], np.mean(angles[j:j+2]), angles[j+1], np.nan] for j in range(angles.size-1)]
			# Append the current slice to the global polar matrix
			endo_polar.append(np.array(endo_bin))
			epi_polar.append(np.array(epi_bin))
			if scar_flag: scar_polar.append(np.array(scar_bin))
				
		# Convert lists to arrays:
		endo_polar = np.array(endo_polar)
		epi_polar = np.array(epi_polar)
		scar_polar = np.array(scar_polar)
		
		# Interpolate nan values in endo and epi polar arrays:
		for cur_slice in range(endo_polar.shape[0]):
			for angle in range(endo_polar.shape[1]):
				angle_less = (angle - 1) % endo_polar.shape[1]
				angle_more = (angle + 1) % endo_polar.shape[1]
				if np.isnan(endo_polar[cur_slice, angle, 3]):
					endo_polar[cur_slice, angle, 3] = (endo_polar[cur_slice, angle_less, 3] + endo_polar[cur_slice, angle_more, 3])/2
				if np.isnan(epi_polar[cur_slice, angle, 3]):
					epi_polar[cur_slice, angle, 3] = (epi_polar[cur_slice, angle_less, 3] + epi_polar[cur_slice, angle_more, 3])/2

		return([endo_polar, epi_polar, scar_polar])
		
	def _plotStacks(self, abs_shifted, ab_list, endo_shifted, epi_shifted, scar_shifted=[], scar=False):
		"""Plot all pinpoints, endocardial contour, epicardial contour, and (if passed) scar points"""
		# Pull and convert values
		ab_x, ab_y, ab_z = ab_list
		endo_shift_arr = np.array(endo_shifted)
		epi_shift_arr = np.array(epi_shifted)
		# Established matplotlib3d figure and axes
		fig = mplt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# Use scatter to plot the apex, base, and septal points
		ax.scatter(abs_shifted[:, 0], abs_shifted[:, 1], abs_shifted[:, 2])
		# Plot the line between the apex and base points
		ax.plot(ab_x, ab_y, ab_z, c='c')
		# Use the scatter to plot the endo and epi contours
		ax.scatter(endo_shift_arr[:, 0], endo_shift_arr[:, 1], endo_shift_arr[:, 2], c='r')
		ax.scatter(epi_shift_arr[:, 0], epi_shift_arr[:, 1], epi_shift_arr[:, 2], c='g')
		if scar:
			# Plot all of the scar points passed
			scar_shift_arr = np.array(scar_shifted)
			ax.scatter(scar_shift_arr[:, 0], scar_shift_arr[:, 1], scar_shift_arr[:, 2], c='k')
			
	def _getEndoEpiFromStack(self, endo_stack, epi_stack, sastruct, rv_insertion_pts, septal_slice, apex_base_pts, scar_stack=np.empty([0])):
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
		
	def _shiftPolarCartesian(self, endo_polar, epi_polar, endo, epi, kept_slices, axis_center, wall_thickness):
		"""Shift polar array into cartesian coordinates
		
		args:
			endo_polar (array): polar endocardial contour
			epi_polar (array): polar epicardial contour
			endo (array): endocardial points from getEndoEpiFromStack
			epi (array): epicardial pionts from getEndoEpiFromStack
			kept_slices (list): the slices with contours from SEGMENT
			axis_center (array): the center point of the apex-base axis in each slice
			wall_thickness (array): wall thickness (polar) in each angle for each slice
		"""
		cine_endo = []
		cine_epi = []
		for slices in kept_slices-1:
		
			# Convert Endo Stack from Polar to Cartesian:
			endo_pol_rho = endo_polar[slices, :, 3]
			endo_pol_theta = endo_polar[slices, :, 1]
			x_endo_cart, y_endo_cart = self._polToCart(endo_pol_rho, endo_pol_theta)
			endo_z = [endo[slices][0, 2] for i in range(len(x_endo_cart))]
			
			# Convert Epi Stack from Polar to Cartesian:
			epi_pol_rho = epi_polar[slices, :, 3]
			epi_pol_theta = epi_polar[slices, :, 1]
			x_epi_cart, y_epi_cart = self._polToCart(epi_pol_rho, epi_pol_theta)
			epi_z = [epi[slices][0, 2] for i in range(len(x_epi_cart))]
			
			# Shift Points back from previous Polar Origin Shift:
			polar_center = np.mean([x_epi_cart, y_epi_cart], axis=1)
			center_diff = [polar_center[0] - axis_center[slices][0], polar_center[1] - axis_center[slices][1]]
			x_endo_cart -= center_diff[0]
			y_endo_cart -= center_diff[1]
			x_epi_cart -= center_diff[0]
			y_epi_cart -= center_diff[1]

			wall = wall_thickness[slices, :, 1]
			cine_endo.append([x_endo_cart, y_endo_cart, endo_z, wall])
			cine_epi.append([x_epi_cart, y_epi_cart, epi_z, wall])
		
		# Swap the axes and reshape to get format the endo and epi traces
		temp_cine_endo = np.swapaxes(np.array(cine_endo), 1, 2)
		new_cine_endo = temp_cine_endo.reshape([temp_cine_endo.shape[0]*temp_cine_endo.shape[1], temp_cine_endo.shape[2]])
		
		temp_cine_epi = np.swapaxes(np.array(cine_epi), 1, 2)
		new_cine_epi = temp_cine_epi.reshape([temp_cine_epi.shape[0]*temp_cine_epi.shape[1], temp_cine_epi.shape[2]])

		return([new_cine_endo, new_cine_epi])
		
	def alignScarCine(self, timepoint=0):
		"""One of the attempts to align the scar and cine meshes.
		
		This function currently does nothing and acts as a placeholder.
		"""
		# If a timepoint is passed, pull the cine from that point
		cine_endo = self.cine_endo[timepoint]
		cine_epi = self.cine_epi[timepoint]
		# Get slice values to section the endo / epi array by slice
		slice_indices = sorted(np.unique(cine_endo[:, 2], return_index=True)[1])
		slice_vals = cine_endo[slice_indices, 2]
		# Set up angle bins
		num_bins = self.scar_ratio.shape[1] + 1
		angles = np.linspace(-math.pi, math.pi, num_bins)
		angles2 = angles[1:]
		angles2 = np.append(angles2, angles[0])
		angles = np.column_stack((angles, angles2))[:-1]
		# Iterate through slices and convert to polar
		full_scar_contour = []
		for i in range(len(slice_vals)):
			# Get indices for the current slice
			cur_slice_ind = np.where(cine_endo[:, 2] == slice_vals[i])[0]
			# Pull current slice endocardial and epicardial cartesian contours
			cur_slice_endo = cine_endo[cur_slice_ind, :]
			cur_slice_epi = cine_epi[cur_slice_ind, :]
			
			# Get the slice center and shift by that value (center slices at 0)
			slice_center = np.mean(cur_slice_epi, axis=0)
			endo_x = cur_slice_endo[:, 0] - slice_center[0]
			endo_y = cur_slice_endo[:, 1] - slice_center[1]
			epi_x = cur_slice_epi[:, 0] - slice_center[0]
			epi_y = cur_slice_epi[:, 1] - slice_center[1]
			
			# Convert the cartesian contours to polar
			endo_theta, endo_rho = self._cartToPol(endo_x, endo_y)
			epi_theta, epi_rho = self._cartToPol(epi_x, epi_y)
			endo_theta = [cur_slice_theta_i - 2*np.pi if cur_slice_theta_i > np.pi else cur_slice_theta_i for cur_slice_theta_i in endo_theta]
			epi_theta = [cur_slice_theta_i - 2*np.pi if cur_slice_theta_i > np.pi else cur_slice_theta_i for cur_slice_theta_i in epi_theta]
			# Get rho values for each angle bin based on theta
			endo_bin_inds = [np.where((endo_theta > angles[i, 0]) & (endo_theta <= angles[i, 1]))[0].tolist() for i in range(angles.shape[0])]
			epi_bin_inds = [np.where((epi_theta > angles[i, 0]) & (epi_theta <= angles[i, 1]))[0].tolist() for i in range(angles.shape[0])]
			endo_rho_mean = [np.mean(endo_rho[endo_bin_inds_i]) for endo_bin_inds_i in endo_bin_inds]
			epi_rho_mean = [np.mean(epi_rho[epi_bin_inds_i]) for epi_bin_inds_i in epi_bin_inds]
			
			# Get the current scar slice
			cur_scar = self.scar_ratio[i, :, :]
			
			# Adjust any values less than 0 or greater than 1 in the ratio
			with np.errstate(invalid='ignore'):
				nonan_inds = np.where(~np.isnan(cur_scar[:, 1]))[0].tolist()
				for j in range(len(nonan_inds)):
					# Check if the value is less than 0
					if cur_scar[nonan_inds[j], 1] <= 0:
						# Set it equal to the average of the adjacent values (to create a smooth scar trace)
						if j == 0:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j], 2], cur_scar[nonan_inds[j+1], 1]])
						elif j == len(nonan_inds) - 1:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j-1], 1], cur_scar[nonan_inds[j], 2]])
						else:
							cur_scar[nonan_inds[j], 1] = np.mean([cur_scar[nonan_inds[j-1], 1], cur_scar[nonan_inds[(j+1) % (len(nonan_inds)-1)], 1]])
					# Check if the value is greater than 1
					if cur_scar[nonan_inds[j], 2] >= 1:
						# Set it equal to the average of the adjacent values (to create a smooth scar trace)
						if j == 0:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j], 1], cur_scar[nonan_inds[j+1], 2]])
						elif j == len(nonan_inds) - 1:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j-1], 2], cur_scar[nonan_inds[j], 1]])
						else:
							cur_scar[nonan_inds[j], 2] = np.mean([cur_scar[nonan_inds[j-1], 2], cur_scar[nonan_inds[(j+1) % (len(nonan_inds)-1)], 2]])
			
			# Get the scar inner and outer rho values based on endo and epi rho values
			scar_inner_rho = [endo_rho_mean[j] + cur_scar[j, 1] * (epi_rho_mean[j] - endo_rho_mean[j]) for j in range(cur_scar.shape[0])]
			scar_outer_rho = [endo_rho_mean[j] + cur_scar[j, 2] * (epi_rho_mean[j] - endo_rho_mean[j]) for j in range(cur_scar.shape[0])]
			# Convert the scar values to cartesian
			scar_inner_x, scar_inner_y = self._polToCart(scar_inner_rho, cur_scar[:, 0])
			scar_outer_x, scar_outer_y = self._polToCart(scar_outer_rho, cur_scar[:, 0])
			# If there is no scar trace here, just move to the next slice and append an empty array for the current contour
			if np.all(np.isnan(scar_inner_x)):
				full_scar_contour.append(np.array([]))
				continue
			
			# Extract the contour and remove NaN values
			# Roll the array so that the first point of the contour is the non-nan value immediately after nan
			nan_boundary = np.where([np.isnan(scar_inner_x[j]) & ~np.isnan(scar_inner_x[j+1]) for j in range(scar_inner_x.size - 1)])[0]
			scar_inner_x = np.roll(scar_inner_x, -nan_boundary[0]-1)
			# Find the end of the contour and slice, removing NaNs
			num_boundary = np.where([~np.isnan(scar_inner_x[j]) & np.isnan(scar_inner_x[j+1]) for j in range(scar_inner_x.size - 1)])[0]
			scar_inner_x = scar_inner_x[:num_boundary[0]+1]
			# Roll each other contour
			scar_outer_x = np.roll(scar_outer_x, -nan_boundary[0]-1)[:num_boundary[0]+1]
			scar_inner_y = np.roll(scar_inner_y, -nan_boundary[0]-1)[:num_boundary[0]+1]
			scar_outer_y = np.roll(scar_outer_y, -nan_boundary[0]-1)[:num_boundary[0]+1]
			# Construct combined arrays in trace order, inner -> outer in a loop
			scar_x = np.append(scar_inner_x, scar_outer_x[::-1])
			scar_y = np.append(scar_inner_y, scar_outer_y[::-1])
			# Re-shift based on center of slice
			scar_x += slice_center[0]
			scar_y += slice_center[1]
			# Stack the x, y, and z values into a slice of the full contour
			full_scar_contour.append(np.column_stack((scar_x, scar_y, [slice_vals[i]]*scar_x.size)))
		
		# Get equal number of points per slice
		for i in range(len(full_scar_contour)):
			scar_layer = full_scar_contour[i]
			if scar_layer.size == 0: continue
			scar_xy_pts = scar_layer[:, :2]
			scar_range_pts = np.linspace(0, 1, scar_xy_pts.shape[0])
			interp_func = sp.interpolate.interp1d(scar_range_pts, scar_xy_pts, kind='cubic', axis=0)
			new_scar_numpts = np.linspace(0, 1, 80)
			new_xy_pts = interp_func(new_scar_numpts)
			new_xy_pts = np.column_stack((new_xy_pts, [scar_layer[0, 2]] * 80))
			full_scar_contour[i] = new_xy_pts
		
		# Interpolate additional scar slices
		interp_scar_contour = []
		for i in range(len(full_scar_contour) - 1):
			# Get scar slices to use for interpolation
			scar_layer = full_scar_contour[i]
			scar_adj_layer = full_scar_contour[i+1]
			# If the layers are edge layers, or non-scar, don't bother
			if scar_layer.size == 0 or scar_adj_layer.size == 0: continue
			# Use the two slices as the edge values for the interpolation function
			interp_arr = [0, 1]
			x_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 0], scar_adj_layer[:, 0])))
			y_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 1], scar_adj_layer[:, 1])))
			z_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_layer[:, 2], scar_adj_layer[:, 2])))
			# Interpolate a number of intermediate slices
			new_interp_arr = np.linspace(0, 1, 5)
			new_x_vals = x_interp_func(new_interp_arr)
			new_y_vals = y_interp_func(new_interp_arr)
			new_z_vals = z_interp_func(new_interp_arr)
			# Reconstruct each slice back together, removing duplicate slices
			for j in range(new_x_vals.shape[1]):
				cur_slice = np.column_stack((new_x_vals[:, j], new_y_vals[:, j], new_z_vals[:, j]))
				slice_stored = any((np.around(cur_slice, 5) == np.around(slice_arr, 5)).all() for slice_arr in interp_scar_contour)
				if not slice_stored: interp_scar_contour.append(np.column_stack((new_x_vals[:, j], new_y_vals[:, j], new_z_vals[:, j])))
		full_scar_contour = interp_scar_contour
		# Append the contour to the total aligned scar data (essentially tracks timepoints)
		if len(self.aligned_scar) == 0:
			self.aligned_scar = [None] * len(self.cine_endo)
		self.aligned_scar[timepoint] = full_scar_contour
		return(full_scar_contour)
		
	def __getTimeIndices(self, contour, time_pts, timepoint=0):
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
		
	def alignDense(self, cine_timepoint=0):
		"""Align DENSE data to a cine slice by selected timepoint.
		"""
		# Get data at specified timepoints.
		try:
			cine_endo_timepoint = self.cine_endo[cine_timepoint]
			cine_epi_timepoint = self.cine_epi[cine_timepoint]
		except:
			print("Invalid timepoint selected. Try again.")
			return(False)
		
		dense_aligned_pts = [False]*len(self.dense_pts)
		scaled_strain = [False]*len(self.dense_displacement)
		
		# Iterate through DENSE slices
		for slice_num in range(len(self.dense_endo)):
			# Extract slice-based values
			dense_slice_endo = self.dense_endo[slice_num]
			dense_slice_epi = self.dense_epi[slice_num]
			dense_slice_pts = self.dense_pts[slice_num]
			cur_slice = self.dense_slices[slice_num]
			slice_in_cine = np.where(round(cur_slice, 1) == np.round(cine_endo_timepoint[:, 2], 1))[0]
			
			# Get cine contours matching DENSE slice
			cine_endo_slice = cine_endo_timepoint[slice_in_cine, :] - np.mean(cine_epi_timepoint[slice_in_cine, :], axis=0)
			cine_epi_slice = cine_epi_timepoint[slice_in_cine, :] - np.mean(cine_epi_timepoint[slice_in_cine, :], axis=0)
			
			#print(np.mean(cine_epi_timepoint[slice_in_cine, :]))
			
			# Convert both sets of slices to polar
			dense_endo_theta, dense_endo_rho = self._cartToPol(dense_slice_endo[:, 0], dense_slice_endo[:, 1])
			dense_epi_theta, dense_epi_rho = self._cartToPol(dense_slice_epi[:, 0], dense_slice_epi[:, 1])
			cine_endo_theta, cine_endo_rho = self._cartToPol(cine_endo_slice[:, 0], cine_endo_slice[:, 1])
			cine_epi_theta, cine_epi_rho = self._cartToPol(cine_epi_slice[:, 0], cine_epi_slice[:, 1])
			
			# Generate interpolation equations to ensure order and number of points is the same
			theta_interp_pts = np.linspace(0, 2*math.pi, 100)[:-1]

			dense_endo_eq = sp.interpolate.interp1d(dense_endo_theta, dense_endo_rho, fill_value="extrapolate")
			dense_epi_eq = sp.interpolate.interp1d(dense_epi_theta, dense_epi_rho, fill_value="extrapolate")
			cine_endo_eq = sp.interpolate.interp1d(cine_endo_theta, cine_endo_rho, fill_value="extrapolate")
			cine_epi_eq = sp.interpolate.interp1d(cine_epi_theta, cine_epi_rho, fill_value="extrapolate")
			
			# Get interpolated rho values at new theta points
			dense_endo_interp_rho = dense_endo_eq(theta_interp_pts)
			dense_epi_interp_rho = dense_epi_eq(theta_interp_pts)
			cine_endo_interp_rho = cine_endo_eq(theta_interp_pts)
			cine_epi_interp_rho = cine_epi_eq(theta_interp_pts)
			
			# Get endo and epi values together, in order, since transform must be universal
			dense_interp_vals = np.append(np.column_stack((theta_interp_pts, dense_endo_interp_rho)), np.column_stack((theta_interp_pts, dense_epi_interp_rho)), axis=0)
			cine_interp_vals = np.append(np.column_stack((theta_interp_pts, cine_endo_interp_rho)), np.column_stack((theta_interp_pts, cine_epi_interp_rho)), axis=0)
			
			# Convert back to x and y to get a 2d transform field
			dense_interp_x, dense_interp_y = self._polToCart(dense_interp_vals[:, 1], dense_interp_vals[:, 0])
			cine_interp_x, cine_interp_y = self._polToCart(cine_interp_vals[:, 1], cine_interp_vals[:, 0])
			
			# Calculate the linear distance in x and y between 
			x_dist = cine_interp_x - dense_interp_x
			y_dist = cine_interp_y - dense_interp_y
			
			# Pull slice of points
			dense_pts = self.dense_pts[slice_num]
			
			# Interpolate new grid points
			dense_pts_new_x = dense_pts[:, 0] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), x_dist, dense_pts[:, :2], method='cubic')
			dense_pts_new_y = dense_pts[:, 1] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), y_dist, dense_pts[:, :2], method='cubic')
			
			# Get points outside of the hull and use nearest-neighbor to calculate positional change
			isnan_x = np.where(np.isnan(dense_pts_new_x))[0]
			isnan_y = np.where(np.isnan(dense_pts_new_y))[0]
			
			dense_pts_new_x[isnan_x] = dense_pts[isnan_x, 0] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), x_dist, dense_pts[isnan_x, :2], method='nearest')
			dense_pts_new_y[isnan_y] = dense_pts[isnan_y, 1] + sp.interpolate.griddata(np.column_stack((dense_interp_x, dense_interp_y)), y_dist, dense_pts[isnan_y, :2], method='nearest')
			
			# Get the difference in scale between the old and new positions to scale strain
			scale_diff_x = (np.max(dense_pts_new_x) - np.min(dense_pts_new_x)) / (np.max(dense_pts[:, 0]) - np.min(dense_pts[:, 0]))
			scale_diff_y = (np.max(dense_pts_new_y) - np.min(dense_pts_new_y)) / (np.max(dense_pts[:, 1]) - np.min(dense_pts[:, 1]))
			
			# Scale strain
			for time in range(len(self.dense_displacement)):
				dense_time_disp = self.dense_displacement[time] if not isinstance(self.dense_displacement[time], list) else self.dense_displacement[time][slice_num]
				dense_time_x_scale = dense_time_disp[:, 0] * scale_diff_x
				dense_time_y_scale = dense_time_disp[:, 1] * scale_diff_y
				if not scaled_strain[time]:
					scaled_strain[time] = [np.column_stack((dense_time_x_scale, dense_time_y_scale))]
				else:
					scaled_strain[time].append(np.column_stack((dense_time_x_scale, dense_time_y_scale)))
			
			# Store loop variables
			dense_aligned_pts[slice_num] = np.column_stack((dense_pts_new_x, dense_pts_new_y))
			
		# Set globals upon loop completion
		self.dense_aligned_displacement = scaled_strain
		self.dense_aligned_pts = dense_aligned_pts
			
		return(True)