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
	
	"""Contains several functions to assist with import functionality for modeling and mesh data
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