# -*- coding: utf-8 -*-
"""
Contains information necessary for import and alignment of confocal microscopy images.

Created on Mon Jan 27 12:22:15 2018

@author: cdw2be

Planar Image Brightness Correction based on: https://imagej.nih.gov/ij/plugins/plane-brightness/2010_Michalek_Biosignal.pdf
"""

# Imports
import math
import tkinter as tk
from tkinter import filedialog
import scipy as sp
import numpy as np
import matplotlib.pyplot as mplt
import glob
from PIL import Image
import os
from cardiachelpers import importhelper
from cardiachelpers import confocalhelper

class ConfocalModel():
	"""Model class to hold confocal microscopy images and format them to generate a mesh to align with MRI data.
	"""

	def __init__(self, top_dir):
		"""Initialize the model made to import confocal microscopy data.
		
		args:
			confocal_dir (string): The path to the directory containing tif image files
		"""
		
		# Get all TIFF files in the directory
		dirs = [(top_dir + '/' + sub_dir) for sub_dir in os.listdir(top_dir) if os.path.isdir(top_dir + '/' + sub_dir)]
		self.slices = [ConfocalSlice(im_dir) for im_dir in dirs]
	
	def generateStitchedImages(self, slices, sub_slices=[0], overlap=0.1, compress_ratio=0.25, channel=-1):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		for slice_num in slices:
			for sub_slice in sub_slices:
				self.slices[slice_num].createStitchedImage(overlap=overlap, compress_ratio=compress_ratio, channel=channel, frame=sub_slice)
		
class ConfocalSlice():
	"""Class to hold information for a single biological slice of tissue.
	
	Biological slices are stored in individual directories.
	"""
	def __init__(self, confocal_dir):
		self.confocal_dir = confocal_dir
		self.slice_name = os.path.split(confocal_dir)[1]
		
		self.tif_files = glob.glob(os.path.join(confocal_dir, '*.tif')).copy()
		self.raw_images = [None]*len(self.tif_files)
		
		for file_num, image_file in enumerate(self.tif_files):
			self.raw_images[file_num] = Image.open(image_file)
			
		self.num_slices = self.raw_images[0].n_frames
		
		self.compressed_images = [None]*self.num_slices
		
		self.stitched_image_files = [None]*self.num_slices
	
	def createStitchedImage(self, overlap=0.1, compress_ratio=0.25, channel=-1, frame=0, stitched_file=False):
		"""Stitch images together in a grid.
		"""
		self.im_locs, im_locs_dict = confocalhelper.getImagePositions(self.tif_files)
		
		self.im_grid = confocalhelper.getImageGrid(self.tif_files, self.im_locs, im_locs_dict)
		
		if not self.compressed_images[frame]:
			self.compressed_images[frame] = [None]*len(self.raw_images)
			
		for image_num, raw_image in enumerate(self.raw_images):
			image_frames = confocalhelper.splitImageFrames(raw_image)
			print(len(image_frames))
			image_frame = image_frames[frame]
			img_channels = confocalhelper.splitChannels(image_frame)
			if channel >= 0:
				img_channels = [img_channels[channel]]
			if compress_ratio < 1:
				compressed_channels = [None]*len(img_channels)
				for channel_num, cur_channel in enumerate(img_channels):
					compressed_channels[channel_num] = confocalhelper.compressImages(cur_channel, image_scale=compress_ratio)
				if len(compressed_channels) > 1:
					compressed_image = Image.merge('RGB', compressed_channels)
					self.compressed_images[frame][image_num] = compressed_image
				else:
					self.compressed_images[frame][image_num] = compressed_channels[0]
		
		if not stitched_file:
			stitched_file = self.confocal_dir + '/' + self.slice_name + 'slice' + str(frame) + 'chan' + str(channel) + '.tif'
			
		stitched_success = confocalhelper.stitchImages(self.compressed_images[frame], self.im_grid[:, 0], self.im_grid[:, 1], save_pos=stitched_file)