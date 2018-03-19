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
		self.top_dir = top_dir
		self.slices = [ConfocalSlice(im_dir) for im_dir in dirs]
		self.slice_names = [confocal_slice.slice_name for confocal_slice in self.slices]
	
	def generateStitchedImages(self, slices, sub_slices, channel_list, overlap=0.1, compress_ratio=0.25, force_file=False):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		# Set subslices list to mid-slice if none is indicated.
		for slice_ind, slice_num in enumerate(slices):
			# Determine if the sub_slices list is per-slice, or a general list.
			cur_sub_slices = sub_slices[slice_ind] if isinstance(sub_slices[slice_ind], list) else sub_slices
			cur_slice_chans = channel_list[slice_ind] if isinstance(channel_list[slice_ind], list) else channel_list
			for sub_slice in cur_sub_slices:
				file_name = self.top_dir + '/' + self.slice_names[slice_num] + 'Frame' + str(sub_slice) + 'Stitched.tif'
				if (not os.path.isfile(file_name)) or force_file:
					self.slices[slice_num].createStitchedImage(overlap=overlap, compress_ratio=compress_ratio, channel=cur_slice_chans, frame=sub_slice, stitched_file=file_name, force_file=force_file)
				else:	
					print('Image already exists! No need to overwrite.')
					
	def getSubsliceList(self, top_slice):
		"""Gather information about which sub-slices are present within each image.
		"""
		return(list(range(self.slices[top_slice].num_slices)))
		
	def getChannelList(self, top_slice):
		"""Gather information about which channels are present within each image.
		"""
		return(list(self.slices[top_slice].channels))
		
class ConfocalSlice():
	"""Class to hold information for a single biological slice of tissue.
	
	Biological slices are stored in individual directories.
	"""
	def __init__(self, confocal_dir):
		# Store the directory of images used for this model and parse out the folder name.
		self.confocal_dir = confocal_dir
		self.slice_name = os.path.split(confocal_dir)[1]
		
		# Record filenames for all tiff image files in the directory.
		self.tif_files = glob.glob(os.path.join(confocal_dir, '*.tif')).copy()
		self.raw_images = [None]*len(self.tif_files)
		
		# Iterate through and create image objects for each tiff file
		for file_num, image_file in enumerate(self.tif_files):
			self.raw_images[file_num] = Image.open(image_file)
		
		# Define how many sub-slices (confocal slices) are contained within the overall slice
		self.num_slices = self.raw_images[0].n_frames
		
		# Define the channels within the image
		self.channels = self.raw_images[0].getbands()
		
		# Set up a list to hold compressed versions of the images, formatted as slices at top level
		#		This is due to compression only working on a single slice
		self.compressed_images = [None]*self.num_slices
		
		# Assign a file to contain positional information for the images for stitching purposes
		#		This file may or may not exist already in the directory, but is either referenced or created during stitching
		self.image_grid_file = self.confocal_dir + '/stitch_grid.txt'
	
	def createStitchedImage(self, overlap=0.1, compress_ratio=0.25, channel=-1, frame=0, stitched_file=False, force_file=False):
		"""Stitch images together in a grid.
		"""
		# Either read or create the image grid file
		if os.path.isfile(self.image_grid_file):
			self.im_grid = confocalhelper.readImageGrid(self.image_grid_file)
		else:
			# If the image grid file does not exist, create it for future use.
			self.im_locs, im_locs_dict = confocalhelper.getImagePositions(self.tif_files)
			self.im_grid = confocalhelper.getImageGrid(self.tif_files, self.im_locs, im_locs_dict)
			confocalhelper.writeImageGrid(self.im_grid, self.image_grid_file)
		
		# Define a name for the file to use to store the stitched image
		if not stitched_file:
			stitched_file = self.confocal_dir + '/StitchedImages/' + self.slice_name + 'slice' + str(frame) + 'chan' + str(channel) + '.tif'
		if os.path.isfile(stitched_file) and not force_file:
			# If the file exists and overwriting isn't forced, don't bother stitching
			print('File exists! No need to overwrite.')
			return(True)
		
		# If there is nothing in the compressed images (i.e. it hasn't already been compressed), instantiate it
		if not self.compressed_images[frame]:
			self.compressed_images[frame] = [None]*len(self.raw_images)
		
		# Iterate through the images to generate compressed versions
		for image_num, raw_image in enumerate(self.raw_images):
			# Pull the desired frame and channels
			image_frames = confocalhelper.splitImageFrames(raw_image)
			image_frame = image_frames[frame]
			img_channels_complete = confocalhelper.splitChannels(image_frame)
			if isinstance(channel, list):
				img_channels = [img_channels_complete[channel_num] for channel_num in channel]
			# If being told to compress the images, compress to desired ratio
			if compress_ratio < 1:
				compressed_channels = [None]*len(img_channels)
				for channel_num, cur_channel in enumerate(img_channels):
					compressed_channels[channel_num] = confocalhelper.compressImages(cur_channel, image_scale=compress_ratio)
				if len(compressed_channels) > 1:
					compressed_image = Image.merge('RGB', compressed_channels)
					self.compressed_images[frame][image_num] = compressed_image
				else:
					self.compressed_images[frame][image_num] = compressed_channels[0]

		# Pass the compressed images, image grid information, and stitched file to save to the image stitching function
		stitched_success = confocalhelper.stitchImages(self.compressed_images[frame], self.im_grid[:, 0], self.im_grid[:, 1], save_pos=stitched_file)