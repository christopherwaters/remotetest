# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 12:48:42 2017

@author: cdw2be
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
import warnings
import mrimodel
import confocalmodel
import mesh

class modelGUI(tk.Frame):
	"""Generates a GUI to control the Python-based cardiac modeling toolbox.
	"""
	
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)
		self.grid()
		self.createWidgets()
		
		# Set row and column paddings
		master.rowconfigure(0, pad=5)
		master.rowconfigure(1, pad=5)
		master.rowconfigure(2, pad=5)
		master.rowconfigure(3, pad=5)
		master.rowconfigure(4, pad=5)
		master.columnconfigure(8, pad=5)
		master.columnconfigure(9, pad=5)
		
		# On window close by user
		master.protocol('WM_DELETE_WINDOW', self.destroy())

	def createWidgets(self):
		# Establish tk variables
		sa_filename = tk.StringVar()
		la_filename = tk.StringVar()
		lge_filename = tk.StringVar()
		dense_filenames = tk.StringVar()
		confocal_dir = tk.StringVar()
		cine_time_opts = ['N/A']

		# Create Entry objects for files
		sa_file_entry = ttk.Entry(width=80, textvariable=sa_filename)
		la_file_entry = ttk.Entry(width=80, textvariable=la_filename)
		lge_file_entry = ttk.Entry(width=80, textvariable=lge_filename)
		dense_file_entry = ttk.Entry(width=80, textvariable=dense_filenames)
		confocal_dir_entry = ttk.Entry(width=80, textvariable=confocal_dir)
		
		# Creat Combobox objects for timepoints
		cine_timepoint_cbox = ttk.Combobox(values=cine_time_opts, state='disabled', width=5)
		cine_timepoint_cbox.current(0)
		
		# Create Entry objects for mesh settings
		num_rings_entry = ttk.Entry(width=10)
		elem_per_ring_entry = ttk.Entry(width=10)
		elem_thru_wall_entry = ttk.Entry(width=10)
		mesh_type_cbox = ttk.Combobox(values=['4x2', '4x4', '4x8'], state='readonly', width=10)
		mesh_type_cbox.current(0)
		
		# Settings for combobox selections
		cine_timepoint_cbox.bind('<<ComboboxSelected>>', lambda _ : self.cineTimeChanged(cine_timepoint_cbox))
		
		# Grid placement of separators
		ttk.Separator(orient='vertical').grid(column=9, row=0, rowspan=7, sticky='NS')
		
		# Create labels for entries
		#   Filename labels
		importLabel = ttk.Label(text='Model Import Options and Settings')
		importLabel.grid(row=0, column=0, columnspan=9)
		ttk.Label(text='Short-Axis File:').grid(row=1, sticky='W')
		ttk.Label(text='Long-Axis File:').grid(row=2, sticky='W')
		ttk.Label(text='LGE File:').grid(row=3, sticky='W')
		ttk.Label(text='DENSE Files:').grid(row=4, sticky='W')
		ttk.Label(text='Confocal Directory:').grid(row=5, sticky='W')
		#   Indicator Labels
		ttk.Label(text='Primary Cine Timepoint:').grid(row=1, column=7, sticky='W')
		self.progLabel = ttk.Label(text='Ready')
		self.progLabel.grid(row=6, column=0, columnspan=8)
		#   Mesh setting labels
		meshLabel = ttk.Label(text='Mesh Settings')
		meshLabel.grid(row=0, column=10, columnspan=2)
		ttk.Label(text='Number of Rings:').grid(row=1, column=10, sticky='W')
		ttk.Label(text='Elements per Ring:').grid(row=2, column=10, sticky='W')
		ttk.Label(text='Elements through Wall:').grid(row=3, column=10, sticky='W')
		ttk.Label(text='Mesh Type:').grid(row=4, column=10, sticky='W')
		
		# Set specific label fonts
		f = font.Font(meshLabel, meshLabel.cget('font'))
		f.configure(underline=True)
		meshLabel.configure(font=f)
		importLabel.configure(font=f)
		
		# Place entry object
		sa_file_entry.grid(row=1, column=1, columnspan=5)
		la_file_entry.grid(row=2, column=1, columnspan=5)
		lge_file_entry.grid(row=3, column=1, columnspan=5)
		dense_file_entry.grid(row=4, column=1, columnspan=5)
		confocal_dir_entry.grid(row=5, column=1, columnspan=5)
		cine_timepoint_cbox.grid(row=1, column=8)
		num_rings_entry.grid(row=1, column=11)
		elem_per_ring_entry.grid(row=2, column=11)
		elem_thru_wall_entry.grid(row=3, column=11)
		num_rings_entry.insert(0, '14')
		elem_per_ring_entry.insert(0, '25')
		elem_thru_wall_entry.insert(0, '5')
		mesh_type_cbox.grid(row=4, column=11)
		
		# Add browse buttons for file exploration, to pass to entry boxes
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(sa_file_entry)).grid(row=1, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(la_file_entry)).grid(row=2, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(lge_file_entry)).grid(row=3, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(dense_file_entry, multi='True')).grid(row=4, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(confocal_dir_entry, multi='Dir')).grid(row=5, column=6)
		
		# Add buttons to form model components
		ttk.Button(text='Generate MRI Model', command= lambda: self.createMRIModel(sa_filename, la_filename, lge_filename, dense_filenames, cine_timepoint_cbox)).grid(row=2, column=7, columnspan=2)
		self.meshButton = ttk.Button(text='Generate MRI Mesh', state='disabled', command= lambda: self.createMRIMesh(cine_timepoint_cbox))
		self.meshButton.grid(row=5, column=10, columnspan=2)
		
	def openFileBrowser(self, entry_box, multi='False'):
		"""Open a file browser window and assign the file name to the passed entry box.
		"""
		if multi == 'True':
			file_name = filedialog.askopenfilenames(title='Select Files')
		elif multi == 'Dir':
			file_name = filedialog.askdirectory(title='Select Folder')
		else:
			file_name = filedialog.askopenfilename(title='Select File')
		entry_box.delete(0, 'end')
		entry_box.insert(0, file_name)
		return(file_name)
		
	def createMRIModel(self, sa_filename, la_filename, lge_filename, dense_filenames, cine_timepoint_cbox):
		# Create model, import initial cine stack
		self.progLabel['text'] = 'Generating MRI Model'
		self.mri_model = mrimodel.MRIModel(sa_filename.get(), la_filename.get(), scar_file=lge_filename.get(), dense_file=dense_filenames.get())
		self.progLabel['text'] = 'Importing Cine Stack'
		self.mri_model.importCine(timepoint=0)
		# Import LGE, if included, and generate full alignment array
		if self.mri_model.scar:
			self.progLabel['text'] = 'Importing Scar Stack'
			self.mri_model.importLGE()
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				for i in range(len(self.mri_model.cine_endo)):
					self.mri_model.alignScarCine(timepoint=i)
		# Import DENSE, if included
		if self.mri_model.dense:
			self.progLabel['text'] = 'Importing DENSE Stack'
			self.mri_model.importDense()
			self.mri_model.alignDense(align_timepoint=0)
		# Update GUI elements
		cine_timepoint_cbox.configure(values=list(range(len(self.mri_model.cine_endo))), state='readonly')
		cine_timepoint_cbox.current(0)
		self.progLabel['text'] = 'MRI Model Generated Successfully'
		self.meshButton.configure(state='normal')

	def createMRIMesh(self, cine_timepoint_cbox):
		self.mri_mesh = mesh.Mesh()
		time_point = int(cine_timepoint_cbox.get())
		cine_endo_mat, cine_epi_mat = self.mri_mesh.fitContours(self.mri_model.cine_endo[time_point], self.mri_model.cine_epi[time_point], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts, '4x2')
	
	def cineTimeChanged(self, cine_timepoint_cbox):
		self.progLabel.configure(text='Updating timepoint in model.')
		try:
			new_timepoint = int(cine_timepoint_cbox.get())
		except:
			self.progLabel.configure(text='Timepoint selection failed. Probably a NaN timepoint.')
			return(False)
		self.mri_model.importCine(timepoint = new_timepoint)
		if self.mri_model.dense:
			self.mri_model.alignDense(align_timepoint = new_timepoint)
		self.progLabel.configure(text='Timepoint successfully updated!')
			
root = tk.Tk()
gui = modelGUI(master=root)
gui.master.title('Cardiac Modeling Toolbox')
gui.mainloop()