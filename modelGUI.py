# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 12:48:42 2017

@author: cdw2be
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
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
		
		# On window close by user
		master.protocol('WM_DELETE_WINDOW', self.destroy())

	def createWidgets(self):
		# Establish tk variables
		sa_filename = tk.StringVar()
		la_filename = tk.StringVar()
		lge_filename = tk.StringVar()
		dense_filenames = tk.StringVar()
		confocal_dir = tk.StringVar()
		mesh_timepoint = tk.IntVar()
		dense_timepoint = tk.IntVar()
		mesh_time_opts = [0]
		dense_time_opts = [0]
		
		# Create Entry objects for files
		sa_file_entry = ttk.Entry(width=80, textvariable=sa_filename)
		la_file_entry = ttk.Entry(width=80, textvariable=la_filename)
		lge_file_entry = ttk.Entry(width=80, textvariable=lge_filename)
		dense_file_entry = ttk.Entry(width=80, textvariable=dense_filenames)
		confocal_dir_entry = ttk.Entry(width=80, textvariable=confocal_dir)
		
		# Creat Combobox objects for timepoints
		mesh_timepoint_cbbox = ttk.Combobox(values=mesh_time_opts, state='readonly', width=5)
		dense_timepoint_cbbox = ttk.Combobox(values=dense_time_opts, state='readonly', width=5)
		mesh_timepoint_cbbox.current(0)
		dense_timepoint_cbbox.current(0)
		
		# Create labels for entries
		ttk.Label(text='Short-Axis File:').grid(row=0, sticky='W')
		ttk.Label(text='Long-Axis File:').grid(row=1, sticky='W')
		ttk.Label(text='LGE File:').grid(row=2, sticky='W')
		ttk.Label(text='DENSE Files:').grid(row=3, sticky='W')
		ttk.Label(text='Confocal Directory:').grid(row=4, sticky='W')
		ttk.Label(text='DENSE Timepoint').grid(row=1, column=7, sticky='W')
		ttk.Label(text='Mesh Timepoint:').grid(row=0, column=7, sticky='W')
		
		# Place entry object
		sa_file_entry.grid(row=0, column=1, columnspan=5)
		la_file_entry.grid(row=1, column=1, columnspan=5)
		lge_file_entry.grid(row=2, column=1, columnspan=5)
		dense_file_entry.grid(row=3, column=1, columnspan=5)
		confocal_dir_entry.grid(row=4, column=1, columnspan=5)
		mesh_timepoint_cbbox.grid(row=0, column=8)
		dense_timepoint_cbbox.grid(row=1, column=8)
		
		# Add browse buttons for file exploration, to pass to entry boxes
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(sa_file_entry)).grid(row=0, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(la_file_entry)).grid(row=1, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(lge_file_entry)).grid(row=2, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(dense_file_entry, multi='True')).grid(row=3, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(confocal_dir_entry, multi='Dir')).grid(row=4, column=6)
		
		# Add buttons to form model components
		ttk.Button(text='Generate MRI Model', command= lambda: self.createMRIModel(sa_filename, la_filename, lge_filename, dense_filenames, mesh_timepoint_cbbox, dense_timepoint_cbbox)).grid(row=5, column=1, sticky='W')
		ttk.Button(text='Generate MRI Mesh', command= lambda: self.createMRIMesh()).grid(row=5, column=2, sticky='W')
		
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
		
	def createMRIModel(self, sa_filename, la_filename, lge_filename, dense_filenames, mesh_timepoint_cbbox, dense_timepoint_cbbox):
		self.mri_model = mrimodel.MRIModel(sa_filename.get(), la_filename.get(), scar_file=lge_filename.get(), dense_file=dense_filenames.get())
		if self.mri_model.scar:
			self.mri_model.importLGE()
		print(list(range(len(self.mri_model.cine_endo))))
		mesh_timepoint_cbbox['values'] = list(range(len(self.mri_model.cine_endo)))
		dense_timepoint_cbbox['values'] = list(range(len(self.mri_model.cine_endo)))

	def createMRIMesh(self):
		self.mri_mesh = mesh.Mesh()
		cine_endo_mat, cine_epi_mat = self.mri_mesh.fitContours(self.mri_model.cine_endo[time_point], self.mri_model.cine_epi[time_point], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts, '4x2')
			
root = tk.Tk()
gui = modelGUI(master=root)
gui.master.title('Cardiac Modeling Toolbox')
gui.mainloop()