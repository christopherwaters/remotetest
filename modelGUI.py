# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 12:48:42 2017

@author: cdw2be
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
from tkinter import messagebox
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
		self.scar_plot_bool = tk.IntVar(value=0)
		self.dense_plot_bool = tk.IntVar(value=0)
		self.nodes_plot_bool = tk.IntVar(value=0)

		# Create Entry objects for files
		sa_file_entry = ttk.Entry(width=80, textvariable=sa_filename)
		la_file_entry = ttk.Entry(width=80, textvariable=la_filename)
		lge_file_entry = ttk.Entry(width=80, textvariable=lge_filename)
		dense_file_entry = ttk.Entry(width=80, textvariable=dense_filenames)
		confocal_dir_entry = ttk.Entry(width=80, textvariable=confocal_dir)
		self.postview_file_entry = ttk.Entry()
		
		# Creat Combobox objects for timepoints
		self.cine_timepoint_cbox = ttk.Combobox(state='disabled', width=5)
		self.dense_timepoint_cbox = ttk.Combobox(state='disabled', width=5)
		
		# Create Entry objects for mesh settings
		num_rings_entry = ttk.Entry(width=10)
		elem_per_ring_entry = ttk.Entry(width=10)
		elem_thru_wall_entry = ttk.Entry(width=10)
		mesh_type_cbox = ttk.Combobox(values=['4x2', '4x4', '4x8'], state='readonly', width=10)
		mesh_type_cbox.current(0)
		
		# Create checkboxes for plot options
		self.scar_cbutton = ttk.Checkbutton(variable = self.scar_plot_bool, state='disabled')
		self.dense_cbutton = ttk.Checkbutton(variable = self.dense_plot_bool, state='disabled')
		self.nodes_cbutton = ttk.Checkbutton(variable = self.nodes_plot_bool, state='disabled')
		self.scar_cbutton.grid(row=10, column=1, sticky='W')
		self.dense_cbutton.grid(row=11, column=1, sticky='W')
		self.nodes_cbutton.grid(row=9, column=1, sticky='W')
		
		# Settings for combobox selections
		self.cine_timepoint_cbox.bind('<<ComboboxSelected>>', lambda _ : self.cineTimeChanged())
		
		# Grid placement of separators
		ttk.Separator(orient='vertical').grid(column=9, row=0, rowspan=14, sticky='NS')
		ttk.Separator(orient='horizontal').grid(column=0, row=8, columnspan=9, sticky='EW')
		ttk.Separator(orient='vertical').grid(row=8, column=1, rowspan=6, sticky='NSE')
		
		# Create labels for entries
		#   Filename labels
		import_label = ttk.Label(text='Model Import Options and Settings')
		import_label.grid(row=0, column=0, columnspan=9)
		ttk.Label(text='Short-Axis File:').grid(row=1, sticky='W')
		ttk.Label(text='Long-Axis File:').grid(row=2, sticky='W')
		ttk.Label(text='LGE File:').grid(row=3, sticky='W')
		ttk.Label(text='DENSE Files:').grid(row=4, sticky='W')
		ttk.Label(text='Confocal Directory:').grid(row=5, sticky='W')
		postview_label = ttk.Label(text='Postview Options')
		postview_label.grid(row=9, column=2, columnspan=7)
		ttk.Label(text='Postview filename:').grid(row=10, column=2, sticky='W')
		#   Indicator Labels
		ttk.Label(text='Primary Cine Timepoint:').grid(row=1, column=7, sticky='W')
		self.progLabel = ttk.Label(text='Ready')
		self.progLabel.grid(row=6, column=0, columnspan=8)
		#   Mesh setting labels
		mesh_label = ttk.Label(text='Mesh Settings')
		mesh_label.grid(row=0, column=10, columnspan=2)
		ttk.Label(text='Number of Rings:').grid(row=1, column=10, sticky='W')
		ttk.Label(text='Elements per Ring:').grid(row=2, column=10, sticky='W')
		ttk.Label(text='Elements through Wall:').grid(row=3, column=10, sticky='W')
		ttk.Label(text='Mesh Type:').grid(row=4, column=10, sticky='W')
		#   Plot setting labels
		ttk.Label(text='Plot nodes in mesh?').grid(row=9, column=0, sticky='W')
		ttk.Label(text='Plot Scar?').grid(row=10, column=0, sticky='W')
		ttk.Label(text='Plot DENSE?').grid(row=11, column=0, sticky='W')
		ttk.Label(text='DENSE Timepoint:').grid(row=12, column=0, sticky='W')
		
		# Set specific label fonts
		f = font.Font(mesh_label, mesh_label.cget('font'))
		f.configure(underline=True)
		mesh_label.configure(font=f)
		import_label.configure(font=f)
		postview_label.configure(font=f)
		
		# Place entry object
		sa_file_entry.grid(row=1, column=1, columnspan=5)
		la_file_entry.grid(row=2, column=1, columnspan=5)
		lge_file_entry.grid(row=3, column=1, columnspan=5)
		dense_file_entry.grid(row=4, column=1, columnspan=5)
		confocal_dir_entry.grid(row=5, column=1, columnspan=5)
		self.postview_file_entry.grid(row=10, column=3, columnspan=5, sticky='WE')
		self.cine_timepoint_cbox.grid(row=1, column=8)
		num_rings_entry.grid(row=1, column=11)
		elem_per_ring_entry.grid(row=2, column=11)
		elem_thru_wall_entry.grid(row=3, column=11)
		num_rings_entry.insert(0, '14')
		elem_per_ring_entry.insert(0, '25')
		elem_thru_wall_entry.insert(0, '5')
		num_rings_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		elem_per_ring_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		elem_thru_wall_entry.configure(validate='key', validatecommand=(num_rings_entry.register(self.intValidate), '%P'))
		mesh_type_cbox.grid(row=4, column=11)
		self.dense_timepoint_cbox.grid(row=12, column=1, sticky='W')
		
		# Add browse buttons for file exploration, to pass to entry boxes
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(sa_file_entry)).grid(row=1, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(la_file_entry)).grid(row=2, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(lge_file_entry)).grid(row=3, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(dense_file_entry, multi='True')).grid(row=4, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(confocal_dir_entry, multi='Dir')).grid(row=5, column=6)
		ttk.Button(text='Browse', command= lambda: self.openFileBrowser(self.postview_file_entry, multi='Feb')).grid(row=10, column=8, sticky='W')
		
		# Add buttons to form model components
		ttk.Button(text='Generate MRI Model', command= lambda: self.createMRIModel(sa_filename, la_filename, lge_filename, dense_filenames)).grid(row=2, column=7, columnspan=2)
		self.meshButton = ttk.Button(text='Generate Model First', state='disabled', command= lambda: self.createMRIMesh(num_rings_entry, elem_per_ring_entry, elem_thru_wall_entry, mesh_type_cbox))
		self.meshButton.grid(row=9, column=10, columnspan=2)
		
		# Add buttons to plot model
		self.plot_mri_button = ttk.Button(text='Plot MRI Model', command= lambda: self.plotMRIModel(), state='disabled')
		self.plot_mri_button.grid(row=13, column=0, sticky='W')
		self.plot_mesh_button = ttk.Button(text='Plot MRI Mesh', command= lambda: self.plotMRIMesh(), state='disabled')
		self.plot_mesh_button.grid(row=13, column=1, sticky='W')
		
		# Add mesh options (post-creation adjustments)
		self.conn_mat_cbox = ttk.Combobox(state='disabled', values=['hex', 'pent'], width=10)
		self.conn_mat_cbox.current(0)
		self.conn_mat_cbox.grid(row=5, column=11)
		ttk.Label(text='Select conn matrix:').grid(row=5, column=10, sticky='W')
		self.scar_fe_button = ttk.Button(text='Identify scar nodes', state='disabled', command= lambda: self.scarElem())
		self.scar_fe_button.grid(row=10, column=10, columnspan=2)
		self.dense_fe_button = ttk.Button(text='Assign element displacements', state='disabled', command= lambda: self.denseElem())
		self.dense_fe_button.grid(row=11, column=10, columnspan=2)
		
		# Add buttons to generate FEBio files and open postview
		self.feb_file_button = ttk.Button(text='Generate FEBio File', state='disabled', command= lambda: self.genFebFile())
		self.feb_file_button.grid(row=11, column=3)
		self.postview_open_button = ttk.Button(text='Launch PostView', state='disabled', command= lambda: self.openPostview())
		self.postview_open_button.grid(row=11, column=4)
		
	def openFileBrowser(self, entry_box, multi='False'):
		"""Open a file browser window and assign the file name to the passed entry box.
		"""
		if multi == 'True':
			file_name = filedialog.askopenfilenames(title='Select Files')
		elif multi == 'Dir':
			file_name = filedialog.askdirectory(title='Select Folder')
		elif multi == 'Feb':
			file_name = filedialog.asksaveasfilename(title='Save File', filetypes=(('FEBio files','*.feb'), ('All files', '*')))
			if not (file_name.split('.')[-1] == 'feb'):
				file_name += '.feb'
		else:
			file_name = filedialog.askopenfilename(title='Select File')
		entry_box.delete(0, 'end')
		entry_box.insert(0, file_name)
		return(file_name)
		
	def createMRIModel(self, sa_filename, la_filename, lge_filename, dense_filenames):
		# Create model, import initial cine stack
		if sa_filename.get() == '' or la_filename.get() == '':
			self.progLabel['text'] = 'Please select long-axis and short-axis files first'
			return(False)
		self.progLabel['text'] = 'Generating MRI Model'
		if not(dense_filenames.get() == ''):
			dense_filenames_parsed = dense_filenames.get().split('} {')
			list_replacements = {ord('{') : None, ord('}') : None}
			dense_filenames_replaced = [temp_str.translate(list_replacements) for temp_str in dense_filenames_parsed]
		else:
			dense_filenames_replaced = dense_filenames.get()
		self.mri_model = mrimodel.MRIModel(sa_filename.get(), la_filename.get(), scar_file=lge_filename.get(), dense_file=dense_filenames_replaced)
		self.progLabel['text'] = 'Importing Cine Stack'
		self.mri_model.importCine(timepoint=0)
		# Import LGE, if included, and generate full alignment array
		if self.mri_model.scar:
			self.scar_cbutton.configure(state='normal')
			self.progLabel['text'] = 'Importing Scar Stack'
			self.mri_model.importLGE()
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				for i in range(len(self.mri_model.cine_endo)):
					self.mri_model.alignScarCine(timepoint=i)
		# Import DENSE, if included
		if self.mri_model.dense:
			self.dense_cbutton.configure(state='normal')
			self.progLabel['text'] = 'Importing DENSE Stack'
			self.mri_model.importDense()
			self.mri_model.alignDense(cine_timepoint=0)
			self.dense_timepoint_cbox.configure(values=list(range(len(self.mri_model.dense_aligned_displacement))), state='readonly')
			self.dense_timepoint_cbox.current(0)
		# Update GUI elements
		self.cine_timepoint_cbox.configure(values=list(range(len(self.mri_model.cine_endo))), state='readonly')
		self.cine_timepoint_cbox.current(0)
		self.progLabel['text'] = 'MRI Model Generated Successfully'
		self.meshButton.configure(state='normal', text='Generate MRI Mesh')

	def createMRIMesh(self, num_rings_entry, elem_per_ring_entry, elem_thru_wall_entry, mesh_type_cbox):
		# Pull variables from GUI entry fields
		num_rings = int(num_rings_entry.get())
		elem_per_ring = int(elem_per_ring_entry.get())
		elem_in_wall = int(elem_thru_wall_entry.get())
		time_point = int(self.cine_timepoint_cbox.get())
		
		# Create base mesh
		self.mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
		
		# Fill out mesh components and fields from model
		self.mri_mesh.fitContours(self.mri_model.cine_endo[time_point], self.mri_model.cine_epi[time_point], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts, mesh_type_cbox.get())
		self.mri_mesh.feMeshRender()
		self.mri_mesh.nodeNum(self.mri_mesh.meshCart[0], self.mri_mesh.meshCart[1], self.mri_mesh.meshCart[2])
		self.mri_mesh.getElemConMatrix()
		
		# Update GUI elements
		self.plot_mri_button.configure(state='normal')
		self.plot_mesh_button.configure(state='normal')
		self.feb_file_button.configure(state='normal')
		self.nodes_cbutton.configure(state='normal')
		self.conn_mat_cbox.configure(state='readonly')
		if self.mri_model.scar:
			self.scar_fe_button.configure(state='normal')
		if self.mri_model.dense:
			self.dense_fe_button.configure(state='normal')
	
	def cineTimeChanged(self):
		self.progLabel.configure(text='Updating timepoint in model.')
		try:
			new_timepoint = int(self.cine_timepoint_cbox.get())
		except:
			self.progLabel.configure(text='Timepoint selection failed. Probably a NaN timepoint.')
			return(False)
		self.mri_model.importCine(timepoint = new_timepoint)
		if self.mri_model.dense:
			self.mri_model.alignDense(cine_timepoint = new_timepoint)
		self.progLabel.configure(text='Timepoint successfully updated!')
	
	def plotMRIModel(self):
		time_point = int(self.cine_timepoint_cbox.get())
		mri_axes = self.mri_mesh.segmentRender(self.mri_model.cine_endo[time_point], self.mri_model.cine_epi[time_point], self.mri_model.cine_apex_pt, self.mri_model.cine_basal_pt, self.mri_model.cine_septal_pts)
		if self.scar_plot_bool.get():
			mri_axes = self.mri_mesh.displayScarTrace(self.mri_model.aligned_scar[time_point], ax=mri_axes)
		if self.dense_plot_bool.get():
			mri_axes = self.mri_mesh.displayDensePts(self.mri_model.dense_aligned_pts, self.mri_model.dense_slices, self.mri_model.dense_aligned_displacement, dense_plot_quiver=1, timepoint=int(self.dense_timepoint_cbox.get()), ax=mri_axes)
	
	def plotMRIMesh(self):
		mesh_axes = self.mri_mesh.surfaceRender(self.mri_mesh.endo_node_matrix)
		mesh_axes = self.mri_mesh.surfaceRender(self.mri_mesh.epi_node_matrix, mesh_axes)
		if self.nodes_plot_bool.get():
			mesh_axes = self.mri_mesh.nodeRender(self.mri_mesh.nodes, mesh_axes)
		if self.scar_plot_bool.get() and self.mri_mesh.nodes_in_scar.size:
			mesh_axes = self.mri_mesh.nodeRender(self.mri_mesh.nodes[self.mri_mesh.nodes_in_scar, :], ax=mesh_axes)
		elif self.scar_plot_bool.get() and not self.mri_mesh.nodes_in_scar.size:
			messagebox.showinfo('Warning', 'Identify scar nodes before plotting to view.')
	
	def scarElem(self):
		time_point = int(self.cine_timepoint_cbox.get())
		self.mri_mesh.assignScarElems(self.mri_model.aligned_scar[time_point], conn_mat = self.conn_mat_cbox.get())
	
	def denseElem(self):
		time_point = int(self.cine_timepoint_cbox.get())
		self.mri_mesh.assignDenseElems(self.mri_model.dense_aligned_pts, self.mri_model.dense_slices, self.mri_model.dense_aligned_displacement)
	
	def genFebFile(self):
		feb_file_name = self.postview_file_entry.get()
		self.mri_mesh.generateFEFile(feb_file_name, self.conn_mat_cbox.get())
		self.postview_open_button.configure(state='normal')
		
	def openPostview(self):
		feb_file_name = self.postview_file_entry.get()
		try:
			open(feb_file_name)
		except:
			messagebox.showinfo('File Warning', 'FEBio File not found. Check file name and try again.')
			return(False)
		self.mri_mesh.displayMeshPostview(feb_file_name)
	
	def intValidate(self, new_value):
		if new_value == '':
			return(True)
		try:
			int(new_value)
			return(True)
		except:
			return(False)
	
root = tk.Tk()
gui = modelGUI(master=root)
gui.master.title('Cardiac Modeling Toolbox')
gui.mainloop()