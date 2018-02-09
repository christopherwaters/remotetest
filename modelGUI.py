# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 12:48:42 2017

@author: cdw2be
"""

import tkinter as tk
from tkinter import filedialog
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

	def createWidgets(self):
		# Create Entry object
		sa_file_entry = tk.Entry(width=100)
		la_file_entry = tk.Entry(width=100)
		lge_file_entry = tk.Entry(width=100)
		dense_file_entry = tk.Entry(width=100)
		confocal_dir_entry = tk.Entry(width=100)
		
		# Create labels for entries
		tk.Label(text='Short-Axis File:').grid(row=0)
		tk.Label(text='Long-Axis File:').grid(row=1)
		tk.Label(text='LGE File:').grid(row=2)
		tk.Label(text='DENSE Files:').grid(row=3)
		tk.Label(text='Confocal Directory:').grid(row=4)
		
		# Place entry object
		sa_file_entry.grid(row=0, column=1)
		la_file_entry.grid(row=1, column=1)
		lge_file_entry.grid(row=2, column=1)
		dense_file_entry.grid(row=3, column=1)
		confocal_dir_entry.grid(row=4, column=1)
		
		# Add browse buttons for file exploration, to pass to entry boxes
		tk.Button(text='Browse', command= lambda: self.openFileBrowser(sa_file_entry)).grid(row=0, column=2)
		tk.Button(text='Browse', command= lambda: self.openFileBrowser(la_file_entry)).grid(row=1, column=2)
		tk.Button(text='Browse', command= lambda: self.openFileBrowser(lge_file_entry)).grid(row=2, column=2)
		tk.Button(text='Browse', command= lambda: self.openFileBrowser(dense_file_entry, multi='True')).grid(row=3, column=2)
		tk.Button(text='Browse', command= lambda: self.openFileBrowser(confocal_dir_entry, multi='Dir')).grid(row=4, column=2)
		
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

root = tk.Tk()
gui = modelGUI(master=root)
gui.mainloop()
root.destroy()