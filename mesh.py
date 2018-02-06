# -*- coding: utf-8 -*-
"""
Creates the class and functions for implementing mesh formation and visualization.
Based on MRI Processing pipeline by Thien-Khoi Phung.

Created on Wed Oct 4 2:54:15 2017

@author: cdw2be
"""

# Imports
import numpy as np
import scipy as sp
import math
import warnings
import matplotlib.pyplot as mplt
import matplotlib.path as path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import subprocess

class Mesh():

	"""Defines meshes based on endocardial and epicardial contours from MRIModel objects.
	
	This class can fit contours to a prolate mesh, and visualize the meshes and contours
		for simplified interpretation and inspection. Also can be used to generate the 
		element connectivity matrices to be passed in to FEBio.
		
	Attributes:
		num_rings (int): Number of elements in the longitudinal axis of the mesh.
		elem_per_ring (int): Number of elements circumferentially around the wall.
		elem_in_wall (int): Number of elements radially through the wall.
		endo_node_matrix (array): The nodal mesh matrix from bicubic fitting for endocardial contour.
		epi_node_matrix (array): The nodal mesh matrix from bicubic fitting for epicardial contour.
		focus (float): The focus point to be used for the prolate spheroid calculations.
		origin (array): The shift values for x, y, z to center at 0.
		transform (array): The second element of the shifting process to center cartesian at the origin.
		hex (array): The element connectivity matrix for hexahedron-shaped elements
		pent (array): The element connectivity matrix for pentahedron-shaped elements
		
	TODO:
		Align multiple meshes (define as a class function returning a transformation matrix).
	"""

	def __init__(self, num_rings=14, elem_per_ring=24, elem_in_wall=5):
		"""Initialization method for new Mesh class instances.
		
		args:
			num_rings (int): The number of rings vertically along the heart.
			elem_per_ring (int): The number of elements circumferentially per ring.
			elem_in_wall (int): The number of layers through the wall.
		"""
		# Set class variables based on passed values and initialize object fields
		self.num_rings = num_rings
		self.elem_per_ring = elem_per_ring
		self.elem_in_wall = elem_in_wall
		self.endo_node_matrix = []
		self.epi_node_matrix = []
		self.focus = []
		self.transform = []
		self.origin = []
		self.nodes = []
		self.hex = []
		self.pent = []
		self.scar_elems = []
		self.dense_x_displacements = []
		self.dense_y_displacements = []
	
	def fitContours(self, all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts, mesh_type):
		"""Function to Perform the Contour Fitting from the Passed Endo and Epi Contour data.
		
		args:
			all_data_endo (array): Full stack of endocardial contour data contained in an MRIModel object.
			all_data_epi (array): Full stack of epicardial contour data contained in an MRIModel object.
			apex_pt (array): Apical point by long-axis of an MRIModel object.
			basal_pt (array): Basal point by long-axis of an MRIModel object.
			septal_pts (array): Septal points by short-axis of an MRIModel object.
			mesh_type (string): String indicating the mesh type to be used in contour fitting.
		returns:
			endo_node_matrix (array): An array of the endocardial nodes defining the mesh.
			epi_node_matrix (array): An array of the epicardial nodes defining the mesh.
		"""
		
		# Set up variables
		data_endo, data_epi, self.focus, self.transform, self.origin = self._prepData(all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts)
		
		# Fit using a bicubic interpolation.
		self.endo_node_matrix, _ = self._fitBicubicData(data_endo, self.focus, mesh_density=mesh_type)
		self.epi_node_matrix, _ = self._fitBicubicData(data_epi, self.focus, mesh_density=mesh_type)
		
		return([self.endo_node_matrix, self.epi_node_matrix])
		
	def segmentRender(self, all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts, landmarks=True, ax=None, scar=None):
		"""Display the segmentation contours and user-indicated points
		
		args:
			all_data_endo (array): All endocardial contour data including slice values.
			all_data_epi (array): All epicardial contour data including slice values.
			apex_pt (array): Apical point
			basal_pt (array): Basal point
			septal_pts (array): Septal points (including midpoint)
			landmarks (bool): Whether to plot the pinpoints.
			ax (mplotlib3d axes): Axes on which to plot (default None)
		returns:
			ax (mplotlib3d axes): The axes of the contour
		"""
		if not ax:
			fig = mplt.figure()
			ax = fig.add_subplot(111, projection='3d')
		data_endo, data_epi, _, _, _ = self._prepData(all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts)
		# Subtract origin and transform data
		apex_transform = np.dot((apex_pt - self.origin), np.transpose(self.transform))
		basal_transform = np.dot((basal_pt - self.origin), np.transpose(self.transform))
		septal_transform1 = np.dot((septal_pts[0, :] - self.origin), np.transpose(self.transform))
		septal_transform2 = np.dot((septal_pts[1, :] - self.origin), np.transpose(self.transform))
		septal_transform3 = np.dot((septal_pts[2, :] - self.origin), np.transpose(self.transform))
		# Set up bins as the unique data in all_data_endo third column (the slices)
		bins = np.unique(all_data_endo[:, 2])
		for jz in range(bins.size):
			# Get the indices that match the current bin and append then append the first value
			tracing = np.where(all_data_endo[:, 2] == bins[jz])[0]
			tracing = np.append(tracing, tracing[0])
			# Pull x, y, z from endo and epi and plot
			x = data_endo[tracing, 2]
			y = data_endo[tracing, 1]
			z = data_endo[tracing, 0]
			ax.plot(x, y, -z, 'y-')
			# Epi plotting
			x = data_epi[tracing, 2]
			y = data_epi[tracing, 1]
			z = data_epi[tracing, 0]
			ax.plot(x, y, -z, 'c-')
		if landmarks:
			# Plot the apex, basal, and septal points
			ab = np.array([apex_transform, basal_transform])
			si = np.array([septal_transform2, septal_transform3])
			
			ax.plot(ab[:, 2], ab[:, 1], -ab[:, 0], 'k-.')
			ax.scatter(si[:, 2], si[:, 1], -si[:, 0], 'bo', s=50)
			ax.scatter(septal_transform1[2], septal_transform1[1], -septal_transform1[0], 'ro', s=50)
		
		return(ax)
		
	def surfaceRender(self, nodal_mesh, ax=None):
		"""Plot surface mesh on optionally-passed axes
		
		args:
			nodal_mesh (array): Mesh to be plotted
			ax (mplot3d axes object): Axes on which to plot the mesh
		returns:
			ax (mplot3d axes object): Axes containing the surface plot contained in a figure
		"""
		# If no axes were passed, generate new set of axes
		if not ax:
			fig = mplt.figure()
			ax = fig.add_subplot(111, projection='3d')
	
		# Sort the mesh by first 3 columns
		nodal_mesh = nodal_mesh[nodal_mesh[:, 0].argsort()]
		nodal_mesh = nodal_mesh[nodal_mesh[:, 1].argsort(kind='mergesort')]
		nodal_mesh = nodal_mesh[nodal_mesh[:, 2].argsort(kind='mergesort')]
		
		# Set up number of divisions and calculate e for each division (as a ratio)
		num_div = 20
		e = [i/num_div for i in range(num_div + 1)]
		# Convert angular values from degrees to radians
		rads = math.pi/180
		nodal_mesh[:, 1:3] *= rads
		# Store the shapes and sizes of the mesh values
		m = nodal_mesh.shape[0]
		size_nodal_nu = np.where(nodal_mesh[:, 2] == 0)[0].size
		size_nodal_phi = m/size_nodal_nu
		# Get the mu and theta values from the mesh
		nodal_nu = nodal_mesh[:size_nodal_nu, 1]
		nodal_phi = nodal_mesh[::size_nodal_nu, 2]
		# Convert apex node from prolate to cartesian, then plot with scatter
		if min(nodal_nu) == 0:
			x, y, z = self._prolateToCart(nodal_mesh[0, 0], nodal_mesh[0, 1], nodal_mesh[0, 2], self.focus)
			ax.scatter(z, y, -x)
			start_nu = 1
		else:
			start_nu = 0
		# Plot circumferential element boundaries
		for i in range(start_nu, size_nodal_nu):
			for j in range(int(size_nodal_phi)):
				# Define nodal values for interpolation
				if j == size_nodal_phi-1:
					ind0 = i
					p0 = 2*math.pi
				else:
					ind0 = (j+1)*size_nodal_nu + i
					p0 = nodal_phi[j+1]
				ind1 = (j)*size_nodal_nu + i
				p1 = nodal_phi[j]
				# Get mu and dM/dm1
				m0 = nodal_mesh[ind0, 0]
				dm0 = nodal_mesh[ind0, 3]
				m1 = nodal_mesh[ind1, 0]
				dm1 = nodal_mesh[ind1, 3]
				# Convert to cartesian
				n0x, n0y, n0z = self._prolateToCart(nodal_mesh[ind0, 0], nodal_mesh[ind0, 1], nodal_mesh[ind0, 2], self.focus)
				# Plot the node
				ax.scatter(n0z, n0y, -n0x)
				# Plot the arc segments
				for k in range(2, len(e)):
					# Determine starting point to use
					if k == 2:
						pt_x, pt_y, pt_z = n0x, n0y, n0z
					else:
						pt_x, pt_y, pt_z = x_here, y_here, z_here
					# Get lambda
					hm0 = 1 - 3*(e[k]**2) + 2*(e[k]**3)
					hdm0 = e[k]*(e[k] - 1)**2
					hm1 = (e[k]**2)*(3 - 2*e[k])
					hdm1 = (e[k]**2)*(e[k] - 1)
					m = hm0 * m0 + hdm0 * dm0 + hm1 * m1 + hdm1 * dm1
					# Get theta
					p_here = p0 - e[k]*(p0 - p1)
					# Convert to cartesian
					x_here, y_here, z_here = self._prolateToCart(m, nodal_nu[i], p_here, self.focus)
					# Create vectors
					x = np.append(pt_x, x_here)
					y = np.append(pt_y, y_here)
					z = np.append(pt_z, z_here)
					# Plot segments
					ax.plot(z, y, -x, 'k-.')
		# Plot longitudinal element boundaries
		for i in range(int(size_nodal_phi)):
			for j in range(size_nodal_nu-1):
				# Define nodal values needeed for interpolation
				ind0 = i*size_nodal_nu + j
				ind1 = ind0 + 1
				n0 = nodal_nu[j]
				n1 = nodal_nu[j+1]
				# Get lambda and dL/de2
				m0 = nodal_mesh[ind0, 0]
				dm0 = nodal_mesh[ind0, 4]
				m1 = nodal_mesh[ind1, 0]
				dm1 = nodal_mesh[ind1, 4]
				# Convert nodal points to cartesian
				n0x, n0y, n0z = self._prolateToCart(nodal_mesh[ind0, 0], nodal_mesh[ind0, 1], nodal_mesh[ind0, 2], self.focus)
				# Plot arc
				for k in range(2, len(e)):
					# Determine point to use
					if k == 2:
						pt_x, pt_y, pt_z = n0x, n0y, n0z
					else:
						pt_x, pt_y, pt_z = x_here, y_here, z_here
					# Get lambda
					hm0 = 1 - 3*(e[k]**2) + 2*(e[k]**3)
					hdm0 = e[k]*(e[k] - 1)**2
					hm1 = (e[k]**2)*(3 - 2*e[k])
					hdm1 = (e[k]**2)*(e[k] - 1)
					m = hm0 * m0 + hdm0 * dm0 + hm1 * m1 + hdm1 * dm1
					# Get nu
					n_here = n0 + e[k]*(n1-n0)
					# Convert to cartesian
					x_here, y_here, z_here = self._prolateToCart(m, n_here, nodal_phi[i], self.focus)
					# Append the vectors for plotting
					x = np.append(pt_x, x_here)
					y = np.append(pt_y, y_here)
					z = np.append(pt_z, z_here)
					# Plot the segment
					ax.plot(z, y, -x, 'k-.')
					
		return(ax)
	
	def feMeshRender(self):
		"""Interpolate element density mesh based on endo and epi surfaces.
		
		Does not call any values, operates on the fields of the object.
		returns:
			meshCart (array): The interpolated mesh in cartesian coordinates.
			meshProl (array): The interpolated mesh in prolate coordinates.
		"""
		# Set up initial variables
		endo_fit = self.endo_node_matrix.copy()
		epi_fit = self.epi_node_matrix.copy()
		rads = math.pi/180
		
		# Sort Rows and Do Unit Conversion to Radians
		endo_fit = endo_fit[endo_fit[:, 0].argsort()]
		endo_fit = endo_fit[endo_fit[:, 1].argsort(kind='mergesort')]
		endo_fit = endo_fit[endo_fit[:, 2].argsort(kind='mergesort')]
		
		epi_fit = epi_fit[epi_fit[:, 0].argsort()]
		epi_fit = epi_fit[epi_fit[:, 1].argsort(kind='mergesort')]
		epi_fit = epi_fit[epi_fit[:, 2].argsort(kind='mergesort')]
		
		endo_fit[:, 1:3] *= rads
		epi_fit[:, 1:3] *= rads
		
		# Determine the size of the bicubic mesh
		m, n = endo_fit.shape
		mu_num = np.where(endo_fit[:, 2] == 0)[0].size
		th_num = int(m / mu_num)
		
		# Pad Nodal Mesh
		endo_fit = np.append(endo_fit, endo_fit[:mu_num, :], axis=0)
		endo_fit[m:, 2] = 2*math.pi
		epi_fit = np.append(epi_fit, epi_fit[:mu_num, :], axis=0)
		epi_fit[m:, 2] = 2*math.pi
		th_num += 1
		
		# Reshape mesh for interp function
		endo_data = np.reshape(endo_fit, (mu_num, th_num, n), order='F')
		epi_data = np.reshape(epi_fit, (mu_num, th_num, n), order='F')
		
		# Set up surface mesh for interpolation
		mu_vec = endo_fit[:mu_num, 1]
		
		# Pull unique theta values:
		sort_by_mu = endo_fit[endo_fit[:, 2].argsort()]
		sort_by_mu = sort_by_mu[sort_by_mu[:, 1].argsort(kind='mergesort')]
		th_vec = sort_by_mu[:th_num, 2]
		
		# Mesh Grid of Unique Mu and Theta for OG Nodes
		mu_grid_og, th_grid_og = np.meshgrid(mu_vec, th_vec)
		
		# New mesh to interpolate based on mesh density
		mu_vec_fe = np.linspace(0, mu_grid_og.flatten()[-1], self.num_rings+2)
		th_vec_fe = np.linspace(0, th_grid_og.flatten()[-1], self.elem_per_ring+1)
		th_vec_fe = th_vec_fe[1:]
		mu_grid_fe, th_grid_fe = np.meshgrid(mu_vec_fe, th_vec_fe)
		
		# Interpolate Along New Surface Grid
		endo_interp_surf = self._biCubicInterp(mu_grid_fe, th_grid_fe, endo_data)
		epi_interp_surf = self._biCubicInterp(mu_grid_fe, th_grid_fe, epi_data)
		
		# Convert Endo and Epi Interp to Cartesian
		endo_mu = endo_interp_surf[:, :, 0]
		endo_nu = endo_interp_surf[:, :, 1]
		endo_phi = endo_interp_surf[:, :, 2]
		
		endo_x, endo_y, endo_z = self._prolateToCart(endo_mu, endo_nu, endo_phi, self.focus)
		
		epi_mu = epi_interp_surf[:, :, 0]
		epi_nu = epi_interp_surf[:, :, 1]
		epi_phi = epi_interp_surf[:, :, 2]
		epi_x, epi_y, epi_z = self._prolateToCart(epi_mu, epi_nu, epi_phi, self.focus)
		
		# Interpolate nodes between endo and epi
		step_x = np.reshape((epi_x - endo_x)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		step_y = np.reshape((epi_y - endo_y)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		step_z = np.reshape((epi_z - endo_z)/self.elem_in_wall, [epi_x.shape[0], epi_x.shape[1], 1])
		
		endo_x = np.reshape(endo_x, [endo_x.shape[0], endo_x.shape[1], 1])
		endo_y = np.reshape(endo_y, [endo_y.shape[0], endo_y.shape[1], 1])
		endo_z = np.reshape(endo_z, [endo_z.shape[0], endo_z.shape[1], 1])
		
		epi_x = np.reshape(epi_x, [epi_x.shape[0], epi_x.shape[1], 1])
		epi_y = np.reshape(epi_y, [epi_y.shape[0], epi_y.shape[1], 1])
		epi_z = np.reshape(epi_z, [epi_z.shape[0], epi_z.shape[1], 1])
		
		# Create stepnumber matrix
		steps = np.zeros([endo_x.shape[0], endo_x.shape[1], self.elem_in_wall+1])
		for jz in range(self.elem_in_wall+1):
			steps[:, :, jz] = jz
		
		# Reformat x, y, z arrays
		x = np.tile(endo_x, (1, 1, self.elem_in_wall + 1)) + np.tile(step_x, (1, 1, self.elem_in_wall+1))*steps
		y = np.tile(endo_y, (1, 1, self.elem_in_wall + 1)) + np.tile(step_y, (1, 1, self.elem_in_wall+1))*steps
		z = np.tile(endo_z, (1, 1, self.elem_in_wall + 1)) + np.tile(step_z, (1, 1, self.elem_in_wall+1))*steps
		
		# Convert cartesian back to prolate
		m, n, p = self._cartToProlate(x, y, z, self.focus)
		
		# Rotate each matrix by 90 degrees
		x = np.rot90(x)
		y = np.rot90(y)
		z = np.rot90(z)
		m = np.rot90(m)
		n = np.rot90(n)
		p = np.rot90(p)
		
		# Define the return matrices
		meshCart = [x, y, z]
		meshProl = [m, n, p]
		
		return([meshCart, meshProl])
	
	def nodeNum(self, x, y, z):
		"""Generate node corners as x, y, z points
		
		Numbering order:
			Theta, to 2*pi
			Endocardial to epicardial
			Basal to apical
		returns:
			nodes (array): x, y, z coordinates for each node, ordered by row
				Formatted as (Basal to Apex, y, x)
		"""
		# Get the shape of the x array to determine number of elements (circ, long, rad)
		c, l, r = x.shape
		
		nodes = None
		for lyr in range(r):
			# Pull coordinates from each layer and flip so the first node is the top row
			# Reshape to a column array going from 0 to 360 degrees and basal to apical
			x_lyr = np.reshape(np.fliplr(x[:, :, lyr]), [c*l, 1], order='F')
			y_lyr = np.reshape(np.fliplr(y[:, :, lyr]), [c*l, 1], order='F')
			z_lyr = np.reshape(np.fliplr(z[:, :, lyr]), [c*l, 1], order='F')
			# Store in nodes and remove all apical nodes except one
			if nodes is None:
				nodes = np.column_stack([x_lyr[:-(c-1)], y_lyr[:-(c-1)], z_lyr[:-(c-1)]])
			else:
				nodes = np.append(nodes, np.column_stack([x_lyr[:-(c-1)], y_lyr[:-(c-1)], z_lyr[:-(c-1)]]), axis=0)
		self.nodes = nodes
		return(nodes)
		
	def getElemConMatrix(self):
		"""Get the connectivity matrix for the nodes to define the elements.
		
		Operates on the instance variables defined at creations. The matrix defines
		an element per row, nodes are defined by index in the nodes array.
		returns:
			hex (array): Array for a 6-sided element per node.
			pent (array): Array for a 5-sided element per node.
		"""
		# Calculate the number of nodes and element in each peel layer
		nodes_per_layer = (self.num_rings + 1)*self.elem_per_ring + 1
		elem_per_layer = self.num_rings*self.elem_per_ring
		
		# Lay out the two arrays
		hex = np.zeros([elem_per_layer*self.elem_in_wall, 8])
		pent = np.zeros([self.elem_per_ring*self.elem_in_wall, 6])
		for l in range(self.elem_in_wall):
			# Calculate number of elements
			enum = range(self.elem_per_ring*l + 1, self.elem_per_ring*(l+1) + 1)
			for n in range(elem_per_layer):
				# Calculate which nodes define each element in hex
				nn = range(elem_per_layer*l + 1, elem_per_layer*(l+1) + 1)
				sn = nodes_per_layer*l + n + 1
				hex[nn[n]-1, :] = [sn, sn+1, sn+nodes_per_layer+1, sn+nodes_per_layer, sn+self.elem_per_ring, sn+self.elem_per_ring+1, sn+nodes_per_layer+self.elem_per_ring+1, sn+nodes_per_layer+self.elem_per_ring] if math.fmod((n+1), self.elem_per_ring) != 0 else [sn, sn+1-self.elem_per_ring, sn+nodes_per_layer+1-self.elem_per_ring, sn+nodes_per_layer, sn+self.elem_per_ring, sn+1, sn+nodes_per_layer+1, sn+nodes_per_layer+self.elem_per_ring]
			for n in range(self.elem_per_ring):
				# Calculate which nodes define each element in pent
				pent[enum[n]-1, :] = [nodes_per_layer*(l+2), nodes_per_layer*(l+2)-self.elem_per_ring+n, nodes_per_layer*(l+2)-self.elem_per_ring+n+1, nodes_per_layer*(l+1), nodes_per_layer*(l+1)-self.elem_per_ring+n, nodes_per_layer*(l+1)-self.elem_per_ring+n+1] if math.fmod((n+1), self.elem_per_ring) != 0 else [nodes_per_layer*(l+2), nodes_per_layer*(l+2)-self.elem_per_ring+n, nodes_per_layer*(l+2)-self.elem_per_ring, nodes_per_layer*(l+1), nodes_per_layer*(l+1)-self.elem_per_ring+n, nodes_per_layer*(l+1)-self.elem_per_ring]
		
		# Subtract 1 from each element because of zero-indexing in python
		hex -= 1
		pent -= 1
		
		hex = hex.astype(int)
		pent = pent.astype(int)
		
		# Assign to instance variables
		self.hex = hex
		self.pent = pent
		
		return([hex, pent])
		
	def renderPatchMesh(self, nodes, elem_con_matrix, data=False, color=False, title='Mesh Rendering'):
		"""Should visualize the finite elements. Currently not implemented due to limitations."""
		if data:
			data_col = data / max(data)
		else:
			data_col = np.zeros(np.append(elem_con_matrix.shape, 1))
		
		fig = mplt.figure()
		fig.addsubplot(111, projection='3d')
		
		for jz in range(elem_con_matrix.shape[0]):
			face = np.array([[1, 2, 3, 4], [1, 5, 6, 2], [2, 6, 7, 3], [6, 5, 8, 7], [3, 7, 8, 4], [1, 4, 8, 5]])
	
	def labelScarElems(self, scar_contour, conn_mat = 'hex'):
		"""Get a matrix indicating which elements are contained within the scar contour
		
		args:
			scar_contour (array): The x, y, z array for the scar contour, from the MRIModel object
			conn_mat (string): Indicates which connectivity matrix to use (hex or pent)
		returns:
			scar_node_inds (array): The indices of the element connectivity matrix that are scar
			center_matrix (array): The center of the elements defined by conn_mat that are in the scar
		"""
		# Convert the scar contour to prolate coordinates
		scar_contour = [scar_contour_i for scar_contour_i in scar_contour if scar_contour_i.size > 0]
		scar_vstack = np.vstack(tuple(scar_contour))
		scar_vstack = np.dot((scar_vstack - np.array([self.origin for i in range(scar_vstack.shape[0])])), np.transpose(self.transform))
		
		# Get connectivity matrix and create scar matrix based on size
		elem_con = self.pent if conn_mat == 'pent' else self.hex
		scar_mat = [False] * elem_con.shape[0]
		
		# Create matrix of node centers (center in prolate, convert back to cartesian)
		center_matrix = []
		for i in range(elem_con.shape[0]):
			node_inds = elem_con[i, :]
			cur_elem_nodes = self.nodes[node_inds.tolist()]
			cur_elem_center = np.mean(cur_elem_nodes, axis=0)
			center_matrix.append(cur_elem_center.tolist())
		center_matrix = np.array(center_matrix)
		center_matrix_orig = center_matrix

		# Get scar Z-range
		scar_z_min = np.min(-scar_vstack[:, 0])
		scar_z_max = np.max(-scar_vstack[:, 0])
		
		# Remove nodes outside of the scar z-range
		elems_in_z = np.argwhere((-center_matrix[:, 0] <= scar_z_max) & (-center_matrix[:, 0] >= scar_z_min))
		center_matrix = center_matrix[elems_in_z, :]
		num_pts_slice = scar_contour[0].shape[0]
		center_matrix = np.squeeze(center_matrix)
		elem_inds = []
		
		# Loop through elements remaining and remove if outside of the scar region
		for elem_center in center_matrix.tolist():
			# Define which known slices are established above and below
			ind_above = None
			ind_below = None
			for i in range(len(scar_contour)):
				scar_slice_cur = scar_vstack[i*num_pts_slice:(i+1)*num_pts_slice, :]
				scar_slice_cur_mean = np.mean(scar_slice_cur, axis=0)
				# Define the indices above and below, set equal if you are inside the slice
				if round(-scar_slice_cur_mean[0], 1) == round(-elem_center[0], 1):
					ind_above = i
					ind_below = i
					break
				elif round(-scar_slice_cur_mean[0], 1) > round(-elem_center[0], 1):
					ind_above = i
				else:
					ind_below = i
					break
			# Calculate scar contours based on z-axis position
			if not ind_below and ind_above == len(scar_contour)-1:
				ind_below = ind_above
			if ind_below == ind_above:
				# If currently aligned with a scar layer, use that layer only
				scar_layer = scar_vstack[ind_above*num_pts_slice:(ind_above+1)*num_pts_slice, :]
				scar_pts = np.column_stack((scar_layer[:, 2], scar_layer[:, 1]))
			else:
				scar_above = scar_vstack[ind_above*num_pts_slice:(ind_above+1)*num_pts_slice, :]
				scar_below = scar_vstack[ind_below*num_pts_slice:(ind_below+1)*num_pts_slice, :]
				# Interpolate contours at new z value
				above_z = np.mean(-scar_above[:, 0])
				below_z = np.mean(-scar_below[:, 0])
				# The interpolation bounds are the two layers above and below
				interp_arr = [below_z, above_z]
				# Develop interpolation functions in cartesian coordinates
				x_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_below[:, 2], scar_above[:, 2])))
				y_interp_func = sp.interpolate.interp1d(interp_arr, np.column_stack((scar_below[:, 1], scar_above[:, 1])))
				# Calculate new x and y values at z
				x_new_z = x_interp_func(-elem_center[0])
				y_new_z = y_interp_func(-elem_center[0])
				# Combine new points
				scar_pts = np.column_stack((x_new_z, y_new_z))	
			# Convert the values to polar to compare radial and circumferential extent
			new_z_theta, new_z_rho = self._cartToPol(scar_pts[:, 0], scar_pts[:, 1])
			elem_center_theta, elem_center_rho = self._cartToPol(elem_center[2], elem_center[1])
			theta_min, theta_max, direction = self.__getAngleRange(new_z_theta)
			# If the scar is min->max in normal order, direction is True
			# 	Delete elements outside of the circumferential extent, then move to the next element
			if direction:
				if elem_center_theta > theta_max or elem_center_theta < theta_min:
					center_matrix = np.delete(center_matrix, np.where(np.bitwise_and(center_matrix[:, 0] == elem_center[0], center_matrix[:, 1] == elem_center[1], center_matrix[:, 2] == elem_center[2]))[0], axis=0)
					continue
			else:
				if theta_max < elem_center_theta < theta_min:
					center_matrix = np.delete(center_matrix, np.where(np.bitwise_and(center_matrix[:, 0] == elem_center[0], center_matrix[:, 1] == elem_center[1], center_matrix[:, 2] == elem_center[2]))[0], axis=0)
					continue
			# Get the indices closest on the left and right for both inner and outer traces
			theta_inner = new_z_theta[:int(num_pts_slice/2)]
			theta_outer = new_z_theta[int(num_pts_slice/2):]
			theta_inner_ind = np.where(np.diff(np.sign(theta_inner - elem_center_theta)))[0][0]
			theta_inner_inds = [theta_inner_ind, theta_inner_ind+1]
			theta_outer_ind = np.where(np.diff(np.sign(theta_outer - elem_center_theta)))[0][0]+int(num_pts_slice/2)
			theta_outer_inds = [theta_outer_ind, theta_outer_ind+1]
			# Calculate rho of the inner and outer traces, giving the range that is "in-scar"
			rho_inner = np.mean(new_z_rho[theta_inner_inds])
			rho_outer = np.mean(new_z_rho[theta_outer_inds])
			# Delete elements that are outside of the scar range, then continue to the next element
			if not (rho_inner <= elem_center_rho <= rho_outer):
				center_matrix = np.delete(center_matrix, np.where(np.bitwise_and(center_matrix[:, 0] == elem_center[0], center_matrix[:, 1] == elem_center[1], center_matrix[:, 2] == elem_center[2]))[0], axis=0)
				continue
			# All remaining elements are in the scar trace, so store the indices based on element connectivity matrix index
			elem_inds = np.append(elem_inds, np.where(np.bitwise_and(center_matrix_orig[:, 0] == elem_center[0], center_matrix_orig[:, 1] == elem_center[1], center_matrix_orig[:, 2] == elem_center[2]))[0])
		
		# Convert the indices to integers so that they may be used to automatically index
		scar_node_inds = elem_inds.astype(int)
		return([scar_node_inds, center_matrix])
	
	def assignScarElems(self, scar_contour, conn_mat='hex'):
		"""Get a matrix indicating which elements are contained within the scar contour
		
		args:
			scar_contour (array): The x, y, z array for the scar contour, from the MRIModel object
			conn_mat (string): Indicates which connectivity matrix to use (hex or pent)
		returns:
			scar_node_inds (array): The indices of the element connectivity matrix that are scar
			center_matrix (array): The center of the elements defined by conn_mat that are in the scar
		"""
		# Set number of nodes needed for element to be in scar
		num_nodes = 3
		
		# Convert the scar contour to prolate coordinates
		scar_contour = [scar_contour_i for scar_contour_i in scar_contour if scar_contour_i.size > 0]
		num_slices = len(scar_contour)
		scar_edge_spacing = int(scar_contour[0].shape[0]/2)
		scar_vstack = np.vstack(tuple(scar_contour))
		scar_vstack = np.dot((scar_vstack - np.array([self.origin for i in range(scar_vstack.shape[0])])), np.transpose(self.transform))
		
		nodes_in_scar = self._assignRegionNodes(scar_vstack, scar_edge_spacing, num_slices)
		
		# Get connectivity matrix
		elem_con = self.pent if conn_mat == 'pent' else self.hex
		
		# Determine number of nodes in scar region for each element
		elem_scar_mask = np.reshape(np.in1d(elem_con, nodes_in_scar), elem_con.shape)
		elem_node_ct = np.sum(elem_scar_mask, axis=1)
		elems_in_scar = np.where(elem_node_ct >= num_nodes)[0]
		
		return([nodes_in_scar, elems_in_scar])
	
	def assignDenseElems(self, dense_pts, dense_slices, dense_displacement_all, conn_mat='hex'):
		"""Assign DENSE dx and dy values to elements within the field.
		
		Assignment of DENSE is done based on DENSE value at center of element. Interpolation is
		performed via tricubic interpolation in prolate spheroid coordinates.
		"""
		# Establish overhead lists:
		dense_pts_z = [[dense_slices[i] - self.origin[2]]*dense_pts[i].shape[0] for i in range(len(dense_slices))]
		
		# Convert nodes to prolate, calculate centers, and return to cartesian
		nodes_prol_mu, nodes_prol_nu, nodes_prol_phi = self._cartToProlate(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
		
		elem_con = self.pent if conn_mat == 'pent' else self.hex
		
		elem_cart_centers = [False]*elem_con.shape[0]
		elem_prol_centers = [False]*elem_con.shape[0]
		
		for i in range(elem_con.shape[0]):
			nodes_in_elem = elem_con[i, :]
			elem_prol = np.column_stack((nodes_prol_mu[nodes_in_elem], nodes_prol_nu[nodes_in_elem], nodes_prol_phi[nodes_in_elem]))
			elem_center_prol = np.mean(elem_prol, axis=0)
			elem_prol_centers[i] = [elem_center_prol[0], elem_center_prol[1], elem_center_prol[2]]
			elem_center_cart = self._prolateToCart(elem_center_prol[0], elem_center_prol[1], elem_center_prol[2])
			elem_cart_centers[i] = elem_center_cart
		
		dense_pts_prol = np.array([])
		dense_pts_cart = np.array([])
		
		for slice_num in range(len(dense_slices)):
			cur_slice_pts = np.column_stack((dense_pts[slice_num], dense_pts_z[slice_num]))
			cur_slice_pts = np.dot(cur_slice_pts, np.transpose(self.transform))
			csd_mu, csd_nu, csd_phi = self._cartToProlate(cur_slice_pts[:, 0], cur_slice_pts[:, 1], cur_slice_pts[:, 2])
			if dense_pts_prol.size:
				dense_pts_cart = np.append(dense_pts_cart, cur_slice_pts, axis=0)
				dense_pts_prol = np.append(dense_pts_prol, np.column_stack((csd_mu, csd_nu, csd_phi)), axis=0)
			else:
				dense_pts_prol = np.column_stack((csd_mu, csd_nu, csd_phi))
				dense_pts_cart = cur_slice_pts

		# Find nodes contained within the DENSE measurement volume
		elems_in_dense = np.where((np.min(dense_pts_cart[:, 0]) <= np.array(elem_cart_centers)[:, 0]) & (np.array(elem_cart_centers)[:, 0] <= np.max(dense_pts_cart[:, 0])))[0]
		elem_pts_interp = np.array(elem_prol_centers)[elems_in_dense, :]
		
		# Iterate through each timepoint to calculate an interpolation of DENSE values
		elem_displacements_x = [[np.nan for i in range(len(dense_displacement_all))] for j in range(elem_con.shape[0])]
		elem_displacements_y = [[np.nan for i in range(len(dense_displacement_all))] for j in range(elem_con.shape[0])]
		for time_point in range(len(dense_displacement_all)):
			dense_disp_timepoint = dense_displacement_all[time_point]
			dense_cur_disp = np.array([])
			for slice_num in range(len(dense_disp_timepoint)):
				if dense_cur_disp.size:
					dense_cur_disp = np.append(dense_cur_disp, dense_disp_timepoint[slice_num], axis=0)
				else:
					dense_cur_disp = dense_disp_timepoint[slice_num]
			elem_interp_dx = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 0], elem_pts_interp, method='linear')
			elem_interp_dy = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 1], elem_pts_interp, method='linear')
			elem_interp_nans = np.where(np.isnan(elem_interp_dx))[0]
			elem_interp_dx[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 0], elem_pts_interp[elem_interp_nans, :], method='nearest')
			elem_interp_dy[elem_interp_nans] = sp.interpolate.griddata(dense_pts_prol, dense_cur_disp[:, 1], elem_pts_interp[elem_interp_nans, :], method='nearest')
			for elem_num in range(len(elems_in_dense)):
				elem_displacements_x[elems_in_dense[elem_num]][time_point] = elem_interp_dx[elem_num]
				elem_displacements_y[elems_in_dense[elem_num]][time_point] = elem_interp_dy[elem_num]
			
		# Do this as a list, set up list on creation with NaNs and fill here. Allows more open adjustments.
		self.dense_x_displacements = np.array(elem_displacements_x)
		self.dense_y_displacements = np.array(elem_displacements_y)
			
		return(elem_displacements_x, elem_displacements_y)
		
	def generateFEFile(self, input_file, conn_mat='hex'):
		"""Generate FEBio output file for use in FEBio and PostView.
		"""
		# XML Version String
		xml_ver_str = '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
		
		# Set up febio xml tags
		fe_start_str = '<febio_spec version="2.0">\n'
		fe_end_str = '</febio_spec>'
		
		# Set up Module section strings
		module_str = '\t<Module type="solid"/>\n'
		
		'''
		# Set up Material section strings
		material_start_str = '<Material>'
		material_end_str = '</Material>'
		material_child_str = '<material id="1" name="Material 1" type="trans iso Mooney-Rivlin">'
		material_end_str = '</material>'
		'''
		# Set up Geometry strings
		geometry_start_str = '\t<Geometry>\n'
		geometry_end_str = '\t</Geometry>\n'
		node_start_str = '\t\t<Nodes>\n'
		node_end_str = '\t\t</Nodes>\n'
		type_str = 'pent6' if conn_mat == 'pent' else 'hex8'
		element_start_str = '\t\t<Elements type="{}" mat="1" elset="Part14">\n'.format(type_str)
		element_end_str = '\t\t</Elements>\n'
		
		# Set up node base string
		node_str = '\t\t\t<node id="{}">{},{},{}</node>\n'
		elem_str = '\t\t\t<elem id="{}">{},{},{},{},{},{}</elem>\n' if conn_mat == 'pent' else '\t\t\t<elem id="{}">{},{},{},{},{},{},{},{}</elem>\n'
		
		# Select connectivity matrix based on selection
		conn_mat_nodes = self.pent if conn_mat == 'pent' else self.hex
		
		# Set up formatted string lists
		node_strings_formatted = [node_str.format(i+1, self.nodes[i, 0], self.nodes[i, 1], self.nodes[i, 2]) for i in range(self.nodes.shape[0])]
		elem_strings_formatted = [elem_str.format(*tuple(np.insert(conn_mat_nodes[i, :] + 1, 0, i+1))) for i in range(conn_mat_nodes.shape[0])]
		
		# Create file
		output_file_name = input_file.replace('.mat', '.feb')
		with open(output_file_name, 'w') as out_file:
			out_file.writelines([xml_ver_str, fe_start_str, module_str, geometry_start_str, node_start_str])
			out_file.writelines(node_strings_formatted)
			out_file.writelines([node_end_str, element_start_str])
			out_file.writelines(elem_strings_formatted)
			out_file.writelines([element_end_str, geometry_end_str, fe_end_str])
		return(output_file_name)
	
	def displayMeshPostview(self, file_name):
		"""Launch PostView with specific file selected.
		"""
		p = subprocess.Popen(['C://Program Files/postview-2.1.0/PostView2.exe', file_name])
		return(p)
		
	def _assignRegionNodes(self, region_contour, contour_edge_spacing, num_slices):
		"""Generalized method to label which elements fall within a region contour.
		
		Region contour should be spaced evenly per slice, with an equal number of points on the outer layer and inner layer. The point distance between the circumferential extents of the region should be given with contour edge spacing.
		"""
		# Determine mu error buffer should be used
		err_val = 0
		
		# Convert nodes to cartesian for measurement
		nodes_prol_mu, nodes_prol_nu, nodes_prol_phi = self._cartToProlate(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
		nodes_prol = np.column_stack((nodes_prol_mu, nodes_prol_nu, nodes_prol_phi))
		
		# Convert region to prolate
		region_mu, region_nu, region_phi = self._cartToProlate(region_contour[:, 0], region_contour[:, 1], region_contour[:, 2])
		region_prol = np.column_stack((region_mu, region_nu, region_phi))
		region_prol_edges = region_prol[0::contour_edge_spacing, :]
		
		# Get polygonal path for the region edges
		region_prol_polygon = np.vstack((region_prol_edges[0::2, :], region_prol_edges[:0:-2, :], region_prol_edges[0, :]))
		region_polygon = path.Path(np.column_stack((region_prol_polygon[:, 2], region_prol_polygon[:, 1])))
		
		# Determine points inside region contour
		nodes_in_region = np.where(region_polygon.contains_points(np.column_stack((nodes_prol[:, 2], nodes_prol[:, 1]))))[0]
		
		# Get surface plots for inner and outer region surfaces
		base_list = list(range(contour_edge_spacing))
		sum_list = [[contour_edge_spacing*2*i]*contour_edge_spacing for i in range(num_slices)]
		region_inds_inner = []
		for i in range(len(sum_list)):
			region_inds_inner = np.append(region_inds_inner, np.add(base_list, sum_list[i]))
		region_inds_inner = [int(s_i) for s_i in region_inds_inner]
		region_inds_outer = [region_inds_inner_i + contour_edge_spacing for region_inds_inner_i in region_inds_inner]
		
		# Interpolate base on grid placement to get inner and outer values at each node
		inner_pt_vals = sp.interpolate.griddata(np.column_stack((region_prol[region_inds_inner, 2], region_prol[region_inds_inner, 1])), region_prol[region_inds_inner, 0], np.column_stack((nodes_prol[nodes_in_region, 2], nodes_prol[nodes_in_region, 1])), method='cubic')
		outer_pt_vals = sp.interpolate.griddata(np.column_stack((region_prol[region_inds_outer, 2], region_prol[region_inds_outer, 1])), region_prol[region_inds_outer, 0], np.column_stack((nodes_prol[nodes_in_region, 2], nodes_prol[nodes_in_region, 1])), method='cubic')
		mu_range = np.column_stack((inner_pt_vals, outer_pt_vals))
		
		# Error factor if needed
		mu_range_err = [err_val*abs(mu_range[i, 1] - mu_range[i, 0]) for i in range(mu_range.shape[0])]
		
		# Find which nodes are within the mu extent at the specified phi, nu points
		nodes_in_mu = np.where([((np.min(mu_range[i, :])-mu_range_err[i]) <= nodes_prol[nodes_in_region[i], 0]) & ((np.max(mu_range[i, :])+mu_range_err[i]) >= nodes_prol[nodes_in_region[i], 0]) for i in range(len(nodes_in_region))])[0]
		
		# Get final node values within the 3-d region
		final_node_inds = nodes_in_region[nodes_in_mu]
		
		return(final_node_inds)
	
	def displayScarTrace(self, scar, ax=None):
		"""Plots scar trace overlay onto a passed axis.
		"""
		if not ax:
			fig = mplt.figure()
			ax = fig.add_subplot(111, projection='3d')
		for scar_slice in scar:
			# Append the first point to the end to make a circular contour
			cur_scar = np.append(scar_slice, np.expand_dims(scar_slice[0, :], 0), axis=0)
			# Transform the data using the same transformation as endo/epi contours
			data_scar = np.dot((cur_scar - np.array([self.origin for i in range(cur_scar.shape[0])])), np.transpose(self.transform))
			# Plot
			x = data_scar[:, 2]
			y = data_scar[:, 1]
			z = data_scar[:, 0]
			ax.plot(x, y, -z, 'r-')
		return(ax)
	
	def displayDensePts(self, dense_pts, dense_slices, dense_displacement_all=False, dense_plot_quiver=0, timepoint=-1, ax=None):
		"""Shows DENSE pointts and (optionally) displacements in a 3D graph.
		"""
		# If no axes were passed, generate axes
		if not ax:
			fig = mplt.figure()
			ax = fig.add_subplot(111, projection='3d')
		
		# Pull appropriate timepoint (or set to False if displacement is undesired)
		if timepoint >= 0:
			try:
				dense_displacement = dense_displacement_all[timepoint]
			except(IndexError):
				warnings.warn('Timepoint not in range of values! Displacement will be ignored.')
				timepoint = -1
				dense_displacement = False
			except(TypeError):
				warnings.warn('Displacement not passed, but timepoint requested! Displacement will be ignored.')
				dense_displacement = False
		else:
			dense_displacement = False
		
		for i in range(len(dense_slices)):
			cur_slice = dense_slices[i]
			cur_dense_pts = np.column_stack((dense_pts[i], [cur_slice - self.origin[2]]*dense_pts[i].shape[0]))
			data_dense = np.dot(cur_dense_pts, np.transpose(self.transform))
			x = data_dense[:, 2]
			y = data_dense[:, 1]
			z = data_dense[:, 0]
			ax.scatter(x, y, -z, ',')
			
			if dense_displacement:
				dense_displacement_slice = dense_displacement[i]
				if dense_plot_quiver == 1:
					ax.quiver(x, y, -z, dense_displacement_slice[:, 0], dense_displacement_slice[:, 1], [0]*dense_displacement_slice.shape[0])
		return(ax)
	
	def nodeRender(self, nodes, ax=None):
		"""Display the nodes passed in as a 3D scatter plot."""
		if not ax:
			fig = mplt.figure()
			ax = fig.add_subplot(111, projection='3d')
		x = nodes[:, 2]
		y = nodes[:, 1]
		z = nodes[:, 0]
		ax.scatter(x, y, -z)
		return(ax)
	
	def _fitBicubicData(self, data, focus, mesh_density='4x2', smooth=True, constraints=True, compute_errors=True):
		"""Bicubic fit of x,y,z data to a prolate mesh
		
		args:
			data (array): x, y, z points arranged by column
			focus (float): the focal point for the prolate spheroid
			mesh_density (string): Determines mesh type
			smooth (bool): Whether or not to implement smoothing
			contraints (bool): Determines solution method
		returns:
			node_matrix (array): Array containing nodal prolate spheroid values
			rms_error (int): Error estimation for lambda
		"""
		
		if not mesh_density in ['4x2', '4x4', '4x8']:
			mesh_density = '4x2'
			print('Mesh density incorrect. Set to 4x2.')
		
		if constraints:
			if mesh_density == '4x2':
				c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [4, 1, 1, 0, 1, -1, 1, 0, 1], [7, 1, 1, 0, 1, 1, -1, 0, 1], [10, 1, 1, 0, 1, 4, -1, 0, 1]])
			elif mesh_density == '4x4':
				c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [6, 1, 1, 0, 1, -1, 1, 0, 1], [11, 1, 1, 0, 1, 1, -1, 0, 1], [16, 1, 1, 0, 1, 6, -1, 0, 1]])
			elif mesh_density == '4x8':
				c = np.array([[1, -1, 1, 0, 1, -1, 1, 0, 1], [6, 1, 1, 0, 1, -1, 1, 0, 1], [11, 1, 1, 0, 1, -1, 1, 0, 1], [16, 1, 1, 0, 1, -1, 1, 0, 1], [21, 1, 1, 0, 1, 1, -1, 0, 1], [26, 1, 1, 0, 1, 6, -1, 0, 1], [31, 1, 1, 0, 1, 11, -1, 0, 1], [36, 1, 1, 0, 1, 16, -1, 0, 1]])
		
		# Get starting mesh and organize
		nodal_mesh = self._getStarterMesh(mesh_density)
		unsorted_nodal_mesh_deg = nodal_mesh[:, 1:3]
		nodal_mesh = nodal_mesh[nodal_mesh[:, 0].argsort()]
		nodal_mesh = nodal_mesh[nodal_mesh[:, 1].argsort(kind='mergesort')]
		nodal_mesh = nodal_mesh[nodal_mesh[:, 2].argsort(kind='mergesort')]
		
		# Get Sizes
		m = nodal_mesh.shape[0]
		size_nodal_nu = np.where(nodal_mesh[:, 2] == 0)[0].size
		size_nodal_phi = m/size_nodal_nu
		num_elem = size_nodal_phi*(size_nodal_nu-1)
		
		# Put into radians
		rads = math.pi/180
		nodal_nu_deg = nodal_mesh[0:size_nodal_nu, 1]
		nodal_phi_deg = nodal_mesh[0::size_nodal_nu, 2]
		nodal_nu = nodal_nu_deg*rads
		nodal_phi = nodal_phi_deg*rads
		
		# Build initial guess vector
		init_guess = nodal_mesh[:, 0].tolist()
		init_guess.extend(nodal_mesh[:, 3])
		init_guess.extend(nodal_mesh[:, 4])
		init_guess.extend(nodal_mesh[:, 5])
		num_dof_total = len(init_guess)
		num_dof_elem = 16

		if smooth:
			gp_limit = 7
			num_gp = [3, 3]
			if num_gp[0] > gp_limit or num_gp[1] > gp_limit:
				raise(NameError)
		
		# Initialize
		pts_elem = [0] * int(num_elem)
		global_stiff = [[0 for i in range(int(num_dof_total))] for j in range(int(num_dof_total))]
		global_rhs = [0] * int(num_dof_total)
		lhs_matrix = [[0] * int(num_dof_total)] * int(num_dof_total)
		
		# Read Data Points:
		data_x = data[:, 0]
		data_y = data[:, 1]
		data_z = data[:, 2]
		count = data.shape[0]
		data_w = np.ones((count, 3))
		# Convert to Prolate
		data_size = 0
		min_nodal_nu = min(nodal_nu)
		max_nodal_nu = max(nodal_nu)
		max_nodal_phi = max(nodal_phi)
		for i in range(count):
			ind = []
			mu, nu, phi = self._cartToProlate(data_x[i], data_y[i], data_z[i], focus)
			if nu >= min_nodal_nu and nu <= max_nodal_nu:
				data_size += 1
				e, corner13phi, corner24phi, corner12nu, corner34nu = self.__calcCornerThetaMu(nu, phi, nodal_phi, size_nodal_phi, max_nodal_phi, nodal_nu, min_nodal_nu, max_nodal_nu)
				element_number = int((corner12nu-1)*size_nodal_phi + corner24phi)
				pts_elem[element_number - 1] = pts_elem[element_number - 1] + 1
				# Build dof vector.
				# 	ind is the index vector, of length 16.
				ind = self.__generateInd(size_nodal_nu, corner13phi, corner24phi, corner12nu, corner34nu, m)

				dof_model = np.array(init_guess)[ind]
				
				lam_model, h = self._generalFit(dof_model, e)
				
				lam_diff = mu - lam_model

				for h1 in range(num_dof_elem):
					global_rhs[ind[h1]] = global_rhs[ind[h1]] + h[h1]*lam_diff*data_w.flatten()[i]
					for h2 in range(num_dof_elem):
						global_stiff[ind[h1]][ind[h2]] += h[h1]*h[h2]*data_w.flatten()[i]
		
		# Add smoothing if requested
		if smooth:
			smooth_weights = self._getSmoothWeights(mesh_density, num_elem, unsorted_nodal_mesh_deg, nodal_nu_deg, nodal_phi_deg, size_nodal_phi)
			global_damp = self._calcDamping(smooth_weights, num_gp, num_dof_total, num_dof_elem, size_nodal_nu, size_nodal_phi, m, num_dof_total)
			lhs_matrix = global_stiff + global_damp
		else:
			lhs_matrix = global_stiff
		
		# Solve for displacement
		if constraints:
			displacement = self._solveReducedSystem(lhs_matrix, global_rhs, c, num_dof_total, m, num_dof_total)
		else:
			displacement = np.linalg.solve(lhs_matrix, global_rhs)
		
		# Fitted dof values sorted by mu and theta
		optimized_dof = np.add(init_guess, displacement)
		
		# Organize DOF data as though it is written to and read from IPNODE file
		dof_data = self._generateDofData(optimized_dof, m, nodal_nu_deg, nodal_phi_deg, unsorted_nodal_mesh_deg, size_nodal_nu)
		
		# Get the 72x6 matrix with nodal values
		# 	Columns are in the order: lambda, mu, theta, dlds1, dlds2, d2lds1ds2
		node_matrix = np.append(dof_data[:, 0].reshape([dof_data.shape[0], 1]), np.append(unsorted_nodal_mesh_deg, dof_data[:, 1:], axis=1), axis=1)
		
		# Perform the error estaimate
		if compute_errors:
			# Initialize global error
			err = 0
			
			# Iterate through points again to estimate error
			for i in range(count):
				mu, nu, phi = self._cartToProlate(data_x[i], data_y[i], data_z[i], focus)
				if nu >= min_nodal_nu and nu < max_nodal_nu:
					e, corner13phi, corner24phi, corner12nu, corner34nu = self.__calcCornerThetaMu(nu, phi, nodal_phi, size_nodal_phi, max_nodal_phi, nodal_nu, min_nodal_nu, max_nodal_nu)
					ind = self.__generateInd(size_nodal_nu, corner13phi, corner24phi, corner12nu, corner34nu, m)
					dof_model = optimized_dof[ind]
					
					lam_model, _ = self._generalFit(dof_model, e)
					
					err += ((lam_model - mu)*data_w.flatten()[i])**2
			# Get the RMS Error in Lambda
			rms_err = math.sqrt(err / data_size)
		else:
			rms_err = 0
		return([node_matrix, rms_err])
		
	def _getStarterMesh(self, mesh_density):
		"""Return initial starter mesh based on mesh density."""
		# Based on the indicated mesh density, compose starter mesh columns
		if mesh_density == '4x2':
			first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
			second_coord = np.array([120, 120, 120, 120, 60, 60, 60, 60, 0, 0, 0, 0])
			third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270])
			coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		elif mesh_density == '4x4':
			first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
			second_coord = np.array([120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0])
			third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270])
			coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		elif mesh_density == '4x8':
			first_coord = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
			second_coord = np.array([120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0, 120, 120, 120, 120, 90, 90, 90, 90, 60, 60, 60, 60, 30, 30, 30, 30, 0, 0, 0, 0])
			third_coord = np.array([0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315, 45, 135, 225, 315])
			coord1_der1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
			coord1_der3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		# Construct the full y array by stacking the columns
		y = np.column_stack((first_coord, second_coord, third_coord, coord1_der1, coord1_der2, coord1_der3))
		return(y)
		
	def _prepData(self, all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts):
		"""Reorganize endo and epi data for processing.
		
		args:
			all_data_endo (array): Endo data from MRI model
			all_data_epi (array): Epi data from MRI model
			apex_pt (array): Apical point selected from long-axis data
			basal_pt (array): Basal point selected from long-axis data
			septal_pts (array): Septal points selected from short-axis data
		returns:
			data_endo (array): Modified endo contour
			data_epi (array): Modified epi contour
			focus (float): The focal point for prolate coordinates
			transform_basis (array): Combined vector from the 3 calculated basis vectors
			origin (array): Point indicating the origin point
		"""
		endo = all_data_endo[:, 0:3]
		epi = all_data_epi[:, 0:3]
		
		# Calculate first basis vector, (Base - Apex)/Magnitude
		c = apex_pt - basal_pt
		c_norm = self._calcNorm(c)
		e1_basis = [c1 / c_norm for c1 in c]
		
		# Calculate Origin Location
		origin = basal_pt + c/3
		
		# Calculate Focus Length based on c_norm
		focus = (2*c_norm/3)/(math.cosh(1))
		num_points = endo.shape[0]
		
		# Calculate Second basis vector using plane intersects, septal point, and e1
		d1 = septal_pts[0, :] - origin
		d2 = d1 - [np.dot(d1, e1_basis)*e1_elem for e1_elem in e1_basis]
		e2_basis = d2 / self._calcNorm(d2)
		
		# Calculate third basis vector from the first 2 basis vectors
		e3 = np.cross(e1_basis, e2_basis)
		e3_basis = e3 / self._calcNorm(e3)
		
		# Set up transform basis from the 3 calculated basis vectors
		transform_basis = np.array([e1_basis, e2_basis, -e3_basis])
		
		# Set up the modified endo and epi contours
		data_endo = np.dot((endo - np.array([origin for i in range(num_points)])), np.transpose(transform_basis))
		data_epi = np.dot((epi - np.array([origin for i in range(num_points)])), np.transpose(transform_basis))
		
		# Append extra identifying data to the modified contours
		data_endo = np.append(data_endo, np.reshape(all_data_endo[:, 3], [all_data_endo.shape[0], 1]), axis=1)
		data_epi = np.append(data_epi, np.reshape(all_data_epi[:, 3], [all_data_epi.shape[0], 1]), axis=1)
		return([data_endo, data_epi, focus, transform_basis, origin])
		
	def _calcNorm(self, arr_in):
		"""Calculate the norm of an input array"""
		norm = np.sqrt(np.sum(np.square(arr_in)))
		return norm
		
	def _cartToProlate(self, x, y, z, focus=None):
		"""Convert passed x, y, z from cartesian to prolate based on focus."""
		# Check if focus is passed
		if focus is None: focus = self.focus
		# Checks if the data is a matrix to set the loop.
		matrix_flag = False
		if isinstance(x, np.ndarray):
			# Store the shape of the passed array, then flatten them
			input_shape = x.shape
			x = np.reshape(x, x.size, 1)
			y = np.reshape(y, y.size, 1)
			z = np.reshape(z, z.size, 1)
			len_x = x.size
			matrix_flag = True
		else:
			len_x = 1
		m = np.zeros((len_x, 1))
		n = np.zeros((len_x, 1))
		p = np.zeros((len_x, 1))
		# Loop through the vaues and perform equations to convert to prolate.
		for jz in range(len_x):
			# Pull values from array if arrays were passed
			if matrix_flag:
				x1 = x[jz]
				x2 = y[jz]
				x3 = z[jz]
			else:
				x1 = x
				x2 = y
				x3 = z
			a1 = x1**2 + x2**2 + x3**2 - focus**2
			a2 = math.sqrt((a1**2)+4*(focus**2)*((x2**2)+(x3**2)))
			a3 = 2*(focus**2)
			a4 = max([(a1+a2)/a3, 0])
			a5 = max([(a2-a1)/a3, 0])
			a6 = math.sqrt(a4)
			a7 = min([math.sqrt(a5), 1])
			a8 = math.asin(a7) if abs(a7) <= 1 else 0
			if abs(a7) > 1: print('SLH_CMI_C2P: A8 is zero')
			if x3==0 or a6==0 or a7==0:
				a9 = 0
			else:
				a9 = x3 / (focus*a6*a7) if abs(a6*a7)>0 else 0
			a9 = math.pi/2 if a9 >= 1 else -math.pi/2 if a9 <= -1 else math.asin(a9)
			# Set the prolate values lambda (z1), mu (z2), and theta (z3)
			z1 = math.log(a6 + math.sqrt(a4+1))
			z2 = a8 if x1 >= 0 else math.pi - a8
			z3 = math.fmod(a9, 2*math.pi) if x2 >= 0 else math.pi-a9
			# Store the singular values into the array
			if matrix_flag:
				m[jz] = z1
				n[jz] = z2
				p[jz] = z3
			else:
				m = z1
				n = z2
				p = z3
		# Reshape the mu, nu, and phi arrays based on the input arrays
		if matrix_flag:
			m = np.reshape(m, input_shape, order='F')
			n = np.reshape(n, input_shape, order='F')
			p = np.reshape(p, input_shape, order='F')
		return([m, n, p])
		
	def _prolateToCart(self, m, n, p, focus=None):
		"""Convert passed lambda, mu, theta from prolate to cartesian based on focus."""
		if focus is None: focus = self.focus
		x = focus * np.cosh(m) * np.cos(n)
		y = focus * np.sinh(m) * np.sin(n) * np.cos(p)
		z = focus * np.sinh(m) * np.sin(n) * np.sin(p)
		return([x, y, z])
		
	def _generalFit(self, dof_model, e, order=0):
		"""Compute lambda values by bicubic Hermite-Lagrange basis functions.
		
		args:
			dof_model (array): DOF of the model
			e (array): Local coordinates
			order (int): Temporal order
		returns:
			l (array): Lambda values
			h (array): Shape functions for local coordinates
		"""
		# 1-D Shape Functions
		h00 = [1 - 3*(e[0]**2) + 2*(e[0]**3), 1 - 3*(e[1]**2) + 2*(e[1]**3)]
		h10 = [e[0]*((e[0]-1)**2), e[1]*((e[1]-1)**2)]
		h01 = [(e[0]**2)*(3-2*e[0]), (e[1]**2)*(3-2*e[1])]
		h11 = [(e[0]**2)*(e[0]-1), (e[1]**2)*(e[1]-1)]
		
		# Assemble 2-D Shape Functions
		h_init = [h00[0]*h00[1], h01[0]*h00[1], h00[0]*h01[1], h01[0]*h01[1], h10[0]*h00[1], h11[0]*h00[1], h10[0]*h01[1], h11[0]*h01[1], h00[0]*h10[1], h01[0]*h10[1], h00[0]*h11[1], h01[0]*h11[1], h10[0]*h10[1], h11[0]*h10[1], h10[0]*h11[1], h11[0]*h11[1]]
		
		if order > 0: # Temporal Basis Functions
			# Normalize time coordinates
			t = [i/order for i in range(order+1)]
			num_t = order+1
			# Compute Lagrange polynomial
			lg = [1 for i in range(num_t)]
			lg = [lg[i]*(e[2]-t[j])/(t[i]-t[j]) for i in range(num_t) for j in range(num_t) if not(j==i)]
			# Rearrange to put element nodes in first 2 positions
			if order > 1:
				temp = lg[1:num_t]
				lg[1] = lg[order]
				lg[2:order] = temp
			# Assemble h and get l
			l = 0
			h = []
			for i in range(num_t):
				l = l + lg[i]*np.dot(h_init, dof_model[:, i])
				h.append([lg[i]*h_init_i for h_init_i in h_init])
		else: # Space-only fit
			h = h_init
			l = np.dot(h, dof_model)
		return([l, h])
		
	def _getSmoothWeights(self, mesh_density, num_elem, unsorted_mesh_deg, nodal_mu_deg, nodal_theta_deg, size_nodal_theta):
		"""Parse the template files to get the smoothing weights
		
		args:
			ipfit_file (string): IPNODE template file
			elem_file (string): Starting mesh file
			num_elem (int): The number of elements
			unsorted_mesh_deg (array): The unsorted mesh, with angles in degrees
			nodal_mu_deg (array): Nodal mu values, based on angles in degrees
			nodal_theta_deg (array): Nodal theta values, with angles in degrees
			size_nodal_theta (array): Size of the nodal theta arrays
		returns:
			smooth_weights (array): The sorted mesh values
		"""
		# Pre-allocate the array for smooth_weights to allow indexing
		smooth_weights = np.zeros([int(num_elem), 5])
		# Generate columns for temp array
		if mesh_density == '4x2':
			col1 = np.array([0.01] * 8)
			col2 = np.array([0.02] * 8)
			col3 = np.array([0.01] * 8)
			col4 = np.array([0.02] * 8)
			col5 = np.array([0.04] * 8)
		elif mesh_density == '4x4':
			col1 = np.array([0.01] * 16)
			col2 = np.array([0.02] * 16)
			col3 = np.array([0.01] * 16)
			col4 = np.array([0.02] * 16)
			col5 = np.array([0.04] * 16)
		elif mesh_density == '4x8':
			col1 = np.array([0.1] * 32)
			col2 = np.array([0.2] * 32)
			col3 = np.array([0.1] * 32)
			col4 = np.array([0.2] * 32)
			col5 = np.array([0.4] * 32)
		# Form temp array from columns
		temp = np.column_stack((col1, col2, col3, col4, col5))
		# Generate Element Index Lists
		if mesh_density == '4x2':
			n1_list = [6, 7, 8, 5, 10, 11, 12, 9]
			n2_list = [5, 6, 7, 8, 9, 10, 11, 12]
		elif mesh_density == '4x4':
			n1_list = [6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13, 18, 19, 20, 17]
			n2_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
		elif mesh_density == '4x8':
			n1_list = [25, 6, 26, 7, 27, 8, 28, 5, 29, 10, 30, 11, 31, 12, 32, 9, 33, 14, 34, 15, 35, 16, 36, 13, 37, 18, 38, 19, 39, 20, 40, 17]
			n2_list = [5, 25, 6, 26, 7, 27, 8, 28, 9, 29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 37, 18, 38, 19, 39, 20, 40]
		# Fill out smooth_weights array
		for i in range(len(n1_list)):
			# Pull n1 and n2 from the lists
			n1 = n1_list.pop(0)
			n2 = n2_list.pop(0)
			# Pull mu and theta based on n1 and n2
			mu = unsorted_mesh_deg[n1-1, 0]
			theta = unsorted_mesh_deg[n2-1, 1]
			# Get indices of nodal mu / theta array where less than mu / theta
			mu_ind = np.where(nodal_mu_deg <= mu)[0].size
			theta_ind = np.where(nodal_theta_deg <= theta)[0].size
			# Calculate e index based on mu and theta
			e_ind = int((mu_ind-1)*size_nodal_theta + theta_ind)
			# Copy last row in temp to smooth weights, then remove row from temp
			smooth_weights[e_ind-1] = temp[-1, :]
			temp = temp[:-1, :]
		return(smooth_weights)
		
	def _calcDamping(self, smooth_weights, num_gp, num_dof_total, num_dof_elem, size_nodal_mu, size_nodal_theta, num_nodes, num_dof_mesh, num_time_nodes=1):
		"""Calculate the global damping array from passed values
		
		Assumes that there are 5 derivatives (1, 11, 2, 22, 12)
		args:
			smooth_weights (array): Array output from reduced system solution
			num_gp (int)
			num_dof_total (int): Total number of degrees of freedom
			num_dof_elemn (int): Number of degrees of freedom per element
			size_nodal_mu (int): Size of the nodal_mu array
			size_nodal_theta (float): Size of the nodal_theta array
			num_nodes (int): Number of nodes
			num_dof_mesh (int): Number of degrees of freedom in the mesh
			num_time_nodes (int, default 1): Number of timepoints per node
		returns:
			global_damp (array): Array with global damping and mass matrix used for smoothing
		"""
		# Initialize the output matrix and convert size_nodal_theta to an integer
		size_nodal_theta = int(size_nodal_theta)
		global_damp = np.zeros([num_dof_total, num_dof_total])
		
		p = np.array([[0, 0, 0, 0, 0, 0, 0], [-0.2886751345948130, 0.2886751345948130, 0, 0, 0, 0, 0], [-0.3872983346207410, 0, 0.3872983346207410, 0, 0, 0, 0], [-0.4305681557970260, -0.1699905217924280, 0.1699905217924280, 0.4305681557970260, 0, 0, 0], [-0.4530899229693320, -0.2692346550528410, 0, 0.2692346550528410, 0.4530899229693320, 0, 0], [-0.4662347571015760, -0.3306046932331330, -0.1193095930415990, 0.1193095930415990, 0.3306046932331330, 0.4662347571015760, 0], [-0.4745539561713800, -0.3707655927996970, -0.2029225756886990, 0, 0.2029225756886990, 0.3707655927996970, 0.4745539561713800]]) + 0.5
		
		w = np.array([[1, 0, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0, 0], [0.2777777777777780, 0.4444444444444440, 0.2777777777777780, 0, 0, 0, 0], [0.1739274225687270, 0.3260725774312730, 0.3260725774312730, 0.1739274225687270, 0, 0, 0], [0.1184634425280940, 0.2393143352496830, 0.2844444444444440, 0.2393143352496830, 0.1184634425280940, 0, 0], [0.0856622461895850, 0.1803807865240700, 0.2339569672863460, 0.2339569672863460, 0.1803807865240700, 0.0856622461895850, 0], [0.0647424830844350, 0.1398526957446390, 0.1909150252525600, 0.2089795918367350, 0.1909150252525600, 0.1398526957446390, 0.0647424830844350]])
		
		# Step through each element and integrate at Gauss points
		#	Same number of Gauss points are used for all elements
		for mu in range(size_nodal_mu-1):
			for theta in range(size_nodal_theta):
				# Calculate element number
				elem_num = (mu - 1)*size_nodal_theta + theta
				# Get the nearest theta and mu indices
				corner24theta = theta+1
				corner13theta = 1 if theta == size_nodal_theta-1 else corner24theta+1
				corner12mu = mu+1
				corner34mu = corner12mu + 1
				# Generate the index array of relevant degrees of freedom
				ind = self.__generateInd(size_nodal_mu, corner13theta, corner24theta, corner12mu, corner34mu, num_nodes)
				# Step through each Gauss point
				for e1 in range(num_gp[0]):
					for e2 in range(num_gp[1]):
						# Calculate weight of gauss point and position
						wgp = w[num_gp[0]-1, e1]*w[num_gp[1]-1, e2]
						e = [p[num_gp[0]-1, e1], p[num_gp[1]-1, e2]]
						for der in range(5):
							# Get smoothing weighting factor for specific derivative
							wder = smooth_weights[elem_num, der]
							# Get the basis derivatives in the same order as ind
							h = self._calcBasisDerivs(e, der+1)
							for h1 in range(num_dof_elem):
								for h2 in range(num_dof_elem):
									for i in range(num_time_nodes):
										# Calculate the global damping matrix point
										global_damp[num_dof_mesh*(i-1) + ind[h1], num_dof_mesh*(i-1)+ind[h2]] += h[h1]*h[h2]*wgp*wder
										
		return(global_damp)
							
	def _calcBasisDerivs(self, e, deriv_num, order=0):
		"""Compute derivatives of the bicubic hermite and lagrange basis function coefficients
		
		The order of the basis derivatives is the order of the time polynomial.
		
		args:
			e (array): The points at which to calculate the derivatives.
			deriv_num (int): Determines which derivatives to calculate
			order (int): Determines space-time derivative calculation
		returns:
			h (array): Array containing coefficient derivatives
		"""
		# 1D Shape Functions:
		h00 = [1 - 3*(e_i**2) + 2*(e_i**3) for e_i in e]
		h10 = [e_i*((e_i-1)**2) for e_i in e]
		h01 = [(e_i**2)*(3-2*e_i) for e_i in e]
		h11 = [(e_i**2)*(e_i-1) for e_i in e]
		
		# First Derivatives
		dh00 = [-6*e_i + 6*(e_i**2) for e_i in e]
		dh10 = [2*e_i*(e_i-1) + (e_i-1)**2 for e_i in e]
		dh01 = [-2*(e_i**2) + 2*e_i*(3-2*e_i) for e_i in e]
		dh11 = [(e_i**2) + 2*e_i*(e_i-1) for e_i in e]
		
		# Second Derivatives
		d2h00 = [12*e_i - 6 for e_i in e]
		d2h10 = [6*e_i - 4 for e_i in e]
		d2h01 = [-12*e_i + 6 for e_i in e]
		d2h11 = [6*e_i - 2 for e_i in e]
		
		# Assemble Spatial Derivative Coefficients
		if deriv_num == 0:
			h_init = [h00[0]*h00[1], h01[0]*h00[1], h00[0]*h01[1], h01[0]*h01[1], h10[0]*h00[1], h11[0]*h00[1], h10[0]*h01[1], h11[0]*h01[1], h00[0]*h10[1], h01[0]*h10[1], h00[0]*h11[1], h01[0]*h11[1], h10[0]*h10[1], h11[0]*h10[1], h10[0]*h11[1], h11[0]*h11[1]]
		elif deriv_num == 1:
			h_init = [dh00[0]*h00[1], dh01[0]*h00[1], dh00[0]*h01[1], dh01[0]*h01[1], dh10[0]*h00[1], dh11[0]*h00[1], dh10[0]*h01[1], dh11[0]*h01[1], dh00[0]*h10[1], dh01[0]*h10[1], dh00[0]*h11[1], dh01[0]*h11[1], dh10[0]*h10[1], dh11[0]*h10[1], dh10[0]*h11[1], dh11[0]*h11[1]]
		elif deriv_num == 2:
			h_init = [d2h00[0]*h00[1], d2h01[0]*h00[1], d2h00[0]*h01[1], d2h01[0]*h01[1], d2h10[0]*h00[1], d2h11[0]*h00[1], d2h10[0]*h01[1], d2h11[0]*h01[1], d2h00[0]*h10[1], d2h01[0]*h10[1], d2h00[0]*h11[1], d2h01[0]*h11[1], d2h10[0]*h10[1], d2h11[0]*h10[1], d2h10[0]*h11[1], d2h11[0]*h11[1]]
		elif deriv_num == 3:
			h_init = [h00[0]*dh00[1], h01[0]*dh00[1], h00[0]*dh01[1], h01[0]*dh01[1], h10[0]*dh00[1], h11[0]*dh00[1], h10[0]*dh01[1], h11[0]*dh01[1], h00[0]*dh10[1], h01[0]*dh10[1], h00[0]*dh11[1], h01[0]*dh11[1], h10[0]*dh10[1], h11[0]*dh10[1], h10[0]*dh11[1], h11[0]*dh11[1]]
		elif deriv_num == 4:
			h_init = [h00[0]*d2h00[1], h01[0]*d2h00[1], h00[0]*d2h01[1], h01[0]*d2h01[1], h10[0]*d2h00[1], h11[0]*d2h00[1], h10[0]*d2h01[1], h11[0]*d2h01[1], h00[0]*d2h10[1], h01[0]*d2h10[1], h00[0]*d2h11[1], h01[0]*d2h11[1], h10[0]*d2h10[1], h11[0]*d2h10[1], h10[0]*d2h11[1], h11[0]*d2h11[1]]
		elif deriv_num == 5:
			h_init = [dh00[0]*dh00[1], dh01[0]*dh00[1], dh00[0]*dh01[1], dh01[0]*dh01[1], dh10[0]*dh00[1], dh11[0]*dh00[1], dh10[0]*dh01[1], dh11[0]*dh01[1], dh00[0]*dh10[1], dh01[0]*dh10[1], dh00[0]*dh11[1], dh01[0]*dh11[1], dh10[0]*dh10[1], dh11[0]*dh10[1], dh10[0]*dh11[1], dh11[0]*dh11[1]]
			
		if order == 0:
			h = h_init
		else:
			# 0-order time derivative:
			
			# Normalized time coordinates for elemtn of this order
			t = [i/order for i in range(order+1)]
			num_t = order+1
			
			# Compute Lagrange Polynomial, 0-order time derivative
			lg = [1 for i in range(num_t)]
			lg = [lg[i]*(e[2]-t[j])/(t[i]-t[j]) for i in range(num_t) for j in range(num_t) if not(j==i)]
			
			# Rearrange to put Element nodes in first 2 pos
			if order > 1:
				temp = lg[1:num_t]
				lg[1] = lg[num_t]
				lg[2:num_t+1] = temp
				
			# Assemble First row of H (0-order time derivative)	
			h_temp = [lg[i]*h_init for i in range(num_t)]
			h[0, :] = h_temp
			
			# First order derivative:
			
			# Compute First-order time derivatives (velocity)
			vel = []
			for i in range(num_t):
				vel.append(0)
				for j in range(num_t):
					if j != i:
						vel_temp = 1
						for k in range(num_t):
							if k != j and k != i:
								vel_temp *= e[2] - t[k]
						vel[i] += vel_temp
				for m in range(num_t):
					if m != i:
						vel[i] /= t[i]-t[m]
			
			# Rearrange to put Elemtn nodes in first 2 pos
			if order > 1:
				temp = vel[1:num_t]
				vel[1] = vel[num_t]
				vel[2:num_t+1] = temp
				
			# Assemble Second row of H
			h_temp = [vel[i]*h_init for i in range(num_t)]
			h[1, :] = h_temp
			
			# Second order derivative:
			
			# Compute second-order time derivatives
			acc = []
			if order > 1:
				for i in range(num_t):
					acc.append(0)
					for j in range(num_t):
						if j != i:
							for k in range(num_t):
								if not k in [i, j]:
									acc_temp = 1
									for m in range(num_t):
										if not m in [k, j, i]:
											acc_temp *= e[2] - t[m]
									acc[i] += acc_temp
					for n in range(num_t):
						if n != i:
							acc[i] /= t[i]-t[n]
				temp = acc[1:num_t]
				acc[1] = acc[num_t]
				acc[2:num_t] = temp
				
				h_temp = [acc[i]*h_init for i in range(num_t)]
				
				h[2, :] = h_temp
		return(h)
		
	def __generateInd(self, size_nodal_mu, corner13theta, corner24theta, corner12mu, corner34mu, num_nodes):
		"""Generates the ind list based on nearest nodal points and the size of the nodal mu array
		
		args:
			size_nodal_mu (int): Size of nodal mu array
			corner13theta - corner34mu (ints): Indices of nearest nodal points
			num_nodes (int): Total number of nodes
		returns:
			ind (list): List of indices for nearest nodal points
		"""
		ind = []
		# Append nearest nodal points
		ind.append(size_nodal_mu*(corner13theta - 1) + corner12mu - 1)
		ind.append(size_nodal_mu*(corner24theta - 1) + corner12mu - 1)
		ind.append(size_nodal_mu*(corner13theta - 1) + corner34mu - 1)
		ind.append(size_nodal_mu*(corner24theta - 1) + corner34mu - 1)
		ind.append(ind[0] + num_nodes)
		ind.append(ind[1] + num_nodes)
		ind.append(ind[2] + num_nodes)
		ind.append(ind[3] + num_nodes)
		ind.append(ind[4] + num_nodes)
		ind.append(ind[5] + num_nodes)
		ind.append(ind[6] + num_nodes)
		ind.append(ind[7] + num_nodes)
		ind.append(ind[8] + num_nodes)
		ind.append(ind[9] + num_nodes)
		ind.append(ind[10] + num_nodes)
		ind.append(ind[11] + num_nodes)
		# Convert list from floats to ints
		ind = [int(ind_i) for ind_i in ind]
		return(ind)
		
	def _solveReducedSystem(self, lhs_matrix, global_rhs, c, num_dof_total, nn, num_dof_mesh, num_time_nodes=1):
		"""Use the constraints encoded in c to reduce linear system of equations and solve
		
		args:
			lhs_matrix (array): The left-hand side of the equation
			global_rhs (array): The right-hand side of the equation
			c (array): Array of constraints on the equation
			num_dof_total (int): Total number of degrees of freedom
			nn
			num_dof_mesh (int): Number of dofs in the mesh
			num_time_nodes (int): How many timepoints to use for each node
		returns:
			displacement (array): Solution to the equation between lhs and rhs
		"""
		# Set up the mapping vector v
		v = np.zeros([num_dof_total, 2])
		v[:, 0] = range(num_dof_total)
		v += 1
		
		# Arrange degrees of freedom
		m = c.shape[0]
		k = num_dof_total
		
		for i in range(m):
			for j in range(1,8,2):
				if c[i, j] > 0: # Coupling
					k -= num_time_nodes
					mult = c[i, j+1]
					search_flag = True
					target = c[i, j]
					# Cycle until you reach the end of a link
					while search_flag:
						p = np.where(target == c[:, 0])[0]
						if c[p, j] == -1:
							for tnn in range(num_time_nodes):
								v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 0] = num_dof_mesh*tnn+c[p,0]+(((j+1)/2)-1)*nn
								v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 1] = mult
							search_flag = False
						else:
							target = c[p, j]
							mult = mult*c[p, j+1]
				elif c[i, j] == 0: # Fixing
					k -= num_time_nodes
					for tnn in range(num_time_nodes):
						v[int(num_dof_mesh*tnn+c[i, 0]+(((j+1)/2)-1)*nn)-1, 0] = 0
		# Eliminate zeros and redundancies
		true_dof = np.unique(v[np.nonzero(v[:, 0]), 0][0])
		
		# Reduction and Solve:
		red_rhs = np.zeros(k)
		red_lhs = np.zeros([k, k])
		ind1 = 0
		
		for i in range(num_dof_total):
			p1 = v[i, 0]
			mult1 = v[i, 1]
			
			if p1 != 0: # Ensure p1 is unfixed
				if p1 == i+1: # Ensure p1 is a free degree of freedom
					red_rhs[ind1] += global_rhs[i]
					ind2 = 0
					for j in range(num_dof_total):
						p2 = v[j, 0]
						mult2 = v[j, 1]
						if p2 != 0: # Ensure p2 is unfixed
							if p2 == j+1: # Ensure p2 is a free degree of freedom
								red_lhs[ind1, ind2] += lhs_matrix[i, j]
								ind2 += 1
							else: # If p2 is coupled:
								m = np.where(p2 == true_dof)[0]
								red_lhs[ind1, m] += lhs_matrix[i, j]*mult2
					ind1 += 1
				else: # If p1 is coupled:
					m = np.where(p1 == true_dof)[0]
					red_rhs[m] += global_rhs[i]*mult1
					ind2 = 0
					for j in range(num_dof_total):
						p2 = v[j, 0]
						mult2 = v[j, 1]
						if p2 != 0: # Ensure p2 is unfixed
							if p2 == j+1: # Ensure p2 is a free degree of freedom
								red_lhs[m, ind2] += lhs_matrix[i, j]*mult1
								ind2 += 1
							else: # If p2 is coupled:
								n = np.where(p2 == true_dof)[0]
								red_lhs[m, n] += lhs_matrix[i, j]*mult1*mult2
		# Solving the reduced system
		red_disp = np.linalg.solve(red_lhs, red_rhs)
		
		# Reassemble
		displacement = []
		for i in range(num_dof_total):
			p = v[i, 0]
			mult = v[i, 1]
			if p != 0: # P is unfixed
				ind = np.where(p == true_dof)[0]
				displacement.append((red_disp[ind]*mult)[0])
			else: # P is fixed
				displacement.append(0)

		return(displacement)
		
	def _dofNodeWrite(self, write_file, read_file, dof, unsorted_nodal_mesh, nodal_mu, nodal_theta, num_nodes, size_nodal_mu, focus):
		"""Write the IPNODE file based on the calculate dof array
		
		args:
			write_file (string): Location of file to write.
			read_file (string): Location of template IPNODE file.
			dof (array): DOF array.
			unsorted_nodal_mesh (array)
			nodal_mu (float)
			nodal_theta (float)
			num_nodes (int)
			size_nodal_mu (int)
			focus (float)
		returns:
			dof_data (array): The dof_data array generated by the generateDofData function and used to write the IPNODE file.
		"""
		# Calculate the actual DOF data based on input values.
		dof_data = self._generateDofData(dof, num_nodes, nodal_mu, nodal_theta, unsorted_nodal_mesh, size_nodal_mu)
		# Open the write and read files and create the write file based on the template of the read file.
		#		Changes are made on specific files, pre-determined by practice.
		with open(write_file, 'w+') as fw:
			with open(read_file) as fr:
				i = -1
				for line in fr:
					if 'The Xj(1) coordinate' in line:
						i += 1
						fw.write(' The Xj(1) coordinate is [ 0.00000E+00]: {:12.5E} \n'.format(dof_data[i, 0]))
					elif 'The Xj(1) derivative wrt s(1) is' in line:
						fw.write(' The Xj(1) derivative wrt s(1) is [ 0.00000E+00]: {:12.5E} \n'.format(dof_data[i, 1]))
					elif 'The Xj(1) derivative wrt s(2) is' in line:
						fw.write(' The Xj(1) derivative wrt s(2) is [ 0.00000E+00]: {:12.5E} \n'.format(dof_data[i, 2]))
					elif 'The Xj(1) derivative wrt s(1) & s(2)' in line:
						fw.write(' The Xj(1) derivative wrt s(1) & s(2) is [ 0.00000E+00]: {:12.5E} \n'.format(dof_data[i, 3]))
					elif 'Specify the focus position' in line:
						fw.write(' Specify the focus position [1.0]: {:12.5E} \n'.format(focus))
					else:
						fw.write(line)
		return(dof_data)
		
	def _generateDofData(self, dof, num_nodes, nodal_mu, nodal_theta, unsorted_nodal_mesh, size_nodal_mu):
		"""Return the unified dof_data array based on the passed inputs.
		
		This dof_data array is what is used to generate the IPNODE output file.
		"""
		dof_data = []
		for i in range(num_nodes):
			# Set m1 as the number of points in the unsorted nodal mesh greater than nodal mu.
			m1 = np.where(nodal_mu <= unsorted_nodal_mesh[i, 0])[0].size
			# Set t as the number of points in the unsorted nodal mesh greater than nodal theta.
			t = np.where(nodal_theta <= unsorted_nodal_mesh[i, 1])[0].size
			# Calculate indices and used those to pull from input dof and append to dof data.
			lam_ind = size_nodal_mu * (t-1) + m1 - 1
			d1 = lam_ind + num_nodes
			d2 = d1 + num_nodes
			d3 = d2 + num_nodes
			dof_data.append([dof[lam_ind], dof[d1], dof[d2], dof[d3]])
		return(np.array(dof_data))
		
	def __calcCornerThetaMu(self, z2, z3, nodal_theta, size_nodal_theta, max_nodal_theta, nodal_mu, min_nodal_mu, max_nodal_mu):
		"""Find the 4 nearest nodal points
		Uses a convention from Hashima et al
		
		args:
			z2 (float): mu
			z3 (float): theta
			nodal_theta (array): Theta values for each node
			size_nodal_theta (int): Size of the nodal theta array
			max_nodal_theta (float): Maximum value in nodal theta
			nodal_mu (array): Mu values for each node
			min_nodal_mu (float): Minimum value in nodal mu
			max_nodal_mu (float): Maximum value in nodal mu
		"""
		e = [0, 0]
		# Special theta cases
		if z3 >= 2*math.pi: # If theta is above 2pi
			z3 -= 2*math.pi
			t13 = nodal_theta[1]
			t24 = nodal_theta[0]
			e[0] = (t13 - z3)/(t13 - t24)
			# Indices for theta
			corner13theta = 2
			corner24theta = 1
		elif z3 >= max_nodal_theta: # If theta = 0 is e[0] = 0
			t13 = 2*math.pi
			t24 = max_nodal_theta
			e[0] = (t13 - z3)/(t13 - t24)
			# Indices for theta
			corner13theta = 1
			corner24theta = size_nodal_theta
		else: # General theta case
			min_t = np.where(nodal_theta <= z3)[0].size
			t13 = nodal_theta[min_t]
			t24 = nodal_theta[min_t-1]
			e[0] = (t13 - z3) / (t13 - t24)
			# Indices for theta
			corner13theta = min_t + 1
			corner24theta = min_t
		# Special mu cases
		if z2 == min_nodal_mu: # Smallest mu value possible
			corner12mu = 1
			corner34mu = 2
		elif z2 == max_nodal_mu: # Largest mu value possible
			corner12mu = size_nodal_mu - 1
			corner34mu = size_nodal_mu
		else: # General mu case
			min_m = np.where(nodal_mu < z2)[0].size
			m12 = nodal_mu[min_m - 1]
			m34 = nodal_mu[min_m]
			e[1] = (z2 - m12)/(m34 - m12)
			corner12mu = min_m
			corner34mu = min_m + 1
		return([e, corner13theta, corner24theta, corner12mu, corner34mu])
		
	def _biCubicInterp(self, x_data, y_data, node_data, scale_der=0):
		"""Interpolate the x and y data to a bicubic fit
		
		args:
			x_data (array): x values to fit
			y_data (array): y values to fit
			node_data (array): Nodal values to use in the fit
			scale_der (int)
		
		returns:
			a (array): The interpolated values in order lambda, mu, theta, derivatives
		"""
		# Pull the mu and theta vectors from x, y, and node arrays
		new_mu_vec = x_data[0, :]
		new_theta_vec = y_data[:, 0]
		old_mu_vec = node_data[:, 0, 1]
		old_theta_vec = node_data[0, :, 2]
		
		# Pull the sizes of the arrays established earlier
		size_new_mu = new_mu_vec.size
		size_new_theta = new_theta_vec.size
		size_old_mu, size_old_theta, n = node_data.shape
		rads = math.pi/180
		
		# Set up A (output matrix of lambda, mu, theta, and derivatives, same order as node_data)
		a = np.zeros([size_new_mu, size_new_theta, n])
		a[:, :, 1] = np.transpose(x_data)
		a[:, :, 0] = np.transpose(y_data)
		
		for i in range(size_new_mu):
			for j in range(size_new_theta):
				# Pull the current mu and theta
				mu = new_mu_vec[i]
				theta = new_theta_vec[j]
				# Determine if you are currently at a nodal position
				found_nodal_point = False
				for k in range(size_old_mu):
					for m in range(size_old_theta):
						# If both mu and theta are in the old vectors, at a nodal point
						if (mu == old_mu_vec[k]) and (theta == old_theta_vec[m]):
							# Store the data in the a array if at a nodal point
							a[i, j, 2] = node_data[k, m, 0]
							a[i, j, 3:n] = node_data[k, m, 3:n]
							found_nodal_point = True
							break
				
				# If not at nodal position, do interpolation
				if not found_nodal_point:
					if i == 0:
						# Set min_t to the number of points in the old theta vector less than current theta
						min_t = np.where(old_theta_vec <= theta)[0].size
						# Set corner array based on nodal data
						corner = np.array([node_data[0, min_t, :], node_data[0, min_t-1, :], node_data[1, min_t, :], node_data[1, min_t, :]])
						# Calculate e array based on corner and theta values
						e = [(corner[0, 2]-theta)/(corner[0,2]-corner[1,2]), 0]
					elif j == (size_new_theta-1):
						# Set min_m to the number of point in the old mu vector less than current mu
						min_m = np.where(old_mu_vec < mu)[0].size
						# Set corner array based on nodal data
						corner = np.array([node_data[min_m-1, 1, :], node_data[min_m-1, 0, :], node_data[min_m, 1, :], node_data[min_m, 0, :]])
						# Calculate e array based on corner and mu values
						e = [1, (mu-corner[1,1])/(corner[3,1]-corner[1,1])]
					else:
						# Set min_t and min_m based on number of points in respective arrays less than current values
						min_t = np.where(old_theta_vec <= theta)[0].size
						min_m = np.where(old_mu_vec < mu)[0].size
						# Set corner array based on nodal data
						corner = np.array([node_data[min_m-1, min_t, :], node_data[min_m-1, min_t-1, :], node_data[min_m, min_t, :], node_data[min_m, min_t-1, :]])
						# Calculate e array based on corner, theta, and mu values
						e = [(corner[0, 2]-theta)/(corner[0, 2]-corner[1, 2]), (mu-corner[1, 1])/(corner[3, 1]-corner[1, 1])]
					
					if scale_der > 0:
						# Fill out the rest of the corner array
						corner[:, 3] *= (corner[0, 2]-corner[1, 2])*rads
						corner[:, 4] *= (corner[3, 1]-corner[1, 1])*rads
						corner[:, 5] *= (math.pi/180)*(corner[0, 2]-corner[1, 2])*(corner[3, 1]-corner[1, 1])*rads
					# Set a array points equal to lambda
					a[i, j, 2:n] = self._getLambda(corner, e)
					
					if scale_der > 0:
						# Modify the a array values based on corner values
						a[i, j, 3] /= (corner[0,2] - corner[1,2])*rads
						a[i, j, 4] /= (corner[3,1] - corner[1,1])*rads
						a[i, j, 5] /= (corner[0,2] - corner[1,2])*(corner[3,1]-corner[1,1])*(math.pi/180)*rads
		# Flip the a array first and third slices				
		temp = np.copy(a[:, :, 0])
		a[:, :, 0] = a[:, :, 2]
		a[:, :, 2] = temp
		return(a)
					
	def _getLambda(self, c, e):
		"""Compute Lambda Values and Derivatives by Cubic Hermite Function"""
		
		l = []
		
		h00 = [1 - 3*(e_i**2) + 2*(e_i**3) for e_i in e]
		h10 = [e_i*((e_i-1)**2) for e_i in e]
		h01 = [(e_i**2)*(3-2*e_i) for e_i in e]
		h11 = [(e_i**2)*(e_i-1) for e_i in e]
		
		dh00 = [6*((e_i**2)-e_i) for e_i in e]
		dh10 = [3*(e_i**2)-4*e_i+1 for e_i in e]
		dh01 = [6*(e_i-e_i**2) for e_i in e]
		dh11 = [3*(e_i**2)-2*e_i for e_i in e]
		
		l1 = h00[0]*h00[1]*c[0,0] + h01[0]*h00[1]*c[1,0] + h00[0]*h01[1]*c[2,0] + h01[0]*h01[1]*c[3,0]
		l2 = h10[0]*h00[1]*c[0,3] + h11[0]*h00[1]*c[1,3] + h10[0]*h01[1]*c[2,3] + h11[0]*h01[1]*c[3,3]
		l3 = h00[0]*h10[1]*c[0,4] + h01[0]*h10[1]*c[1,4] + h00[0]*h11[1]*c[2,4] + h01[0]*h11[1]*c[3,4]
		l4 = h10[0]*h10[1]*c[0,5] + h11[0]*h10[1]*c[1,5] + h10[0]*h11[1]*c[2,5] + h11[0]*h11[1]*c[3,5]
		l = np.append(l, l1 + l2 + l3 + l4)
		
		dl1wrt1 = dh00[0]*h00[1]*c[0,0] + dh01[0]*h00[1]*c[1,0] + dh00[0]*h01[1]*c[2,0] + dh01[0]*h01[1]*c[3,0]
		dl2wrt1 = dh10[0]*h00[1]*c[0,3] + dh11[0]*h00[1]*c[1,3] + dh10[0]*h01[1]*c[2,3] + dh11[0]*h01[1]*c[3,3]
		dl3wrt1 = dh00[0]*h10[1]*c[0,4] + dh01[0]*h10[1]*c[1,4] + dh00[0]*h11[1]*c[2,4] + dh01[0]*h11[1]*c[3,4]
		dl4wrt1 = dh10[0]*h10[1]*c[0,5] + dh11[0]*h10[1]*c[1,5] + dh10[0]*h11[1]*c[2,5] + dh11[0]*h11[1]*c[3,5]
		l = np.append(l, dl1wrt1 + dl2wrt1 + dl3wrt1 + dl4wrt1)
		
		dl1wrt2 = h00[0]*dh00[1]*c[0,0] + h01[0]*dh00[1]*c[1,0] + h00[0]*dh01[1]*c[2,0] + h01[0]*dh01[1]*c[3,0]
		dl2wrt2 = h10[0]*dh00[1]*c[0,3] + h11[0]*dh00[1]*c[1,3] + h10[0]*dh01[1]*c[2,3] + h11[0]*dh01[1]*c[3,3]
		dl3wrt2 = h00[0]*dh10[1]*c[0,4] + h01[0]*dh10[1]*c[1,4] + h00[0]*dh11[1]*c[2,4] + h01[0]*dh11[1]*c[3,4]
		dl4wrt2 = h10[0]*dh10[1]*c[0,5] + h11[0]*dh10[1]*c[1,5] + h10[0]*dh11[1]*c[2,5] + h11[0]*dh11[1]*c[3,5]
		l = np.append(l, dl1wrt2 + dl2wrt2 + dl3wrt2 + dl4wrt2)
		
		dl1wrt12 = dh00[0]*dh00[1]*c[0,0] + dh01[0]*dh00[1]*c[1,0] + dh00[0]*dh01[1]*c[2,0] + dh01[0]*dh01[1]*c[3,0]
		dl2wrt12 = dh10[0]*dh00[1]*c[0,3] + dh11[0]*dh00[1]*c[1,3] + dh10[0]*dh01[1]*c[2,3] + dh11[0]*dh01[1]*c[3,3]
		dl3wrt12 = dh00[0]*dh10[1]*c[0,4] + dh01[0]*dh10[1]*c[1,4] + dh00[0]*dh11[1]*c[2,4] + dh01[0]*dh11[1]*c[3,4]
		dl4wrt12 = dh10[0]*dh10[1]*c[0,5] + dh11[0]*dh10[1]*c[1,5] + dh10[0]*dh11[1]*c[2,5] + dh11[0]*dh11[1]*c[3,5]
		l = np.append(l, dl1wrt12 + dl2wrt12 + dl3wrt12 + dl4wrt12)

		return(l)
		
	def __getAngleRange(self, angles):
		"""Get the leftmost and rightmost values from a passed series of angles.
		
		The angles should be circular, with a total possible range of 2*pi. The purpose of this function
		is to allow finding the circular extent of angles that cross the origin point.
		
		args:
			angles (float arr-like): The angles for which ranges are being determined.
		returns:
			angle_min (float): The "minimum" angle (clockwise)
			angle_max (float): The "maximum" angle (clockwise)
			direction (bool): True if the scar does not pass through the origin (angle flip point)
		"""
		# Get maximum and minimum angle values and subtract
		angle_max = np.max(angles)
		angle_min = np.min(angles)
		init_range = angle_max - angle_min
		# If the range is less than 2*pi, you don't cross the origin
		if init_range < 6:
			direction = True
			return([angle_min, angle_max, direction])
		else:
			# Sort angles from minimum -> maximum, append initial value to the end, increased by 2*pi
			angles_sorted = np.sort(angles)
			angles_sorted = np.append(angles_sorted, angles_sorted[0] + 2*math.pi)
			# Calculate the moving differential
			angles_diff = [angles_sorted[i+1] - angles_sorted[i] for i in range(len(angles))]
			# The "true minimum" (most counterclockwise angle) is immediately after the largest gap
			angle_min = angles_sorted[np.argmax(angles_diff, axis=0) + 1]
			angle_max = angles_sorted[np.argmax(angles_diff, axis=0)]
			# Determine directionality to ensure that scar values are between the minimum and maximum appropriately
			direction = np.all(np.bitwise_and(angles >= angle_min, angles <= angle_max))
		return([angle_min, angle_max, direction])
		
	def _cartToPol(self, x, y):
		"""Convert cartesian (x, y) coordinates to polar (theta, rho) coordinates"""
		rho = np.sqrt(np.square(x) + np.square(y))
		theta = np.arctan2(y,x)
		theta = np.where(theta < 0, theta + 2*np.pi, theta)
		return np.array([theta, rho])
		
	def _polToCart(self, theta, rho):
		"""Convert polar (theta, rho) coordinates to cartesian (x, y) coordinates"""
		x = rho*np.cos(theta)
		y = rho*np.sin(theta)