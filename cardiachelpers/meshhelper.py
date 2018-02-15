import numpy as np
import scipy as sp
import math
from cardiachelpers import mathhelper
from matplotlib import path

def prepData(all_data_endo, all_data_epi, apex_pt, basal_pt, septal_pts):
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
	
	calcNorm = lambda arr_in: np.sqrt(np.sum(np.square(arr_in)))
	
	# Calculate first basis vector, (Base - Apex)/Magnitude
	c = apex_pt - basal_pt
	c_norm = calcNorm(c)
	e1_basis = [c1 / c_norm for c1 in c]
	
	# Calculate Origin Location
	origin = basal_pt + c/3
	
	# Calculate Focus Length based on c_norm
	focus = (2*c_norm/3)/(math.cosh(1))
	num_points = endo.shape[0]
	
	# Calculate Second basis vector using plane intersects, septal point, and e1
	d1 = septal_pts[0, :] - origin
	d2 = d1 - [np.dot(d1, e1_basis)*e1_elem for e1_elem in e1_basis]
	e2_basis = d2 / calcNorm(d2)
	
	# Calculate third basis vector from the first 2 basis vectors
	e3 = np.cross(e1_basis, e2_basis)
	e3_basis = e3 / calcNorm(e3)
	
	# Set up transform basis from the 3 calculated basis vectors
	transform_basis = np.array([e1_basis, e2_basis, -e3_basis])
	
	# Set up the modified endo and epi contours
	data_endo = np.dot((endo - np.array([origin for i in range(num_points)])), np.transpose(transform_basis))
	data_epi = np.dot((epi - np.array([origin for i in range(num_points)])), np.transpose(transform_basis))
	
	# Append extra identifying data to the modified contours
	data_endo = np.append(data_endo, np.reshape(all_data_endo[:, 3], [all_data_endo.shape[0], 1]), axis=1)
	data_epi = np.append(data_epi, np.reshape(all_data_epi[:, 3], [all_data_epi.shape[0], 1]), axis=1)
	return([data_endo, data_epi, focus, transform_basis, origin])

def getLambda(c, e):
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
	
def nearestNodalPoints(z2, z3, nodal_theta, size_nodal_theta, max_nodal_theta, nodal_mu, min_nodal_mu, max_nodal_mu):
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
	
def generateInd(size_nodal_mu, corner13theta, corner24theta, corner12mu, corner34mu, num_nodes):
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
	
def assignRegionNodes(nodes, region_contour, contour_edge_spacing, num_slices, focus):
	"""Generalized method to label which elements fall within a region contour.
	
	Region contour should be spaced evenly per slice, with an equal number of points on the outer layer and inner layer. The point distance between the circumferential extents of the region should be given with contour edge spacing.
	"""
	# Determine mu error buffer should be used
	err_val = 0
	
	# Convert nodes to cartesian for measurement
	nodes_prol_mu, nodes_prol_nu, nodes_prol_phi = mathhelper.cart2prolate(nodes[:, 0], nodes[:, 1], nodes[:, 2], focus)
	nodes_prol = np.column_stack((nodes_prol_mu, nodes_prol_nu, nodes_prol_phi))
	
	# Convert region to prolate
	region_mu, region_nu, region_phi = mathhelper.cart2prolate(region_contour[:, 0], region_contour[:, 1], region_contour[:, 2], focus)
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