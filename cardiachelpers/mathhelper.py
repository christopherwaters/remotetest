import numpy as np
import scipy as sp

def pol2cart(theta, rho):
	"""Convert polar (theta, rho) coordinates to cartesian (x, y) coordinates"""
	x = rho * np.cos(theta)
	y = rho * np.sin(theta)
	return ([x, y])

def cart2pol(x, y):
	"""Convert cartesian (x,y) coordinates to polar (theta, rho) coordinates"""
	rho = np.sqrt(np.square(x) + np.square(y))
	theta = np.arctan2(y,x)
	theta = np.where(theta < 0, theta + 2*np.pi, theta)
	return np.array([theta, rho])

def calcNorm(arr_in):
	"""Calculates the norm of the passed array as the square root of the sum of squares"""
	norm = np.sqrt(np.sum(np.square(arr_in)))
	return norm

def findMidPt(endo_pins, time_id, septal_slice, endo_x, endo_y):
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
	polar_coords = cart2pol(cur_slice[0,:], cur_slice[1,:])[:,1:]
	
	# Get the theta values for the perpindicular line (polar theta)
	perp_dot = np.dot([1, perp_slope], [1, 0])
	perp_norm = calcNorm([1, 1*perp_slope])
	th1 = np.arccos(perp_dot/perp_norm)
	th2 = th1 + np.pi;
	
	# Calculate the rho values for the two theta values by interpolation
	r_interp = sp.interpolate.interp1d(polar_coords[0,:], polar_coords[1,:])
	r1 = r_interp(th1)
	r2 = r_interp(th2)
	r = r1 if r1<r2 else r2
	theta = th1 if r1<r2 else th2
	
	# Reconvert the interpolated rho and theta to cartesian
	mid_pt = (pol2cart(theta, r) + mean_pt.reshape([1,2])).reshape(2)
	return(mid_pt)