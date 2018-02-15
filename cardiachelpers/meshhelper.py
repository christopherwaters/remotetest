import numpy as np
import math

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