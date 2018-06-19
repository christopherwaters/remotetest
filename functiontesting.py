import mrimodel
import confocalmodel
import mesh
import numpy as np
import warnings

sa_filename = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-Chris-PinPts.mat'
la_filename = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-LAPinPts.mat'
lge_filename = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-Scar.mat'
dense_filenames = ['C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-DENSE-Z5_4.mat', 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-DENSE-Z6_4.mat']

mri_model = mrimodel.MRIModel(sa_filename, la_filename, scar_file=lge_filename, dense_file=dense_filenames)
mri_model.importCine(timepoint=0)
mri_model.importLGE()
with warnings.catch_warnings():
	warnings.simplefilter('ignore')
	for i in range(len(mri_model.cine_endo)):
		mri_model.alignScarCine(timepoint=i)
mri_model.importDense()
mri_model.alignDense(cine_timepoint=0)

num_rings = 14
elem_per_ring = 25
elem_in_wall = 5
mesh_type = '4x2'
time_point = 0

mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
mri_mesh.fitContours(mri_model.cine_endo[time_point], mri_model.cine_epi[time_point], mri_model.cine_apex_pt, mri_model.cine_basal_pt, mri_model.cine_septal_pts, mesh_type)
mri_mesh.feMeshRender()
mri_mesh.nodeNum(mri_mesh.meshCart[0], mri_mesh.meshCart[1], mri_mesh.meshCart[2])
mri_mesh.getElemConMatrix()
mri_mesh.assignScarElems(mri_model.aligned_scar[time_point], conn_mat = 'hex')
mri_mesh.assignDenseElems(mri_model.dense_aligned_pts, mri_model.dense_slices, mri_model.dense_aligned_displacement, mri_model.radial_strain, mri_model.circumferential_strain)