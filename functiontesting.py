import mrimodel
import confocalmodel
import mesh
import numpy as np
import warnings
from cardiachelpers import displayhelper

mesh_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/LVGEOM_8x4_noshift.mat'
feb_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/temp_feb.feb'

num_rings = 24
elem_per_ring = 48
elem_in_wall = 5
mesh_type = '4x8'
time_point = 0

mri_mesh = mesh.Mesh(num_rings, elem_per_ring, elem_in_wall)
mri_mesh.importPremadeMesh(mesh_filename)
mri_mesh.generateFEFile(feb_filename)
#displayhelper.displayMeshPostview(feb_filename)

sa_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/SA_LGE_Scar_Pnpts.mat'
la_pinpt_filename = 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_2CH_Pnpts.mat'
la_lge_filenames = ['C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_2CH_Scar.mat', 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_3CH_Scar.mat', 'C:/Users/cdw2be/Downloads/Code4Chris/data/LA_LGE_4CH_Scar.mat']

mri_model = mrimodel.MRIModel(sa_filename, la_pinpt_filename, sa_scar_file=sa_filename, la_scar_files=la_lge_filenames)
mri_model.importCine()
mri_model.importLGE()
mri_model.importScarLA()

mri_model.convertDataProlate(mri_mesh.focus)
mri_mesh.rotateNodesProlate()

mri_model.alignScar()
test_interp_return = mri_mesh.interpScarData(mri_model.interp_data)

'''
mri_model = mrimodel.MRIModel(sa_filename, la_filename, scar_file=lge_filename)
mri_model.importCine(timepoint=0)
mri_mesh.fitContours(mri_model.cine_endo[time_point], mri_model.cine_epi[time_point], mri_model.cine_apex_pt, mri_model.cine_basal_pt, mri_model.cine_septal_pts, mesh_type)
mri_mesh.feMeshRender()
mri_mesh.nodeNum(mri_mesh.meshCart[0], mri_mesh.meshCart[1], mri_mesh.meshCart[2])
mri_mesh.getElemConMatrix()
'''
# Splitting
'''
dense_filenames = ['C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-DENSE-Z5_4.mat', 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/E67-D28-DENSE-Z6_4.mat']

mri_model = mrimodel.MRIModel(sa_filename, la_filename, scar_file=lge_filename)
mri_model.importCine(timepoint=0)
mri_model.importLGE()
with warnings.catch_warnings():
	warnings.simplefilter('ignore')
	for i in range(len(mri_model.cine_endo)):
		mri_model.alignScarCine(timepoint=i)
mri_model.importDense()
mri_model.alignDense(cine_timepoint=0)


mri_mesh.fitContours(mri_model.cine_endo[time_point], mri_model.cine_epi[time_point], mri_model.cine_apex_pt, mri_model.cine_basal_pt, mri_model.cine_septal_pts, mesh_type)
mri_mesh.feMeshRender()
mri_mesh.nodeNum(mri_mesh.meshCart[0], mri_mesh.meshCart[1], mri_mesh.meshCart[2])
mri_mesh.getElemConMatrix()
mri_mesh.assignScarElems(mri_model.aligned_scar[time_point], conn_mat = 'hex')
mri_mesh.assignDenseElems(mri_model.dense_aligned_pts, mri_model.dense_slices, mri_model.dense_aligned_displacement, mri_model.radial_strain, mri_model.circumferential_strain)

dense_elems = np.where(~np.isnan(mri_mesh.dense_radial_strain[:, 0]))[0]

mesh_axes = displayhelper.surfaceRender(mri_mesh.endo_node_matrix, mri_mesh.focus)
mesh_axes = displayhelper.surfaceRender(mri_mesh.epi_node_matrix, mri_mesh.focus, mesh_axes)

dense_nodes = np.unique(mri_mesh.hex[dense_elems, :])

mesh_axes = displayhelper.nodeRender(mri_mesh.nodes[dense_nodes, :], mesh_axes)
'''