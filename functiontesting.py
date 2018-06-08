from cardiachelpers import confocalhelper
from cardiachelpers import lcn
import numpy as np
from PIL import Image

for slice_num in range(204):
	
	image_file = 'C:/Users/cdw2be/Documents/LCNTest/Slice 3/HeartPIMicrospheres-031418-Slice3 (raw tile '+str(slice_num+1) + ').tif'

	lcn_test_arr = np.array(confocalhelper.openModelImage(image_file)[19])

	lcn_test_kernel = lcn.create_kernel(25, 25)
	lcn_post_arr = lcn.lcn(lcn_test_arr, lcn_test_kernel)
	lcn_post_image = Image.fromarray(lcn_post_arr)
	file_name = 'C:/Users/cdw2be/Documents/LCNTest/SingleFrame/Frame'+str(slice_num+1)+'.tif'
	lcn_post_image.save(file_name)