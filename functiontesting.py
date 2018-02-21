import glob
import os
confocal_dir = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/ConfocalTestLargest'
tif_files = glob.glob(os.path.join(confocal_dir, '*.tif'))
raw_images = [None]*len(tif_files)
for file_num in range(len(tif_files)):
	raw_images[file_num] = Image.open(tif_files[file_num])
	
im_locs, im_locs_dict = confocalhelper.getImagePositions(tif_files)
im_grid = confocalhelper.getImageGrid(tif_files, im_locs, im_locs_dict)

compressed_images = [None]*len(raw_images)
for image_num in range(len(raw_images)):
	image_frame_split = confocalhelper.splitImageFrames(raw_images[image_num])
	image_red_chan = confocalhelper.splitChannels(image_frame_split[10], pull_channel=0)
	compressed_images[image_num] = confocalhelper.compressImages(image_red_chan, image_scale=0.33)

stitched_image_file = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/testimage.tif'
confocalhelper.stitchImages(compressed_images, im_grid[:, 0], im_grid[:, 1], stitched_image_file)