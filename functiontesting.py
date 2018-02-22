import glob
import os
import confocalmodel

confocal_dir = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/ConfocalTestLargest'

con_slice = confocalmodel.ConfocalSlice(confocal_dir)

con_slice.createStitchedImage(channel=0)