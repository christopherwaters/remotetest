import confocalmodel

confocal_dir = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/ConfocalTestLargest'

con_model = confocalmodel.ConfocalModel(confocal_dir)

con_model.generateStitchedImages([0, 1], sub_slices=list(range(19)), channel=0)