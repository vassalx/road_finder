import h5py
import numpy as np
import os
import cv2
import glob
save_path = './numpy.hdf5'
img_path = '../mass_roads/mass_roads/Images/'
print('image size: %d bytes'%os.path.getsize(img_path))
hf = h5py.File(save_path, 'a') # open a hdf5 file
img_np = np.array([cv2.imread(path) for path in glob.glob(img_path+'*.tiff')])

print(img_np.shape)

dset = hf.create_dataset('default', data=img_np, compression="gzip", compression_opts=4)  # write the data to hdf5 file
hf.close()  # close the hdf5 file
print('hdf5 file size: %d bytes'%os.path.getsize(save_path))