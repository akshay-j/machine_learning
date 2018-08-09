from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import data
from skimage import transform
from skimage.color import rgb2gray

# the below function will create train/test data with images and labels
def load_data(data_dir):
	sub_dirs=[d for d in os.listdir(data_dir)
		if os.path.isdir(os.path.join(data_dir, d))]

	# labels and images
	labels = []
	images = []
	
	# extract image files and labels
	for d in sub_dirs:
		label_dir = os.path.join(data_dir, d)	# extract directory
		filenames = [os.path.join(label_dir, f)
			for f in os.listdir(label_dir)
			if f.endswith(".ppm")]
	
		# create list of file names and labels
		for f in filenames:
			images.append(data.imread(f))
			labels.append(int(d))

	# return images with their labels
	return labels, images


# resizing and grayscaling
def prepare_image_for_network(images):
	# resizing the image to 28 X 28
	image28 = [transform.resize(image, (28, 28)) for image in images]

	# converting list type images to array type
	image28 = np.array(image28)

	# converting to grayscale
	image28 = rgb2gray(image28)

	return image28
