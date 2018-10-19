import os
from skimage.io import imread
attributes_dir='CUB/CUB_200_2011' \
	'/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'

attributes=[]
with open(attributes_dir) as f:
	for line in f:
		attributes.append(line)

# print(len(attributes))
# image_dir='CUB/CUB_200_2011' \
# 	'/CUB_200_2011/images'
#
# image=[]
# for folder in os.listdir(image_dir):
# 	for file in os.listdir(image_dir+'/'+folder):
# 		image.append(imread(image_dir+'/'+folder+'/'+file))
# print(len(image))