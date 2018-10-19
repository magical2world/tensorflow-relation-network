import os
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate

target_folder=[]
num_class=0
for python_folder in os.listdir('omniglot-master'):
	if os.path.isdir(os.path.join('omniglot-master',python_folder)):
		for image_folder in os.listdir(os.path.join('omniglot-master',python_folder)):
			if (image_folder=='images_background')or(image_folder=='images_evaluation'):
				for class_folder in os.listdir(os.path.join('omniglot-master',python_folder,image_folder)):
					if os.path.isdir(os.path.join('omniglot-master',python_folder,image_folder,class_folder)):
						for char_folder in os.listdir(os.path.join('omniglot-master',python_folder,image_folder,class_folder)):
							target_folder.append(os.path.join('omniglot-master',python_folder,image_folder,class_folder,char_folder))

for folder in target_folder:
	for image_file in os.listdir(folder):
		image_name=os.path.join(folder,image_file)
		image=imread(image_name)
		image_90=rotate(image,90)
		imsave(image_name[:-4]+'left.png',image_90)
		image_180=rotate(image,180)
		imsave(image_name[:-4]+'down.png',image_180)
		image_270=rotate(image,270)
		imsave(image_name[:-4]+'right.png',image_270)






