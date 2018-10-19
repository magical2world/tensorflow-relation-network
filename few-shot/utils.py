import os
import random
import numpy as np

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

train_folder=target_folder[:1200]
test_folder=target_folder[1200:]

def dev2onehot(idx,way):
	onehot=np.zeros(way)
	onehot[idx]=1
	return onehot

def next_batch(way,shot,query,mode='train'):
	sample_image=[]
	query_image=[]
	target=[]
	if mode=='train':
		batch_folder=random.sample(train_folder,way)
	else:
		batch_folder=random.sample(test_folder,way)

	for idx,folder in enumerate(batch_folder):
		image_files=[]
		for file in os.listdir(folder):
			image_files.append(os.path.join(folder,file))
		random.shuffle(image_files)
		sample_image.append(image_files[:shot][0])
		for num in range(shot,shot+query):
			query_image.append(image_files[num])
			target.append(idx)
	return sample_image,query_image,list(np.int32(target))


# sample,query,target=next_batch(5,1,19)
# print(len(target))


