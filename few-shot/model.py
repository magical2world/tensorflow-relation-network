import tensorflow as tf
import numpy as np
from utils import next_batch
from skimage.io import imread
from skimage.transform import resize

class relation_network():
	def __init__(self):
		self.sample_image=tf.placeholder(tf.float32,[5,28,28,1])
		self.query_image=tf.placeholder(tf.float32,[19*5,28,28,1])
		self.target=tf.placeholder(tf.int64,[19*5])
		self.mode=tf.placeholder_with_default(False,shape=[])
	def conv_block(self,feature_in,filters,kernel_size,padding='same'):
		conv=tf.layers.conv2d(feature_in,filters=filters,kernel_size=kernel_size,padding=padding)
		bn=tf.layers.batch_normalization(conv,training=self.mode)
		return tf.nn.relu(bn)
	def embedding(self,image_in):
		conv1=self.conv_block(image_in,64,3,padding='valid')
		pool1=tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)
		conv2=self.conv_block(pool1,64,3,padding='valid')
		pool2=tf.layers.max_pooling2d(conv2,pool_size=2,strides=2)
		conv3=self.conv_block(pool2,64,3)
		conv4=self.conv_block(conv3,64,3)
		return conv4
	def feature_concatenation(self):
		sample_feature=self.embedding(self.sample_image)
		sample_feature=tf.expand_dims(sample_feature,[0])
		sample_feature=tf.tile(sample_feature,[19*5,1,1,1,1])
		query_feature=self.embedding(self.query_image)
		query_feature=tf.expand_dims(query_feature,[0])
		query_feature=tf.tile(query_feature,[5,1,1,1,1])
		query_feature=tf.transpose(query_feature,[1,0,2,3,4])
		concatenation_map=tf.concat([sample_feature,query_feature],axis=-1)
		concatenation_map=tf.reshape(concatenation_map,[-1,5,5,128])
		# print(concatenation_map)
		return concatenation_map
	def relation(self):
		feature_map=self.feature_concatenation()
		conv_rela1=self.conv_block(feature_map,64,3)
		pool_rela1=tf.layers.max_pooling2d(conv_rela1,pool_size=2,strides=2)
		conv_rela2=self.conv_block(pool_rela1,64,3)
		pool_rela2=tf.layers.max_pooling2d(conv_rela2,pool_size=2,strides=2)
		fc1=tf.layers.flatten(pool_rela2)
		fc2=tf.layers.dense(fc1,8,activation=tf.nn.relu)
		fc3=tf.layers.dense(fc2,1,activation=tf.nn.sigmoid)
		return tf.reshape(fc3,[-1,5])
	def read_image(self,file):
		images=[]
		for file_name in file:
			image=imread(file_name)/255.0
			image=resize(image,(28,28))
			image=(image-0.92206)/0.08426
			images.append(np.expand_dims(image,axis=-1))
		return images
	def accuracy(self,predict,label):
		correct=tf.equal(tf.argmax(predict,1),label)
		return tf.reduce_mean(tf.cast(correct,tf.float32))
	def train(self):
		logits=self.relation()
		mse_loss=tf.reduce_mean(tf.square(tf.one_hot(self.target,5)-logits))
		optimizer=tf.train.AdamOptimizer(0.001).minimize(mse_loss)
		accuracy=self.accuracy(logits,self.target)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for step in range(1000):
				sample_image,query_image,target=next_batch(5,1,19)
				sample_image=self.read_image(sample_image)
				query_image=self.read_image(query_image)
				idx=np.arange(0,len(query_image))
				np.random.shuffle(idx)
				query_image=np.array(query_image)[idx]
				target=np.array(target)[idx]
				loss,_,acc=sess.run([mse_loss,optimizer,accuracy],feed_dict={self.sample_image:sample_image,
																self.query_image:query_image,
																self.target:target,
																self.mode:True})
				if step%10==0:
					print("number step %d,loss is %f"%(step,loss))
					print("number step %d,accuracy is %f"%(step,acc))
