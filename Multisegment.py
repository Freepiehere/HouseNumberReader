# Profiler Command:
# python -m cProfile -s time Multisegment.py > Stats.txt

from __future__ import division, print_function, absolute_import
from random import randint
from threading import Thread, Event
from queue import Queue
from matplotlib import style

import tensorflow as tf
import numpy as np
import scipy.ndimage as sio
import matplotlib.pyplot as plt

import logging
import cv2
import h5py
import tensorflow
import time
import sys

num_epoch = 4000
batch_size = 12
learning_rate = 0.01
display_step = 100

testing_filepath = "D:/Coding/HouseNumber/test"
training_filepath = "D:/Coding/HouseNumber/train"

def loadImg(filename):
    return cv2.imread(filename)

def showImg(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# Create connection to database
# Accepts:
#   - Database filepath
# Returns:
#   - Image sample datastructure (ds)
#   - Sample bounding box details (bbox)
def databaseConnection(filepath):
    print("Connecting to Database")
    arrays = {}
    f = h5py.File(filepath+'/digitStruct.mat','r')
    for k,v in f.items():
        arrays[k] = np.array(v)
    f_keys = list(arrays)
    ds = f[f_keys[-1]]
    ds_keys = list(ds.keys())
    bbox = ds[ds_keys[0]][...]
    return ds, bbox

# Retrieves variables from image samples:
# Accepts:
#   - Image sample datastructure (ds)
#   - Sample bounding box details (bbox)
#   - Index of desired sample (ind)
#   - Desired variable (variable) set to "all" by default
#       Options: "top", "height", "left", "width", "label", "all"
# Returns:
#   - Desired sample variable (variable) set to "all" by default
#       Options: "top", "height", "left", "width", "label", "all"
def retrieveVariables(ds,bbox,ind,variable="all"):
    # Note: Make preparations to allow multiple variables chosen without having to utilize "all". e.g. ["top","height"]
    variables = {'top':['top'],'height':['height'],'left':['left'],'width':['width'],'label':['label'],'all':['top','height','left','width','label']}
    desired = variables[variable]
    result = {var:list() for var in desired}
    grp = ds[bbox[ind][0]]
    if(grp['label'].shape[0]>1):
        for next_variable in desired:
            for k in range(len(np.asarray(grp[next_variable]))):
                result[next_variable].append(np.asarray(ds[np.asarray(grp[next_variable])[k][0]])[0][0])
    else:
        for next_variable in desired:
            result[next_variable].append(np.asarray(grp[next_variable])[0][0])
    return result

# Compares given list of variables (rows & cols) to gloabl variables (max_rows & max_cols)
# and overwrites global variables if given variabels exceed them, respectively.
# Accepts:
#   - list of float row lengths (rows)
#   - list of float column lengths (cols)
max_rows = 0
max_cols = 0
def networkDims(rows,cols):
    global max_rows
    global max_cols
    for n_rows,n_cols in zip(rows,cols):
        if(max_rows<n_rows):
            max_rows = n_rows
        if(max_cols<n_cols):
            max_cols = n_cols
    return

# Populates dictionary (datamap) with image index in database, and associated HDF5 reference object index
# Accepts:
#   - integer index of bounding box group (bbox_ind)
#   - list of integer labels of image samples contained within the bounding box group
def mapLabel(bbox_ind,labels,datamap):
    for label in labels:
        datamap[label].append((bbox_ind,labels.index(label)))
    return datamap

# Converts integer label of training batch into a onehot label array with dims (batch_size,10)
def label_oneHot(label):
    if not label:
        return np.zeros([batch_size,10])
    onehot_labels = np.zeros([batch_size,10])
    for i in range(len(onehot_labels)):
        onehot_labels[i,label-1] = 1.
    return onehot_labels

def oneHot_label(oneHot):
    oneHot = oneHot[0]
    label = np.where(oneHot>0.99)[0][0]+1
    return int(label)

# Generate and populate a datamap dictionary whose values are bbox and grp indices of an image sample, 
# and keys are the labels associated with the image samples.
# Accepts:
#   - HDF5 datastructure (ds)
#   - HDF5 bounding box dataset (bbox)
# Returns:
#   - datamap dictionary
def prepareData(ds,bbox):
    print("Preparing Data")
    datamap = {i:list() for i in range(1,11)}
    N = len(bbox)
    for n in range(N):
        var = retrieveVariables(ds,bbox,n,'all')
        labels = var['label']
        rows, cols = var['height'], var['width']
        datamap = mapLabel(n,labels,datamap)
        networkDims(rows, cols)
    return datamap

def cropImg(img,ds,bbox,ind):
    var = retrieveVariables(ds,bbox,ind[0])
    top,height,left,width = var['top'][ind[1]],var['height'][ind[1]],var['left'][ind[1]],var['width'][ind[1]]
    
    sample = img[int(top):int(top)+int(height)+1,int(left):int(left)+int(width)+1]
    return sample

def getBatchSample(sample_ind,ds,bbox,filepath):
    img = loadImg(filepath + '/'+str(sample_ind[0]+1)+'.png') 
    sample = cropImg(img,ds,bbox,sample_ind)
    gray_sample = rgb_gray(sample)
    prepared_sample = resizeImage(gray_sample)
    return prepared_sample

def resizeImage(img):
    global max_rows
    global max_cols
    return cv2.resize(img,(int(max_rows),int(max_cols)))

def rgb_gray(img):
    try:
        img = img.astype(np.uint8)
        gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    except:
        showImg(img)
    return gray_img

def convModel(X):
    conv_1 = tf.layers.conv2d(
        inputs = X,
        filters = 32,
        kernel_size = (7,7),
        strides = (2,2),
        padding = 'same'
    )
    pool_1 = tf.layers.max_pooling2d(
        inputs = conv_1,
        pool_size = (2,2),
        strides = (1,1)
    )
    conv_2 = tf.layers.conv2d(
        inputs = pool_1,
        filters = 32,
        kernel_size = (3,3),
        strides = (2,2),
        padding = 'same'
    )
    pool_2 = tf.layers.max_pooling2d(
        inputs = conv_2,
        pool_size = (3,3),
        strides = (2,2)
    )
    conv_3 = tf.layers.conv2d(
        inputs = pool_2,
        filters = 32,
        kernel_size = (5,5),
        strides = (2,2),
        padding = 'same'
    )
    pool_3 = tf.layers.max_pooling2d(
        inputs = conv_3,
        pool_size = (2,2),
        strides = (1,1)
    )
    conv_4 = tf.layers.conv2d(
        inputs = pool_3,
        filters = 32,
        kernel_size = (5,5),
        strides = (1,1),
        padding = 'same'
    )
    pool_4 = tf.layers.max_pooling2d(
        inputs = conv_4,
        pool_size = (3,3),
        strides = (2,2)
    )
    conv_5 = tf.layers.conv2d(
        inputs = pool_4,
        filters = 16,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same'
    )
    flatten = tf.layers.flatten(
        inputs = conv_5
    )
    dense = tf.layers.dense(
        inputs = flatten,
        units = 100,
        activation = tf.nn.sigmoid
    )
    logits = tf.layers.dense(
        inputs = dense,
        units = 10,
        activation = tf.nn.sigmoid
    )
    return logits

class BatchGenerator(Thread):
    def __init__(self,datamap,batchsize,ds,bbox,filepath,queue):
        print("Init BatchGenerator")
        super(BatchGenerator,self).__init__()
        Thread.__init__(self)
        self.stop_event = Event()
        self.batch_q = queue
        self.datamap = datamap
        self.batchsize = batchsize
        self.ds = ds
        self.bbox = bbox
        self.filepath = filepath
        self.myseed = 0

    def stop(self):
        self.stop_event.set()

    # Assembles a batch of training samples, all of the same randomly generated label
    # Accepts:
    #   --
    # Returns:
    #   - Batch of training images and associated batch class label
    def requestBatch(self):
        batch_class = randint(1,10)
        sample_inds = self.datamap[batch_class]
        np.random.shuffle(sample_inds)
        sample_inds = sample_inds[0:self.batchsize]
        batch = np.asarray([getBatchSample(ind,self.ds,self.bbox,self.filepath) for ind in sample_inds])
        batch = np.expand_dims(batch,axis=3)
        print(batch.shape)
        quit()

        self.myseed += 1
        return (batch,label_oneHot(batch_class))

    def run(self):
        while self.myseed < num_epoch and not self.stop_event.is_set():
            if(not self.batch_q.full()):
                self.batch_q.put(self.requestBatch())
                
class BatchManager:
    def __init__(self,MAX_CAPACITY):
        print("Init BatchManager")
        self.batch_q = Queue(maxsize=MAX_CAPACITY)
        self.batchsize = batch_size
        #self.batchlimit
          
        self.ds_test, self.bbox_test = databaseConnection(testing_filepath)
        self.ds_train, self.bbox_train = databaseConnection(training_filepath)
        self.datamap_test = prepareData(self.ds_test,self.bbox_test)
        self.datamap_train = prepareData(self.ds_train,self.bbox_train)
        self.threads = []
        for x in range(3):
            print("Generating generators")
            generator = BatchGenerator(self.datamap_train,self.batchsize,self.ds_train,self.bbox_train,training_filepath,self.batch_q)
            generator.daemon = True
            self.threads.append(generator)
            generator.start()
            self.run()
        
    def next_batch(self):
        return self.batch_q.get()
    
    #def close(self,timeout=5):
    #    pass
        
    def run(self):
        import tensorflow as tf
        X = tf.placeholder(dtype=np.float64,shape=[self.batchsize,max_cols,max_rows,1])

        y_pred = convModel(X)
        y_true = tf.placeholder(dtype=y_pred.dtype,shape=[batch_size,10])

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels = y_true,
            logits = y_pred
        )
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        sess = tf.Session()
        sess.run(init_op)
        try:
            saver.restore(sess,"D:/Coding/HouseNumber/model.ckpt")
            print("Model Restored")
        except:
            print("Initializing new model")

        while(self.batch_q.empty()):
            continue
        # Begin Training
        print('Begin Training...')
        
        

        losses = {i:list() for i in range(1,11)}

        plt.ion()
        fig = plt.figure()
        #plt.axis([0,num_epoch,0,3])
        i=0
        x = list()
        colors = ['b','g','r','c','m','y','k','darkolivegreen','skyblue','darkorange']
        while(not self.batch_q.empty()):
            input_batch,input_class = self.next_batch()
            print(input_batch.shape)
            quit()
            _,l = sess.run([train_op,loss],feed_dict={X:input_batch,y_true:input_class})
            
            #if(i<20):
            #    input_batch,input_class= self.next_batch()        
            #    showImg(input_batch[0,:,:])
            #    print(oneHot_label(input_class))

            x.append(i)
            losses[oneHot_label(input_class)].append(int((sum(l)/len(l)*100)+0.5)/100.0)
            for k,v in losses.items():
                plt.plot(x[:len(v)],v,color=colors[k-1],marker='o')
            plt.title('Training By Class')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.show()
            plt.pause(0.0001)

            i+=1
            print("Minibatch %i trained. Batch Class: %i" % (i,oneHot_label(input_class)))
            if i % display_step == 0 or i ==1:
                print('Step %i: Minibatch Loss: %f' % (i,sum(l)/len(l)))
            
        for generator in self.threads:
            generator.stop()
            generator.join()
        save_path = saver.save(sess,"D:/Coding/HouseNumber/model.ckpt")
        
        plt.savefig('training_results.png')
        print("Model saved in path: %s" % save_path)
        print()
        print(losses)
        print()
        # Begin Testing
        print('Begin Testing...')
        L = {i:list() for i in range(1,11)}
        for k in self.datamap_test.keys():
            class_data = self.datamap_test[k]
            for inds in class_data:
                
                sample = getBatchSample(inds,self.ds_test,self.bbox_test,testing_filepath)
                inp = np.zeros([int(self.batchsize),int(max_cols),int(max_rows)])
                inp[0,:,:] = sample
                lab = label_oneHot(None)
                lab[0][k-1]=1.
                l = sess.run(loss,feed_dict={X:inp,y_true:lab})
                L[k].append(l[0])
        x = L.keys()
        for i in x:
            L[i] = np.asarray(L[i])
        y = [sum(L[i])/len(L[i]) for i in x]
        y_err = [np.std(L[i])/(len(L[i])**(0.5)) for i in x]
        plt.bar(x,y,y_err=y_err)
        plt.xlabel('Classes')
        plt.ylabel('MSE')
        plt.title('Testing Results')
        plt.show()
        plt.pause(0.0001)
        plt.savefig('testing_results.png')
        


#self.batchlimit
if __name__ == '__main__':
    bm = BatchManager(10)
    
