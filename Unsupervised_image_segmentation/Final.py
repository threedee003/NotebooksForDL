import numpy as np
import skimage.io as io
import operator
import math

"""

from matplotlib import patches
from spectral import imshow as spyShow
from spectral import spy_colors
"""


"""
Author : T. Dhar
Date : 18.07.2022
Objective : Unsupervised K-Means clustering for Image Segmentation

"""





img = io.imread(image_file_path)  





"""

The count_pixel and merge_pix are the helper functions for the bin_generator function so they have to executed before 
bin_generator

"""
def count_pixel(arr,R,flag,bit):
    r = bit//R                       #range size 
    out_ = []                                        
    x = []
    y = []
    temp = 0
    for i in range(0,R):
        for j in range(i*r,(i+1)*r+1):
            temp = (arr == j).sum()             
            out_.append(temp)
        x.append(np.argmax(out_))
        y.append(np.max(out_))
        out_.clear()
    return np.array(x)+np.arange(0,bit,r) ,y, flag*np.ones((np.shape(x)))



def merge_pix(*args): 
    merged = []
    for arg in args:
        merged = np.concatenate((merged,arg),axis = None)
    return merged







"""
This is the function to be called for bin values generation. Make sure to execute count_pixel and merge_pix 
before executing this function
"""
def bin_generator(img,R,bit):
	"""
	args:
	    img : input image should be a numpy array of shape ( channel_height X channel_width X number of channels) ie, HxCxN
	    img.dtype : 'unit8'
	    R : should be integer , number of partitions for the histogram of the image. R < 2^bit-1
	    bit : maximum value for a bit type of image. For example in an 8 bit image 2^8-1 = 255 Hence enter bit = 255
	    returns : bin values :
	                          new_X : the most occuring pixel values 
	                          new_Y : the frequency of their occurences
	                          new_F : a flag list to keep track of their origin channel
	"""
    ch = np.shape(img)[2]        
    x_ = []
    y_ = []
    f_ = []
    for i in range(ch):
        x_ = merge_pix(x_,count_pixel(img[:,:,i].flatten(),R,i,bit)[0])
        y_ = merge_pix(y_,count_pixel(img[:,:,i].flatten(),R,i,bit)[1])
        f_ = merge_pix(f_,count_pixel(img[:,:,i].flatten(),R,i,bit)[2])
    sorT = sorted(zip(x_,y_,f_), key=operator.itemgetter(1))
    new_X,new_Y,new_F = zip(*sorT)
    return new_X,new_Y,new_F
   





"""
These locate and take_mean are the helper functions for the centroid_generator function so they have to be called 
before centroid generator
"""
def locate(arr,element):
    stash = []
    for i in range(len(arr)):
        if arr[i] == element:
            stash.append(i)
    return stash


def take_mean(arr,locs):
    sum_ = 0
    for i in locs:
    	sum_ += arr[i]
    return sum_//len(locs)





"""
This is the function to generate cluster centers for K-Means algorithm for image segmentation.
Make sure to run the functions locate and take_mean before executing this function
"""

def centroid_generator(img,flag,x,y,K):
	"""
	args:
	     img : input image should be a numpy array of shape ( channel_height X channel_width X number of channels) ie, HxCxN
	     img.dtype : 'unit8'
	     flag : new_F generated before from the bin_generator 
	     x : new_X generated before from the bin_generator
	     y : new_Y generated before from the bin_generator
	     K : for the K-Means algorithm enter the desired value for K for image segmentation into K distinct classes.
	     returns : a matrix of shape (K x number of channels of image) contains the centroids for cluster centers 
	"""
    if K<=0:
        print("Enter a valid value for K !!")
    new_flag = flag[-K:]
    new_x = x[-K:]
    means_ = np.ones((K,img.shape[2]))
    for i in range(K):
        stash_ = locate(img[:,:,int(new_flag[i])].flatten(),new_x[i])
        for j in range(img.shape[2]):
            if j == int(new_flag[i]):
                means_[i][j] *= new_x[i]
                continue
            means_[i][j] = take_mean(img[:,:,j].flatten(),stash_)
    return means_



"""
A custom made L2 norm. Helper function for mask_gen, execute this before mask_generator

"""

def euclidean(means_,values):
    sum_ = 0
    for i in range(len(means_)):
        sum_ += (means_[i]-values[i])**2
    return np.sqrt(sum_)





"""
The mask generator function generates a segmentation map
"""

def mask_gen(img,mean):
	"""
	   args : 
	         img : input image should be a numpy array of shape ( channel_height X channel_width X number of channels) ie, HxCxN
	         mean : the centroid values matrix generated from centroid_generator
	         returns : generates a segmentation mask of size (HxC) with K unique values
	"""
    stash = []
    h,w = img.shape[0],img.shape[1]
    mask_ = np.ones((h,w))
    mean = list(mean)
    for i in range(h):
        for j in range(w):
            for means in mean:
                stash.append(euclidean(means,list(img[i,j,0:int(img.shape[2])])))
            mask_[i][j] = mask_[i][j]*np.argmin(stash)
            stash.clear()
    return mask_








"""
mat_locate and mat_mean are the two helper functions to execute the learner function. Execute these before training .
"""

def mat_locate(mat,element):
    stash_ = []
    for i in range(int(mat.shape[0])):
        for j in range(int(mat.shape[1])):
            if mat[i][j] == element:
                stash_.append((i,j))
    return stash_

def mat_mean(mat,locs):
    value_ = 0
    for tups in locs:
            value_ += mat[int(tups[0])][int(tups[1])]
    return value_//len(locs)



"""
learner function improves the segmentation mask by training. 
"""

def learner(mask,img,initial_centers,epochs,epsilon):
	"""
	   args : 
	        mask : the initial mask generated from the mask_gen function
	        img : input image should be a numpy array of shape ( channel_height X channel_width X number of channels) ie, HxCxN
	        inital_centers : the inital cluster centers from the centroid_generator
	        epochs : number of iterations for training
	        epsilon : to bind the training process with a stopping criteria i.e,  loss < epsilon stops the training
	        returns : 
	                mask : improved mask after training
	                final_centroids : improved cluster centers generated during training.
	"""             

    loss = []
    stash = []
    idx = np.unique(mask)
    start = time.time()
    for i in range(epochs):
        final_centroids = np.ones((len(idx),img.shape[2]))
        new_centroids = np.ones((len(idx),img.shape[2]))
        for l in idx:
            stash = mat_locate(mask,int(l))
            #print(stash)
            for k in range(img.shape[2]):
                new_centroids[int(l)][k] *= mat_mean(img[:,:,int(k)],stash)
            stash.clear()
                
        error = np.absolute(new_centroids-initial_centers).sum(axis = 1).sum(axis = 0)
        if  error < epsilon :
            loss.append(error)
            final_centroids = new_centroids
            end_time = time.time()-start
            break
        elif error > epsilon:
            loss.append(error)
            initial_centers = new_centroids
            mask = mask_gen(img,initial_centers)
        
        print("epoch {}/{}---------------------------------------> Error : {} ---- > Time : {} ".format(i+1,epochs,error,time.time()-start))
        
    end_time = time.time()-start    
    
    print("Time elapsed : {} ".format(end_time))
    
    return mask,final_centroids
        



def master(img,R,K,bit,epochs,epsilon):
    X,Y,Z = bin_generator(img,R,bit)
    mean_ = centroid_generator(img,Z,X,Y,K)
    mask_ = mask_gen(img,mean_)
    imp_msk, imp_cent = learner(mask_,img,mean_,epochs,epsilon)
    






"""
Time complexity : O(n*c*h)
"""
