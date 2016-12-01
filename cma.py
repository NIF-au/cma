
# coding: utf-8

# In[1]:

# libraries 
import nibabel as nib
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from __future__ import division
import sys
import subprocess, shlex
from scipy import ndimage


# In[2]:

# definitions
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def rescale(values, new_min = 0, new_max = 1):
    output = []
    old_min, old_max = min(values), max(values)
    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)
    return output


# In[3]:

# MASK GENERATION 


# In[ ]:

# format-conversion mnc2nii 
input_im = "/.../model_contrast1.mnc" # insert path to MRI model with contrast 1, to which the other model's contrast (contrast 2) is matched to
output_im = "/.../contrast1.nii"
shell_cmd = "mnc2nii" + " " + input_im + " " + output_im 
args = shlex.split(shell_cmd)
p = subprocess.Popen(args) 

# bet
input_im = "/.../contrast1.nii"
output_im = "/.../contrast1_bet.nii"
shell_cmd = "bet" + " " + input_im + " " + output_im + " " + "-R -m -f 0.5 -v"
args = shlex.split(shell_cmd)
p = subprocess.Popen(args) 

# format-conversion nii2mnc 
input_im = "/.../contrast1_bet_mask.nii"
output_im = "/.../contrast1_bet_mask.mnc"
shell_cmd = "nii2mnc" + " " + input_im + " " + output_im
args = shlex.split(shell_cmd)
p = subprocess.Popen(args) 


# In[ ]:

# RESAMPLING 


# In[ ]:

# resample one model to the other
resampleToModel = "/.../rough_aligned_model_contrast2.mnc" # insert path to MRI model with contrast 2, which is roughly aligned to model with contrast 1
input_im = "/.../model_contrast1.mnc"
output_im = "/.../contrast1_resampled2contrast2.mnc"
shell_cmd = "mincresample -like" + " " + resampleToModel + " " + input_im + " " + output_im
args = shlex.split(shell_cmd)
p = subprocess.Popen(args) 

# resampling of mask
input_im = "/.../contrast1_bet_mask.mnc"
output_im = "/.../contrast1_bet_mask_resampled2contrast2.mnc"
shell_cmd = "mincresample -like" + " " + resampleToModel + " " + input_im + " " + output_im
args = shlex.split(shell_cmd)
p = subprocess.Popen(args) 


# In[ ]:

# LOADING AND BLURRING OF MODEL WITH CONTRAST 1 


# In[14]:

# load models
contrast1 = nib.load('/.../contrast1_resampled2contrast2.mnc')
contrast2 = nib.load('/.../rough_aligned_model_contrast2.mnc')

# load mask
mask = nib.load('/.../contrast1_bet_mask_resampled2contrast2.mnc')

# get image data 
contrast1Data = contrast1.get_data()
contrast2Data = contrast2.get_data()
maskData = mask.get_data()

# convert to numpy array
arr_contrast1 = np.array(contrast1Data)
arr_contrast2 = np.array(contrast2Data)
arr_mask = np.array(maskData)

# apply (resampled) mask to both models  
contrast1_masked = arr_contrast1 * arr_mask
contrast2_masked = arr_contrast2 * arr_mask

# smooth model with contrast 1 (the contrast to which the other model's contrast will be matched to) 
contrast1_masked_smoothed = ndimage.uniform_filter(contrast1_masked, size=[9, 9, 9])


# In[ ]:

# PREPROCESSING


# In[19]:

# convert from 3D to 1D
reshape_size_contrast1 = contrast1_masked_smoothed.size
reshape_size_contrast2 = contrast2_masked.size

arr1D_contrast1 = np.reshape(contrast1_masked_smoothed.data, reshape_size_contrast1)
arr1D_contrast2 = np.reshape(contrast2_masked.data, reshape_size_contrast2)

# max val 
max_contrast1 = np.amax(arr1D_contrast1)
max_contrast2 = np.amax(arr1D_contrast2)
max_val = 300.0

# scaling   
scaleFactor_1 = max_val / max_contrast1
arr1D_contrast1_scaled = arr1D_contrast1 * scaleFactor_1
 
scaleFactor_2 = max_val / max_contrast2
arr1D_contrast2_scaled = arr1D_contrast2 * scaleFactor_2

# convert data type (float64 --> uint32)
contrast1_uint32 = np.uint32(arr1D_contrast1_scaled)
contrast2_uint32 = np.uint32(arr1D_contrast2_scaled)

# unique val and indeces of contrast 2 
unique, indeces = np.unique(contrast2_uint32, return_index=True)

# only picking out unique val 
contrast2_unique = contrast2_uint32[indeces]

# only including unique values above 0 
M = np.where(contrast2_unique > 0)
contrast2_unique_zero = contrast2_unique[M] 

# allocating some space 
targetVal = np.array([])
logicMatch = np.array([])
valMatch_contrast1 = np.array([])
len_contrast2_unique = len(contrast2_unique_zero)


# In[20]:

# CORE FUNCTION: VOXEL INTENSITY LOOKUP


# In[ ]:

finalValMatch = np.array([])
for i in range(len_contrast2_unique):
    targetVal = contrast2_unique_zero[i]
    L = np.where(contrast2_uint32 == targetVal)
    valMatch_contrast1 = arr1D_contrast1[L]
    
    # decreasing size of valMatch_contrast1 by only 
    # (1) including positive values and 
    # (2) taking every 15th element
    array_np = np.asarray(valMatch_contrast1)
    positive_valMatch_contrast1 = array_np > 0 
    decreased_valMatch_contrast1 = array_np[positive_valMatch_contrast1]     
    decreased_nth_valMatch_contrast1 = decreased_valMatch_contrast1[::15].copy()     
    
    # a list that contains the matched values and their frequencies
    Blist = stats.itemfreq(decreased_nth_valMatch_contrast1).tolist()
    
    # finding the values with the highest frequency 
    max_count = max(Blist, key=lambda x: x[1])
    max_val_list = [x for x in Blist if x[1] == max_count[1]]
    max_vals = [l[0] for l in max_val_list]
    mean_val = np.mean(max_vals)
    finalValMatch = np.append(finalValMatch, mean_val)
    
# data type conversion and rescaling of contrast 2
contrast2_unique_fl64 = np.float64(contrast2_unique_zero) 
contrast2_unique_rescaled_fl64 = contrast2_unique_fl64 / scaleFactor_2


# In[26]:

# SPLINE FITTING


# In[ ]:

x = contrast2_unique_rescaled_fl64 # intensities of contrast 2
y = finalValMatch # matching intensities of contrast 1
plt.plot(x, y, '.') 

# creating the spline and adjusting the amount of smoothing
spl = UnivariateSpline(x, y)
spl.set_smoothing_factor(400)

x_converted = spl(x)
plt.plot(x, x_converted, 'g', lw=1) 
plt.xlabel('Intensity values of contrast 2')
plt.ylabel('Intensity values of contrast 1')
plt.show()


# In[ ]:

# LOOKUP TABLE GENERATION


# In[33]:

firstLutColumn = x
secondLutColumn = x_converted

# rescaling of intensity values to the range 0-1
firstLutColumn = rescale(firstLutColumn, 0, 1)
secondLutColumn = rescale(secondLutColumn, 0, 1)

# saving lookup table (lut) as .txt
lut = open("lookuptable.txt","w")
for j in range(len(firstLutColumn)):
    firstLutColumn_str = str(firstLutColumn[j])
    secondLutColumn_str = str(secondLutColumn[j])
    lut.write(firstLutColumn_str + " " + secondLutColumn_str + "\n")
lut.close()


# In[35]:

# CONVERSION OF MODEL WITH CONTRAST 2 USING LOOKUP TABLE


# In[ ]:

lut = "/.../lookuptable.txt"
input_im = "/.../rough_aligned_model_contrast2.mnc"
output_im = "/.../model_contrast2_lookupConverted2contrast1.mnc"
shell_cmd = "minclookup -continuous -lookup_table" + " " + lut + " " + input_im + " " + output_im + " " + "-2"
args = shlex.split(shell_cmd)
p = subprocess.Popen(args)

