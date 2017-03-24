''' Contrast Matching Algorithm (CMA).

This algorithm takes two MRI models of different contrast as input
, matches the contrast of one model to that of the other
, and returns a contrast-matched model.
The algorithm thereby enables non-linear coregistration
using cross correlation of multi-modal minimum deformation averaged MRI models.

[Usage]
python3 cma.py <dir/target.mnc> <dir/source.mnc> <optional:working_dir>

target.mnc:
source.mnc:
working_dir: tmp folder will be created here. Defaults to current_dir/tmp

[Requirements]
python3, mincnorm, mnc2nii, nii2mnc, bet, mincresample, gunzip

'''

# Libraries
import time
import datetime
import tempfile
import os
import subprocess
import sys
import shutil
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy import stats
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def main():

    ''' main '''
    print('=' * 79)
    print('Contrast Matching Algorithm (CMA)')
    print('=' * 79)

    if sys.argv[3]:
        tmp_dir = create_tmp_dir(sys.argv[3])
    else:
        tmp_dir = create_tmp_dir()
    step('Created temp dir at', tmp_dir)

    target_mnc = Filepath.load(sys.argv[1])
    step('Input', target_mnc.abspath)
    source_mnc = Filepath.load(sys.argv[2])
    step('input', source_mnc.abspath)

    target_norm_res_mnc, source_norm_mnc, bet_mask_res_mnc = generate_masks(
        target_mnc, source_mnc, tmp_dir)

    arr1d_contrast1, contrast2_uint32, contrast2_unique_zero, \
        contrast2_unique_rescaled_fl64 = arr_preprocessing(
            bet_mask_res_mnc,
            target_norm_res_mnc,
            source_norm_mnc)

    # allocating some space
    source_val = np.array([])
    val_match_contrast1 = np.array([])
    len_contrast2_unique = len(contrast2_unique_zero)

    # CORE FUNCTION: VOXEL INTENSITY LOOKUP

    final_val_match = np.array([])
    for i in range(len_contrast2_unique):
        source_val = contrast2_unique_zero[i]
        val_match_contrast1 = arr1d_contrast1[
            np.where(contrast2_uint32 == source_val)]

        # decreasing size of val_match_contrast1 by only
        # (1) including positive values and
        # (2) taking every 15th element
        array_np = np.asarray(val_match_contrast1)
        positive_val_match_contrast1 = array_np > 0
        decreased_val_match_contrast1 = array_np[positive_val_match_contrast1]
        decreased_nth_val_match_contrast1 = decreased_val_match_contrast1[
            ::15].copy()

        # a list that contains the matched values and their frequencies
        Blist = stats.itemfreq(decreased_nth_val_match_contrast1).tolist()

        # finding the values with the highest frequency
        max_count = max(Blist, key=lambda x: x[1])
        max_val_list = [x for x in Blist if x[1] == max_count[1]]
        max_vals = [l[0] for l in max_val_list]
        mean_val = np.mean(max_vals)
        final_val_match = np.append(final_val_match, mean_val)

    step('Core function: matched intensities')
    # data type conversion and rescaling of contrast 2

    x = contrast2_unique_rescaled_fl64
    y = final_val_match
    plt.plot(x, y, '.')

    # creating the spline and adjusting the amount of smoothing
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(400)

    x_converted = spl(x)
    plt.plot(x, x_converted, 'g', lw=1)
    plt.xlabel('Intensity values of contrast 2')
    plt.ylabel('Intensity values of contrast 1')
    intensity_plot = Filepath.create(source_mnc, '_intensity.png')
    plt.savefig(intensity_plot.abspath)
    step('Spline fit. Saved to', intensity_plot.abspath)

    firstLutColumn = x
    secondLutColumn = x_converted

    # rescaling of intensity values to the range 0-1
    firstLutColumn = rescale(firstLutColumn, 0, 1)
    secondLutColumn = rescale(secondLutColumn, 0, 1)

    # saving lookup table (lut) as .txt
    lut = open(os.path.join(tmp_dir, 'lookuptable.txt'), "w")
    for j in range(len(firstLutColumn)):
        firstLutColumn_str = str(firstLutColumn[j])
        secondLutColumn_str = str(secondLutColumn[j])
        lut.write(firstLutColumn_str + " " + secondLutColumn_str + "\n")
    lut.close()
    step('Loopkup table generated. Saved to', )

    source_lookup_mnc = Filepath.create(
        source_mnc, '_lookup.mnc')

    run_cmd(
        'minclookup', '-continuous', '-lookup_table',
        os.path.join(tmp_dir, 'lookuptable.txt'),
        source_mnc.abspath,
        source_lookup_mnc.abspath, '-2')
    print('=' * 79)


def step(*args):
    ''' Prints steps '''
    status = '[' + str(datetime.datetime.now()) + '][Complete] '
    print(status + ' '.join(args))


def run_cmd(*args):
    ''' Runs shell command '''
    subprocess.call(args)
    step(' '.join(args))


class Filepath(object):
    ''' Handles file paths '''
    def __init__(self, input_file):
        self.abspath = os.path.abspath(input_file)
        (self.dirname, self.filename) = os.path.split(self.abspath)
        (self.root, self.ext) = os.path.splitext(self.abspath)
        (self.name, _) = os.path.splitext(self.filename)

    @classmethod
    def load(cls, input_file, output_dir=False):
        ''' Load path for existing file '''
        if output_dir:
            input_file = shutil.copy2(input_file, output_dir)
        return cls(input_file)

    @classmethod
    def create(cls, self, ext, output_dir=False):
        ''' Create path for new file '''
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.dirname
        output_file = os.path.join(output_dir, self.name + ext)
        return cls(output_file)

    def exist(self):
        ''' Does this file exist '''
        return os.path.lexists(self.abspath)


def generate_masks(target_mnc, source_mnc, tmp_dir):
    ''' Normalisation, Mask Generation and Resampling '''
    target_norm_mnc = Filepath.create(target_mnc, '_norm.mnc', tmp_dir)
    source_norm_mnc = Filepath.create(source_mnc, '_norm.mnc', tmp_dir)

    target_nii = Filepath.create(
        target_mnc, '.nii', os.path.join(tmp_dir, 'bet'))
    bet_nii = Filepath.create(target_nii, '_bet.nii')
    bet_mask_nii = Filepath.create(target_nii, '_bet_mask.nii')
    bet_mask_mnc = Filepath.create(target_nii, '_bet_mask.mnc', tmp_dir)
    target_norm_res_mnc = Filepath.create(target_norm_mnc, '_res.mnc')
    bet_mask_res_mnc = Filepath.create(bet_mask_mnc, '_res.mnc')

    run_cmd('mincnorm', target_mnc.abspath, target_norm_mnc.abspath)
    run_cmd('mincnorm', source_mnc.abspath, source_norm_mnc.abspath)
    run_cmd('mnc2nii', target_norm_mnc.abspath, target_nii.abspath)
    run_cmd('bet', target_nii.abspath, bet_nii.abspath,
            '-R', '-m', '-f', '0.5', '-v')
    if not bet_mask_nii.exist():
        bet_mask_nii_gz = Filepath.create(target_nii, '_bet_mask.nii.gz')
        run_cmd('gunzip', bet_mask_nii_gz.abspath)
    run_cmd(
        'nii2mnc', bet_mask_nii.abspath, bet_mask_mnc.abspath)
    run_cmd('mincresample', '-like',
            source_norm_mnc.abspath,
            target_norm_mnc.abspath,
            target_norm_res_mnc.abspath)
    run_cmd('mincresample', '-like',
            source_norm_mnc.abspath,
            bet_mask_mnc.abspath,
            bet_mask_res_mnc.abspath)
    return target_norm_res_mnc, source_norm_mnc, bet_mask_res_mnc


def create_tmp_dir(path=os.path.join(os.getcwd(), 'tmp')):
    ''' Creates tmp dir. Defaults to current_dir/tmp '''
    os.makedirs(path, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=path)
    # tmp_dir = os.path.abspath(path)
    return tmp_dir


def rescale(values, new_min=0, new_max=1):
    ''' Rescale algorithm '''
    output = []
    old_min, old_max = min(values), max(values)
    for val in values:
        new_val = (new_max - new_min) / (old_max - old_min) * (val - old_min) \
            + new_min
        output.append(new_val)
    return output


def arr_image(input_im):
    ''' Convert image to numpy array '''
    image = nib.load(input_im.abspath)
    image_data = image.get_data()
    step('Loaded into array', input_im.abspath)
    return np.array(image_data)


def arr_preprocessing(img_mask, img1, img2):
    ''' Applying mask, smoothing model, intensity lookup '''
    max_val = 300.0
    arr_mask = arr_image(img_mask)
    contrast1_masked = np.multiply(arr_image(img1), arr_mask)
    contrast2_masked = np.multiply(arr_image(img2), arr_mask)

    contrast1_masked_smoothed = ndimage.uniform_filter(
        contrast1_masked, size=[9, 9, 9])
    step('Smoothing of model with contrast 1')

    # INTENSITY LOOKUP
    reshape_size_contrast1 = contrast1_masked_smoothed.size
    reshape_size_contrast2 = contrast2_masked.size

    arr1d_contrast1 = np.reshape(
        contrast1_masked_smoothed.data, reshape_size_contrast1)
    arr1d_contrast2 = np.reshape(contrast2_masked.data, reshape_size_contrast2)
    step('Converted from 3D to 1D')

    contrast2_uint32 = np.uint32(
        arr1d_contrast2 * (max_val / np.amax(arr1d_contrast2)))
    step('convert data type from float64 to uint32')

    _, indeces = np.unique(contrast2_uint32, return_index=True)
    contrast2_unique = contrast2_uint32[indeces]
    contrast2_unique_zero = contrast2_unique[np.where(contrast2_unique > 0)]
    contrast2_unique_fl64 = np.float64(contrast2_unique_zero)
    contrast2_unique_rescaled_fl64 = contrast2_unique_fl64/(
        max_val/np.amax(arr1d_contrast2))
    step('Picked unique value above 0')

    return arr1d_contrast1, contrast2_uint32, contrast2_unique_zero, \
        contrast2_unique_rescaled_fl64


if __name__ == '__main__':
    main()
