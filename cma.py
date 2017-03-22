''' module docstring '''
# coding: utf-8
# pylint: disable=I0011,E0401,C0103,R0914,R0915,C0200

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
# import code # code.interact(local=locals())


def run_cmd(*args):
    ''' create '''
    subprocess.call(args)


class Image(object):
    ''' Class describing image files '''
    def __init__(self, input_im):
        self.abspath = os.path.abspath(input_im)
        (self.dirname, self.filename) = os.path.split(self.abspath)
        (self.root, self.ext) = os.path.splitext(self.abspath)
        (self.name, _) = os.path.splitext(self.filename)

    @classmethod
    def load(cls, input_im, output_dir=False):
        ''' load '''
        if output_dir:
            input_im = shutil.copy2(input_im, output_dir)
        return cls(input_im)

    @classmethod
    def create(cls, self, ext, output_dir=False):
        ''' create '''
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.dirname
        return cls(os.path.join(output_dir, self.name + ext))


def generate_masks(source_mnc, target_mnc, tmp_dir):
    ''' Compilation of mask generation '''
    source_norm_mnc = Image.create(source_mnc, '.mnc', tmp_dir)
    target_norm_mnc = Image.create(target_mnc, '.mnc', tmp_dir)

    source_nii = Image.create(source_mnc, '.nii', os.path.join(tmp_dir, 'bet'))
    bet_nii_gz = Image.create(source_nii, '_bet.nii.gz')
    bet_mask_nii_gz = Image.create(source_nii, '_bet_mask.nii.gz')
    bet_mask_nii = Image.create(source_nii, '_bet_mask.nii')
    bet_mask_mnc = Image.create(source_nii, '_bet_mask.mnc', tmp_dir)
    source_norm_res_mnc = Image.create(source_norm_mnc, '_res.mnc')
    bet_mask_res_mnc = Image.create(bet_mask_mnc, '_res.mnc')

    run_cmd('cp', source_mnc.abspath, source_norm_mnc.abspath)
    run_cmd('cp', target_mnc.abspath, target_norm_mnc.abspath)
    run_cmd('mnc2nii', source_norm_mnc.abspath, source_nii.abspath)
    run_cmd('bet', source_nii.abspath, bet_nii_gz.abspath,
            '-R', '-m', '-f', '0.5', '-v')
    run_cmd('gunzip', bet_mask_nii_gz.abspath)
    run_cmd(
        'nii2mnc', bet_mask_nii.abspath, bet_mask_mnc.abspath)
    run_cmd('mincresample', '-like',
            target_norm_mnc.abspath,
            source_norm_mnc.abspath,
            source_norm_res_mnc.abspath)
    run_cmd('mincresample', '-like',
            target_norm_mnc.abspath,
            bet_mask_mnc.abspath,
            bet_mask_res_mnc.abspath)
    return source_norm_res_mnc, target_norm_mnc, bet_mask_res_mnc


def create_tmp_dir(path=os.path.join(os.getcwd(), 'tmp')):
    ''' creating tmp dir. Defaults to working_dir/tmp '''
    os.makedirs(path, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=path)
    # tmp_dir = os.path.abspath(path)
    return tmp_dir


def rescale(values, new_min=0, new_max=1):
    ''' rescale '''
    output = []
    old_min, old_max = min(values), max(values)
    for v in values:
        new_v = (new_max - new_min)/(old_max - old_min)*(v - old_min) + new_min
        output.append(new_v)
    return output


def arr_image(input_im):
    ''' arr_image '''
    image = nib.load(input_im.abspath)
    image_data = image.get_data()
    image.uncache()
    return np.array(image_data)


def arr_preprocessing(img_mask, img1, img2):
    ''' combined all pre-processing manipulation '''
    MAX_VAL = 300.0
    arr_mask = arr_image(img_mask)
    contrast1_masked = np.multiply(arr_image(img1), arr_mask)
    contrast2_masked = np.multiply(arr_image(img2), arr_mask)
    # smooth model with contrast 1 (the contrast to which the other model's
    # contrast # will be matched to)
    contrast1_masked_smoothed = ndimage.uniform_filter(
        contrast1_masked, size=[9, 9, 9])

    # PREPROCESSING
    reshape_size_contrast1 = contrast1_masked_smoothed.size
    reshape_size_contrast2 = contrast2_masked.size

    arr1d_contrast1 = np.reshape(
        contrast1_masked_smoothed.data, reshape_size_contrast1)
    arr1d_contrast2 = np.reshape(contrast2_masked.data, reshape_size_contrast2)

    contrast2_uint32 = np.uint32(
        arr1d_contrast2 * (MAX_VAL / np.amax(arr1d_contrast2)))

    _, indeces = np.unique(contrast2_uint32, return_index=True)
    contrast2_unique = contrast2_uint32[indeces]
    contrast2_unique_zero = contrast2_unique[np.where(contrast2_unique > 0)]
    contrast2_unique_fl64 = np.float64(contrast2_unique_zero)
    contrast2_unique_rescaled_fl64 = contrast2_unique_fl64/(
        MAX_VAL/np.amax(arr1d_contrast2))

    return arr1d_contrast1, contrast2_uint32, contrast2_unique_zero, \
        contrast2_unique_rescaled_fl64


def main():

    ''' main '''

    # make tmpdir
    if sys.argv[3]:
        tmp_dir = create_tmp_dir(os.path.join(sys.argv[3], 'tmp'))
    else:
        tmp_dir = create_tmp_dir()

    source_mnc = Image.load(sys.argv[1])
    target_mnc = Image.load(sys.argv[2])

    source_norm_res_mnc, target_norm_mnc, bet_mask_res_mnc = generate_masks(
        source_mnc, target_mnc, tmp_dir)

    # MASK GENERATION
    arr1d_contrast1, contrast2_uint32, contrast2_unique_zero, \
        contrast2_unique_rescaled_fl64 = arr_preprocessing(
            bet_mask_res_mnc,
            source_norm_res_mnc,
            target_norm_mnc)

    # allocating some space
    target_val = np.array([])
    val_match_contrast1 = np.array([])
    len_contrast2_unique = len(contrast2_unique_zero)

    # CORE FUNCTION: VOXEL INTENSITY LOOKUP

    final_val_match = np.array([])
    for i in range(len_contrast2_unique):
        target_val = contrast2_unique_zero[i]
        val_match_contrast1 = arr1d_contrast1[
            np.where(contrast2_uint32 == target_val)]

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
    plt.show()

    firstLutColumn = x
    secondLutColumn = x_converted

    # rescaling of intensity values to the range 0-1
    firstLutColumn = rescale(firstLutColumn, 0, 1)
    secondLutColumn = rescale(secondLutColumn, 0, 1)

    # saving lookup table (lut) as .txt
    lut = open("lookuptable.txt", "w")
    for j in range(len(firstLutColumn)):
        firstLutColumn_str = str(firstLutColumn[j])
        secondLutColumn_str = str(secondLutColumn[j])
        lut.write(firstLutColumn_str + " " + secondLutColumn_str + "\n")
    lut.close()

    target_lookup_mnc = Image.create(
        target_mnc, '_lookup.mnc')

    run_cmd(
        'minclookup', '-continuous', '-lookup_table', 'lookuptable.txt',
        target_mnc.abspath,
        target_lookup_mnc.abspath, '-2')

if __name__ == '__main__':
    main()
