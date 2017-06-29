""" Contrast Matching Algorithm (CMA).

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

"""

# Libraries
import sys
import shutil
import argparse
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy import stats
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pathlib
from helpers import create_tmpdir
from helpers import step
from helpers import do_cmd
import datetime
import re


def get_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', help='the target contrast', required=True)
    parser.add_argument('--source', help='the source contrast', required=True)
    parser.add_argument('--tmpdir', help='temporary directory', required=True)
    parser.add_argument('--bet', help='bet/bet2/auto, default auto',
                        default='auto')
    parser.add_argument('--maxval', help='max val, default 300', default=300.0)
    myargs = parser.parse_args(args)
    return myargs


def main():

    ''' main '''
    print('\n')
    print('=' * 79)
    print('Contrast Matching Algorithm (CMA)')
    print('=' * 79)
    print(str(datetime.datetime.now().strftime('[%H:%M:%S] [%Y-%m-%d]')))

    myargs = get_args(sys.argv[1:])
    tmpdir = create_tmpdir(pathlib.Path(myargs.tmpdir))

    step('Created temp dir at', tmpdir)

    target_mnc = pathlib.Path(myargs.target)
    step('Input Target Contrast', target_mnc)
    source_mnc = pathlib.Path(myargs.source)
    step('Input Source Contrast', source_mnc)

    target_norm_res_mnc, source_norm_mnc, bet_mask_res_mnc = generate_masks(
        target_mnc, source_mnc, tmpdir, myargs)

    arr1d_contrast1, contrast2_uint32, contrast2_unique_zero, \
    contrast2_unique_rescaled_fl64 = arr_preprocessing(
        bet_mask_res_mnc, target_norm_res_mnc, source_norm_mnc,
        float(myargs.maxval))

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
    # plt.plot(x, x_converted, 'g', lw=1)
    plt.xlabel('Intensity values of contrast 2')
    plt.ylabel('Intensity values of contrast 1')

    intensity_plot = source_mnc.with_name(source_mnc.stem + '_intensity.png')
    plt.savefig(str(intensity_plot))
    step('Spline fit. Saved to', intensity_plot)

    firstLutColumn = x
    secondLutColumn = x_converted

    # rescaling of intensity values to the range 0-1
    firstLutColumn = rescale(firstLutColumn, 0, 1)
    secondLutColumn = rescale(secondLutColumn, 0, 1)

    # saving lookup table (lut) as .txt
    lut = open(str(tmpdir / 'lookuptable.txt'), "w")
    for j in range(len(firstLutColumn)):
        firstLutColumn_str = str(firstLutColumn[j])
        secondLutColumn_str = str(secondLutColumn[j])
        lut.write(firstLutColumn_str + " " + secondLutColumn_str + "\n")
    lut.close()
    step('Loopkup table generated. Saved to', )

    source_lookup_mnc = source_mnc.with_name(source_mnc.stem + '_lookup.mnc')

    do_cmd('minclookup', '-continuous', '-lookup_table',
           tmpdir/'lookuptable.txt', source_mnc, source_lookup_mnc, '-2')
    print('=' * 79)


def get_dimension(img):
    imginfo, _ = do_cmd('mincinfo', img)
    imginfo = imginfo.decode('utf-8')
    dim = 0
    if re.compile('xspace').search(imginfo):
        dim += 1
    if re.compile('yspace').search(imginfo):
        dim += 1
    if re.compile('zspace').search(imginfo):
        dim += 1
    return dim


def generate_masks(target_mnc, source_mnc, tmpdir, myargs):
    ''' Normalisation, Mask Generation and Resampling '''
    betdir = tmpdir/'bet'
    if not betdir.exists():
        betdir.mkdir()

    target_norm_mnc = tmpdir/(target_mnc.stem + '_norm.mnc')
    source_norm_mnc = tmpdir/(source_mnc.stem + '_norm.mnc')
    target_nii = betdir/(target_mnc.stem + '.nii')
    target_norm_res_mnc = tmpdir/(target_norm_mnc.stem + '_res.mnc')
    bet_mask_nii_gz = betdir/(target_nii.stem + '_mask.nii.gz')
    bet_mask_nii = betdir/(target_nii.stem + '_mask.nii')
    bet_mask_mnc = betdir/(target_nii.stem + '_mask.mnc')
    bet_mask_res_mnc = betdir/(bet_mask_mnc.stem + '_res.mnc')

    if get_dimension(source_mnc) == get_dimension(target_mnc):
        step('Source and target have matching dimensions')
    else:
        step('[Error] Dimension mismatch, terminating')

    do_cmd('mincnorm', target_mnc, target_norm_mnc)
    do_cmd('mincnorm', source_mnc, source_norm_mnc)
    do_cmd('mnc2nii', target_norm_mnc, target_nii)

    if myargs.bet == 'auto' and shutil.which('bet2'):
        bet = 'bet2'
    elif myargs.bet == 'auto' and shutil.which('bet'):
        bet = 'bet'
    elif myargs.bet == 'bet2':
        bet = 'bet2'
    elif myargs.bet == 'bet':
        bet = 'bet'
    else:
        step('bet/bet2 not found, terminating')
        quit()

    if bet == 'bet2':
        do_cmd('bet2', target_nii, betdir / target_nii.stem,
               '-m', '-f', '0.5', '-v')
    elif bet == 'bet':
        do_cmd('bet', target_nii, betdir / target_nii.stem,
               '-R', '-m', '-f', '0.5', '-v')

    if bet_mask_nii_gz.exists():
        do_cmd('gunzip', bet_mask_nii_gz)
    do_cmd('nii2mnc', bet_mask_nii, bet_mask_mnc)
    do_cmd('mincresample', '-like', source_norm_mnc, target_norm_mnc,
           target_norm_res_mnc)
    do_cmd('mincresample', '-like', source_norm_mnc, bet_mask_mnc,
           bet_mask_res_mnc)
    return target_norm_res_mnc, source_norm_mnc, bet_mask_res_mnc


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
    image = nib.load(str(input_im))
    image_data = image.get_data()
    step('Loaded into array', input_im)
    return np.array(image_data)


def arr_preprocessing(img_mask, img1, img2, max_val):
    ''' Applying mask, smoothing model, intensity lookup '''
    arr_mask = arr_image(str(img_mask))
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
