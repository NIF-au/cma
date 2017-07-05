# Contrast Matching Algorithm (CMA)

This algorithm takes two MRI models of different contrast as input, matches the contrast of one model to that of the other, and returns a contrast-matched model. The algorithm thereby enables non-linear coregistration using cross correlation of multi-modal minimum deformation averaged MRI models. 

For the algorithm to be successful, the two models must initially be roughly aligned.

The algorithm contains the following steps: 

  1. The intensity range of both models is normalised to be between 0-100
  2. A mask, generated using BET2, is applied to both models
  2. The model with the contrast, to which the other model is matched to, is blurred
  3. Both models are preprocessed in preparation for the lookup 
  4. A voxel intensity value lookup between the two models, based on voxel location and majority decision, is performed
  5. A spline function, that describes the voxel intensity relation between the two models, is determined
  6. A lookup table is generated on the basis of the spline function
  7. One of the models is converted using minclookup and the generated lookup table

The algorithm is implemented in Python. 

## Software requirements

The script uses external software, so for it to be executable, the following software is required installed:  

- Brain Extraction Tool (BET2). Can be found here: https://github.com/liangfu/bet2
- MINC toolkit version 1.9.11. Can be found here: https://bic-mni.github.io/#v2-version-19111911

## Team Members

- Julie Broni Munk
- Nina Jacobsen
- Maciej Plocharski
- Lasse Riis Østergaard
- Lars Marstaller
- David Reutens 
- Markus Barth
- Andrew Janke
- Aswin Narayanan
- Steffen Bollmann
