# Convolutional neural network for centrality analysis

## Conv nets for NA61/SHINE

NA61/SHINE experiment in CERN researches the strong fundamental interaction, specifically quark-gluon plasma to hadron matter phase transition. 
The research is based on measuring fluctuations of the number of "spectators" (remaining parts of nuclei from the beam after collision). 
This fluctuations affect energy of the spectators, which we can register with the Projectile Spectator Detector (PSD), and topology of tracks they leave inside it. 
In fact, the PSD is a cubic lattice construction that can be imagined as a 3D matrix with cubic pixels or, more precisely, voxels. 
In this way, it is interesting to apply a convolutional neural networks with 3D kernels to the analysis of data from PSD and compare the results with classical methods, such as decision trees and "cut-based analysis". 

Check our paper: https://link.springer.com/article/10.1134/S1063779620030259

About NA61: http://shine.web.cern.ch, http://luhep.spbu.ru/na61-shine/ (ru), https://arxiv.org/abs/1803.01692

## Data and code

Unfortunately, both the data and the trained models are classified.
The structure of the datasets is, however, free and is provided in terms of numpy shapes: 
1. (N, 2) &ndash; labels of N samples (one-hot encoded for binary classification);
2. (N, 4, 4, 10, 1) &ndash; features of N samples with dimensionality 4x4x10 and a single "color" channel;
Note that you need two 4x4x10 tensors of features for a single sample: in the NA61/SHINE, the first one corresponts to the central section of the detector, while the latter contains data from the peripheral section.

### TF version

The scripts were run with tensorflow 1.9 and are not optimized for tf 2.x

### Dependencies

1. TensorFlow
2. NumPy
3. Matplotlib (for data visualization scripts)

##

Feel free to make code reviews! I would be very glad to receive your feedback.
