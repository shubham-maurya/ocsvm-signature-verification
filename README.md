# ocsvm-signature-verification 

Effectively using one-class SVM to verify offline signatures on writer-independent parameters. 

I use image processing techniques like Otsu's binarization, bounding box calculation and thinning for preprocessing the image. 

I then generate features by splitting the image into 8 parts, and calculate the tan inverse of centre of gravity of each part. 

Out of 55 writers in the CEDAR database, I use 30 writers to tune the parameters and appropriate threshold, while evaluating performance of the tuned SVM on the other 25 writers. 

This is an attempt at reproducing the below - 

Guerbai, Chibani and Hadjadji (2015). The effective use of the one-class SVM classifier for handwritten signature verification based on writer-independent parameters.
