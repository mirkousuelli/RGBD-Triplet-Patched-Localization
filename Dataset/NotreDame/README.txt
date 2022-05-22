###############################################################################
#                                                                             #
# Notre Dame dataset                                                          #
#   Created by Noah Snavely (snavely@cs.cornell.edu)                          #
#                                                                             #
###############################################################################

This archive contains a reconstruction of the Notre Dame cathedral
from photos on Flickr.  This data is free for non-commercial use.
There are three parts to this reconstruction:

  - The list of images (`list.txt').

  - The reconstructed cameras and points (`notredame.out').  (See the
    Bundler User's Manual for information on parsing this file:
    http://phototour.cs.washington.edu/bundler/bundler-v0.3-manual.html#S6).

  - The images themselves, in the `image' subdirectory.  In addition
    to the original images, this contains images with radial
    distortion correction applied according to the estimated
    distortion parameters for each image (these corrected images have
    the extension `.rd.jpg').

These images are licensed with Creative Commons licensed.  Please
respect these licenses, and give proper attribution when, for
instance, publishing one of these images in a paper.  Each image in
this archive has a Flickr user and Flickr image ID embedded in the
filename (which is of the form `<username>_<imageID>.jpg').  To link
to the original image on Flickr, use the following URL:

   http://www.flickr.com/username/imageID/

Questions?  Please send them to Noah Snavely at
snavely@cs.cornell.edu.
