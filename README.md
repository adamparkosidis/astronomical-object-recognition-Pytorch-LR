# A single layer perceptron using logististic regression to classify observed astronomical objects into: star, galaxy, or quasar.

## The Data

We will be using data collected by the Sloan Digital Sky Survey (SDSS) and released as part of Data Release 14. SDSS is a multi-spectral and spectroscopic survey telescope at Apache Point Observatory in New Mexico, USA.

The telescope uses a camera of 30 CCDs which observe in different optical filter bands (u,g,r,i,z). The data used here is freely publicly available.

Therefore we will be importing a table of 10,000 objects, for each of which we have the following information:

* class = tells us whether the object is a GALAXY, STAR, or QSO ("quasi-stellar object" or quasar)

* ra = [Right Ascension](https://en.wikipedia.org/wiki/Right_ascension)

* dec = Declination

* [redshift](https://en.wikipedia.org/wiki/Redshift)

Magnitude in each of the following filters:

* u
* g
* r
* i
* z

We will not be needing almost any of the imaging data, so you don't need to understand how SDSS works, but if you're curious, you can read about it [here](http://www.sdss3.org/dr9/imaging/imaging_basics.php).

The data we will be using has already been preprocessed in the following ways:

* The 'class' feature has been converted to a set of labels (target array) `T`. Here, 0, 1, and 2 correspond to STAR, GALAXY, and QUASAR respectively.

* The rest of the features have been converted to a scaled array `X`.
