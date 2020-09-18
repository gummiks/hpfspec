
# HPFspec

A versatile hpf spectrum class

Capabilities:

- Easily calculate barycentric correction and BJD times
- Calculate CCFs for different orders
- Calculate absolute RVs for different orders using CCFs
- Calculate vsinis using a CCF method (uses a slowly rotating calibration star)

# Installation instructions

> git clone git@github.com:gummiks/hpfspec.git

> python setup.py install

# Dependencies
Depends on 
- the crosscorr fortran module which has to be in the normal python path
- barycorrpy

# Todo
Finish adding HPFSpecMatch, which builds on this class.

This will give another capability to do vsini, which uses a chi2 method instead of a CCF method
