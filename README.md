# Fortran installation instructions

You will need to compile two fortran files.

To install run the following commands:

## Linux

> f2py -c -m CCF_1d CCF_1d.f

> f2py -c -m CCF_3d CCF_3d.f


## Mac
On mac you might need to try sudo (verified that this works on my mac).

> sudo f2py -c -m CCF_1d CCF_1d.f

> sudo f2py -c -m CCF_3d CCF_3d.f
