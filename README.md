
# HPFspec

A versatile hpf spectrum class

Capabilities:

- Easily calculate barycentric correction and BJD times
- Calculate CCFs for different orders
- Calculate absolute RVs for different orders using CCFs
- Calculate vsinis using a CCF method (uses a slowly rotating calibration star)

# Dependencies
Depends on the ccf fortran module which has to be in the normal python path

# Todo
Finish adding HPFSpecMatch, which builds on this class.

This will give another capability to do vsini, which uses a chi2 method instead of a CCF method
