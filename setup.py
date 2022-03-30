from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='hpfspec',
      version='0.1.2',
      description='Package to work with HPF Spectra',
      long_description=readme(),
      url='https://github.com/gummiks/hpfspec/',
      author='Gudmundur Stefansson',
      author_email='gummiks@gmail.com',
      install_requires=['barycorrpy>=0.3.4','astroquery','crosscorr'],
      # install_requires=['barycorrpy>=0.3.4','astroquery'],#,'crosscorr'],
      packages=['hpfspec'],
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='HPF Spectra Astronomy',
      include_package_data=True
      )
