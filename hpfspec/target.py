import barycorrpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
from astroquery.mast import Catalogs
from . import bary
DIRNAME = os.path.dirname(__file__)
PATH_TARGETS = os.path.join(DIRNAME,'data/target_files')

class Target(object):
    """
    Simple target class. Capable of querying SIMBAD. Can calculate barycentric corrections.
    
    EXAMPLE:
        H = HPFSpectrum(fitsfiles[1])
        H.plot_order(14,deblazed=True)
        T = Target('G 9-40')
        T.calc_barycentric_velocity(H.jd_midpoint,'McDonald Observatory')
        T = Target('G 9-40')
    """
    
    def __init__(self,name,config_folder=PATH_TARGETS,verbose=False,obsname='McDonald Observatory'):
        self.config_folder = config_folder
        self.config_filename = self.config_folder + os.sep + name + '.config'
        if name=='Teegarden':
            name = "Teegarden's star"
        if name=='HR8926-4':
            name = 'GL87'
        if name=='GJ_1151':
            name = 'GJ 1151'
        if name=='GJ_324_A':
            name = 'GJ_324A'
        if name=='HD_68988':
            name = 'HD 68988'
        if name=='NLTT_51984':
            name = 'GJ_9751'
        self.name = name
        try:
            self.data = self.from_file(verbose=verbose)
        except Exception as e:
            print(e,'File does not exist!')
            if 'TIC' in name:
                print('Querying TIC for data')
                self.data = self.query_tic(name)
            else:
                print('Querying SIMBAD for data')
                self.data, self.warning = barycorrpy.utils.get_stellar_data(name)
            self.to_file(self.data)
        self.ra = self.data['ra']
        self.dec = self.data['dec']
        self.pmra = self.data['pmra']
        self.pmdec = self.data['pmdec']
        self.px = self.data['px']
        self.epoch = self.data['epoch']
        if self.data['rv'] is None:
            self.rv = 0.
        else:
            self.rv = self.data['rv']/1000.# if self.data['rv'] < 1e20 else 0.
        self.obsname = obsname


    def query_tic(self,ticname):
        """
        Query the TESS Input Catalog for data
        """
        name = ticname.replace('-',' ').replace('_',' ')
        df = Catalogs.query_object(name, radius=0.0003, catalog="TIC").to_pandas()[0:1]
        data = {}
        data['ra'] = df.ra.values[0]
        data['dec'] = df.dec.values[0]
        data['pmra'] = df.pmRA.values[0]
        data['pmdec'] = df.pmDEC.values[0]
        data['px'] = df.plx.values[0]
        data['epoch'] = 2451545.0
        data['rv'] = 0.
        return data

    def from_file(self,verbose=False):
        if verbose:
            print('Reading from file {}'.format(self.config_filename))
        #if os.path.exists(self.config_filename):
        config = configparser.ConfigParser()
        config.read(self.config_filename)
        data = dict(config.items('targetinfo'))
        for key in data.keys():
            data[key] = float(data[key])
        return data

    def to_file(self,data):
        print('Saving to file {}'.format(self.config_filename))
        config = configparser.ConfigParser()
        config.add_section('targetinfo')
        for key in data.keys():
            config.set('targetinfo',key,str(data[key]))
            print(key,data[key])
        with open(self.config_filename,'w') as f:
            config.write(f)
        print('Done')
        
    def calc_barycentric_velocity(self,jdtime,obs):
        """
        OUTPUT:
            BJD_TDB
            berv in km/s
        
        EXAMPLE:
            bjd, berv = bary.bjdbrv(H.jd_midpoint,T.ra,T.dec,obsname='McDonald Observatory',
                           pmra=T.pmra,pmdec=T.pmdec,rv=T.rv,parallax=T.px,epoch=T.epoch)
        """
        #bjd, berv = bary.bjdbrv(jdtime,self.ra,self.dec,obsname=self.obsname,
        #                           pmra=self.pmra,pmdec=self.pmdec,rv=self.rv,parallax=self.px,epoch=self.epoch)
        bjd, berv = bary.bjdbrv(jdtime,self.ra,self.dec,obsname=obs,
                                   pmra=self.pmra,pmdec=self.pmdec,rv=self.rv,parallax=self.px,epoch=self.epoch)
        return bjd, berv/1000.
    
    def __repr__(self):
        return "{}, ra={:0.4f}, dec={:0.4f}, pmra={}, pmdec={}, rv={:0.4f}, px={:0.4f}, epoch={}".format(self.name,
                                            self.ra,self.dec,self.pmra,self.pmdec,self.rv,self.px,self.epoch)
