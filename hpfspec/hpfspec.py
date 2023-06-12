from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io
import scipy.interpolate
import scipy.optimize
from scipy.signal import medfilt
import os
import crosscorr
from . import stats_help
from . import utils
from . import spec_help
from . import rotbroad_help
from . import target
from .priors import PriorSet, UP, NP, JP
DIRNAME = os.path.dirname(__file__)
PATH_FLAT_DEBLAZED = os.path.join(DIRNAME,"data/hpf/flats/alphabright_fcu_sept18_deblazed.fits")
PATH_FLAT_BLAZED = os.path.join(DIRNAME,"data/hpf/flats/alphabright_fcu_sept18.fits")
PATH_TELLMASK = os.path.join(DIRNAME,"data/masks/telluric/telfit_telmask_conv17_thres0.995_with17area.dat")
PATH_SKYMASK = os.path.join(DIRNAME,"data/masks/sky/HPF_SkyEmmissionLineWavlMask_broadened_11111_Compressed.txt")
PATH_CCF_MASK = crosscorr.mask.HPFSPECMATCHMASK
PATH_WAVELENGTH = os.path.join(DIRNAME,"data/hpf/wavelength_solution/LFC_wavecal_scifiber_v2.fits")
PATH_TARGETS = target.PATH_TARGETS

class HPFSpectrum(object):
    """
    Yet another HPF Spectrum object. Can work with deblazed spectra.
    
    EXAMPLE:
        H = HPFSpectrum(fitsfiles[1])
        H.plot_order(14,deblazed=True)
    """
    path_flat_deblazed = PATH_FLAT_DEBLAZED
    path_flat_blazed = PATH_FLAT_BLAZED
    path_tellmask = PATH_TELLMASK
    path_skymask = PATH_SKYMASK
    path_ccf_mask = PATH_CCF_MASK
    path_wavelength_solution = PATH_WAVELENGTH
    
    def __init__(self,filename,targetname='',deblaze=True,tell_err_factor=1.,ccf_redshift=True,
                 sky_err_factor=1.,sky_scaling_factor=1.0,verbose=False,setup_he10830=False,rv=0.,degrade_snr=None):
        self.filename = filename
        self.basename = filename.split(os.sep)[-1]
        self.sky_scaling_factor = sky_scaling_factor
        self.degrade_snr = degrade_snr
        
        # Read science frame
        self.hdu = astropy.io.fits.open(filename)
        self.header = self.hdu[0].header
        self.exptime = self.header["EXPLNDR"]
        self.object = self.header["OBJECT"]
        try: self.qprog = self.header["QPROG"]
        except Exception as e: self.qprog = np.nan
        midpoint_keywords = ['JD_FW{}'.format(i) for i in range(28)]
        self.jd_midpoint = np.median(np.array([self.header[i] for i in midpoint_keywords]))
        
        # Read Flat
        self.hdu_flat = astropy.io.fits.open(self.path_flat_deblazed)
        self.header_flat = self.hdu_flat[0].header
        self.flat_sci = self.hdu_flat[1].data
        self.flat_sky = self.hdu_flat[2].data
        
        self.e_sci = np.sqrt(self.hdu[4].data)*self.exptime
        self.e_sky = np.sqrt(self.hdu[5].data)*self.exptime*self.sky_scaling_factor
        self.e_cal = np.sqrt(self.hdu[6].data)*self.exptime
        self.e = np.sqrt(self.hdu[4].data + self.hdu[5].data)*self.exptime

        self.f_sky = (self.hdu[2].data*self.exptime/self.flat_sky)*self.sky_scaling_factor
        self._f_sky = self.hdu[2].data*self.exptime
        
        self._f_sci = self.hdu[1].data*self.exptime
        self.f_sci = self.hdu[1].data*self.exptime/self.flat_sci
        self.f = self.f_sci - self.f_sky
        if self.degrade_snr != None:
            self.f_degrade, self.v_degrade = np.zeros_like(self.f), np.zeros_like(self.e)
            for o in range(28):
                self.f_degrade[o], self.v_degrade[o] = DegradeSNR(self.f[o], self.e[o]**2, self.degrade_snr)

        # Read in wavelength
        self.w = self.hdu[7].data
        self.w_sky = self.hdu[8].data
        self.w_cal = self.hdu[9].data
        self.drift_corrected = True

        # Inflate errors around tellurics and sky emission lines
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()
        if tell_err_factor == sky_err_factor:
            mm = mt | ms
            self.e[mm] *= tell_err_factor
        else:
            self.e[mt] *= tell_err_factor
            self.e[ms] *= sky_err_factor

        self.sn5 = np.nanmedian(self.f[5]/self.e[5])
        self.sn6 = np.nanmedian(self.f[6]/self.e[6])
        self.sn14 = np.nanmedian(self.f[14]/self.e[14])
        self.sn15 = np.nanmedian(self.f[15]/self.e[15])
        self.sn16 = np.nanmedian(self.f[16]/self.e[16])
        self.sn17 = np.nanmedian(self.f[17]/self.e[17])
        self.sn18 = np.nanmedian(self.f[18]/self.e[18])
        self.sn = self.f/self.e
        
        if targetname=='':
            targetname = self.object
        self.target = target.Target(targetname,verbose=verbose)
        self.bjd, self.berv = self.target.calc_barycentric_velocity(self.jd_midpoint,'McDonald Observatory')


        if ccf_redshift:
            if verbose:
                print('Barycentric shifting')
            v = np.linspace(-125,125,1501)
            _, rabs = self.rvabs_for_orders(v,orders=[5,6,16,17],plot=False,verbose=verbose)
            self.rv = np.median(rabs)
            self.redshift(rv=self.rv)
        else:
            self.rv = rv
            if verbose:
                print('Barycentric shifting, RV={:0.3f}'.format(self.rv))
            self.redshift(rv=self.rv)

        if deblaze:
            self.deblaze()
        #self.hdu.close()
        if setup_he10830:
            self._setup_he10830()

    def _setup_he10830(self,nmedfilt=7,interp='linear'):
        """
        Run some useful calculations for He 10830
        """
        print('Setting up He 10830, and f_debl_lownoise attributes')
        ms = self.get_sky_mask()
        self.deblaze()
        self.e_lownoise = np.sqrt(self.hdu[4].data)*self.exptime
        self.e_lownoise[ms] = self.e[ms]

        self.sn_lownoise = self.f/self.e_lownoise
        self.f_sky_debl_lownoise = np.copy(self.f_sky_debl)
        self.f_debl_lownoise = np.zeros_like(self.f)
        for o in range(28):
            # median filter everything outside the OH lines to minimize read noise when subtracting
            self.f_sky_debl_lownoise[o][~ms[o]] = medfilt(self.f_sky_debl[o][~ms[o]],nmedfilt)
            # interpolate the sky fiber onto the science fiber
            mm = np.isfinite(self.f_sky_debl_lownoise[o])
            ww = self.w[o]
            ff = scipy.interpolate.interp1d(self.w_sky[o][mm],self.f_sky_debl_lownoise[o][mm],kind=interp,fill_value="extrapolate")(ww)
            self.f_debl_lownoise[o] = self.f_sci_debl[o] - ff*self.sky_scaling_factor
            self.f_debl_lownoise[o] = self.f_debl_lownoise[o]/np.nanmedian(self.f_debl_lownoise[o])
        self.e_debl_lownoise = self.f_debl_lownoise/self.sn_lownoise


    def __repr__(self):
        return 'HPFSpec({},sn18={:0.1f})'.format(self.object,self.sn18)

    def get_telluric_mask(self,w=None,o=None):
        """
        Return telluric mask interpolated onto a given grid.
        
        INPUT:
            w - wavelength grid to interpolate on
            o - 
            
        OUTPUT:
        
        EXAMPLE:
        """
        if w is None:
            w = self.w
        mask = np.genfromtxt(self.path_tellmask)
        m = scipy.interpolate.interp1d(mask[:,0],mask[:,1])(w) > 0.01
        if o is None:
            return m
        else:
            m[o]

    def get_sky_mask(self,w=None,o=None):
        """
        Return sky mask interpolated onto a given grid.
        
        INPUT:
            w - wavelength grid to interpolate on
            o - 
            
        OUTPUT:
        
        EXAMPLE:
        """
        if w is None:
            w = self.w
        mask = np.genfromtxt(self.path_skymask)
        m = scipy.interpolate.interp1d(mask[:,0],mask[:,1],fill_value="extrapolate")(w) > 0.01
        if o is None:
            return m
        else:
            m[o]

    def calculate_ccf_for_orders(self,v,orders=[3,4,5,6,14,15,16,17,18],plot=True):
        """
        Calculate CCF for given orders

        INPUT:
            v - velocities

        EXAMPLE:
            H0 = astropylib.hpfspec.HPFSpectrum(df[df.name=="G_9-40"].filename.values[0])
            v = np.linspace(-0,25,161)
            H0.calculate_ccf_for_orders(v)

        NOTES: Calculates on barycentric shifted (NOT ABS RV SHIFTED) and undeblazed version
        """

        self.M = crosscorr.mask.Mask(self.path_ccf_mask)
        w = spec_help.redshift(self.w,vo=self.berv,ve=0.)
        self.ccf = crosscorr.calculate_ccf_for_hpf_orders(w,self.f,v,self.M,berv=0.,orders=orders,plot=plot)
        return self.ccf

    def rvabs_for_orders(self,v,orders,v2_width=25.0,plot=True, ax=None, bx=None, verbose=True, n_points=40):
        """
        Calculate absolute RV for different orders using two iterations (course + fine fitting Gaussian)

        INPUT:
            

        OUTPUT:
            rv1 - from 1st course iteration
            rv2 - from 2nd fine iteration

        EXAMPLE:
            v = np.linspace(-125,125,161)
            H0.rvabs_for_orders(v,orders=[4,5,6])

        NOTES: Calculates on barycentric shifted (NOT ABS RV SHIFTED) and undeblazed version
        """
        self.M = crosscorr.mask.Mask(self.path_ccf_mask)
        w = spec_help.redshift(self.w,vo=self.berv,ve=0.)
        #m = np.isfinite(self.f)
        rv1, rv2 = spec_help.rvabs_for_orders(w,self.f,orders,v,self.M,v2_width,plot,ax,bx,verbose,n_points)
        return rv1, rv2

    def resample_order(self,ww,p=None,vsini=None,shifted=True):
        """
        Resample order to a given grid. Useful when comparing spectra and calculating chi2

        NOTES:
            dt = 0.04/4 = 0.01 for HPF
        """
        
        if shifted: w = self.w_shifted
        else: w = self.w
            
        m = (w> ww.min()-2.)&(w<ww.max()+2.)
        w = w[m]
        f = self.f_debl[m]
        e = self.e_debl[m]
        if self.degrade_snr != None:
                f = self.f_degrade_debl[m]
        m = np.isfinite(f)
        w = w[m]
        f = f[m]
        e = e[m]
        
        ff = scipy.interpolate.interp1d(w,f,kind='linear')(ww)
        ee = scipy.interpolate.interp1d(w,e,kind='linear')(ww)

        if p is not None:
            print('Applying Chebychev polynomial',p)
            ff*=np.polynomial.chebyshev.chebval(ww,p)
            ee*=np.polynomial.chebyshev.chebval(ww,p)
        if vsini is not None:
            print('Applying vsini: {}km/s'.format(vsini))
            ff = rotbroad_help.broaden(ww,ff,vsini)
        return ff, ee
            
    def deblaze(self):
        """
        Deblaze spectrum, make available with self.f_debl
        """
        hdu = astropy.io.fits.open(self.path_flat_blazed)
        self.f_sci_debl = self.hdu[1].data*self.exptime/hdu[1].data
        self.f_sky_debl = self.hdu[2].data*self.exptime/hdu[2].data
        self.f_debl = self.f_sci_debl-self.f_sky_debl*self.sky_scaling_factor
        if self.degrade_snr != None:
            self.f_degrade_debl = self.f_degrade/hdu[1].data
        for i in range(28): 
            self.f_debl[i] = self.f_debl[i]/np.nanmedian(self.f_debl[i])
            if self.degrade_snr != None:
                self.f_degrade_debl[i] = self.f_degrade_debl[i]/np.nanmedian(self.f_degrade_debl[i])
        self.e_debl = self.f_debl/self.sn
            
    def redshift(self,berv=None,rv=None):
        """
        Redshift spectrum correcting for both berv and rv

        INPUT:
            berv in km/s
            rv in km/s
        """
        if berv is None:
            berv = self.berv
        if rv is None:
            rv = self.target.rv
        self.w_shifted = spec_help.redshift(self.w,vo=berv,ve=rv)
        self.rv = rv

    def rotbroad(self,ww,vsini,eps=0.6,plot=False):
        """
        Broaden with vsini
        """
        ff, ee = self.resample_order(ww)
        _f = rotbroad_help.broaden(ww,ff,vsini,u1=eps)
        return _f
        
    def plot_order(self,o,deblazed=False,shifted=False,ax=None,color=None,plot_shaded=True,alpha=1.):
        """
        Plot spectrum deblazed or not
        
        EXAMPLE:
            
        """
        mask_tell = np.genfromtxt(self.path_tellmask)
        mask_sky = np.genfromtxt(self.path_skymask)
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()

        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if deblazed:
            self.deblaze()
            f = self.f_debl[o]
            e = self.e_debl[o]
            f_mt = mask_tell[:,1]
            f_ms = mask_sky[:,1]
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        else:
            f = self.f[o]
            e = self.e[o]
            f_mt = mask_tell[:,1]*np.nanmax(f)
            f_ms = mask_sky[:,1]*np.nanmax(f)
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        ax.errorbar(w,f,e,marker='o',lw=0.5,capsize=2,mew=0.5,elinewidth=0.5,markersize=2,color=color,alpha=alpha)
        if plot_shaded:
            ax.plot(w[mt[o]],f[mt[o]],lw=0,marker='.',markersize=2,color='blue')
            ax.plot(w[ms[o]],f[ms[o]],lw=0,marker='.',markersize=2,color='red')
            ax.fill_between(w_mt,f_mt,color='blue',alpha=0.1)
            ax.fill_between(w_ms,f_ms,color='red',alpha=0.1)

        utils.ax_apply_settings(ax)
        ax.set_title('{}, {}, order={}, SN18={:0.2f}\nBJD={}, BERV={:0.5f}km/s'.format(self.object,
                                                                                       self.basename,o,self.sn18,self.bjd,self.berv))
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.nanmin(self.w[o]),np.nanmax(self.w[o]))

    def plot_order2(self,o,deblazed=False,shifted=False,ax=None,color=None,plot_shaded=True,alpha=1.,sep=0.,errorbar=True):
        """
        Plot spectrum deblazed or not
        
        EXAMPLE:
            
        """
        mask_tell = np.genfromtxt(self.path_tellmask)
        mask_sky = np.genfromtxt(self.path_skymask)
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()

        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if deblazed:
            self.deblaze()
            f = self.f_debl[o]
            e = self.e_debl[o]
            f_mt = mask_tell[:,1]
            f_ms = mask_sky[:,1]
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        else:
            f = self.f[o]
            e = self.e[o]
            f_mt = mask_tell[:,1]*np.nanmax(f)
            f_ms = mask_sky[:,1]*np.nanmax(f)
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        if errorbar:
            ax.errorbar(w,f+sep,e,marker='o',lw=0.5,capsize=2,mew=0.5,elinewidth=0.5,markersize=2,color=color,alpha=alpha)
        else:
            ax.plot(w,f+sep,marker='.',lw=0.5,markersize=2,color=color,alpha=alpha)
        if plot_shaded:
            ax.plot(w[mt[o]],f[mt[o]],lw=0,marker='.',markersize=2,color='blue')
            ax.plot(w[ms[o]],f[ms[o]],lw=0,marker='.',markersize=2,color='red')
            ax.fill_between(w_mt,f_mt,color='blue',alpha=0.1)
            ax.fill_between(w_ms,f_ms,color='red',alpha=0.1)

        utils.ax_apply_settings(ax)
        ax.set_title('{}, {}, order={}, SN18={:0.2f}\nBJD={}, BERV={:0.5f}km/s'.format(self.object,
                                                                                       self.basename,o,self.sn18,self.bjd,self.berv))
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.nanmin(self.w[o]),np.nanmax(self.w[o]))

    def plot_order2ln(self,o,deblazed=False,shifted=False,ax=None,color=None,plot_shaded=True,alpha=1.,sep=0.,errorbar=True):
        """
        Plot spectrum deblazed or not
        
        EXAMPLE:
            
        """
        mask_tell = np.genfromtxt(self.path_tellmask)
        mask_sky = np.genfromtxt(self.path_skymask)
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()

        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if deblazed:
            self.deblaze()
            f = self.f_debl_lownoise[o]
            e = self.e_debl_lownoise[o]
            f_mt = mask_tell[:,1]
            f_ms = mask_sky[:,1]
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        else:
            f = self.f[o]
            e = self.e[o]
            f_mt = mask_tell[:,1]*np.nanmax(f)
            f_ms = mask_sky[:,1]*np.nanmax(f)
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        if errorbar:
            ax.errorbar(w,f+sep,e,marker='o',lw=0.5,capsize=2,mew=0.5,elinewidth=0.5,markersize=2,color=color,alpha=alpha)
        else:
            ax.plot(w,f+sep,marker='.',lw=0.5,markersize=2,color=color,alpha=alpha)
        if plot_shaded:
            ax.plot(w[mt[o]],f[mt[o]],lw=0,marker='.',markersize=2,color='blue')
            ax.plot(w[ms[o]],f[ms[o]],lw=0,marker='.',markersize=2,color='red')
            ax.fill_between(w_mt,f_mt,color='blue',alpha=0.1)
            ax.fill_between(w_ms,f_ms,color='red',alpha=0.1)

        utils.ax_apply_settings(ax)
        ax.set_title('{}, {}, order={}, SN18={:0.2f}\nBJD={}, BERV={:0.5f}km/s'.format(self.object,
                                                                                       self.basename,o,self.sn18,self.bjd,self.berv))
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.nanmin(self.w[o]),np.nanmax(self.w[o]))


class HPFSpecList(object):

    def __init__(self,splist=None,filelist=None,tell_err_factor=1.,sky_err_factor=1.,targetname='',
                 sky_scaling_factor=1.,verbose=False):
        if splist is not None:
            self.splist = splist
        else:
            if np.size(sky_scaling_factor)==1:
                sky_scaling_factor = np.ones(len(filelist))*sky_scaling_factor
            self.splist = [HPFSpectrum(f,tell_err_factor=tell_err_factor,
                                       sky_err_factor=sky_err_factor,
                                       targetname=targetname,
                                       sky_scaling_factor=s,verbose=verbose) for f,s in zip(filelist,sky_scaling_factor)]

    @property
    def sn18(self):
        return [sp.sn18 for sp in self.splist]

    @property
    def filenames(self):
        return [sp.filename for sp in self.splist]

    @property
    def objects(self):
        return [sp.object for sp in self.splist]

    @property
    def exptimes(self):
        return [sp.exptime for sp in self.splist]

    @property
    def qprog(self):
        return [sp.qprog for sp in self.splist]

    @property
    def rv(self):
        return [sp.rv for sp in self.splist]

    @property
    def df(self):
        d = pd.DataFrame(zip(self.objects,self.filenames,self.exptimes,self.sn18,self.qprog,self.rv),
                         columns=['OBJECT_ID','filename','exptime','sn18','qprog','rv'])
        return d

    
    def resample_order(self,ww,p=None,shifted=True):
        """
        Resample, and can apply cheb polynomials
        """
        ff = []
        ee = []
        for i, H in enumerate(self.splist):
            if p is not None:
                _f, _e = H.resample_order(ww,p=p[i],shifted=shifted)
            else:
                _f, _e = H.resample_order(ww,shifted=shifted)
            ff.append(_f)
            ee.append(_e)
        return ff, ee

    def get_cheb_coeffs(self,ww,plot=False,verbose=False,mask=None):
        """
        Calculate chebychev coefficients. Compares the spectra together.
        Masks out tellurics in both spectra.

        NOTES:
            Assumes that spectra are blaze-corrected
        """
        if mask is None:
            mask = np.zeros(len(ww),dtype=bool)
        coeffs = []
        # As we are skipping the first one
        coeffs.append(np.array([1.,0.,0.,0.,0.,0.]))
        H1 = self.splist[0] # target, other stars get scaled to this
        ff1, ee1 = H1.resample_order(ww,shifted=True)
        for i, H2 in enumerate(self.splist[1:]):
            if verbose: 
                print(i)
            ff2, ee2 = H2.resample_order(ww,shifted=True)
            m = H1.get_telluric_mask(ww) | H2.get_telluric_mask(ww) | mask
            C = Chi2Function(ww,ff1,ee1,ff2,ee2,mask=m)
            FC2 = FitChi2(C)
            FC2.minimize_AMOEBA(verbose=verbose)
            if plot:
                FC2.plot_model(FC2.min_pv)
            coeffs.append(FC2.min_pv)
        return coeffs

class Chi2Function(object):
    def __init__(self,w,f1,e1,f2,e2,mask):
        self.w = w
        self.mask = mask
        self.data_target = {'f': f1,
                            'e': e1}
        self.data_ref    = {'f': f2,
                            'e': e2}
        
        self.priors = [UP( -1e10  , 1e10        , 'c0'   , 'c_0'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c1'   , 'c_1'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c2'   , 'c_2'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c3'   , 'c_3'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c4'   , 'c_4'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c5'   , 'c_5'       ,priortype="model")]
        self.ps     = PriorSet(self.priors)
        
    def compute_model(self,pv):
        coeffs = pv
        ff_ref = self.data_ref['f']*np.polynomial.chebyshev.chebval(self.w,coeffs)
        return ff_ref
    
    def __call__(self,pv,verbose=False):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            print('Outside')
            return np.inf 
        flux_model = self.compute_model(pv)
        flux_target = self.data_target['f']
        dummy_error = np.ones(len(flux_target))
        chi2 = stats_help.chi2(flux_target[~self.mask]-flux_model[~self.mask],dummy_error[~self.mask],1,verbose=verbose)
        return chi2

class FitChi2(object):
    
    def __init__(self,Chi2Function):
        self.chi2f = Chi2Function
        
    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.chi2f.ps.labels,self.chi2f.ps.centers,
                                               self.chi2f.ps.bounds[:,0],self.chi2f.ps.bounds[:,1],pv,
                                               self.chi2f.ps.centers-pv),
                                               columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics
        
    def minimize_AMOEBA(self,verbose=True):
        if verbose:
            print('Performing first Chebfit')
        centers_coeffs = np.polynomial.chebyshev.chebfit(self.chi2f.w,self.chi2f.data_target['f']-self.chi2f.data_ref['f']+1.,5)
        if verbose:
            print('Found centers:',centers_coeffs)        
        centers = list(centers_coeffs)
        if verbose:
            print('With CHI',self.chi2f(centers))
            print(len(centers),len(centers_coeffs))
        
        self.res = scipy.optimize.minimize(self.chi2f,centers,method='Nelder-Mead',tol=1e-7,
                                   options={'maxiter': 10000, 'maxfev': 50000})# 'disp': True})
        
        self.min_pv = self.res.x
        
    def plot_model(self,pv):
        coeffs = pv

        fig, (ax, bx) = plt.subplots(nrows=2,dpi=200,sharex=True,gridspec_kw={'height_ratios':[5,2]})
        ax.plot(self.chi2f.w,self.chi2f.data_target['f'],color='black',label='Target',lw=1)
        ax.plot(self.chi2f.w,self.chi2f.data_ref['f'],color='grey',label='Reference',lw=1)
        ff = self.chi2f.compute_model(pv)
        ax.plot(self.chi2f.w,np.polynomial.chebyshev.chebval(self.chi2f.w,coeffs))
        ax.plot(self.chi2f.w,ff,color='crimson',label='Reference*cheb',alpha=0.5,lw=1)
        bx.plot(self.chi2f.w,self.chi2f.data_target['f']-ff,lw=1)
        bx.set_xlabel('Wavelength [A]',fontsize=12)
        bx.set_ylabel('Residual',fontsize=12)
        title = '$\chi^2$={}, coeffs={}'.format(self.chi2f(pv),coeffs)
        ax.set_title(title,fontsize=10)
        for xx in (ax,bx):
            utils.ax_apply_settings(xx,ticksize=10)
        fig.subplots_adjust(hspace=0.05)
        bx = ax.twinx()
        bx.plot(self.chi2f.w,self.chi2f.mask)

def DegradeSNR(Flux, Variance, DesiredSNR):
	"""
	Degrade the SNR, by increasing the variance based on the DesiredSNR
	
	INPUTS:
		Flux: 1D Flux array
		Variance: 1D Variance array
		DesiredSNR: Median SNR for the 1D array, calculated as Flux/sqrt(Variance)
	OUTPUTS:
		NewFlux: Gaussian distributed new flux using a new variance based on DesiredSNR
		NewVariance: New variance based on degraded SNR
		
	Shubham Kanodia
	4th March 2022
	"""
	
	OriginalSNR = np.nanmedian(Flux/np.sqrt(Variance))
	ScaleSNR = OriginalSNR/DesiredSNR
	NewVariance = Variance * (ScaleSNR**2)
	NewFlux = np.random.normal(loc=Flux, scale=np.sqrt(NewVariance))
	
	return NewFlux, NewVariance

#def chi2spectra(ww,H1,H2,rv1=None,rv2=None,plot=False,verbose=False):
#    """
#    
#    EXAMPLE:
#        H1 = HPFSpectrum(df[df.name=='G_9-40'].filename.values[0])
#        H2 = HPFSpectrum(df[df.name=='AD_Leo'].filename.values[0])
#
#        wmin = 10280.
#        wmax = 10380.
#        ww = np.arange(wmin,wmax,0.01)
#        chi2spectra(ww,H1,H2,rv1=14.51,plot=True)
#        
#    EXAMPLE loop through chi2 rv space:
#        wmin = 10280.
#        wmax = 10380.
#        ww = np.arange(wmin,wmax,0.01)
#
#        H1 = HPFSpectrum(df[df.name=='G_9-40'].filename.values[0])
#        H2 = HPFSpectrum(df[df.name=='AD_Leo'].filename.values[0])
#
#        chis = []
#        rvs = np.linspace(14,15,200)
#
#        for i, rv in enumerate(rvs):
#            chi = chi2spectra(ww,H1,H2,rv1=rv,plot=False)
#            chis.append(chi)
#            print(i,rv,chi)
#    """
#    H1.deblaze()
#    H1.redshift(rv=rv1)
#    ff1, ee1 = H1.resample_order(ww)
#
#    H2.deblaze()
#    H2.redshift(rv=rv2)
#    ff2, ee2 = H2.resample_order(ww)
#    
#    chi2 = stats.chi2(ff1-ff2,np.sqrt(ee1**2.+ee2**2.),verbose=verbose)
#
#    if plot:
#        fig, (ax,bx) = plt.subplots(dpi=200,nrows=2,sharex=True,gridspec_kw={'height_ratios':[4,2]})
#        if rv1 is None: rv1 = H1.rv
#        if rv2 is None: rv2 = H2.rv
#        ax.plot(ww,ff1,lw=1,color='black',label="{}, rv={:0.2f}km/s".format(H1.object,rv1))
#        ax.plot(ww,ff2,lw=1,color='crimson',label="{}, rv={:0.2f}km/s".format(H2.object,rv2))
#        bx.errorbar(ww,ff1-ff2,ee1+ee2,elinewidth=1,marker='o',markersize=2,lw=0.,color='crimson')
#        fig.subplots_adjust(hspace=0.05)
#        [utils.ax_apply_settings(xx,ticksize=10) for xx in (ax,bx)]
#        bx.set_xlabel('Wavelength [A]')
#        ax.set_ylabel('Flux')
#        bx.set_ylabel('Residuals')
#        ax.set_title('{} vs {}: $\chi^2=${:0.3f}'.format(H1.object,H2.object,chi2))
#        ax.legend(loc='upper right',fontsize=8,bbox_to_anchor=(1.4,1.))
#        
#    return chi2
