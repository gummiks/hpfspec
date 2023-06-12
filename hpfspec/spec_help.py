from __future__ import print_function
import scipy
import os
import barycorrpy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
import pandas as pd
import scipy.ndimage.filters
from scipy.interpolate import interp1d, CubicSpline, griddata, interpn
from scipy.ndimage.filters import correlate1d
import astropy.constants as aconst
import seaborn as sns
import scipy.interpolate
import h5py
import astropy.time
import astropy.io
import crosscorr
import hpfspec
from . import utils
from . import rv_utils
from . import rotbroad_help
c = 299792.4580   # [km/s]
cp = sns.color_palette("colorblind")
HPFGJ699MASK = crosscorr.mask.HPFGJ699MASK

def vacuum_to_air(wl):
    """
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: numpy.ndarray

    :returns: numpy.ndarray

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.

    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f

def airtovac(wave_air):
    """
    taken from idl astrolib
    ;+
    ; NAME:
    ;       AIRTOVAC
    ; PURPOSE:
    ;       Convert air wavelengths to vacuum wavelengths 
    ; EXPLANATION:
    ;       Wavelengths are corrected for the index of refraction of air under 
    ;       standard conditions.  Wavelength values below 2000 A will not be 
    ;       altered.  Uses relation of Ciddor (1996).
    ;
    ; CALLING SEQUENCE:
    ;       AIRTOVAC, WAVE_AIR, [ WAVE_VAC]
    ;
    ; INPUT/OUTPUT:
    ;       WAVE_AIR - Wavelength in Angstroms, scalar or vector
    ;               If this is the only parameter supplied, it will be updated on
    ;               output to contain double precision vacuum wavelength(s). 
    ; OPTIONAL OUTPUT:
    ;        WAVE_VAC - Vacuum wavelength in Angstroms, same number of elements as
    ;                 WAVE_AIR, double precision
    ;
    ; EXAMPLE:
    ;       If the air wavelength is  W = 6056.125 (a Krypton line), then 
    ;       AIRTOVAC, W yields an vacuum wavelength of W = 6057.8019
    ;
    ; METHOD:
    ;	Formula from Ciddor 1996, Applied Optics 62, 958
    ;
    ; NOTES:
    ;       Take care within 1 A of 2000 A.   Wavelengths below 2000 A *in air* are
    ;       not altered.
    ; REVISION HISTORY
    ;       Written W. Landsman                November 1991
    ;       Use Ciddor (1996) formula for better accuracy in the infrared 
    ;           Added optional output vector, W Landsman Mar 2011
    ;       Iterate for better precision W.L./D. Schlegel  Mar 2011
    ;-
    """
    wave_vac = wave_air * 1.0
    g = wave_vac > 2000     #Only modify above 2000 A

    if np.sum(g):
       for iter in [0, 1]:
          if isinstance(g, np.ndarray):
             sigma2 = (1e4/wave_vac[g])**2.     #Convert to wavenumber squared
             # Compute conversion factor
             fact = 1. + 5.792105e-2 / (238.0185 - sigma2) + \
                                1.67917e-3 / (57.362 - sigma2)
             wave_vac[g] = wave_air[g] * fact              #Convert Wavelength
          else: # scalar version
             sigma2 = (1e4/wave_vac)**2.     #Convert to wavenumber squared
             # Compute conversion factor
             fact = 1. + 5.792105e-2 / (238.0185 - sigma2) + \
                                1.67917e-3 / (57.362 - sigma2)
             wave_vac = wave_air * fact              #Convert Wavelength
    return wave_vac

def get_flux_from_file(filename,o=None,ext=1):
    """
    Get flat flux for a given order

    NOTES:
        f_flat = get_flat_flux('MASTER_FLATS/20180804/alphabright_fcu_march02_july21_deblazed.fits',5)
    """
    hdu = astropy.io.fits.open(filename)
    if o is None:
        return hdu[ext].data
    else:
        return hdu[ext].data[o]

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize=12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.5,alpha=0.5)

def jd2datetime(times):
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])

def read_rv_results_from_hdf5(filename='0_RESULTS/rv_results.hdf5'):
    '''
    Reads a hdf5 file that contains rvs

    Returns a hdf5 object. Remember to close.
    '''
    print('Reading file: {}'.format(filename))
    hf = h5py.File(filename,'r')
    print('This file has the following keys')
    print(hf.keys())
    #print(hf['description'])
    return hf


def group_tracks(bjds,threshold=0.05,plot=False):
    """
    Group HET tracks
    
    INPUT:
        bjds - bjd times
        threshold - the threshold to define a new track
                    assumes a start of a new track if it is more than threshold apart
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(bjds))
    diff = np.diff(bjds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(bjds))
        for i in x:
            plt.plot(i,bjds[i],marker='o',color=cc[groups[i]])
    return groups

def group_inds(inds,threshold=1,plot=False):
    """
    Group indices
    
    INPUT:
        inds - indices times
        threshold - the threshold to define a new group
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(inds))
    diff = np.diff(inds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(inds))
        for i in x:
            plt.plot(i,inds[i],marker='o',color=cc[groups[i]])
    return groups

def detrend_maxfilter_gaussian(flux,n_max=300,n_gauss=500,plot=False):
    """
    A function useful to estimate spectral continuum

    INPUT:
        flux: a vector of fluxes
        n_max: window for max filter
        n_gauss: window for gaussian filter smoothing

    OUTPUT:
        flux/trend - the trend corrected flux
        trend - the estimated trend

    EXAMPLE:
        f_norm, trend = detrend_maxfilter_gaussian(df_temp.flux,plot=True)
    """
    flux_filt = scipy.ndimage.filters.maximum_filter1d(flux,n_max)
    trend = scipy.ndimage.filters.gaussian_filter1d(flux_filt,sigma=n_gauss)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(flux)
        ax.plot(trend)
        fig, ax = plt.subplots()
        ax.plot(flux/trend)
    return flux/trend, trend

def average_ccf(ccfs):
    """
    A function to average ccfs
    
    INPUT:
        An array of CCFs
        
    OUTPUT:
    
    """
    ccfs = np.sum(ccfs,axis=0)
    ccfs /= np.nanmedian(ccfs)
    return ccfs

def barshift(x, v=0.,def_wlog=False):
   """
   Convenience function for redshift.

   x: The measured wavelength.
   v: Speed of the observer [km/s].

   Returns:
      The true wavelengths at the barycentre.

   """
   return redshift(x, vo=v,def_wlog=def_wlog)


def redshift(x, vo=0., ve=0.,def_wlog=False):
   """
   x: The measured wavelength.
   v: Speed of the observer [km/s].
   ve: Speed of the emitter [km/s].

   Returns:
      The emitted wavelength l'.

   Notes:
      f_m = f_e (Wright & Eastman 2014)

   """
   if np.isnan(vo): vo = 0     # propagate nan as zero (@calibration in fib B)
   a = (1.0+vo/c) / (1.0+ve/c)
   if def_wlog:
      return x + np.log(a)   # logarithmic
      #return x + a          # logarithmic + approximation v << c
   else:
      return x * a
      #return x / (1.0-v/c)


def bin_rvs_by_track(bjd,RV,RV_err):
    """
    Bin RVs in an HET track

    INPUT:
        bjd
        RV
        RV_err

    OUTPUT:

    """
    track_groups = group_tracks(bjd,threshold=0.05,plot=False)
    date = [str(i)[0:10] for i in jd2datetime(bjd)]
    df = pd.DataFrame(zip(date,track_groups,bjd,RV,RV_err),
            columns=['date','track_groups','bjd','RV','RV_err'])
    g = df.groupby(['track_groups'])
    ngroups = len(g.groups)

    # track_bins
    nbjd, nRV, nRV_err = np.zeros(ngroups),np.zeros(ngroups),np.zeros(ngroups)
    for i, (source, idx) in enumerate(g.groups.items()):
        cut = df.loc[idx]
        #nRV[i], nRV_err[i] = wsem(cut.RV.values,cut.RV_err)
        nRV[i], nRV_err[i] = weighted_rv_average(cut.RV.values,cut.RV_err.values)
        nbjd[i] = np.mean(cut.bjd.values)
    return nbjd, nRV, nRV_err



def weighted_rv_average(x,e):
    """
    Calculate weigted average

    INPUT:
        x
        e

    OUTPUT:
        xx: weighted average of x
        ee: associated error
    """
    xx, ww = np.average(x,weights=e**(-2.),returned=True)
    return xx, np.sqrt(1./ww)

def weighted_average(error):
    """
    Weighted average. Useful to add rv errors together.
    """
    w = np.array(error)**(-2.)
    return np.sum(w)**(-0.5)


def vsini_calcurve(w_all,f_all,vsinis,v,M,rv_abs,orders=[3,4,5,6,14,15,16,17,18],eps=0.6,debug=False,n_points=50):
    """
    Generate a vsini calcurve for CCFs
    
    INPUT:
        w_all - wavelength matrix with order indices
        f_all - flux matrix with order indices
        vsinis - vsinis to loop over
        M - mask object
        v - an array of velocities to use for CCF generation (in km/s)
        rv_abs - absolute RV of mask
        orders: orders to use
        
    OUTPUT:
        sigmas - array of standard deviations of the best-fit Gaussians
        rvs - array of rvs in km/s
        amplitudes - array of the amplitudes of the Gaussians
        ccfs - an array of the ccfs, one for each vsini
    
    EXAMPLE:
        rv_abs = barycorrpy.get_stellar_data("GJ 699")[0]['rv']/1000.
        ww_all = hf["template/ww"].value
        ff_all = hf["template/ff"].value
        v = np.linspace(-25,25.,161)
        M = crosscorr.mask.Mask("0_CCFS/GJ699/20190309/MASK/0_COMBINED/combined_stellarframe.mas")
        vsinis = [0.01,1.,2.,3.,4.,5.]
        s,r,a,c = broadening_calcurve(ww_all,ff_all,vsinis,RV_abs,orders=[14,15])
    """
    sigmas = []
    ccfs = []
    rvs = []
    amplitudes = []
    colors = utils.get_cmap_colors(N=len(vsinis),cmap="coolwarm")
    for i, vs in enumerate(vsinis):
        _ccfs = []
        for o in orders:
            print("it={:3.0f}, vsini={:6.3f}km/s, o={:2.0f}".format(i,vs,o))
            w, f = w_all[o], f_all[o]
            m = np.isfinite(f)
            w, f = w[m], f[m]
            # BEFORE 20190730
            #wb, fb = utils.rot_broaden_spectrum(w,f,eps=eps,vsini=vs,plot=False)
            #c = ccf.ccf.calculate_ccf(wb,fb,v,M.wi,M.wf,M.weight,-rv_abs)
            fb = rotbroad_help.broaden(w,f,vsini=vs)
            c = crosscorr.calculate_ccf(w,fb,v,M.wi,M.wf,M.weight,-rv_abs)
            _ccfs.append(c)
            #ax.plot(v,c/np.max(c),label="vsini={:0.2f}km/s".format(vs),color=colors[i])
        _ccf_all = np.sum(_ccfs,axis=0)
        _ccf_all = _ccf_all/np.max(_ccf_all)
        amp, rv, sigm, _ = rv_utils.rv_gaussian_fit_single_ccf(v,_ccf_all,
                                                                p0=[0, 0.0, 3.0, 0],
                                                                debug=debug,n_points=n_points)
        sigmas.append(sigm)
        rvs.append(rv)
        amplitudes.append(amp)
        ccfs.append(_ccf_all)
    return np.array(sigmas), np.array(rvs), np.array(amplitudes), np.array(ccfs)

def vsini_from_calcurve_for_orders_old(hf_targ,hf_cal,rv_abs_targ,rv_abs_cal,
				   vsinis,v,M,orders=[4,5,6,14,15,16,17,18]):
    """
    Calculate vsini from a calcurve

    INPUT:

    OUTPUT:
    	calc_vsini, s1_cals, c1_cals, c1_targs

    EXAMPLE:
    """
    ww_all_cal = hf_cal["template/ww"].value
    ff_all_cal = hf_cal["template/ff"].value
    ww_all_targ = hf_targ["template/ww"].value
    ff_all_targ = hf_targ["template/ff"].value
    calc_vsini = []
    c1_cals = []
    s1_cals = []
    c1_targs = []
    for oo in orders:
        # Calculate cal curve
        s1_cal,r1_cal,a1_cal,c1_cal     = vsini_calcurve(ww_all_cal,
                                 ff_all_cal,vsinis,v,M,rv_abs_cal,orders=[oo],eps=0.5)
        # Calcculate target curve: single number
        s1_targ,r1_targ,a1_targ,c1_targ = vsini_calcurve(ww_all_targ,
                                 f_all_targ,[0.0001],v,M,rv_abs_targ,orders=[oo],eps=0.5)
        _calc_vsini = scipy.interpolate.interp1d(s1_cal,vsinis,kind='cubic')(s1_targ)[0]
        print("Order={}, vsini={}km/s".format(oo,_calc_vsini))
        calc_vsini.append(_calc_vsini)
        s1_cals.append(s1_cal)
        c1_cals.append(c1_cal)
        c1_targs.append(c1_targ)
    return calc_vsini, s1_cals, c1_cals, c1_targs


def vsini_from_calcurve_for_orders(hf_targ,hf_cal,rv_abs_targ,rv_abs_cal,
				   vsinis,v,M,orders=[4,5,6,14,15,16,17,18],
				   eps=0.6,plot=False,plot_master=True,title="",master_savename=''):
    """
    Calculate vsini from a calcurve

    INPUT:

    OUTPUT:
    	calc_vsini, s1_cals, c1_cals, c1_targs

    EXAMPLE:
    """
    ww_all_cal = hf_cal["template/ww"].value
    ff_all_cal = hf_cal["template/ff"].value
    ww_all_targ = hf_targ["template/ww"].value
    ff_all_targ = hf_targ["template/ff"].value
    calc_vsini = []
    c_cals = []
    s_cals = []
    c_targs = []
    if plot_master:	
        N = len(orders)
        if N>5:
            nrows = 2
            ncols = 5
        else:
            nrows = 1
            ncols = N
        fig, axx = plt.subplots(ncols=ncols,nrows=nrows,figsize=(18,5*nrows),dpi=200)
    for i,o in enumerate(orders):
        # Calculate cal curve
        ww_cal = ww_all_cal[o]
        ff_cal = ff_all_cal[o]
        ww_targ = ww_all_targ[o]
        ff_targ = ff_all_targ[o]
        s_cal,r_cal,a_cal,c_cal     = vsini_calcurve_for_wf(ww_cal,ff_cal,vsinis,v,M,
                                                            rv_abs_cal,eps=eps,plot=plot)
        # Calcculate target curve: single number
        s_targ,r_targ,a_targ,c_targ = vsini_calcurve_for_wf(ww_targ,ff_targ,[0.0],v,M,
                                                             rv_abs_targ,eps=eps,plot=plot)
        print('fitted sigma: {}km/s compared to s_cal[0]: {}'.format(s_targ[0],s_cal[0]))
        try:
           _calc_vsini = scipy.interpolate.interp1d(s_cal,vsinis,kind='cubic')(s_targ)[0]
        except Exception as e:
           print(e)
           print('Setting vsini=0km/s')
           _calc_vsini= 0.
        print("Order={}, vsini={}km/s".format(o,_calc_vsini))
        if _calc_vsini < 0.:
           print('WARNING vsini <0, setting=0km/s')
           _calc_vsini=0.
        calc_vsini.append(_calc_vsini)
        s_cals.append(s_cal)
        c_cals.append(c_cal)
        c_targs.append(c_targ)
        if plot_master:
            # Calculate broadened CCF
            print('calculating broadened CCF, vsini={:0.2f}km/s'.format(_calc_vsini))
            _s_cal,_r_cal,_a_cal,_c_cal     = vsini_calcurve_for_wf(ww_cal,ff_cal,[_calc_vsini],v,M,
                                                                    rv_abs_cal,eps=eps,plot=plot)
            min_t = 1.-np.min(c_targ[0])
            _min_c = 1.-np.min(_c_cal[0])
            min_c = 1.-np.min(c_cal[0])
            axx.flatten()[i].plot(v+r_targ[0]-r_cal[0],(c_cal[0]-1.)*(min_t/min_c)+1.,label='Scaled Unbroadened Cal',color='black',alpha=0.3)
            axx.flatten()[i].plot(v+r_targ[0]-_r_cal[0],(_c_cal[0]-1.)*(min_t/_min_c)+1.,label='Scaled Broadened Cal: vsini={:0.2f}km/s'.format(_calc_vsini),color='black',alpha=1.)
            axx.flatten()[i].plot(v,c_targ[0],color='red',label='Unscaled Unbroadened Target')
            axx.flatten()[i].legend(loc='lower left',fontsize=8)
            axx.flatten()[i].set_xlabel('v [km/s]',fontsize=20)
            axx.flatten()[i].set_title('o={}, vsini={:0.2f}km/s'.format(o,_calc_vsini),fontsize=20)
            ylim = axx.flatten()[i].get_ylim()
            axx.flatten()[i].set_ylim(ylim[0]*0.9,ylim[1])
            utils.ax_apply_settings(axx.flatten()[i])
    
    calc_vsini = np.array(calc_vsini)
    if plot_master:
        axx.flatten()[0].set_ylabel('CCF',fontsize=20)
        fig.suptitle(title+", vsini={:.3f}+-{:0.3f}km/s".format(np.median(calc_vsini),np.std(calc_vsini)),y=0.95,fontsize=24)
        fig.tight_layout()
        if master_savename!="":
            direct = os.path.dirname(os.path.abspath(master_savename))
            utils.make_dir(direct)
            fig.subplots_adjust(top=0.8)
            fig.savefig(master_savename,dpi=200)
            print('Saved figure to {}'.format(master_savename))

    return np.array(calc_vsini), np.array(s_cals), np.array(c_cals), np.array(c_targs)


def vsini_calcurve_for_wf(w,f,vsinis,v,M,rv_abs,eps=0.6,debug=False,n_points=50,plot=False,ax=None,bx=None,verbose=True):
    """
    Generate a vsini calcurve for CCFs
    
    INPUT:
        w_all - wavelength matrix with order indices
        f_all - flux matrix with order indices
        vsinis - vsinis to loop over
        M - mask object
        v - an array of velocities to use for CCF generation (in km/s)
        rv_abs - absolute RV of mask
        orders: orders to use
        
    OUTPUT:
        sigmas - array of standard deviations of the best-fit Gaussians
        rvs - array of rvs in km/s
        amplitudes - array of the amplitudes of the Gaussians
        ccfs - an array of the ccfs, one for each vsini
    
    EXAMPLE:
        rv_abs = barycorrpy.get_stellar_data("GJ 699")[0]['rv']/1000.
        ww_all = hf["template/ww"].value
        ff_all = hf["template/ff"].value
        v = np.linspace(-25,25.,161)
        M = crosscorr.mask.Mask("0_CCFS/GJ699/20190309/MASK/0_COMBINED/combined_stellarframe.mas")
        vsinis = [0.01,1.,2.,3.,4.,5.]
        s,r,a,c = broadening_calcurve(ww_all,ff_all,vsinis,RV_abs,orders=[14,15])
        
        vsinis = [0.00001]
        ww_all_cal = hf_targ["template/ww"].value
        ff_all_cal = hf_targ["template/ff"].value
        s,r,a,c = vsini_calcurve_for_wf(ww_all_cal[14],ff_all_cal[14],vsinis,v,M,rv_abs_targ,plot=True)
        
        vsinis = [0.0001,1.,2.,3.,4.,5.,6.,10.,15.]
        ww_all_cal = hf_cal["template/ww"].value
        ff_all_cal = hf_cal["template/ff"].value
        s,r,a,c = vsini_calcurve_for_wf(ww_all_cal[14],ff_all_cal[14],vsinis,v,M,rv_abs_cal,plot=True)
    """
    sigmas = []
    ccfs = []
    rvs = []
    amplitudes = []
    m = np.isfinite(f)
    w, f = w[m], f[m]
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        colors = utils.get_cmap_colors(N=len(vsinis),cmap="coolwarm")
    for i, vs in enumerate(vsinis):
        if verbose: print("it={:3.0f}, vsini={:6.3f}km/s".format(i,vs))
        if vs==0:
            if verbose: print('vsini=0, skipping broadening!')
            wb, fb = w, f
        else:
            #wb, fb = utils.rot_broaden_spectrum(w,f,eps=eps,vsini=vs,plot=False)
            fb = rotbroad_help.broaden(w,f,vsini=vs,u1=eps)
            wb = w
        c = crosscorr.calculate_ccf(wb,fb,v,M.wi,M.wf,M.weight,-rv_abs)
        c = c/np.max(c)
        if plot:
            ax.plot(v,c,label="vsini={:0.2f}km/s".format(vs),color=colors[i])
        if plot: plot_fit=True
        else: plot_fit=False
        amp, rv, sigm, _ = rv_utils.rv_gaussian_fit_single_ccf(v,c,
                                                                p0=[0, 0.0, 3.0, 0],
                                                                debug=debug,n_points=n_points,
                                                                plot_fit=plot_fit,ax=ax)
        sigmas.append(sigm)
        rvs.append(rv)
        amplitudes.append(amp)
        ccfs.append(c)
    if plot:
        ax.legend(bbox_to_anchor=(1.1,1.))
        ax.set_xlabel('v [km/s]')
        ax.set_ylabel('CCF')
        if bx is None:
            fig, bx = plt.subplots()
        bx.plot(vsinis,sigmas,'h',lw=1)
        bx.set_xlabel('vsini [km/s]')
        bx.set_ylabel('Gaussian sigma [km/s]')
    return np.array(sigmas), np.array(rvs), np.array(amplitudes), np.array(ccfs)


def rvabs(ww,ff,v,M,v2_width=25.,plot=True,ax=None,bx=None,verbose=True,n_points=40):
    """
    Get absolute RVs. Note, this is relative to the mask provided

    INPUT:
       ww - wavelengths
       ff - fluxes
       v - velocities for CCF generation
       M - Mask object
    
    EXAMPLE:
        # loop through all rvabs for given HPF orders
        M = crosscorr.mask.Mask("0_CCFS/GJ699/20190309/MASK/0_COMBINED/combined_stellarframe.mas")
        for o in [5,6,14,15,16,17,18]:
            ww = ww_all_targ[o]
            ff = ff_all_targ[o] 
            v = np.linspace(-120,120,2000)
            rvabs(ww,ff,v,M,plot=False)
    """
    # 1st iteration
    c = crosscorr.calculate_ccf(ww,ff,v,M.wi,M.wf,M.weight,0.) # THE LAST ARGUMENT CHANGES GJ 905 FROM -75.8 to -77.8km/s
    c = c/np.nanmax(c)
    imin = np.argmin(c)
    vmin = v[imin]
    if verbose:
        print('First iteration:  RVabs = {:0.5f}km/s'.format(vmin))
    # 2nd iteration
    v2 = np.linspace(vmin-v2_width,vmin+v2_width,161)
    c2 = crosscorr.calculate_ccf(ww,ff,v2,M.wi,M.wf,M.weight,0.)
    c2 = c2/np.nanmax(c2)
    if plot:
        if ax is None and bx is None:
            fig, (ax,bx)  = plt.subplots(ncols=2,figsize=(16,5))
        ax.plot(v,c)
        ax.plot(vmin,c[imin],marker='o',color='crimson')
        ax.set_xlabel('v [km/s]')
        bx.set_xlabel('v [km/s]')
        ax.set_ylabel('CCF')
        bx.set_ylabel('CCF')
        ax.set_title('RV-abs = {:0.4f}km/s'.format(vmin))
        utils.ax_apply_settings(ax)
        utils.ax_apply_settings(bx)
        bx.plot(v2,c2)
    amp, vmin2, sigm, _ = rv_utils.rv_gaussian_fit_single_ccf(v2,c2,
                                                                p0=[0, vmin, 3.0, 0],
                                                                debug=False,n_points=int(n_points),
                                                                plot_fit=plot,ax=bx)
    if verbose:
        print('Second iteration: RVabs = {:0.5f}km/s, sigma={:0.5f}'.format(vmin2,sigm))
    return vmin, vmin2

def rvabs_for_orders(ww_all,ff_all,orders,v,M,v2_width=25.,plot=True,ax=None,bx=None,verbose=True,n_points=40):
    """
    Same as rvabs, except loop for different orders. Useful for error estimation
    
    EXAMPLE:
        v = np.linspace(-120,120,2000)
        rr1, rr2 = rvabs_for_orders(ww_all_targ,ff_all_targ,[4,5,6,14,15,16,17,18],v,M,plot=True,verbose=True)
        np.mean(rr1), np.std(rr1), np.mean(rr2), np.std(rr2)
    """
    rv_abs1 = []
    rv_abs2 = []
    for o in orders:
        ww = ww_all[o]
        ff = ff_all[o] 
        r1, r2 = rvabs(ww,ff,v,M,v2_width=v2_width,plot=plot,ax=ax,bx=bx,verbose=False,n_points=n_points)#SEJ verbose=verbose
        rv_abs1.append(r1)
        rv_abs2.append(r2)
    if verbose:
        print('RVabs iteration #1: {:8.5f}+-{:8.5f}km/s'.format(np.mean(rv_abs1),np.std(rv_abs1)))
        print('RVabs iteration #2: {:8.5f}+-{:8.5f}km/s'.format(np.mean(rv_abs2),np.std(rv_abs2)))
    return np.array(rv_abs1), np.array(rv_abs2)

def vsini_from_calcurve_for_orders_with_rvabs(hf_cal,hf_targ,name_cal,name_targ,v_rvabs,v,M,vsinis,
                                              orders=[4,5,6,14,15,16,17,18],savedir="0_vsini/20190311/",rvabs_cal=None,rvabs_targ=None):
    """
    Function to calculate vsini from a calcurve file
    
    EXAMPLE:
        hf_cal = serval_help.read_master_results_hdf5("0_RESULTS/GJ_699/20181203_gj699_optica/GJ_699_results.hdf5")
        hf_targ = serval_help.read_master_results_hdf5("../site/hpfgto/docs/data/targets/GJ_3323/results/GJ_3323_results.hdf5")
        name_cal = 'GJ 699'
        name_targ = "GJ 3323"
        v_rvabs = np.linspace(-120,120,1000)
        v = np.linspace(-25,25.,161)
        vsinis = [0.000,0.25,0.5,1.,1.5,2.,3.,4.,5.]
        M = crosscorr.mask.Mask("0_CCFS/GJ699/20190309/MASK/0_COMBINED/combined_stellarframe.mas")
        
    """
    ww_all_targ = hf_targ['template/ww'].value
    ff_all_targ = hf_targ['template/ff'].value
    if rvabs_targ is None:
        rr1, rr2 = rvabs_for_orders(ww_all_targ,ff_all_targ,orders,v_rvabs,M,plot=False,verbose=True)
        rvabs_targ = np.median(rr2)
        rvabs_targ_err = np.std(rr2)
    else:
        rvabs_targ_err = 0.
    rvabs_str = 'RV_abs = {:0.5f}+-{:0.5f}km/s'.format(rvabs_targ,rvabs_targ_err)
    print(rvabs_str)
    if rvabs_cal is None:
        rvabs_cal = barycorrpy.get_stellar_data(name_cal)[0]['rv']/1000.
        print("Using following rvabs for cal",rvabs_cal)
    else:
        _rr1, _rr2 = rvabs_for_orders(ww_all_targ,ff_all_targ,orders,v_rvabs,M,plot=False,verbose=True)
        rvabs_cal = np.median(_rr2)
    vs, ss, cc_cal, cc_targ = vsini_from_calcurve_for_orders(hf_targ,
                                hf_cal,rvabs_targ,rvabs_cal,vsinis,v,
                                M,orders=orders,plot_master=True,plot=False,
                                title=name_targ+" "+rvabs_str,
                                master_savename=savedir+'{}_vsini.png'.format(name_targ))
    return np.median(vs), np.std(vs), rvabs_targ, rvabs_targ_err

def bin_data_with_groups(x,y,yerr,group):
    """
    A function to bin a data according to groups
    
    EXAMPLE:
        group = group_tracks(df_all.time,0.002)
        x = df_all.time.values
        y = df_all.rv.values
        yerr = df_all.e_rv.values
        group = df_all.groups.values
        df_bin = bin_data_with_groups(x,y,yerr,group)
    """
    df = pd.DataFrame(zip(x,y,yerr,group),columns=['x','y','yerr','group'])
    xx = []
    yy = []
    yyerr = []
    for i in df.group.unique():
        #print(i)
        _d = df[group==i]

        if len(_d)==1:
            #print('len',len(_d))
            #print(_d.x.values)
            xx.append(_d.x.values[0])
            yy.append(_d.y.values[0])
            yyerr.append(_d.yerr.values[0])

        if len(_d)>1:
            #print('len',len(_d))
            xx.append(np.mean(_d.x.values))
            #print('mean',np.mean(_d.x.values))
            _y, _yerr = weighted_rv_average(_d.y.values,_d.yerr.values)
            yy.append(_y)
            yyerr.append(_yerr)
        #print('lenxx=',len(xx))
    df_bin = pd.DataFrame(zip(xx,yy,yyerr),columns=['x','y','yerr'])
    return df_bin

def bin_data_with_errors(x,y,yerr,nbin):
    """
    Bin data with errorbars
    
    EXAMPLE:
        bin_data_with_errors(df_bin.x.values,df_bin.y.values,df_bin.yerr.values,2)
    """
    xx = []
    yy = []
    yyerr = []
    nbin = int(nbin)
    #print(len(x)/nbin)
    for i in range(len(x)/nbin):
        #print(x[i*nbin:(i+1)*nbin])
        xx.append(np.mean(x[i*nbin:(i+1)*nbin]))
        _y, _yerr = weighted_rv_average(y[i*nbin:(i+1)*nbin],yerr[i*nbin:(i+1)*nbin])
        yy.append(_y)
        yyerr.append(_yerr)
    #print(x[(i+1)*nbin:])
    if len(x[(i+1)*nbin:])>0:
        xx.append(np.mean(x[(i+1)*nbin:]))
        _y, _yerr = weighted_rv_average(y[(i+1)*nbin:],yerr[(i+1)*nbin:])
        yy.append(_y)
        yyerr.append(_yerr)
    df_bin = pd.DataFrame(zip(xx,yy,yyerr),columns=['x','y','yerr'])
    return df_bin

def vsini_from_hpf_spectra(ftarg,fcal,eps=0.6,
    v=np.linspace(-35.,35.,161),
    vsinis2 = [0.05,0.5,1.,2.,3.,4,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,18.,20.],
    orders = [4,5,6,14,15,16,17],plot=False,
    targname="",calname="",savedir="out_vsini/",
    M=crosscorr.mask.Mask(filename=HPFGJ699MASK)):
    """
    Calculate vsinis using CCFs. Requires using a slowly rotating template/calibration star.
    
    INPUT:
        ftarg - HPF spectrum filename for target star
        fcal  - HPF spectrum filename for calibration star (needs to be slowly rotating, normally I use GJ 699)
        v - the velocities to create the CCF used for the vsini calculation
        eps - linear limbarkening coeff
        vsinis2 - the vsinis used to generate the calibration curve
        orders - the orders used to calculate the CCF
        plot - If True, make a useful visualization plot
        targname - Target name to add to the plot
        calname - calibration star name to add to the plot
        savedir - Save directory for the plot
        M - ccf mask object
        
    OUTPUT:
        vsini - vsini in km/s (median of the vsini values for the different orders)
        vsini_err - the error on the vsini (estimated from the standard deviation of the vsini point estimates for the different orders)
        
    NOTES:
        - Generally the target star and the calibration star should be of similar spectral type
        - No tellurics are corrected for
        - Generally order 5 is the best behaved (generally I only use orders 4,5,6,14,15,16,17)
        --- 16, 17 and 18 are sometimes finicky
    """
    H1 = hpfspec.HPFSpectrum(ftarg,targetname=targname)
    H2 = hpfspec.HPFSpectrum(fcal,targetname=calname)
    v_rvabs = np.linspace(-120.,120.,1001)
    vsinis1 = [0.0]
    
    v_calc_for_orders = []
    ccfs2 = []

    sigmas1 = []
    rvs1 = []
    amps1 = []
    ccfs1 = []
    
    sigmas2 = []
    rvs2 = []
    amps2 = []
    ccfs2 = []
    
    for i,o in enumerate(orders):
        # Target
        w1 = H1.w_shifted[o]
        f1 = H1.f_debl[o]
        m = np.isfinite(f1)
        w1 = w1[m]
        f1 = f1[m]
        # Cal
        w2 = H2.w_shifted[o]
        f2 = H2.f_debl[o]
        m = np.isfinite(f2)
        w2 = w2[m]
        f2 = f2[m]
        
        _sigmas1, _rvs1, _amps1, _ccfs1 = vsini_calcurve_for_wf(w1,f1,vsinis1,v,M,0.,eps=eps,plot=False,verbose=False)
        _sigmas2, _rvs2, _amps2, _ccfs2 = vsini_calcurve_for_wf(w2,f2,vsinis2,v,M,0.,eps=eps,plot=False,verbose=False)
        print(np.min(_sigmas2),np.max(_sigmas2),_sigmas1)
        try:
            v1 = scipy.interpolate.interp1d(_sigmas2,vsinis2)(_sigmas1)
        except Exception as e:
            print(e)
            print("Setting v1=0")
            v1 = np.array([0.])
        
        v_calc_for_orders.append(v1[0])
        sigmas1.append(_sigmas1)
        rvs1.append(_rvs1)
        amps1.append(_amps1)
        ccfs1.append(_ccfs1)
        sigmas2.append(_sigmas2)
        rvs2.append(_rvs2)
        amps2.append(_amps2)
        ccfs2.append(_ccfs2)
        print("o={}, vsini= {}km/s".format(o,v1[0]))
        
    vmean = np.median(v_calc_for_orders)
    vsigma = np.std(v_calc_for_orders)
    if plot:
        N = 2
        L = 5
        fig, axx = plt.subplots(nrows=N,ncols=L,dpi=200,sharex=True,sharey=True,figsize=(10,4))
        for i, o in enumerate(orders):
            ax = axx.flatten()[i]         
            if i%L == 0:
                ax.set_ylabel("Flux",fontsize=12)
            ax.set_title("o = {}, vsini = {:0.3f}km/s".format(o,v_calc_for_orders[i]),fontsize=7,y=0.97)
                              
            vsini = v_calc_for_orders[i]                              
            # Calculating broadened vsini at the calculated vsini
            w2 = H2.w_shifted[o]
            f2 = H2.f_debl[o]
            m = np.isfinite(f2)
            w2 = w2[m]
            f2 = f2[m]
            _s, _rv, _amp, _ccf = vsini_calcurve_for_wf(w2,f2,[vsini],v,M=M,rv_abs=0.,eps=eps,plot=False,verbose=False)
            min_t = 1.-np.min(ccfs1[i][0])
            ax.plot(v,ccfs1[i][0],color="crimson",lw=1)
            ax.plot(v+rvs1[i][0]-_rv[0],(_ccf[0]-1.)*(min_t/(1.-np.min(_ccf[0])))+1.,color="black",alpha=1.,ls="-",lw=1) # reference, broadened
            utils.ax_apply_settings(ax,ticksize=6)
        if len(orders) < N*L:
            for i in range(len(orders),N*L):
                ax = axx.flatten()[i]
                ax.set_visible(False)
        fig.subplots_adjust(hspace=0.1,wspace=0.05,left=0.05,right=0.97)
        fig.suptitle("{}: RV={:0.3f}km/s, vsini={:0.3f}+-{:0.3f}km/s".format(H1.target.name,H1.rv,vmean,vsigma),y=0.95)
        utils.make_dir(savedir)
        savename = os.path.join(savedir,"{}_vsini.png".format(H1.target.name))
        fig.savefig(savename,dpi=200)
        print("Saved to {}".format(savename))
    
    return vmean, vsigma

def calculate_ew_w_errors(wl,fl,e,limit_left,limit_right,N=100):
    """
    Main EW function to use, performs an MC sampling of spec_help.calculate_ew() to get errors

    INPUT:
        start- start wavelength in A
        stop - stop wavelength in A

    OUTPUT:
        med - Median value
        low - Lower errorbar
        up  - Upper errorbar
        ews - Posterior

    EXAMPLE:
        _EW0 = [M.get_ew(LIM[0],LIM[1]) for M in MT0.splist]
    """
    ews = []
    for i in range(N):
        f_w_err = np.array([f + e*np.random.randn(1)[0] for f, e in zip(fl,e) ])
        _e = calculate_ew(wl,f_w_err,limit_left,limit_right)
        ews.append(_e)
    ews = np.array(ews)
    med = np.percentile(ews,50)
    low = med - np.percentile(ews,16)
    up  = np.percentile(ews,84) - med
    return med, low, up, ews

def calculate_ew(wl,fl,limit_left,limit_right):
    """
    This amounts to calculating INT(1 - F / F_continuum)*d_wl with the bounds as the feature limits
    Our F_continuum is assumed to be 0 and we have a discrete sampling so use a sum.

    This is correct if fl is normalized to unity.

    NOTES:
        This seems to be more correct than specutils:
            N = 200
            w = np.arange(4900,5100,0.1)
            f = np.ones(len(w))
            m = (w>4990.) & (w<5010.)
            f[m] = 0.
            fig, ax = plt.subplots()
            ax.plot(w,f)
            S = Spectrum1D(spectral_axis=w*u.AA,flux=f*u.Unit('erg cm-2 s-1 AA-1'))
            print('EW',equivalent_width(S,regions=SpectralRegion(4900.*u.AA,5100*u.AA))) # 19.9
            print(hpfspec.spec_help.calculate_ew(w,f,4902,5098)) # 20.0, correct
    """
    
    # for now just force it to be that we have the feature entirely within the bounds
    assert limit_left > np.nanmin(wl)
    assert limit_right < np.nanmax(wl)
    
    # need to calculate the wavelength bin sizes to match against limits
    # each wavelength bin has a center, left, and right. We assume that we are given the center
    # need to calculate left and right
    bin_size = np.diff(wl)
    # assuming that the bin size doesn't change meaningfully from one bin to the next one
    bin_size = np.concatenate(([bin_size[0]],bin_size))
    bin_left = wl - bin_size/2.
    bin_right = wl + bin_size/2.
    
    # check to make sure which pixels are finite (i.e. not NaN) values to work with
    condition_finite = np.isfinite(fl)
    
    # handle pixels entirely within the bounds:
    condition_all_in = (bin_left >= limit_left) & (bin_right <= limit_right)
    
    # select the pixels that are finite and those that are all in
    use = np.nonzero(condition_finite & condition_all_in)[0]
    wluse = wl[use]
    fluse = fl[use]
    
    # recalculate bin boundaries, just in case we lost any pixels due to NaN
    bins = np.diff(wluse)
    bins = np.concatenate(([bins[0]],bins))
    
    # do the calculation and sum
    sub = (1. - fluse) * bins
    
    # add the left extra bin
    leftmost_index = use[0]
    left_extra_bin = bin_right[leftmost_index-1] - limit_left
    left_extra_val = (1. - fl[leftmost_index-1]) * left_extra_bin
    #print(use)
    
    # right extra bin
    rightmost_index = use[-1]
    right_extra_bin = limit_right - bin_left[rightmost_index+1]
    right_extra_val = (1. - fl[rightmost_index+1]) * right_extra_bin
    
    return(np.sum(sub) + left_extra_val + right_extra_val)
