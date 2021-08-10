import numpy as np
import astropy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss_function(x, a, x0, sigma, offset):
    '''it's a Gaussian.'''
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def rv_gaussian_fit_single_ccf(velocity, ccf, n_points=40,p0=[0,0.,3.0,0], mask_inner=0, debug=False, all=False,ax=None,plot_fit=False):
    '''
    Fit a single gaussian to a single CCF. From Megan Bedell's code repository / HARPS Hacks.
    
    Parameters
    ----------
        velocity : np.ndarray
        velocity (in km/s)
        ccf : np.ndarray
        ccf value
        n_points : int
        total number of points on either side of the minimum to fit, must be >= 2
        mask_inner : int
        number of points to ignore on either side of the minimum when fitting (including minimum itself)
        (if 1, ignore the minimum point only)
        debug : boolean
        if True then include print-outs and show a plot
        all : boolean
        if True then include the three non-functional orders and the co-added "order"

    Returns
    -------
        order_par : np.ndarray
        best-fit Gaussian parameters for each order: (amplitude, mean, sigma, offset)
    '''
    if (n_points < 1):
        print("Cannot fit a Gaussian to < 4 points! Try n_points = 2")
        return None
    order_par = np.zeros(4)
    if debug:
        print("starting param",p0)
    ind_min = np.argmin(ccf)
    ind_range = np.arange(n_points*2+1) + ind_min - n_points
    if (ind_range > 160).any() or (ind_range < 0).any():
        if debug:
            print("n_points too large, defaulting to all")
        ind_range = np.arange(len(velocity))
    ind_range = np.delete(ind_range, np.where(np.abs(ind_range - ind_min) < mask_inner))
    popt, pcov = curve_fit(gauss_function, velocity[ind_range], ccf[ind_range], p0=p0, maxfev=10000)
    if debug:
        print("Solution Param",popt)
        fig, axx = plt.subplots(nrows=2,sharex=True)
        ax, bx = axx[0], axx[1]
        ax.plot(velocity,ccf,label="CCF Data: all")
        g = gauss_function(velocity[ind_range],popt[0],popt[1],popt[2],popt[3])
        #ax.plot(velocity[ind_range],ccf[ind_range],label="CCF Data: fitted")
        ax.plot(velocity[ind_range],g,color="red",label="Fit")
        bx.scatter(velocity[ind_range],ccf[ind_range]-g)
        s = np.std(ccf[ind_range]-g)
        bx.set_ylim(-3.*s,3.*s)
        ax.set_title("Amp={:0.3f}, RV={:0.1f}m/s, Sigma={:0.3f}, Offs={:0.3f}".format(popt[0],popt[1]*1000.,popt[2],popt[3]))
    if plot_fit:
        if ax is None:
            fig, ax = plt.subplots()
        g = gauss_function(velocity[ind_range],popt[0],popt[1],popt[2],popt[3])
        ax.plot(velocity[ind_range],g,color="red",label="Amp={:0.3f}, RV={:0.1f}m/s, Sigma={:0.3f}, Offs={:0.3f}".format(popt[0],popt[1]*1000.,popt[2],popt[3]))
    return popt
