import barycorrpy
from astropy.time import Time
from astropy import time, coordinates as coord, units as u

def bjdbrv(jd_utc, ra=None, dec=None, obsname=None, lat=0., lon=0., elevation=None,
        pmra=0., pmdec=0., parallax=0., rv=0., zmeas=0.,
        epoch=2451545.0, tbase=0., leap_update=True,**kwargs):
    """
    Wrapper to barycorrpy.py and utc2bjd. Computes the barycentric
    velocity correction and julian date in one call.
    Keyword obsname refers to observatory.pro in the IDL Astronomy User Library

    See also: http://astroutils.astronomy.ohio-state.edu/exofast/barycorr.html

    :param jd_utc: Julian date (UTC)
    :param ra: RA (J2000) [deg]
    :param dec: Dec (J2000) [deg]
    :param obsname: Observatory name (overrides coordinates if set)
    :param lat: Observatory latitude  [deg]
    :param lon: Observatory longitude (E) [+/-360 deg]
    :param elevation: Observatory elevation [m]
    :param pmra: Proper motion (RA*cos(Dec)) [mas/yr]
    :param pmdec: Proper motion (Dec) [mas/yr]
    :param parallax: Parallax [mas]
    :param rv: Radial velocity (within 100 km/s) [m/s]
    :param zmeas: Measured redshift
    :param epoch: Epoch (default 2448348.56250, J2000)
    :param tbase: Baseline subtracted from times (default 0.0)
    :return: Barycentric correction for zmeas

    Example:
    --------
    >>> from brv_we14py import bjdbrv
    >>> print bjdbrv(2457395.24563, 4.585590721,  44.02195596, 'ca')
    (2457395.247062386, -23684.54364462639)

    """
    if obsname=="McDonald Observatory": # Same as used in SERVAL
        lat = 30.6814
        lon = -104.0147
        elevation = 2025.
    if obsname=='APO':
        lat = 32.78000000000001 # astropy.coordinates.EarthLocation.of_site('APO').lat.deg
        lon = -105.82000000000002 # astropy.coordinates.EarthLocation.of_site('APO').lon.deg
        elevation = 2798. # astropy.coordinates.EarthLocation.of_site('APO').height

    # Barycentric Julian Date
    # adapted from http://docs.astropy.org/en/stable/time/#barycentric-and-heliocentric-light-travel-time-corrections
    targ = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    loc = coord.EarthLocation.from_geodetic(lon, lat, height=elevation)
    #times = time.Time(jd_utc, format='jd', scale='utc', location=loc)
    #ltt_bary = times.light_travel_time(targ)
    JDUTC = Time(jd_utc, format='jd', scale='utc')
    ltt_bary = JDUTC.light_travel_time(targ, location=loc)
    bjd = JDUTC.tdb + ltt_bary
   
    # we should be JDUTC
    if leap_update is False:
        print('WARNING: LEAP UPDATE=FALSE')
    brv, warning, status = barycorrpy.get_BC_vel(JDUTC, ra=ra, dec=dec, epoch=epoch, pmra=pmra,
                    pmdec=pmdec, px=parallax, lat=lat, longi=lon, alt=elevation,leap_update=leap_update,**kwargs)
    if len(brv) > 1:
        return bjd.value, brv
    else:
       return bjd.value, brv[0]
