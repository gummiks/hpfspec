import numpy as np
import os, sys
import datetime
import pandas as pd
from astropy.io import ascii, fits
from scipy.interpolate import interp1d

from astropy.constants import c

ProjectDirectory = r"/storage/home/szk381/cfb12_link/default/SpectralAnalysis"
ProcessedFITSDirectory = os.path.join(ProjectDirectory, 'ProcessedFITS')

try:
	pwd = os.path.dirname(__file__)
except:
	pwd = os.path.join(ProjectDirectory, 'SpectraWorking','src')
sys.path.append(pwd)


from . import target



def CombineHPFSpectra(ObjectName, TemplateFilePath, TemplateFileList, NormalizationOrder=5, TelluricFudgeFactor=1):
	""" 
	Combine spectra to create template for HPF
	
	INPUTS:
		ObjectName: ObjectName to query target parameters for (required for barycentric correction)
		TemplateFilePath: File Path to the template files that will be combined
		TemplateFileList: List of file names to be combined. It is assumed that all files are for the same target
		NormalizationOrder: Normalize all the spectra to this order using their 95% percentile
		TelluricFudgeFactor: Telluric fudge factor to increase the variance where tel mask is true.
		
	OUTPUTS: 
		HDU object corresponding to template file. This is the HDU object for the first template file, with the Science wl, flux and variance replaced by those for the template.
			The template wavelength used is in vacuum in the stellar rest frame.
		
	"""
	
	NumOrders = 28
	NumPixels = 2048
	
	
	SciFluxMaster = np.zeros((len(TemplateFileList), NumOrders, NumPixels))
	SciFluxVarianceMaster = np.zeros((len(TemplateFileList), NumOrders, NumPixels))
	NormalizationFactor = np.zeros((len(TemplateFileList)))

	ReferenceBCVels = np.zeros(NumOrders)

	for i, t in enumerate(TemplateFileList):
		print("Processing and weighting frame = "+t)
		
		HDU = fits.open(TemplateFilePath, t)
		Header = HDU[0].header
		SciWavelength = HDU[7].data
		SciFlux = HDU[1].data
		SciVariance = HDU[4].data
		
		FluxWeightedJD = np.array([Header['JD_FW{}'.format(m)] for m in range(28)])
		IntegrationTime = Header['EXPLNDR']
		
		
		# Replace the dummy grid below with the real telluric mask which should be true where tellurics are present.
		TelluricModel = np.zeros(np.shape(SciFlux)).astype(bool) 
		TelMask = TelluricModel < 0.9*np.nanmedian(TelluricModel, axis=1)[:, None]
		SciVariance[TelMask] *= TelluricFudgeFactor # Inflate variance in Sci flux for telluric line regions 
		
		self.target = target.Target(ObjectName)
		self.bjd, self.berv = self.target.calc_barycentric_velocity(self.jd_midpoint,'McDonald Observatory')
		zbary = self.berv*1000/c.value
		zbulk = self.target.rv*1000/c.value
		
		SciWlStellar = np.zeros(np.shape(SciWavelength))

		# Multiply for zbary because observer is moving
		# Divide by zbulk because emitter is moving
		for o in range(NumOrders):
			SciWlStellar[o] = SciWavelength[o]*(1 +zbary[o])/(1+zbulk) 
					
		if i==0:
			WavelengthReference = np.copy(SciWlStellar)
			TelluricMaskGrid = TelluricModel
			ReferenceBCVels = self.berv
			RefHDU = HDU
		
		HDU.close()
	
		# Normalization constant for each order in each epoch to be scaled by
		NormalizationRaw = np.zeros(NumOrders)

		# Need to interpolate spectra on to the reference grid
		SciFluxInterp = np.zeros((np.shape(SciFlux)))
		SciVarianceInterp = np.zeros((np.shape(SciFlux)))
		
		# Use just one normalization for each epoch. NOT separate for each order.
		NormalizationRaw = np.percentile(SciFlux[NormalizationOrder], q=95)
		NormalizationFactor[i] = NormalizationRaw
		NormalizationFactor[i] /= NormalizationFactor[0] # Multiply each epoch by a normalization factor which is equal to 1 for the 1st epoch
		
		for Order in range(NumOrders):
			nanmask = np.isnan(SciFlux[Order])
			SciFluxInterp[Order] = interp1d(SciWlStellar[Order][~nanmask], SciFlux[Order][~nanmask], kind='linear', fill_value='extrapolate')(WavelengthReference[Order])
			SciVarianceInterp[Order] = interp1d(SciWlStellar[Order][~nanmask], SciVariance[Order][~nanmask], kind='linear', fill_value='extrapolate')(WavelengthReference[Order])
		
		SciFluxMaster[i] = SciFluxInterp/NormalizationFactor[i]
		SciFluxVarianceMaster[i] = SciVarianceInterp / (NormalizationFactor[i] **2)
		
	SciFluxAveraged = np.average(SciFluxMaster, axis=0, weights=1/SciFluxVarianceMaster, returned=True)
	
	WeightedSciFlux = SciFluxAveraged[0]
	WeightedSciVariance = 1/SciFluxAveraged[1]
	
	RefHDU[1].data = WeightedSciFlux
	RefHDU[4].data = WeightedSciVariance
	RefHDU[7].data = WavelengthReference # Wavelength in vacuum in stellar rest frame
	
	RefHDU[0].header['FilesAveraged'] = TemplateFileList
	
	return RefHDU
	
	
