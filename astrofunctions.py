#all necessary functions needed in notebook 
# call an 'import functions.py as function'
# to run a function define the function needed as a method of function.  for example 'function.score()'
#you can view the function inputs from the notebook itself for example 'polygon?'


"""
Created on October 18, 2019

by: Rachel Raikar

"""


import numpy as np
import matplotlib.pyplot as py

pathname = './'

#%%

#defines variables that will be used for clipping spectra to the right wavelength window
#goal: isolate the 'W' shaped feature shared by carbon and weak CN stars
lowerThresh = 5840         #represents 7796.0 angstroms in data["LBIN"]
upperThresh = 6550         #represents 8257.5 angstroms in data["LBIN"]
middleThresh = 6184        #represents 8019.6 angstroms in data["LBIN"]; point between the two 'U's

# NORMALIZING SPECTRA FUNCTIONS
def sliceSpec(star, lower = lowerThresh, upper = upperThresh):
    '''
    star: the index of a particular star (with corresponding spectrum) in SPLASH
    lower: the lower boundary of the slice, in terms of wavelength
    upper: the upper boundary of the slice, in terms of wavelength
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    
    Returns newWv, a Numpy array containing every wavelength value between lower and upper for which data exists for a certain 
        star 
    Also returns newFlux, a Numpy array containing the flux values that (for a certain star) correspond to each of the wavelength 
        values in newSpec
    Also returns newIvar, a Numpy array containing the ivar values that (for a certain star) corresopnd to each of the wavelength 
        values in newSpec
    '''
    newWv = data["LBIN"][star][lower:upper]
    newFlux = data["SPEC"][star][lower:upper]
    newIvar = data["IVAR"][star][lower:upper]
    return newWv, newFlux, newIvar

#defines a function that can be used to normalize the spectrum of a star on a certain wavelength range
def normSpec(star = None, spectrum = None, ivars = None, lower = lowerThresh, upper = upperThresh):
    '''
    star: the index of a particular star (with corresponding spectrum) in SPLASH
    spectrum: a Numpy array containing the data that represents the flux values for a spectrum
    ivars: a Numpy array containing the data that represents the ivar values for the same spectrum as above
    lower: the lower boundary of the range the spectrum will be normalized on, in terms of wavelength
    upper: the upper boundary of the range the spectrum will be normalized on, in terms of wavelength
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    star, spectrum, and ivars all default to None so that either a star index or raw spectral/ivar data can be fed into the function with the same result. Depending on what your data looks like, use the corresponding star or spectrum/ivar inputs and leave the other(s) as None
    Only employ one or the other of these inputs when calling this function; do not define both star and spectrum/ivars at the same time 
    
    If the spectrum is found to consist only of NaN values on the range defined by lower and upper:
    Returns np.array(wvSlice), a Numpy array containing every wavelength value between lower and upper
    Also returns np.array(fluxSlice_list), a Numpy array containing only NaN values that is the same length as np.array(wvSlice)
    Also returns np.array(ivarSlice_list), a Numpy array containing only 0's that is the same length as np.array(wvSlice)
    Also returns False, which is intended to be used as an argument in other functions to determine spectrum validity.
    Note that if this happens, the spectrum will not be normalized and therefore cannot be graphed.
    
    Otherwise:
    Returns wvSlice_np, a Numpy array containing every wavelength value between lower and upper
    Also returns normedFlux_np, a Numpy array containing the normalized flux values for a certain star at every wavelength between lower and upper
    Also returns normedIvar_np, a Numpy array containing the changed ivar values that correspond to each of the normalized flux values for a certain star in normedSpec_np
    Also returns True, which is intended to be used as an argument in other functions to determine spectrum validity.
    Note that for normedFlux_np, values within 5 pixels of a nan value that fall outside of 5 standard deviations from the mean flux have been replaced with nan values as well
    '''
    #converting arrays to list for easy iteration and modification
    if star is None and (spectrum is None or ivars is None):
        print("InputError: missing one or more of the necessary arguments.")
        return np.array(wvSlice), np.array(fluxSlice_list), np.array(ivarSlice_list), False
    if star is not None: #if a star index was inputted, get data directly from SPLASH
        wvSlice, fluxSlice, ivarSlice = sliceSpec(star, lower, upper)
    elif np.all(spectrum is not None) and np.all(ivars is not None): #if a spectrum was inputted, use that as the data
        wvSlice, fluxSlice, ivarSlice = data["LBIN"][0][lower:upper], spectrum, ivars
    wvSlice_list = wvSlice.tolist()
    fluxSlice_list = fluxSlice.tolist()
    ivarSlice_list = ivarSlice.tolist()
    
    #creating a version of the spectrum without any NaNs for performing median/standard deviation calculations
    #if the spectrum is all NaNs on the range specified in the function call, will return an error
    #if the spectrum is all 0s on the range specified in the function call, will replace the 0s with NaNs and returm am error
    newFluxSlice = []
    for wv in range(len(fluxSlice_list)):
        #in the case of NaN ivars, exactly 0 ivars, or exactly 0 flux values, flux values should be converted into NaN and ivar values to 0 so that the data is not graphed
        if fluxSlice_list[wv] == 0 or ivarSlice_list[wv] == 0: 
            fluxSlice_list[wv] = np.nan
            ivarSlice_list[wv] = 0 #seems repetitive due to check in if statement, but necessary for any cases where flux = 0 but ivar does not and needs to be set to 0
        elif np.isnan(ivarSlice_list[wv]) or np.isnan(fluxSlice_list[wv]):
            ivarSlice_list[wv] = 0
        else: #add to "non-zero, non-NaN" data
            newFluxSlice.append(fluxSlice_list[wv])
    if newFluxSlice == []:
        print("NormalizationError: The spectrum for star " + str(star) + " contains only NaN values on the specified range. It cannot be normalized.")
        return np.array(wvSlice), np.array(fluxSlice_list), np.array(ivarSlice_list), False
    medianFlux = st.median(newFluxSlice)
    spec_mean = st.mean(newFluxSlice)
    spec_stdev = st.stdev(newFluxSlice)
    upperLim = spec_mean + 5*spec_stdev #limits calculated to clip out values around nan
    lowerLim = spec_mean - 5*spec_stdev
    
    #replaces with np.nan values that are within 5 pixels of a nan value and outside of 5 standard deviations from the mean flux 
    #accomodates for nan values being located on the edges of the spectrum (in positions where 5 pixels out in either direction would be out of range)
    nanIndices = []
    for index in range(len(fluxSlice_list)):
        if np.isnan(fluxSlice_list[index]): 
            if 5 < index < len(wvSlice_list) - 5:
                for i in range(index-5,index+5):
                    if lowerLim > fluxSlice_list[i] or upperLim < fluxSlice_list[i]:
                        fluxSlice_list[i] = np.nan
            elif 5 > index:
                for i in range(0,index+5):
                    if lowerLim > fluxSlice_list[i] or upperLim < fluxSlice_list[i]:
                        fluxSlice_list[i] = np.nan
            elif index > len(wvSlice_list) - 5:
                for i in range(index-5, len(wvSlice)): #this should go upto len(specSlice)-1 due to 0-indexing
                    if lowerLim > fluxSlice_list[i] or upperLim < fluxSlice_list[i]:
                        fluxSlice_list[i] = np.nan

    #at each pixel, normalizes the star's spectrum and modifies the corresponding ivar value
    normedFlux = []
    normedIvar = []
    for wv in range(len(wvSlice_list)):
        normedFlux.append(fluxSlice_list[wv]/medianFlux)
        normedIvar.append(ivarSlice_list[wv]*(medianFlux**2))
    
    #converts the final products back to arrays for easy manipulation later in the code
    normedFlux_np = np.array(normedFlux)
    normedIvar_np = np.array(normedIvar)
    wvSlice_np = np.array(wvSlice)
    
    return wvSlice_np, normedFlux_np, normedIvar_np, True


def graphNormSpec(star = None, spectrum = None, lower = lowerThresh, upper = upperThresh, template = False, templateSpec = None, fileName = None):
    '''
    star: the index of a particular star (with corresponding spectrum) in SPLASH
    spectrum: a Numpy array consisting of the data values that represent the flux values of a spectrum to be graphed
    lower: the lower boundary of the range the spectrum will be normalized on, in terms of wavelength
    upper: the upper boundary of the range the spectrum will be normalized on, in terms of wavelength
    template: a Boolean variable that, if True, will graph the normalized spectrum and the template spectrum on the same graph. If False, only the normalized spectrum will be graphed
    templateSpec: a Numpy array representing the flux values of the spectrum to be used as the template if template is True
    fileName: an optional argument that can be used to save the created graph as a png with the file name designated
    
    Note that in order to use the template, the functions below that are devoted to template creation must be initialized
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    template will default to False, and templateSpec will default to Ctemplate (the carbon coadd)
    star and spectrum will default to None so that either a star index or raw spectral data can be used in the function. Use the inputs that you need based on your data type and leave the other as None
    fileName will default to None, meaning the graph will not be saved as a png
   
    
    Does not return a particular value, but will produce a graph of wavelength versus flux that represents the normalized spectrum of star
    Note that gaps may be present in the spectral graph where NaN values are present in the flux measurements of a certain star
    '''
    #assigning the data to be graphed
    if star is None and np.all(spectrum is None):
        print("InputError: One of 'star' and 'spectrum' must not be None.")
        return None
    if star is not None:
        wvrange, spectrum, ivar = normSpec(star = star, lower = lower, upper = upper)[:3]
    elif np.all(spectrum is not None):
        wvrange, spectrum = data["LBIN"][0][lower:upper], spectrum
    x = wvrange.tolist()
    y = spectrum.tolist()
    
    #this if statement separates out the graphing data that is assigned if template is True
    if template is True:  #Antara - changed the default template to Wtemplate (weak CN coadd)
        y2 = templateSpec
        plt.plot(x, y, color = "b", label = "Normalized Spectrum")
        plt.plot(x, y2, color = "r", label = "Template Spectrum")
        plt.legend(fontsize = 20)
    else:
        plt.plot(x,y)
    
    #formatting the graph so that it is easily readable and executing its creation
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title("Normalized Spectrum of Star #" + str(star), size = 30) 
    plt.ylabel("Flux (normalized)", size = 20)
    plt.xlabel("Wavelength", size = 20) 
    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()

def defaultHist(data, binList, colors, labels, graphLabels, xLim = None, yLim = None, dens=False,alphas = None, step = 0, fileName = None):
    '''
    main note on this function: everything should be passed in as a tuple - even if a single value, put [ ] around
   
    data: a tuple containing all the lists of data to plot on the same histogram
    binList: the Numpy array to use as bins (used for ALL data populations)
    colors: a tuple containing all the colors, corresponding to each sample in data
    labels: a tuple containing all the labels, corresponding to each sample in data
    graphLabels: a tuple of length 3 containing data for the graph: (1) title, (2) x-label, (3) y-label
    dens: a boolean that, if True, will make the histogram a density histogram - default is False
    xLim, yLim: if not None, the limits to put on the viewing rectangle of the graph
    alphas: a tuple of alpha values (transparency amounts) - default is None (no transparency)
    step: if greater than 0, implies a step histogram should be used. value is the line width for the histogram
    fileName: if not None, means this plot should be saved as a figure with this file name.
    '''
    
    for index in range(len(data)):
        a = 1
        if not alphas is None:
            a = alphas[index]
        if step > 0:
            plt.hist(data[index], bins = binList, color = colors[index], label = labels[index], histtype='step', linewidth = step, density = dens, alpha = a)
        else:
            plt.hist(data[index], bins = binList, color = colors[index], label = labels[index], density = dens, alpha = a)
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title(graphLabels[0], size = 50) 
    plt.xlabel(graphLabels[1], size = 40)
    plt.ylabel(graphLabels[2], size = 40)
    plt.legend(fontsize = 35)

    if xLim is not None:
        plt.xlim(xLim)
    if yLim is not None:
        plt.ylim(yLim)
    
    if fileName is not None:
        plt.savefig(fileName)

    plt.show()

#COADDING SPECTRA

#defines a function that clips out outlier data points in a set of spectra 
#outliers are defined as points that lie outside of a certain number of standard deviations away from the median flux value at each wavelength

def sigmaClip(spectra, ivars, nsigma = 3.5):
    '''
    spectra: a list of the spectra of each star that is being clipped
    ivars: a list of the sets of ivars for each star that is being coadded
    nsigma: an integer or float representing the number of standard deviations away from the median to be used in clipping
    
    nsigma will default to 3.5 unless otherwise specified
    spectra and ivar values must be normalized and corresponding for best results
    
    Returns spectra, a list of the spectra of each star in terms of flux
    Also returns ivars, a list of the set of ivars of each star with values corresponding to the invalid values in spectra (those that lie more than nsigma standard deviations away from the median) replaced by 0s
    '''
    sortedSpecList = []
    for value in range(len(spectra[1])): #note that the use of spectra[1] is arbitrary as all of the lists in spectra have the same length
        listToAdd = []
        for spectrum in range(len(spectra)):
            if not np.isnan(spectra[spectrum][value]):
                listToAdd.append(spectra[spectrum][value])
        sortedSpecList.append(listToAdd)
    medians = []
    for eachlist in range(len(sortedSpecList)):
        if len(sortedSpecList[eachlist]) != 0:
            medians.append(st.median(sortedSpecList[eachlist]))
        else: 
            medians.append(np.nan)
        if eachlist%100 == 0 and eachlist != 0:
            print(str(eachlist) + " medians calculated.")
    for spectrum in range(len(spectra)):
        for flux in range(len(spectra[1])): #see above comment
            if np.isnan(medians[flux]):
                continue
            elif ivars[spectrum][flux] != 0 and not np.isnan(spectra[spectrum][flux]):
                testFlux = ((spectra[spectrum][flux] - medians[flux])**2)*ivars[spectrum][flux] 
                if testFlux > nsigma**2:
                    spectra[spectrum][flux] = np.nan
                    ivars[spectrum][flux] = 0          
    return spectra, ivars

#defines a function that will coadd normalized spectra based on provided lists of spectra and ivar weights
def coadd(spectra, ivars, lower = lowerThresh, upper = upperThresh):
    '''
    spectra: a list of the spectra of each star that is being coadded
    ivars: a list of the sets of ivars for each star that is being coadded
    lower: the lower boundary of the range the spectrum have been normalized on, in terms of wavelength
    upper: the upper boundary of the range the spectrum have been normalized on, in terms of wavelength
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    spectra and ivar values must be normalized and corresponding for best results
    
    Returns wvValues_np, a Numpy array containing every wavelength value for which spectral data exists between lower and upper
    Also returns coaddedFlux_np, a Numpy array containing the flux values of the coadded spectrum corresponding to each wavelength in wvValues_np 
    Also returns coaddedIvar_np, a Numpy array containing the ivar values of the coadded spectrum corresponding to each wavelength in  wvValues_np
    '''
    wvValues = data["LBIN"][0][lower:upper].tolist()
    coaddedFlux = []
    coaddedIvar = []
    for wv in range(len(wvValues)):
        fluxSum = 0
        ivarSum = 0
        for star in range(len(spectra)):
            if not np.isnan(spectra[star][wv]): 
                fluxSum += spectra[star][wv] * ivars[star][wv] #multiplying by ivar as a weight, as per formula
                ivarSum += ivars[star][wv]
        if ivarSum == 0:
            coaddedFlux.append(np.nan)
            coaddedIvar.append(0)
        else:
            newFlux = fluxSum/ivarSum #normalizing the weights, as per formula
            coaddedFlux.append(newFlux) #coaddedSpec now contains a coadded spectrum value at each LBIN value
            coaddedIvar.append(ivarSum) #coadded ivar is simply the sum of all ivars for a certain bin, across all stars (based on a mathematical proof)
    for flux in range(len(coaddedFlux)):
        if coaddedFlux[flux] == 0:
            coaddedFlux[flux] = np.nan
    coaddedFlux_np = np.array(coaddedFlux)
    wvValues_np = np.array(wvValues)
    coaddedIvar_np = np.array(coaddedIvar)
    return wvValues_np, coaddedFlux_np, coaddedIvar_np

#defines a function that applys a smoothing function to a spectrum to improve the quality of the spectrum's graph
#used in this program largely for smoothing the graphs of template spectra
def applyGauss(spectrum, gauss = 2):
    '''
    spectrum: a Numpy array containing the flux values of a pre-clipped and normalized (if applicable) star spectrum that will be smoothed
    gauss: an optional numerical argument to be used as the standard deviation of the Gaussian kernel
    
    gauss will default to 2 if no other value is provided (this value was determined based on previous work by A. Kamath)
    
    Returns smoothSpec, a Numpy array containing the modified/smoothed flux values of the original spectrum
    '''
    kernel = Gaussian1DKernel(gauss)
    smoothSpec = convolve(spectrum, kernel)
    return smoothSpec

#defines a function that will coadd normalized spectra based on provided lists of spectra and ivar weights
def coadd(spectra, ivars, lower = lowerThresh, upper = upperThresh):
    '''
    spectra: a list of the spectra of each star that is being coadded
    ivars: a list of the sets of ivars for each star that is being coadded
    lower: the lower boundary of the range the spectrum have been normalized on, in terms of wavelength
    upper: the upper boundary of the range the spectrum have been normalized on, in terms of wavelength
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    spectra and ivar values must be normalized and corresponding for best results
    
    Returns wvValues_np, a Numpy array containing every wavelength value for which spectral data exists between lower and upper
    Also returns coaddedFlux_np, a Numpy array containing the flux values of the coadded spectrum corresponding to each wavelength in wvValues_np 
    Also returns coaddedIvar_np, a Numpy array containing the ivar values of the coadded spectrum corresponding to each wavelength in  wvValues_np
    '''
    wvValues = data["LBIN"][0][lower:upper].tolist()
    coaddedFlux = []
    coaddedIvar = []
    for wv in range(len(wvValues)):
        fluxSum = 0
        ivarSum = 0
        for star in range(len(spectra)):
            if not np.isnan(spectra[star][wv]): 
                fluxSum += spectra[star][wv] * ivars[star][wv] #multiplying by ivar as a weight, as per formula
                ivarSum += ivars[star][wv]
        if ivarSum == 0:
            coaddedFlux.append(np.nan)
            coaddedIvar.append(0)
        else:
            newFlux = fluxSum/ivarSum #normalizing the weights, as per formula
            coaddedFlux.append(newFlux) #coaddedSpec now contains a coadded spectrum value at each LBIN value
            coaddedIvar.append(ivarSum) #coadded ivar is simply the sum of all ivars for a certain bin, across all stars (based on a mathematical proof)
    for flux in range(len(coaddedFlux)):
        if coaddedFlux[flux] == 0:
            coaddedFlux[flux] = np.nan
    coaddedFlux_np = np.array(coaddedFlux)
    wvValues_np = np.array(wvValues)
    coaddedIvar_np = np.array(coaddedIvar)
    return wvValues_np, coaddedFlux_np, coaddedIvar_np

#defines a function that creates a coadded template spectrum by combining several spectra normalized over a certain range
def getTempSpec(starIndices, lower = lowerThresh, upper = upperThresh, nsigma = 3.5):
    '''
    starIndices: an array or list of the indices of the stars to be coadded (which correspond to spectra in SPLASH)
    lower: the lower boundary of the range the spectrum will be normalized on, in terms of wavelength
    upper: the upper boundary of the range the spectrum will be normalized on, in terms of wavelength
    nsigma: a numerical value representing the number of standard deviations away from the flux medians that data for the coaddition will be trimmed to
    
    lower and upper will default to the lowerThresh and upperThresh values defined above, respectively
    nsigma will default to 3.5 unless another value is provided
    
    Returns coadd_wv, the set of all wavelength values for which data exists in the final coadded spectrum
    Also returns coadd_spec, the set of all flux values that correspond to the wavelength values in coadd_wv for the final coadded spectrum
    Also returns coadd_ivar, the number representing the ivar for every wavelength value on the final coadded spectrum
    Note that the final spectrum has been created from nsigma clipped data
    '''
    #creating lists of normalized spectra and ivars for each star to be coadded
    spectra = []
    ivars = []
    for star in starIndices:
        if lower == lowerThresh and upper == upperThresh:
            #spectra.append(applyGauss(splashSpecs_dict[star],gauss=6).tolist())
            spectra.append(splashSpecs_dict[star].tolist())
            ivars.append(splashIvars_dict[star].tolist())
        else:
            starSpec, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
            #starSpec = applyGauss(starSpec,gauss=6)
            spectra.append(starSpec.tolist())
            ivars.append(starIvar.tolist())
    print("Step 1 of 3 complete. Normalized spectra and inverse variances loaded.")

    #takes the original data and replaces with NaN values any data points that are greater than nsigma standard deviations from the median
    #the median is defined as the median flux at each wavelength when the flux values for every spectrum at that wavelength are considered
    spectra_new, ivars_new = sigmaClip(spectra, ivars, nsigma)
    print("Step 2 of 3 complete. Spectral and ivar data clipped to " + str(nsigma) + " sigmas.")

    #performs the coaddition with the edited data from the last step
    coadd_wv, coadd_spec, coadd_ivar = coadd(spectra_new, ivars_new, lower, upper)
    print("Step 3 of 3 complete. Coaddition of spectra and ivars performed and template created.")
    
    return coadd_wv, coadd_spec, coadd_ivar

#helper function for trimStars
#defines a function which will give a fraction (probability) to run through trimStars, based on bin-based-ratio
def ratioLim(sampleData, medIvar, ratio, modelData):   
    '''
    algorithm
    a. find bin which this ivar falls under
    b. multiply the count of weakCN/carbon (model population) by the constant ratio previously created for that bin
    c. divide the result of (b) by the total number of stars in the current bin from the kphOthers population (the one to trim)
    
    sampleData: the dataset that you wish to clip/modify based on ivar
    medIvar: the ivar to operate with
    ratio: a numerical value representing the ratio of other stars population to model carbon/weak CN population previously found 
    modelData: the dataset that you are trying to match the sampleData to through ivar clipping
    buckets: the buckets with which to operate - if the log-ivar histogram bins are changed, then this must be changed as well
    
    Note that buckets will default to an array from 0 to 6 with a step size of 0.25 unless otherwise assigned
    
    Returns a numerical value representing a probability that will be used to choose a certain number of stars from the sample population
    '''
    buckets = sampleData[1]
    logIv = np.log(medIvar)/np.log(10)
    step = buckets[1] - buckets[0]
    index = int((logIv - buckets[0])/step)
    goalCount = ratio*modelData[0][index]
    return goalCount/sampleData[0][index]

#defines a function that takes in a sample of stars and trims that sample according to the ivars of those stars
def trimStars(starSample, medIvDict, modelData, ratio, sampleData):
    '''
    starSample: the sample of stars that you wish to trim based on ivar
    medIvDict: a dictionary of the median ivars for the stars in starSample; keys are the indices of the stars and values are the median ivars
    modelData: the model histogram data for bin-based ratio calculation (wNm_logHighData or carb_logHighData)
    ratio: the ratio of other:model to use for trimming the starSample dataset
    sampleData: the sample histogram data for bin-based ratio calculation (kphO_logHighData); a 2D array representing the histogram frequencies/bins that were used to find the ratio
    
    Returns pickStar, an array of Boolean values corresponding to the stars in starSample. True indicates that the star should be selected and False indicates that it should not be selected
    Also returns trimmed_ivarMeds, an array containing the median ivar values for each of the stars in the new trimmed sample. In other words, these median ivars correspond to the "True"s in the pickStar array
    '''
    pickStar = np.zeros(len(starSample), dtype = bool)
    trimmed_ivarMeds = []
    index = 0
    for star in starSample: #set to choose from is keckPhotoOthers (not the entire others set)
        if star not in medIvDict:
            index += 1 #must actually skip this index in pickStar, on top of just moving to next star
            continue
        starMedIvar = medIvDict[star]
        
        #find the fraction to pick, and thus probability of picking, based on the function: limFunc(starMedIvar)
        pickFraction = ratioLim(sampleData, starMedIvar, ratio, modelData)
        if random.random() < pickFraction: #random.random() generates a floating point number from [0.0, 1.0)
            pickStar[index] = True #this star will be picked for the new sample
            trimmed_ivarMeds.append(starMedIvar) #add this star's median ivar to a list, to plot in the new normalized histogram
        index += 1
    return pickStar, trimmed_ivarMeds

#creates a function that calculates the score (a measure of the difference between two spectra) for a certain star's spectrum compared to a certain template spectrum
#note that the input spectra and ivars must be already sliced and normalized
def getScore(templateSpec, scienceSpec, ivars):
    '''
    templateSpec: a Numpy array containing the flux values of the coadded template spectrum
    scienceSpec: a Numpy array containing the flux values at each wavelength of the normalized spectrum for a specific star
    ivars: a Numpy array containing the normalized ivar values at each wavelength of the spectrum for a specific star
    
    Returns finalScore, a float representing the similarity between the two spectra. A lower score indicates greater similarity and a higher score indicates lower similarity
    Note that spectra that cannot be normalized will not have a score associated with them. The function returns a NaN as the score for these spectra
    '''
    templateSpec_list = templateSpec.tolist()
    scienceSpec_list = scienceSpec[~np.isnan(scienceSpec)].tolist()
    if scienceSpec_list == []:
        print("ScoreError: Because this spectrum cannot be normalized, its score cannot be found.")
        return np.nan
    else:
        summedScore = 0
        for wv in range(len(templateSpec_list)):
            score = ((scienceSpec[wv] - templateSpec[wv])**2) * ivars[wv]
            if not np.isnan(score):
                summedScore += score
        summedIvars = 0
        for ivar in ivars:
            if not np.isnan(ivar):
                summedIvars += ivar
        finalScore = (summedScore/summedIvars)**0.5
        return finalScore

#test execution of function
specStar, ivarStar = splashSpecs_dict[19534],splashIvars_dict[19534]
print("Score: " + str(getScore(Wtemplate, specStar, ivarStar)))


#MODIFIED TEMPLATE FUNCTIONS

#defines a function that calculates the slope of a spectrum's wavelength vs flux graph on a certain interval
def getSlope(spectrum, lower = lowerThresh, upper = upperThresh):
    '''
    spectrum: a list containing the sliced, normalized flux values of the spectrum to be analyzed
    lower: the lower boundary of the range the spectrum will be normalized on, in terms of wavelength index in data["LBIN"]
    upper: the upper boundary of the range the spectrum will be normalized on, in terms of wavelength index in data["LBIN"]
    
    lower and upper will default to the previously defined lowerThresh and upperThresh values, respectively
    
    Returns slope, a numerical value representing the slope of the spectrum on the wavelength range lower to upper
    '''
    lowerInt, upperInt, = spectrum[:slopeWindow], spectrum[-slopeWindow:]
    lowerMedian = np.nanmedian(lowerInt)
    upperMedian = np.nanmedian(upperInt)
    diff = (data['LBIN'][0][upper] - data['LBIN'][0][lower])
    slope = (upperMedian - lowerMedian)/diff
    return slope

#defines a function that modifies the slope of a spectrum by distorting the graph
def getTiltedSpec(spectrum, slope, lower = lowerThresh, upper = upperThresh):
    '''
    spectrum: a Numpy array containing the flux values of a (likely normalized) star spectrum to be modified
    slope: a numerical value (most often a float) containing the slope to be applied to the spectrum
    lower: the index of the lower boundary of spectrum in data["LBIN"]
    upper: the index of the upper boundary of spectrum in data["LBIN"]
    
    Note that lower and upper will default to the predefined lowerThresh and upperThresh, respectively
    
    Returns titled, a Numpy array containing the flux values of the adjusted star spectrum
    '''
    deltaLam = data['LBIN'][0][lower:upper] - data['LBIN'][0][lower] #note that the used of data["LBIN"][0] is arbitrary because all spectra have the same LBIN data values
    tilted = slope*deltaLam + spectrum #creating a copy of the spectrum with the new slope
    tilted = tilted/np.nanmedian(tilted) #normalizes the spectrum again
    return tilted

#defines a function that scales a spectrum by either enhancing or reducing spectral features
def getScaledSpec(spectrum, c):
    '''
    spectrum: a Numpy array containing the flux values of a (likely normalized) star spectrum to be scaled
    c: a numerical value representing the scale factor by which the spectrum will be modified
    
    Returns scaledFlux, a Numpy array containing the flux values of the adjusted star spectrum
    '''
    if c == -1: #accounts for potential divide by zero errors in scaling function
        raise ZeroDivisionError("c cannot be -1 due to the structure of the scaling formula.")
    scaledFlux = (spectrum + c)/(1+c)
    return scaledFlux

#defines a helper function for findOptimalC that produces a list of all the c-values to be tested
def getRanger(start, stop, step, zoomStart = None, zoomStop = None, zoomStep = None):
    '''
    start: a numerical value representing the lowest value of c you want to test
    stop: a numerical value representing the highest value of c you want to test
    step: a numerical value representing the increment at which you want to create new values of c between start and stop
    zoomStart: a numerical value representing the lowest value of c you want to test with a finer step size
    zoomStop: a numerical value representing the highest value of c you want to test with a finer step size
    zoomStep: a numerical value representing the finer step size to be applied between zoomStop and zoomStart
    
    Note that zoomStart, zoomStop, and zoomStep will all default to None.
    
    Returns rangeList, a list of the scale-factor (c) values to be iterated over for a star that is undergoing the template-matching process
    '''
    rangeList = []
    i = start
    if zoomStart == None:
        zoomStart = start 
        zoomStop = stop
        zoomStep = step
    while i < stop:
        rangeList.append(i)
        if i >= zoomStart and i <= zoomStop: #'zooms in' on a part of the c-range to get smaller c increments just on that range
            i += zoomStep 
        else:
            i += step
    return rangeList

#defines a function that sorts through the c-values produced by getRanger and returns the c-value and s-value that result in the lowest score
def findOptimalC(star, template = Ctemplate_full, lower = lowerThresh, upper = upperThresh, lowC = -15, highC = 60, step = 0.5, zoomStart = None, zoomStop = None, zoomStep = None, trackC = False, gauss = False, nsigma = 10):
    '''
    Note that default argument values have been denoted with () in the descriptions below
    
    star: the index of the star for which the template is to be modified to fit and the score is to be calculated
    template: (Ctemplate_full) the template spectrum that will be used to compare the spectrum of star to
    lower: (lowerThresh) the lower boundary for the spectra of star and template to be normalized on/analyzed on
    upper: (upperThresh) the upper boundary for the spectra of star and template to be normalized on/analyzed on
    lowC: (-15) the lower boundary for the range of c-values to be tested
    highC: (60) the upper boundary for the range of c-values to be tested
    step: (0.5) the increment by which c-values between lowC and highC are tested
    zoomStart: (None) the lower boundary for a range of c-values to be tested with a finer increment size
    zoomStop: (None) the upper boundary for a range of c-values to be tested with a finer increment size
    zoomStep: (None) the finer increment size by which c-values between zoomStart and zoomStop are tested
    trackC: (False) a Boolean value that, if True, will produce a graph of c-values and their corresponding scores as the function steps through the c-range; if False, the graph will not be produced
    gauss: (False) a Boolean value that, if True, will apply a Gaussian smoothing kernel to both the template and science spectra before optimal C is found
    nsigma: (10) a numerical value representing the width of the Gaussian kernel that will be used if gauss is True
    
    Returns bestC, the scale factor that yielded the lowest score
    Also returns bestS, the slope that yielded the lowest score (note that this is the version of the slope that has been changed according to the scaled spectrum)
    Also returns bestScore, the "lowest score" referred to above
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("ScoreError: Because this spectrum cannot be normalized, its score cannot be found.")
        return np.nan, np.nan, np.nan
    bestScore = None
    bestC, bestS = 0, 0
    c_coords, sc_coords = [], []
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope(starFlux, lower, upper)
    for c in getRanger(lowC, highC, step, zoomStart, zoomStop, zoomStep):
        if c == -1:
            continue
        scaledFlux = getScaledSpec(clipTemplate, c)
        s = slope - getSlope(scaledFlux, lower, upper)
        tiltedFlux = getTiltedSpec(scaledFlux, s, lower, upper)
        testScore = getScore(tiltedFlux, starFlux, starIvar)
        if trackC:
            c_coords.append(c)
            sc_coords.append(testScore)
        if bestScore == None or testScore < bestScore:
            bestC, bestS = c, s
            bestScore = testScore
    if trackC:
        plt.plot(c_coords, sc_coords)
        plt.rcParams['figure.figsize'] = 30,11 
        plt.title("C-Value vs. Score for Star " + str(star), size = 30) 
        plt.ylabel("Score", size = 20)
        plt.xlabel("C (Scale) Value", size = 20)
        plt.legend(fontsize = 20)
        plt.ylim(0,0.2)
        plt.show()
    return bestC, bestS, bestScore

#defines a function that allows a user to see the modified template "findOptimalC" chose as the best match for a star
def seeModTemplate(star, c , template = Ctemplate_full, lower = lowerThresh, upper = upperThresh, starSpec = True, unmodified = False, gauss = False, nsigma = 10):
    '''
    star: the index number of the star that was analyzed with findOptimalC
    c: the c (scale) value that was produced when findOptimalC was run on star
    template: the template spectrum to be modified
    lower: the lower boundary on which the template spectrum is to be modified
    upper: the upper boundary on which the template spectrum is to be modified
    starSpec: a Boolean value that, if True, will graph star's science spectrum along with the modified template; if False, only the modified template will be graphed
    unmodified: a Boolean value that, if True, will graph the unmodified template along with the modified template; if False, only the modified template will be graphed
    
    Note that template, lower, upper, starSpec, and unmodified default to Ctemplate_full, lowerThresh, upperThresh, True, and False
    
    This function will return None while also producing a graph of the modified template spectrum along with any other auxilary graphs chosen by the user based on inputs
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("NormalizationError: Because this spectrum cannot be normalized, this method cannot be applied to it.")
        return None
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
   # if gauss:
    #    clipTemplate = applyGauss(clipTemplate, nsigma)
      #  starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope(starFlux, lower, upper)
    scaledFlux = getScaledSpec(clipTemplate, c)
    s = slope - getSlope(scaledFlux, lower, upper)
    tiltedFlux = getTiltedSpec(scaledFlux, s, lower, upper)
    plt.plot(data["LBIN"][0][lower:upper], tiltedFlux, color = 'b', label = ('Modified Template (C = ' + str(c) +')'))
    if starSpec:
        plt.plot(data["LBIN"][0][lower:upper], starFlux, color = 'orange', label = 'Science Spectrum')
    if unmodified:
        plt.plot(data["LBIN"][0][lower:upper], clipTemplate, label = 'Unmodified Template')
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title("Modified Template Comparison for Star " + str(star), size = 30) 
    plt.ylabel("Flux", size = 20)
    plt.xlabel("Wavelength", size = 20)
    plt.legend(fontsize = 20)
    plt.show()

#defines a function that sorts through the c-values produced by getRanger and returns the c-value and s-value that result in the lowest score
def findWOptimalC(star, template = Wtemplate_full, lower = lowerThresh, upper = upperThresh, lowC = -15, highC = 60, step = 0.5, zoomStart = None, zoomStop = None, zoomStep = None, trackC = False, gauss = False, nsigma = 10):
    '''
    Note that default argument values have been denoted with () in the descriptions below
    
    star: the index of the star for which the template is to be modified to fit and the score is to be calculated
    template: (Ctemplate_full) the template spectrum that will be used to compare the spectrum of star to
    lower: (lowerThresh) the lower boundary for the spectra of star and template to be normalized on/analyzed on
    upper: (upperThresh) the upper boundary for the spectra of star and template to be normalized on/analyzed on
    lowC: (-15) the lower boundary for the range of c-values to be tested
    highC: (60) the upper boundary for the range of c-values to be tested
    step: (0.5) the increment by which c-values between lowC and highC are tested
    zoomStart: (None) the lower boundary for a range of c-values to be tested with a finer increment size
    zoomStop: (None) the upper boundary for a range of c-values to be tested with a finer increment size
    zoomStep: (None) the finer increment size by which c-values between zoomStart and zoomStop are tested
    trackC: (False) a Boolean value that, if True, will produce a graph of c-values and their corresponding scores as the function steps through the c-range; if False, the graph will not be produced
    gauss: (False) a Boolean value that, if True, will apply a Gaussian smoothing kernel to both the template and science spectra before optimal C is found
    nsigma: (10) a numerical value representing the width of the Gaussian kernel that will be used if gauss is True
    
    Returns bestC, the scale factor that yielded the lowest score
    Also returns bestS, the slope that yielded the lowest score (note that this is the version of the slope that has been changed according to the scaled spectrum)
    Also returns bestScore, the "lowest score" referred to above
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("ScoreError: Because this spectrum cannot be normalized, its score cannot be found.")
        return np.nan, np.nan, np.nan
    bestScore = None
    bestC, bestS = 0, 0
    c_coords, sc_coords = [], []
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope(starFlux, lower, upper)
    for c in getRanger(lowC, highC, step, zoomStart, zoomStop, zoomStep):
        if c == -1:
            continue
        scaledFlux = getScaledSpec(clipTemplate, c)
        s = slope - getSlope(scaledFlux, lower, upper)
        tiltedFlux = getTiltedSpec(scaledFlux, s, lower, upper)
        testScore = getScore(tiltedFlux, starFlux, starIvar)
        if trackC:
            c_coords.append(c)
            sc_coords.append(testScore)
        if bestScore == None or testScore < bestScore:
            bestC, bestS = c, s
            bestScore = testScore
    if trackC:
        plt.plot(c_coords, sc_coords)
        plt.rcParams['figure.figsize'] = 30,11 
        plt.title("C-Value vs. Score for Star " + str(star), size = 30) 
        plt.ylabel("Score", size = 20)
        plt.xlabel("C (Scale) Value", size = 20)
        plt.legend(fontsize = 20)
        plt.ylim(0,0.2)
        plt.show()
    return bestC, bestS, bestScore

#defines a function that allows a user to see the modified template "findOptimalC" chose as the best match for a star
def seeModWTemplate(star, c , template = Wtemplate_full, lower = lowerThresh, upper = upperThresh, starSpec = True, unmodified = False, gauss = False, nsigma = 10):
    '''
    star: the index number of the star that was analyzed with findOptimalC
    c: the c (scale) value that was produced when findOptimalC was run on star
    template: the template spectrum to be modified
    lower: the lower boundary on which the template spectrum is to be modified
    upper: the upper boundary on which the template spectrum is to be modified
    starSpec: a Boolean value that, if True, will graph star's science spectrum along with the modified template; if False, only the modified template will be graphed
    unmodified: a Boolean value that, if True, will graph the unmodified template along with the modified template; if False, only the modified template will be graphed
    
    Note that template, lower, upper, starSpec, and unmodified default to Ctemplate_full, lowerThresh, upperThresh, True, and False
    
    This function will return None while also producing a graph of the modified template spectrum along with any other auxilary graphs chosen by the user based on inputs
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("NormalizationError: Because this spectrum cannot be normalized, this method cannot be applied to it.")
        return None
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
   # if gauss:
    #    clipTemplate = applyGauss(clipTemplate, nsigma)
      #  starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope(starFlux, lower, upper)
    scaledFlux = getScaledSpec(clipTemplate, c)
    s = slope - getSlope(scaledFlux, lower, upper)
    tiltedFlux = getTiltedSpec(scaledFlux, s, lower, upper)
    plt.plot(data["LBIN"][0][lower:upper], tiltedFlux, color = 'b', label = ('Modified Template (C = ' + str(c) +')'))
    if starSpec:
        plt.plot(data["LBIN"][0][lower:upper], starFlux, color = 'orange', label = 'Science Spectrum')
    if unmodified:
        plt.plot(data["LBIN"][0][lower:upper], clipTemplate, label = 'Unmodified Template')
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title("Modified Template Comparison for Star " + str(star), size = 30) 
    plt.ylabel("Flux", size = 20)
    plt.xlabel("Wavelength", size = 20)
    plt.legend(fontsize = 20)
    plt.show()

#MODIFIED UNDILUTED
#defines an important variable that will be used for finding the slopes of spectra
#the slope will not be calculated based on random points, but on the median point in a small "window" at either end of the spectrum
slopeWindow = 25

#defines a function that calculates the slope of a spectrum's wavelength vs flux graph on a certain interval
def getSlope2(spectrum, lower = lowerThresh, upper = upperThresh):
    '''
    spectrum: a list containing the sliced, normalized flux values of the spectrum to be analyzed
    lower: the lower boundary of the range the spectrum will be normalized on, in terms of wavelength index in data["LBIN"]
    upper: the upper boundary of the range the spectrum will be normalized on, in terms of wavelength index in data["LBIN"]
    
    lower and upper will default to the previously defined lowerThresh and upperThresh values, respectively
    
    Returns slope, a numerical value representing the slope of the spectrum on the wavelength range lower to upper
    '''
    lowerInt, upperInt, = spectrum[:slopeWindow], spectrum[-slopeWindow:]
    lowerMedian = np.nanmedian(lowerInt)
    upperMedian = np.nanmedian(upperInt)
    diff = (data['LBIN'][0][upper] - data['LBIN'][0][lower])
    slope = (upperMedian - lowerMedian)/diff
    return slope

#defines a function that modifies the slope of a spectrum by distorting the graph
def getTiltedSpec2(spectrum, slope, lower = lowerThresh, upper = upperThresh):
    '''
    spectrum: a Numpy array containing the flux values of a (likely normalized) star spectrum to be modified
    slope: a numerical value (most often a float) containing the slope to be applied to the spectrum
    lower: the index of the lower boundary of spectrum in data["LBIN"]
    upper: the index of the upper boundary of spectrum in data["LBIN"]
    
    Note that lower and upper will default to the predefined lowerThresh and upperThresh, respectively
    
    Returns titled, a Numpy array containing the flux values of the adjusted star spectrum
    '''
    deltaLam = data['LBIN'][0][lower:upper] - data['LBIN'][0][lower] #note that the used of data["LBIN"][0] is arbitrary because all spectra have the same LBIN data values
    tilted = slope*deltaLam + spectrum #creating a copy of the spectrum with the new slope
    tilted = tilted/np.nanmedian(tilted) #normalizes the spectrum again
    return tilted

#defines a function that scales a spectrum by either enhancing or reducing spectral features
def getScaledSpec2(spectrum, c):
    '''
    spectrum: a Numpy array containing the flux values of a (likely normalized) star spectrum to be scaled
    c: a numerical value representing the scale factor by which the spectrum will be modified
    
    Returns scaledFlux, a Numpy array containing the flux values of the adjusted star spectrum
    '''
    if c == -1: #accounts for potential divide by zero errors in scaling function
        raise ZeroDivisionError("c cannot be -1 due to the structure of the scaling formula.")
    scaledFlux = (spectrum + c)/(1+c)
    return scaledFlux


#defines a helper function for findOptimalC that produces a list of all the c-values to be tested
def getRanger2(start, stop, step, zoomStart = None, zoomStop = None, zoomStep = None):
    '''
    start: a numerical value representing the lowest value of c you want to test
    stop: a numerical value representing the highest value of c you want to test
    step: a numerical value representing the increment at which you want to create new values of c between start and stop
    zoomStart: a numerical value representing the lowest value of c you want to test with a finer step size
    zoomStop: a numerical value representing the highest value of c you want to test with a finer step size
    zoomStep: a numerical value representing the finer step size to be applied between zoomStop and zoomStart
    
    Note that zoomStart, zoomStop, and zoomStep will all default to None.
    
    Returns rangeList, a list of the scale-factor (c) values to be iterated over for a star that is undergoing the template-matching process
    '''
    rangeList = []
    c = 0
    rangeList.append(c)
    return rangeList
    
    #Previous code from 2.6
    ''' 
    if zoomStart == None:
        zoomStart = start 
        zoomStop = stop
        zoomStep = step
    while i < stop:
        rangeList.append(i)
        if i >= zoomStart and i <= zoomStop: #'zooms in' on a part of the c-range to get smaller c increments just on that range
            i += zoomStep 
        else:
            i += step
    return rangeList
    '''

#defines a function that sorts through the c-values produced by getRanger and returns the c-value and s-value that result in the lowest score
def findWOptimalC2(star, template = Wtemplate_full, lower = lowerThresh, upper = upperThresh, lowC = -15, highC = 60, step = 0.5, zoomStart = None, zoomStop = None, zoomStep = None, trackC = False, gauss = False, nsigma = 10):
    '''
    Note that default argument values have been denoted with () in the descriptions below
    
    star: the index of the star for which the template is to be modified to fit and the score is to be calculated
    template: (Ctemplate_full) the template spectrum that will be used to compare the spectrum of star to
    lower: (lowerThresh) the lower boundary for the spectra of star and template to be normalized on/analyzed on
    upper: (upperThresh) the upper boundary for the spectra of star and template to be normalized on/analyzed on
    lowC: (-15) the lower boundary for the range of c-values to be tested
    highC: (60) the upper boundary for the range of c-values to be tested
    step: (0.5) the increment by which c-values between lowC and highC are tested
    zoomStart: (None) the lower boundary for a range of c-values to be tested with a finer increment size
    zoomStop: (None) the upper boundary for a range of c-values to be tested with a finer increment size
    zoomStep: (None) the finer increment size by which c-values between zoomStart and zoomStop are tested
    trackC: (False) a Boolean value that, if True, will produce a graph of c-values and their corresponding scores as the function steps through the c-range; if False, the graph will not be produced
    gauss: (False) a Boolean value that, if True, will apply a Gaussian smoothing kernel to both the template and science spectra before optimal C is found
    nsigma: (10) a numerical value representing the width of the Gaussian kernel that will be used if gauss is True
    
    Returns bestC, the scale factor that yielded the lowest score
    Also returns bestS, the slope that yielded the lowest score (note that this is the version of the slope that has been changed according to the scaled spectrum)
    Also returns bestScore, the "lowest score" referred to above
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("ScoreError: Because this spectrum cannot be normalized, its score cannot be found.")
        return np.nan, np.nan, np.nan
    bestScore = None
    bestC, bestS = 0, 0
    c_coords, sc_coords = [], []
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope2(starFlux, lower, upper)
    for c in getRanger2(lowC, highC, step, zoomStart, zoomStop, zoomStep):
        if c == -1:
            continue
        scaledFlux = getScaledSpec2(clipTemplate, c)
        s = slope - getSlope2(scaledFlux, lower, upper)
        tiltedFlux = getTiltedSpec2(scaledFlux, s, lower, upper)
        testScore = getScore(tiltedFlux, starFlux, starIvar)
        if trackC:
            c_coords.append(c)
            sc_coords.append(testScore)
        if bestScore == None or testScore < bestScore:
            bestC, bestS = c, s
            bestScore = testScore
    if trackC:
        plt.plot(c_coords, sc_coords)
        plt.rcParams['figure.figsize'] = 30,11 
        plt.title("C-Value vs. Score for Star " + str(star), size = 30) 
        plt.ylabel("Score", size = 20)
        plt.xlabel("C (Scale) Value", size = 20)
        plt.legend(fontsize = 20)
        plt.ylim(0,0.2)
        plt.show()
    return bestC, bestS, bestScore

#defines a function that allows a user to see the modified template "findOptimalC" chose as the best match for a star
def seeModWTemplate2(star, c=0, template = Wtemplate_full, lower = lowerThresh, upper = upperThresh, starSpec = True, unmodified = False, gauss = False, nsigma = 10):
    '''
    star: the index number of the star that was analyzed with findOptimalC
    c: the c (scale) value that was produced when findOptimalC was run on star
    template: the template spectrum to be modified
    lower: the lower boundary on which the template spectrum is to be modified
    upper: the upper boundary on which the template spectrum is to be modified
    starSpec: a Boolean value that, if True, will graph star's science spectrum along with the modified template; if False, only the modified template will be graphed
    unmodified: a Boolean value that, if True, will graph the unmodified template along with the modified template; if False, only the modified template will be graphed
    
    Note that template, lower, upper, starSpec, and unmodified default to Ctemplate_full, lowerThresh, upperThresh, True, and False
    
    This function will return None while also producing a graph of the modified template spectrum along with any other auxilary graphs chosen by the user based on inputs
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("NormalizationError: Because this spectrum cannot be normalized, this method cannot be applied to it.")
        return None
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope2(starFlux, lower, upper)
    scaledFlux = getScaledSpec2(clipTemplate, c)
    s = slope - getSlope2(scaledFlux, lower, upper)
    tiltedFlux = getTiltedSpec2(scaledFlux, s, lower, upper)
    plt.plot(data["LBIN"][0][lower:upper], tiltedFlux, color = 'b', label = ('Modified Template (C = ' + str(c) +')'))
    if starSpec:
        plt.plot(data["LBIN"][0][lower:upper], starFlux, color = 'orange', label = 'Science Spectrum')
    if unmodified:
        plt.plot(data["LBIN"][0][lower:upper], clipTemplate, label = 'Unmodified Template')
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title("Modified Template Comparison for Star " + str(star), size = 30) 
    plt.ylabel("Flux", size = 20)
    plt.xlabel("Wavelength", size = 20)
    plt.legend(fontsize = 20)
    plt.show()

#defines a function that sorts through the c-values produced by getRanger and returns the c-value and s-value that result in the lowest score
def findCOptimalC2(star, template = Ctemplate_full, lower = lowerThresh, upper = upperThresh, lowC = -15, highC = 60, step = 0.5, zoomStart = None, zoomStop = None, zoomStep = None, trackC = False, gauss = False, nsigma = 10):
    '''
    Note that default argument values have been denoted with () in the descriptions below
    
    star: the index of the star for which the template is to be modified to fit and the score is to be calculated
    template: (Ctemplate_full) the template spectrum that will be used to compare the spectrum of star to
    lower: (lowerThresh) the lower boundary for the spectra of star and template to be normalized on/analyzed on
    upper: (upperThresh) the upper boundary for the spectra of star and template to be normalized on/analyzed on
    lowC: (-15) the lower boundary for the range of c-values to be tested
    highC: (60) the upper boundary for the range of c-values to be tested
    step: (0.5) the increment by which c-values between lowC and highC are tested
    zoomStart: (None) the lower boundary for a range of c-values to be tested with a finer increment size
    zoomStop: (None) the upper boundary for a range of c-values to be tested with a finer increment size
    zoomStep: (None) the finer increment size by which c-values between zoomStart and zoomStop are tested
    trackC: (False) a Boolean value that, if True, will produce a graph of c-values and their corresponding scores as the function steps through the c-range; if False, the graph will not be produced
    gauss: (False) a Boolean value that, if True, will apply a Gaussian smoothing kernel to both the template and science spectra before optimal C is found
    nsigma: (10) a numerical value representing the width of the Gaussian kernel that will be used if gauss is True
    
    Returns bestC, the scale factor that yielded the lowest score
    Also returns bestS, the slope that yielded the lowest score (note that this is the version of the slope that has been changed according to the scaled spectrum)
    Also returns bestScore, the "lowest score" referred to above
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("ScoreError: Because this spectrum cannot be normalized, its score cannot be found.")
        return np.nan, np.nan, np.nan
    bestScore = None
    bestC, bestS = 0, 0
    c_coords, sc_coords = [], []
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope2(starFlux, lower, upper)
    for c in getRanger2(lowC, highC, step, zoomStart, zoomStop, zoomStep):
        if c == -1:
            continue
        scaledFlux = getScaledSpec2(clipTemplate, c)
        s = slope - getSlope2(scaledFlux, lower, upper)
        tiltedFlux = getTiltedSpec2(scaledFlux, s, lower, upper)
        testScore = getScore(tiltedFlux, starFlux, starIvar)
        if trackC:
            c_coords.append(c)
            sc_coords.append(testScore)
        if bestScore == None or testScore < bestScore:
            bestC, bestS = c, s
            bestScore = testScore
    if trackC:
        plt.plot(c_coords, sc_coords)
        plt.rcParams['figure.figsize'] = 30,11 
        plt.title("C-Value vs. Score for Star " + str(star), size = 30) 
        plt.ylabel("Score", size = 20)
        plt.xlabel("C (Scale) Value", size = 20)
        plt.legend(fontsize = 20)
        plt.ylim(0,0.2)
        plt.show()
    return bestC, bestS, bestScore

#defines a function that allows a user to see the modified template "findOptimalC" chose as the best match for a star
def seeModCTemplate2(star, c=0, template = Ctemplate_full, lower = lowerThresh, upper = upperThresh, starSpec = True, unmodified = False, gauss = False, nsigma = 10):
    '''
    star: the index number of the star that was analyzed with findOptimalC
    c: the c (scale) value that was produced when findOptimalC was run on star
    template: the template spectrum to be modified
    lower: the lower boundary on which the template spectrum is to be modified
    upper: the upper boundary on which the template spectrum is to be modified
    starSpec: a Boolean value that, if True, will graph star's science spectrum along with the modified template; if False, only the modified template will be graphed
    unmodified: a Boolean value that, if True, will graph the unmodified template along with the modified template; if False, only the modified template will be graphed
    
    Note that template, lower, upper, starSpec, and unmodified default to Ctemplate_full, lowerThresh, upperThresh, True, and False
    
    This function will return None while also producing a graph of the modified template spectrum along with any other auxilary graphs chosen by the user based on inputs
    '''
    if not normSpec(star = star, lower = lower, upper = upper)[3]:
        print("NormalizationError: Because this spectrum cannot be normalized, this method cannot be applied to it.")
        return None
    clipTemplate = template[lower:upper]
    starFlux, starIvar = normSpec(star = star, lower = lower, upper = upper)[1:3]
    if gauss:
        clipTemplate = applyGauss(clipTemplate, nsigma)
        starFlux = applyGauss(starFlux, nsigma)
    while all(np.isnan(starFlux[:slopeWindow])) or all(starFlux[:slopeWindow] == 0):
        starFlux, starIvar, clipTemplate = starFlux[slopeWindow:], starIvar[slopeWindow:], clipTemplate[slopeWindow:]
        lower += slopeWindow
    while all(np.isnan(starFlux[-slopeWindow:])) or all(starFlux[-slopeWindow:] == 0):
        starFlux, starIvar, clipTemplate = starFlux[:-slopeWindow], starIvar[:-slopeWindow], clipTemplate[:-slopeWindow]
        upper -= slopeWindow
    slope = getSlope2(starFlux, lower, upper)
    scaledFlux = getScaledSpec2(clipTemplate, c)
    s = slope - getSlope2(scaledFlux, lower, upper)
    tiltedFlux = getTiltedSpec2(scaledFlux, s, lower, upper)
    plt.plot(data["LBIN"][0][lower:upper], tiltedFlux, color = 'b', label = ('Modified Template (C = ' + str(c) +')'))
    if starSpec:
        plt.plot(data["LBIN"][0][lower:upper], starFlux, color = 'orange', label = 'Science Spectrum')
    if unmodified:
        plt.plot(data["LBIN"][0][lower:upper], clipTemplate, label = 'Unmodified Template')
    plt.rcParams['figure.figsize'] = 30,11 
    plt.title("Modified Template Comparison for Star " + str(star), size = 30) 
    plt.ylabel("Flux", size = 20)
    plt.xlabel("Wavelength", size = 20)
    plt.legend(fontsize = 20)
    plt.show()

#SLOPES

#defines a helper function for the getWSlopes function that determines what the slope of the line of best fit is for a certain graph
def findOptimalM(spectrum, lower, upper, lowM = -0.1, highM = 0.1, step = 0.001, zoomStart = -0.05, zoomStop = 0.05, zoomStep = 0.0001):
    '''
    spectrum: a Numpy array representing the flux values of a normalized, clipped (to the 'W' range) spectrum whose slopes are to be found
    lower: the lower boundary of the range that the slope is to be calculated over (represented by an index of the CLIPPED spectrum; 0 will be lowerThresh and so on)
    upper: the upper boundary of the range that the slope is to be calculated over (represented by an index of the CLIPPED spectrum; 0 will be lowerThresh and so on)
    lowM: the lower boundary of the range of m-values (slope values) to be tested when searching for the line of best fit
    highM: the upper boundary of the range of m-values (slope values) to be tested when searching for the line of best fit
    step: the increment value to use when creating new values of m to test between lowM and highM
    zoomStart: the lower boundary of the range of m-values that you want to test with a finer step size
    zoomStop: the upper boundary of the range of m-values that you want to test with a finer step size
    zoomStep: the finer increment value to be applied between zoomStart and zoomStop
    
    Note that lowM, highM, step, zoomStart, zoomStop, and zoomStep will default to -0.1, 0.1, 0.05, -0.05, 0.05, and 0.001, respectively
    
    Returns bestM, a numerical value representing the slope of the line of best fit for the spectrum on the range lower:upper
    Also returns bestB, the y-intercept value that corresponds to the bestM, to be used for graphing the chosen lines of best fit
    '''
    bestDiff, bestM, bestB = 1000000000000, None, None
    lbin = data["LBIN"][0][lowerThresh:upperThresh][lower:upper]
    mRanger = getRanger(lowM, highM, step, zoomStart, zoomStop, zoomStep) #note that this is the same function earlier defined in score calculation method 2
    for m in mRanger:
        b = ((np.nansum(spectrum)) - (m*(np.nansum(lbin))))/(len(lbin))
        testDiff = np.nansum(np.power((spectrum - m*lbin - b), 2))
        if testDiff < bestDiff:
            bestDiff = testDiff
            bestM, bestB = m, b
    return bestM, bestB

#defines a function that will check to see if the range covered by non-NaN values in a spectrum slice is greater than/equal to 75% of the length of the spectrum
#if the above condition is not met, the slope of the slice cannot be calculated
def isValidWindow(spectrum, windowLower, windowUpper):
    '''
    spectrum: a Numpy array containing the flux values of a normalized, trimmed (clipped to the 'W' range) spectrum whose slope is to be calculated
    windowLower: the lower boundary of the window of spectrum whose validity is to be tested
    windowUpper: the upper boundary of the window of spectrum whose validity is to be tested
    
    Returns False if the window specified has too many NaN values to be used for slope calculation, True otherwise
    '''
    testSpec = spectrum[windowLower:windowUpper]
    count1, count2 = windowLower, windowUpper
    for index in range(len(testSpec)): #finding the index of the first non-NaN value
        if np.isnan(testSpec[index]):
            count1 = index
        else:
            break
    for index in range(len(testSpec), 0, -1): #finding the index of the last non-NaN value
        if np.isnan(testSpec[index - 1]):
            count2 = index
        else:
            break
    if len(testSpec[count1:count2]) < 0.75*len(testSpec): #the range covered by the two indices found above must be greater than or equal to the length of the spectrum to be usable
        return False
    else:
        return True
    
#defines a function that calculates four slopes of a spectral graph
def getWSlopes(spectrum, lowM = -0.1, highM = 0.1, step = 0.001, zoomStart = -0.05, zoomStop = 0.05, zoomStep = 0.0001, getB = False):
    '''
    spectrum: a Numpy array or list containing the flux values for a spectrum whose slopes will be calculated; must already be normalized/clipped to the lowerThresh:upperThresh wavelength range
    lowM: the lower boundary of the range of m-values (slope values) to be tested when searching for the line of best fit
    highM: the upper boundary of the range of m-values (slope values) to be tested when searching for the line of best fit
    step: the increment value to use when creating new values of m to test between lowM and highM
    zoomStart: the lower boundary of the range of m-values that you want to test with a finer step size
    zoomStop: the upper boundary of the range of m-values that you want to test with a finer step size
    zoomStep: the finer increment value to be applied between zoomStart and zoomStop
    getB: a Boolean value that, if set to True, will cause the function to return both slope and y-intercept values for the lines of best fit created (if False, only slopes will be returned)
    
    Note that lowM, highM, step, zoomStart, zoomStop, zoomStep, and getB will default to -0.1, 0.1, 0.05, -0.05, 0.05, 0.001, and False, respectively
    
    Returns slope1, slope2, slope3, and slope4, the slopes of the spectral graph over the window ranges described above; these slopes are intended to represent the four slopes of the 'W' feature in carbon/weak CN spectra
    If getB is set to True, also returns the y-intercept values for the lines of best fit corresponding to slope1, slope2, slope3, and slope4
    '''
    #first, a preliminary check to see if the entire spectrum is NaN; if True, all returned slopes are NaN as well
    if all(np.isnan(spectrum)):
        print("SlopeError: This spectrum is all NaNs. No slopes can be found.")
        return np.nan, np.nan, np.nan, np.nan
    
    #if the above test is passed, the spectrum is divided into four windows
    window1, window2 = spectrum[windows[0][0]:windows[0][1]], spectrum[windows[1][0]:windows[1][1]]
    window3, window4 = spectrum[windows[2][0]:windows[2][1]], spectrum[windows[3][0]:windows[3][1]]
    calc1, calc2, calc3, calc4 = True, True, True, True
    
    #tests are done on each window to make sure that the presence of NaNs will not ruin the slope calculation
    #if the calculation cannot be performed, the calcX variables are set to False (meaning those slopes will not be calculated later) and the slopes to be returned are set to NaN
    if all(np.isnan(window1)) or not isValidWindow(spectrum, windows2[0][0], windows2[0][1]):
        print("SlopeError: This spectrum is dominated by NaNs in window 1. Slope1 could not be found.")
        slope1, calc1 = np.nan, False
    if all(np.isnan(window2)) or not isValidWindow(spectrum, windows2[1][0], windows2[1][1]):
        print("SlopeError: This spectrum is dominated by NaNs in window 2. Slope2 could not be found.")
        slope2, calc2 = np.nan, False
    if all(np.isnan(window3)) or not isValidWindow(spectrum, windows2[2][0], windows2[2][1]):
        print("SlopeError: This spectrum is dominated by NaNs in window 3. Slope3 could not be found.")
        slope3, calc3 = np.nan, False
    if all(np.isnan(window4)) or not isValidWindow(spectrum, windows2[3][0], windows2[3][1]):
        print("SlopeError: This spectrum is dominated by NaNs in window 4. Slope4 could not be found.")
        slope4, calc4 = np.nan, False
    
    #finally, slopes are calculated in each of the valid windows that were not previously weeded out above
    if calc1:
        slope1, b1 = findOptimalM(window1, windows[0][0], windows[0][1], lowM, highM, step, zoomStart, zoomStop, zoomStep)
    if calc2:
        slope2, b2 = findOptimalM(window2, windows[1][0], windows[1][1], lowM, highM, step, zoomStart, zoomStop, zoomStep)
    if calc3:
        slope3, b3 = findOptimalM(window3, windows[2][0], windows[2][1], lowM, highM, step, zoomStart, zoomStop, zoomStep)
    if calc4:
        slope4, b4 = findOptimalM(window4, windows[3][0], windows[3][1], lowM, highM, step, zoomStart, zoomStop, zoomStep)
        
    #the valid slopes and the NaN slopes (if any) are returned together in a tuple, along with the b-values (if specified)
    if getB:
        return slope1, slope2, slope3, slope4, b1, b2, b3, b4
    if not getB:
        return slope1, slope2, slope3, slope4
    
#defines a function that plots any set of star indices on a CMD
#note that the CMD is defined as followed: the x-axis is visible - infrared and the y-axis is infrared
def graphCMD(data, indices, markersize = 20, color = 'b', marker = 'o', label = None, facecolor = True, alpha = None):
    '''
    data: a Numpy array that consists of the dataset your color-magnitude information is contained in
    indices: a Numpy array or list of the star indices to be graphed on the CMD
    markersize: an integer representing the size of the dots to be plotted on the CMD
    color: a string representing the color of the dots to be plotted on the CMD
    marker: a string representing the shape of the dots to be plotted on the CMD
    label: a string representing the label that will be assigned in the legend of the CMD to the dots plotted here
    facecolor: a Boolean value that, if True, will fill in the dots plotted on the CMD and, if False, will plot open circles
    alpha: a numerical value representing the opacity of the markers to be graphed
    
    markersize, color, marker, label, facecolor, and alpha will default to 20, 'b' (blue), 'o', None, True, and None respectively
    
    Note that there is no return value, as well as no plt.show() statement. The plt.show() statement must be included in your code where this function is called
    '''
    if facecolor:
        plt.scatter((data.F814W[indices] - data.F160W[indices]), data.F160W[indices], s = markersize, color = color, marker = marker, label = label, alpha = alpha)
    else:
        plt.scatter((data.F814W[indices] - data.F160W[indices]), data.F160W[indices], s = markersize, color = color, marker = marker, label = label, facecolors = 'none', alpha = alpha)
        
#defines a function that will create a polygon to be graphed based on a set of points provided to be the vertices of the polygon
def polygon(points, color, label, linewidth = 2.5, linestyle = 'solid'):
    '''
    points: a list of points (with the x and y coordinates contained in a list) that will serve as the vertices of the polygon
    color: a string representing the color of the polygon's lines
    label: a string representing the label to be associated with the polygon in the legend of the graph
    linewidth: a numerical value representing the thickness of the polygon's outline
    linestyle: a string describing the style of the polygon's outline
    
    Note that linewidth and linestyle will default to 2.5 and 'solid', respectively.
    
    Returns polygon, a Python object that will show up on graphs that are being created when this function is called
    '''
    polygon = plt.Polygon(points, color = color, label = label, fill = False, linewidth = linewidth, linestyle = linestyle)
    plt.gca().add_patch(polygon)
    return polygon

#%%