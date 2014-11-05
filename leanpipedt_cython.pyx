import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

DTYPE = np.complex64
ctypedef np.complex64_t DTYPE_t

def sigma_clip(np.ndarray[np.float32_t, ndim=1] arr, float sigma=3):
    """ Function takes 1d array of values and returns the sigma-clipped min and max scaled by value "sigma".
    """

    assert arr.dtype == np.float32

    cdef np.ndarray[np.int_t, ndim=1] cliparr = np.arange(len(arr))
    cdef float mean
    cdef float std

    arr = np.append(arr,[np.float32(1)])    # append superfluous item to trigger loop
    while len(cliparr) != len(arr):
        arr = arr[cliparr]
        mean = arr.mean()
        std = arr.std()
        cliparr = np.where((arr < mean + sigma*std) & (arr > mean - sigma*std) & (arr != 0) )[0]
#        print 'Clipping %d from array of length %d' % (len(arr) - len(cliparr), len(arr))
    return mean - sigma*std, mean + sigma*std

cpdef make_triples(d):
    """ Calculates and returns data indexes (i,j,k) for all closed triples.
    """

    ants = d['ants']
    nants = d['nants']
    blarr = d['blarr']

    cdef int t
    cdef int ant1
    cdef int ant2
    cdef int ant3
    cdef np.ndarray[np.int_t, ndim=2] triples = np.zeros((nants*(nants-1)*(nants-2)/6, 3), dtype='int')

    # first make triples indexes in antenna numbering
    anttrips = []
    for i in ants:
        for j in ants[list(ants).index(i)+1:]:
            for k in ants[list(ants).index(j)+1:]:
                anttrips.append([i,j,k])
                
    # next return data indexes for triples
    for t in xrange(len(anttrips)):
        ant1 = anttrips[t][0]
        ant2 = anttrips[t][1]
        ant3 = anttrips[t][2]
        try:
            bl1 = np.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant2) )[0][0]
            bl2 = np.where( (blarr[:,0] == ant2) & (blarr[:,1] == ant3) )[0][0]
            bl3 = np.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant3) )[0][0]
            triples[t,0] = bl1
            triples[t,1] = bl2
            triples[t,2] = bl3
        except IndexError:
            continue

    return triples

@cython.profile(False)
cdef np.ndarray[np.float32_t, ndim=2] fringe_rotation(float dl, float dm, np.ndarray[np.float32_t, ndim=1] u, np.ndarray[np.float32_t, ndim=1] v, np.ndarray[np.float32_t, ndim=1] freq): 
    return np.exp(-2j*3.1415*(dl*np.outer(u,freq) + dm*np.outer(v,freq)))

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef phaseshift(np.ndarray[DTYPE_t, ndim=4] data0, d, float l1, float m1, np.ndarray[np.float32_t, ndim=2] u, np.ndarray[np.float32_t, ndim=2] v, verbose=0):
    """ Shift phase center to (l1, m1).
    Assumes single uv over all times in data0. Reasonable for up to a second or so of data.
    """

    cdef np.ndarray[complex, ndim=2] frot
    cdef np.ndarray[float, ndim=1] freq = d['freq']
    cdef np.ndarray[float, ndim=1] freq_orig = d['freq_orig']
    cdef float dl = l1 - d['l0']
    cdef float dm = m1 - d['m0']
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l

    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]

    if (dl != 0.) or (dm != 0.):
        frot = fringe_rotation(dl, dm, u[len0/2], v[len0/2], freq/freq_orig[0])
        for j in xrange(len1):
            for i in xrange(len0):
                for l in xrange(len3):    # iterate over pols
                    for k in xrange(len2):
                        data0[i,j,k,l] = data0[i,j,k,l] * frot[j,k]
    else:
        if verbose:
            print 'No phase rotation needed'

    d['l0'] = l1
    d['m0'] = m1

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef phaseshift_threaded(np.ndarray[DTYPE_t, ndim=4] data0, d, float l1, float m1, np.ndarray[np.float32_t, ndim=1] u, np.ndarray[np.float32_t, ndim=1] v, verbose=0):
    """ Shift phase center to (l1, m1).
    Assumes single uv over all times in data0. Reasonable for up to a second or so of data.
    """

    cdef np.ndarray[complex, ndim=2] frot
    cdef np.ndarray[float, ndim=1] freq = d['freq']
    cdef np.ndarray[float, ndim=1] freq_orig = d['freq_orig']
    cdef float dl = l1 - d['l0']
    cdef float dm = m1 - d['m0']
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l

    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]

    if (dl != 0.) or (dm != 0.):
        frot = fringe_rotation(dl, dm, u, v, freq/freq_orig[0])
        for j in xrange(len1):
            for i in xrange(len0):
                for l in xrange(len3):    # iterate over pols
                    for k in xrange(len2):
                        data0[i,j,k,l] = data0[i,j,k,l] * frot[j,k]
    else:
        if verbose:
            print 'No phase rotation needed'

cpdef np.ndarray[np.int_t, ndim=1] calc_delay(np.ndarray[float, ndim=1] freq, float inttime, float dm):
    """ Function to calculate delay for each channel, given dm
    freq in GHz, inttime in s.
    """

    cdef float freqref = freq[len(freq)-1]

    return np.round((4.2e-3 * dm * (1/(freq*freq) - 1/(freqref*freqref)))/inttime,0).astype(np.int16)

cpdef calc_resample(float chanwidth, float midfreq, float dm, float inttime):
    """ Function to calculate resmapling factor.
    freqs in GHz. inttime in s. returns intrachannel smearing by number of integrations.
    """

    return np.round(np.sqrt( (8.3e-3 * dm * chanwidth / midfreq**3)**2 + inttime**2)/inttime, 0).astype(np.int16)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse(np.ndarray[DTYPE_t, ndim=4, mode='c'] data0, d, float dm, int verbose=0):
    """ dedisperse the data in place
    replaces data0 in place. dm algorithm on only accurate if moving "up" in dm space.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    cdef int len0
    cdef unsigned int len1
    cdef unsigned int len2
    cdef unsigned int len3
# making copy is slower, but returns actual roll of original data0. without, we can only move "up" in dm space.
#    cdef np.ndarray[DTYPE_t, ndim=4] data1 = np.empty_like(data0)

    # calc relative delay per channel. only shift minimally
    cdef np.ndarray[short, ndim=1] newdelay = calc_delay(d['freq'], d['inttime'], dm)
    cdef np.ndarray[short, ndim=1] relativedelay = newdelay - d['datadelay']

    shape = np.shape(data0)
    len0 = shape[0]
    len1 = shape[1]
    len2 = shape[2]
    len3 = shape[3]

#    print relativedelay
#    for shift in np.unique(relativedelay):
    for j in xrange(len1):
        for k in xrange(len2):
            shift = relativedelay[k]
            if shift > 0:
                for l in xrange(len3):
# following is cython shortcut for 'iprime = np.mod(i+shift, len0)'
                    for i in xrange(len0):
                        iprime = i+shift
#                    print i, iprime
                        if iprime >= 0 and iprime < len0:    # ignore edge cases
                            data0[i,j,k,l] = data0[iprime,j,k,l]
                        elif iprime >= len0:    # set nonsense shifted data to zero
                            data0[i,j,k,l] = 0j
            elif shift < 0:
                print 'negative delay found. this dedispersion algorithm only works for positive delays. ignoring...'

# alternatives
#                            data1[i,j,k,l] = data0[iprime,j,k,l]
#            data0[:,:,indmin:indmax+1,:] = np.roll(data0.take(range(indmin,indmax+1), axis=2), -1*shift, axis=0)

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

    # new delay values
    d['datadelay'] = newdelay
#    return data1        # if doing proper roll, need to return copy

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_threaded(np.ndarray[DTYPE_t, ndim=4, mode='c'] data0, d, float dm, int verbose=0):
    """ dedisperse the data in place
    replaces data0 in place. dm algorithm on only accurate if moving "up" in dm space.
    assumes unshifted data.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
# making copy is slower, but returns actual roll of original data0. without, we can only move "up" in dm space.
#    cdef np.ndarray[DTYPE_t, ndim=4] data1 = np.empty_like(data0)

    # calc relative delay per channel. only shift minimally
    cdef np.ndarray[short, ndim=1] relativedelay = calc_delay(d['freq'], d['inttime'], dm)

#    print relativedelay
#    for shift in np.unique(relativedelay):
    for j in xrange(len1):
        for k in xrange(len2):
            shift = relativedelay[k]
            if shift > 0:
                for l in xrange(len3):
# following is cython shortcut for 'iprime = np.mod(i+shift, len0)'
                    for i in xrange(len0):
                        iprime = i+shift
#                    print i, iprime
                        if iprime >= 0 and iprime < len0:    # ignore edge cases
                            data0[i,j,k,l] = data0[iprime,j,k,l]
                        elif iprime >= len0:    # set nonsense shifted data to zero
                            data0[i,j,k,l] = 0j
            elif shift < 0:
                print 'negative delay found. this dedispersion algorithm only works for positive delays. ignoring...'

# alternatives
#                            data1[i,j,k,l] = data0[iprime,j,k,l]
#            data0[:,:,indmin:indmax+1,:] = np.roll(data0.take(range(indmin,indmax+1), axis=2), -1*shift, axis=0)

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_resample(np.ndarray[DTYPE_t, ndim=4, mode='c'] data0, np.ndarray[float, ndim=1] freq, float inttime, float dm, unsigned int resample, int verbose=0):
    """ dedisperse the data and resample in place. only fraction of array is useful data.
    dm algorithm on only accurate if moving "up" in dm space.
    assumes unshifted data.
    only does resampling by dt. no dm resampling for now.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    # calc relative delay per channel. only shift minimally
    cdef np.ndarray[short, ndim=1] relativedelay = calc_delay(freq, inttime, dm)
    cdef unsigned int newlen0 = len0/resample

    lostsamples = np.mod(len0, resample)
    if lostsamples:
        print 'Warning: resampling drops last %d int(s).' % (lostsamples)

    for j in xrange(len1):
        for l in xrange(len3):
            for k in xrange(len2):
                shift = relativedelay[k]
                for i in xrange(newlen0):
                    iprime = i*resample+shift
                    if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
                        data0[i,j,k,l] = data0[iprime,j,k,l]
                        if resample > 1:
                            for r in xrange(1,resample):
                                data0[i,j,k,l] = data0[i,j,k,l] + data0[iprime+r,j,k,l]
                            data0[i,j,k,l] = data0[i,j,k,l]/resample
                    elif iprime >= len0-(resample):    # set nonsense shifted data to zero
                        data0[i,j,k,l] = 0j

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dedisperse_resample2(np.ndarray[DTYPE_t, ndim=4, mode='c'] data0, np.ndarray[float, ndim=1] freq, float inttime, float dm, unsigned int resample, int verbose=0):
    """ dedisperse the data and resample with fixed value in place. only fraction of array is useful data.
    dm algorithm on only accurate if moving "up" in dm space.
    assumes unshifted data.
    """

    cdef unsigned int i
    cdef unsigned int iprime
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int r
    cdef unsigned int indmin
    cdef unsigned int indmax
    cdef int shift
    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef float chanwidth = freq[1] - freq[0]
    cdef float midfreq = freq[len(freq)/2]
    cdef unsigned int newlen0 = len0/resample

    if np.mod(len0, resample):
        print 'Warning: data length is not an integer number of resamples. Last int(s) will be lost.'

    # calc relative delay per channel. only shift minimally
    cdef np.ndarray[short, ndim=1] relativedelay = calc_delay(freq, inttime, dm)

    for j in xrange(len1):
        for l in xrange(len3):
            for k in xrange(len2):
                shift = relativedelay[k]
                for i in xrange(newlen0):
                    iprime = i*resample+shift
                    if iprime >= 0 and iprime < len0-(resample-1):    # if within bounds of unshifted data with resample stepping
                        data0[i,j,k,l] = data0[iprime,j,k,l]
                        if resample > 1:
                            for r in xrange(1,resample):
                                data0[i,j,k,l] = data0[i,j,k,l] + data0[iprime+r,j,k,l]
                            data0[i,j,k,l] = data0[i,j,k,l]/resample
                    elif iprime >= len0-resample:    # set nonsense shifted data to zero
                        data0[i,j,k,l] = 0j

    if verbose != 0:
        print 'Dedispersed for DM=%d' % dm

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef make_bispectra(np.ndarray[DTYPE_t, ndim=4] data0, np.ndarray[np.int_t, ndim=2] triples):
    """ Makes bispectra by integration, which assumes data have been dedispersed.
    """
    shape = np.shape(data0)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int p
    cdef unsigned int t
    cdef unsigned int l
    cdef unsigned int m
    cdef unsigned int n
    cdef complex sum
    cdef np.ndarray[np.complex64_t, ndim=2] bispectra = np.zeros(shape=(len0, len(triples)), dtype='complex64')
    cdef np.ndarray[np.complex64_t, ndim=2] data0m = np.zeros(shape=(len0,len1), dtype='complex64')

    # mean over chans and pols
    for i in xrange(len0):
        for j in xrange(len1):
            sum = 0.
            for p in xrange(len3):
                for k in xrange(len2):                   
                    sum += data0[i,j,k,p]
            data0m[i,j] = sum/(len2*len3)
#    data0m = data0.mean(axis=2).mean(axis=2)

    for t in xrange(len(triples)):
        l = triples[t, 0]
        m = triples[t, 1]
        n = triples[t, 2]
        for i in xrange(len0):
            bispectra[i, t] = data0m[i, l] * data0m[i, m] * data0m[i, n].conjugate()

    return bispectra

cpdef meantsub(np.ndarray[DTYPE_t, ndim=4, mode='c'] datacal):
    """ Subtract mean visibility, ignoring zeros
    """

    cdef unsigned int i, j, k, l
    sh = datacal.shape
    cdef unsigned int iterint = sh[0]
    cdef unsigned int nbl = sh[1]
    cdef unsigned int nchan = sh[2]
    cdef unsigned int npol = sh[3]
    cdef complex sum
    cdef unsigned int count = 0

    for j in xrange(nbl):
        for k in xrange(nchan):
            for l in xrange(npol):
                sum = 0.
                count = 0
                for i in xrange(iterint):
                    if datacal[i,j,k,l] != 0j:   # ignore zeros
                        sum += datacal[i,j,k,l]
                        count += 1
                if count:
                    for i in xrange(iterint):
                        datacal[i,j,k,l] = datacal[i,j,k,l] - sum/count

cpdef dataflag(np.ndarray[DTYPE_t, ndim=4, mode='c'] datacalfull, d, sigma=4, mode='', convergence=0.2, tripfrac=0.4):
    """ Flagging function
    """

    cdef unsigned int i, j, 
    cdef unsigned int flagged = 0
    cdef unsigned int spw
    cdef unsigned int pol
    cdef unsigned int badpol
    cdef unsigned int badbl
    cdef unsigned int chan
    sh = datacalfull.shape
    cdef unsigned int iterint = sh[0]
    cdef unsigned int nbl = sh[1]
    cdef unsigned int nchan = sh[2]
    cdef unsigned int npol = sh[3]
    cdef unsigned int chperspw = len(d['freq_orig'])/len(d['spw'])
    cdef nspw = len(d['spw'])
    cdef np.ndarray[np.int_t, ndim=1] chans
    cdef np.ndarray[np.float32_t, ndim=1] freqs
    cdef np.ndarray[np.int_t, ndim=1] badbls
    cdef np.ndarray[np.int_t, ndim=1] badpols
    cdef float blstdmed, blstdstd
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] datacal

    if mode == 'blstd':
        for spw in xrange(nspw):
            freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            chans = np.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
            for pol in xrange(npol):
                if np.any(datacalfull[:,:,chans,pol]):
                    datacal = np.ma.masked_array(datacalfull[:,:,chans,pol].copy(), datacalfull[:,:,chans,pol].copy() == 0)
                    blstd = datacal.std(axis=1)

                    # iterate to good median and std values
                    blstdmednew = np.ma.median(blstd)
                    blstdstdnew = blstd.std()
                    blstdstd = blstdstdnew*2
                    while (blstdstd-blstdstdnew)/blstdstd > convergence:
                        blstdstd = blstdstdnew
                        blstdmed = blstdmednew
                        blstd = np.ma.masked_where( blstd > blstdmed + sigma*blstdstd, blstd, copy=False)
                        blstdmednew = np.ma.median(blstd)
                        blstdstdnew = blstd.std()
#                        print blstdstdnew, blstdmednew
                    blstdstd = blstdstdnew
                    blstdmed = blstdmednew

                    # flag blstd too high
                    flagged = 0
                    for i in xrange(iterint):
                        for chan in xrange(len(chans)):
                            if blstd.data[i,chan] > blstdmed + sigma*blstdstd:     # then measure points to flag based on a third std threshold
                                flagged += nbl
                                for j in xrange(nbl):
                                    datacalfull[i,j,chans[chan],pol] = 0j
                    print 'Blstd flagging for (spw %d, pol %d), %d sigma: %3.2f %% of total flagged' % (spw,pol,sigma,float(flagged)/datacalfull.size*100)
                else:
                    print 'Blstd flagging for (spw %d, pol %d): All data flagged.' % (spw,pol)

    elif mode == 'badch':
        for spw in xrange(nspw):
            freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            chans = np.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
            for pol in xrange(npol):
                if np.any(datacalfull[:,:,chans,pol]):
                    datacal = np.ma.masked_array(datacalfull[:,:,chans,pol].copy(), datacalfull[:,:,chans,pol].copy() == 0)
                    meanamp = np.abs(datacal).mean(axis=1)

                    # iterate to good median and std values
                    meanampmednew = np.ma.median(meanamp)
                    meanampstdnew = meanamp.std()
                    meanampstd = meanampstdnew*2
                    while (meanampstd-meanampstdnew)/meanampstd > convergence:
                        meanampstd = meanampstdnew
                        meanampmed = meanampmednew
                        meanamp = np.ma.masked_where(meanamp > meanampmed + sigma*meanampstd, meanamp, copy=False)
                        meanampmednew = np.ma.median(meanamp)
                        meanampstdnew = meanamp.std()
#                    print meanampstdnew, meanampmednew
                    meanampstd = meanampstdnew
                    meanampmed = meanampmednew

                    badch = chans[np.where( (meanamp.mean(axis=0) > meanampmed + sigma*meanampstd) | (meanamp.mean(axis=0).mask==True) )[0]]
#                    print badch, meanamp.mean(axis=0).mask
                    if len(badch):
                        if len(badch) > tripfrac*len(chans):
                            print '%d bad channels in spw. Flagging all data.' % len(badch)
                            badch = chans

                    flagged = 0
                    for chan in badch:
                        flagged += iterint*nbl
                        for i in xrange(iterint):
                            for j in xrange(nbl):
                                datacalfull[i,j,chan,pol] = 0j
                    print 'Bad channel flagging for (spw %d, pol %d), %d sigma: %d chans, %3.2f %% of total flagged' % (spw, pol, sigma, len(badch), float(flagged)/datacalfull.size*100)
                else:
                    print 'Bad channel flagging for (spw %d, pol %d): All data flagged.' % (spw, pol)

    elif mode == 'badap':
        for spw in xrange(nspw):
            freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            chans = np.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])

            datacalfull2 = np.ma.masked_array(datacalfull[:,:,chans,:].copy(), datacalfull[:,:,chans,:].copy() == 0)
            bpa = np.abs(datacalfull2).mean(axis=0).mean(axis=1)
            bpa_ant = np.array([ (bpa[np.where(np.any(d['blarr'] == i, axis=1))[0]]).mean(axis=0) for i in np.unique(d['blarr']) ])
            bpa_ant = np.ma.masked_invalid(bpa_ant)
            ww = np.where(bpa_ant > np.ma.median(bpa_ant) + sigma * bpa_ant.std())
            badants = np.unique(d['blarr'])[ww[0]]
            flagged = 0
            if len(badants):
                badbls = np.where(np.any(d['blarr'] == badants[0], axis=1))[0]   # initialize
                badpols = np.array([ww[1][0]]*len(badbls))
                for i in xrange(1, len(badants)):
                    newbadbls = np.where(np.any(d['blarr'] == badants[i], axis=1))[0]
                    badbls = np.concatenate( (badbls, newbadbls) )
                    badpols = np.concatenate( (badpols, [ww[1][i]]*len(newbadbls)) )

                for j in xrange(len(badbls)):
                    flagged += iterint*len(chans)
                    for i in xrange(iterint):
                        for chan in chans:
                            datacalfull[i,badbls[j],chan,badpols[j]] = 0j
            print 'Bad basepol flagging for spw %d at %d sigma: ants/pols %s/%s, %3.2f %% of total flagged' % (spw, sigma, badants, ww[1], float(flagged)/datacalfull.size*100)

    elif mode == 'ring':
        for spw in xrange(nspw):
            freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            chans = np.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
            for pol in xrange(npol):
                if np.any(datacalfull[:,:,chans,pol]):
                    datacal = np.ma.masked_array(datacalfull[:,:,chans,pol].copy(), datacalfull[:,:,chans,pol].copy() == 0)
                    spfft = np.abs(np.fft.ifft(datacal.mean(axis=0), axis=1))   # delay spectrum of mean data in time
                    spfft = np.ma.masked_array(spfft, spfft = 0)
                    badbls = np.where(spfft[:,len(chans)/2-1:len(chans)/2].mean(axis=1) > sigma*np.ma.median(spfft[:,1:], axis=1))[0]  # find bls with spectral power at max delay. ignore dc in case this is cal scan.
                    if len(badbls) > tripfrac*nbl:    # if many bls affected, flag all
                        print 'Ringing on %d/%d baselines. Flagging all data.' % (len(badbls), nbl)
                        badbls = np.arange(nbl)

                    flagged = 0
                    for badbl in badbls:
                       flagged += iterint*len(chans)
                       for i in xrange(iterint):
                           for chan in chans:
                               datacalfull[i,badbl,chan,pol] = 0j

                    print 'Ringing flagging for (spw %d, pol %d) at %d sigma: %d/%d bls, %3.2f %% of total flagged' % (spw, pol, sigma, len(badbls), nbl, float(flagged)/datacalfull.size*100)
                else:
                    print 'Ringing flagging for (spw %d, pol %d): All data flagged' % (spw, pol)
    else:
                print 'Not flagging based on data quality.'
