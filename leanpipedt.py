##########################################
# functional style, uses multiprocessing #
# this version threads within processing #
##########################################

import numpy as n
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as p
import applytelcal, applycals2
from scipy import signal
from math import ceil
import multiprocessing as mp
import string, os, ctypes, types
import cPickle as pickle
import time as timestamp
import leanpipedt_cython as lib
#import leanpipe_external as lib
import qimg_cython as qimg
#from numpy import fft as fft
import pyfftw.interfaces.NUMPY_fft as fft
import casautil

# set up tools
ms = casautil.tools.ms()
qa = casautil.tools.quanta()

def numpyview(arr, datatype, shape):
    """ Takes mp.Array and returns numpy array with shape of data in MS.
    """

#    return n.frombuffer(arr.get_obj()).view(n.dtype(datatype)).reshape((iterint, nbl, nchan, npol))
    return n.frombuffer(arr.get_obj(), dtype=n.dtype(datatype)).view(n.dtype(datatype)).reshape(shape)

def calc_hexcenters(fwhmsurvey, fwhmfield, show=0):
    """ Tile a large circular area with a small circular areas. sizes are assumed to be fwhm. assumes flat sky.
    """

    large = fwhmsurvey
    small = fwhmfield
    centers = []
    (l0,m0) = (0.,0.)

    centers.append((l0,m0))
    l1 = l0-(small/2.)*n.cos(n.radians(60))
    m1 = m0-(small/2.)*n.sin(n.radians(60))
    ii = 0
    while ( n.sqrt((l1-l0)**2+(m1-m0)**2) < large/2.): 
        l1 = l1+((-1)**ii)*(small/2.)*n.cos(n.radians(60))
        m1 = m1+(small/2.)*n.sin(n.radians(60))
        l2 = l1+small/2
        m2 = m1
        while ( n.sqrt((l2-l0)**2+(m2-m0)**2) < large/2.): 
            centers.append((l2,m2))
            l2 = l2+small/2
        l2 = l1-small/2
        m2 = m1
        while ( n.sqrt((l2-l0)**2+(m2-m0)**2) < large/2.): 
            centers.append((l2,m2))
            l2 = l2-small/2
        ii = ii+1
    l1 = l0
    m1 = m0
    ii = 0
    while ( n.sqrt((l1-l0)**2+(m1-m0)**2) < large/2.): 
        l1 = l1-((-1)**ii)*(small/2.)*n.cos(n.radians(60))
        m1 = m1-(small/2.)*n.sin(n.radians(60))
        l2 = l1
        m2 = m1
        while ( n.sqrt((l2-l0)**2+(m2-m0)**2) < large/2.): 
            centers.append((l2,m2))
            l2 = l2+small/2
        l2 = l1-small/2
        m2 = m1
        while ( n.sqrt((l2-l0)**2+(m2-m0)**2) < large/2.): 
            centers.append((l2,m2))
            l2 = l2-small/2
        ii = ii+1

    delaycenters = n.array(centers)

    if len(delaycenters) == 1:
        plural = ''
    else:
        plural = 's'
    print 'For a search area of %.3f and delay beam of %.3f, we will use %d delay beam%s' % (fwhmsurvey, fwhmfield, len(delaycenters), plural)
    return delaycenters

def detect_bispectra(ba, d, sigma=5., tol=1.3, Q=0, show=0, save=0, verbose=0):
    """ Function to detect transient in bispectra
    sigma gives the threshold for SNR_bisp (apparent). 
    tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
    Q is noise per baseline and can be input. Otherwise estimated from data.
    save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
    """

    # using s=S/Q
    mu = lambda s: 1.  # for bispectra formed from visibilities
    sigbQ3 = lambda s: n.sqrt((1 + 3*mu(s)**2) + 3*(1 + mu(s)**2)*s**2 + 3*s**4)  # from kulkarni 1989, normalized by Q**3, also rogers et al 1995
    s = lambda basnr, ntr: (2.*basnr/n.sqrt(ntr))**(1/3.)  # see rogers et al. 1995 for factor of 2

    # measure SNR_bl==Q from sigma clipped times with normal mean and std of bispectra. put into time,dm order
    bamean = ba.real.mean(axis=1)
    bastd = ba.real.std(axis=1)

    (meanmin,meanmax) = lib.sigma_clip(bamean)  # remove rfi to estimate noise-like parts
    (stdmin,stdmax) = lib.sigma_clip(bastd)
    clipped = n.where((bamean > meanmin) & (bamean < meanmax) & (bastd > stdmin) & (bastd < stdmax) & (bamean != 0.0))[0]  # remove rfi and zeros
    bameanstd = ba[clipped].real.mean(axis=1).std()
    basnr = bamean/bameanstd    # = S**3/(Q**3 / n.sqrt(n_tr)) = s**3 * n.sqrt(n_tr)

    if Q and verbose:
        print 'Using given Q =', Q
    else:
        Q = ((bameanstd/2.)*n.sqrt(d['ntr']))**(1/3.)
        if verbose:
            print 'Estimating noise per baseline from data. Q (per DM) =', Q

    # detect
    cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, d['ntr']))) & (basnr > sigma) )  # get compact sources with high snr

    if show or save:
        p.figure()
        ax = p.axes()
        p.subplot(211)
        p.title(str(d['nskip']) + ' nskip, ' + str(len(cands))+' candidates', transform = ax.transAxes)
        p.plot(basnr, 'b.')
        if len(cands[0]) > 0:
            p.plot(cands, basnr[cands], 'r*')
            p.ylim(-2*basnr[cands].max(),2*basnr[cands].max())
        p.xlabel('Integration',fontsize=12,fontweight="bold")
        p.ylabel('SNR_b',fontsize=12,fontweight="bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 30))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        p.subplot(212)
        p.plot(bastd/Q**3, basnr, 'b.')

        # plot reference theory lines
        smax = s(basnr.max(), d['nants'])
        sarr = smax*n.arange(0,101)/100.
        p.plot(sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(d['ntr']), 'k')
        p.plot(tol*sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(d['ntr']), 'k--')
        if len(cands[0]) > 0:
            p.plot(bastd[cands]/Q**3, basnr[cands], 'r*')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 30))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        if len(cands[0]) > 0:
            p.axis([0, tol*sigbQ3(s(basnr[cands].max(), d['nants'])), -0.5*basnr[cands].max(), 1.1*basnr[cands].max()])

            # show spectral modulation next to each point
            for candint in cands:
                sm = n.single(round(specmod(data, d, candint),1))
                p.text(bastd[candint]/Q**3, basnr[candint], str(sm), horizontalalignment='right', verticalalignment='bottom')
            p.xlabel('sigma_b/Q^3',fontsize=12,fontweight="bold")
            p.ylabel('SNR_b',fontsize=12,fontweight="bold")
            if save:
                if save == 1:
                    savename = d['filename'].split('.')[:-1]
                    savename.append(str(d['nskip']) + '_bisp.png')
                    savename = string.join(savename,'.')
                elif isinstance(save, types.StringType):
                    savename = save
                print 'Saving file as ', savename
                p.savefig(self.pathout+savename)

    return cands[0], basnr, bastd, Q

def estimate_noiseperbl(data0):
    """ Takes large data array and sigma clips it to find noise per bl for input to detect_bispectra.
    Takes mean across pols and channels for now, as in detect_bispectra.
    """
    
    # define noise per baseline for data seen by detect_bispectra or image
    data0mean = data0.mean(axis=2).imag                      # use imaginary part to estimate noise without calibrated, on-axis signal
    (data0meanmin, data0meanmax) = lib.sigma_clip(data0mean.flatten())
    good = n.where( (data0mean>data0meanmin) & (data0mean<data0meanmax) )
    noiseperbl = data0mean[good].std()   # measure single noise for input to detect_bispectra
    print 'Sigma clip of %.3f to %.3f keeps %d%% of data' % (data0meanmin, data0meanmax, (100.*len(good[0]))/len(data0mean.flatten()))
    print 'Estimate of noise per baseline: %.3f' % noiseperbl
    return noiseperbl

def save(d, cands, verbose=0):
    """ Save all candidates in pkl file for later aggregation and filtering.
    """

    if len(cands):
        loclist = []
        proplist = []
        for cand in cands:
            # first unpack
#            (beamnum, dtind, i, dmind, snr, img, specpol) = cand  # big set
            (beamnum, dtind, i, dmind, snr, lm, snr2, lm2) = cand   # midsize set

            # then build list to dump
            loclist.append( [beamnum, dtind, i, dmind] )
#            proplist.append( (snr, img, specpol) )     # full set
            proplist.append( [snr, lm[0], lm[1], snr2, lm2[0], lm2[1]] )     # midsize set

        if verbose:
            print loclist, proplist

        # save candidate info in pickle file
        pkl = open(d['candsfile'], 'a')
        pickle.dump((loclist, proplist), pkl)
        pkl.close()
    else:
        if verbose:
            print 'No cands to save...'

def imgallloop(d, dmind, dtind, beamnum):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    runs cython qimg library for image, then filters results based on spectral modulation of candidate.
    """

    # THIS ONE does the uv gridding and searches for the candidates
    # NOTE, the qimg_cython.pyx defines all the different imaging algorithms Casey has written.
    twindow = 15       # window to save for plotting data in pickle

    # dedisperse using global 'data'
    data0 = dataprep(d, dmind, dtind)    # returns masked array of dedispersed and stitched data

    ims,snr,candints = qimg.imgallfullfilterxy(n.outer(u[d['iterint']/2], d['freq']/d['freq_orig'][0]), n.outer(v[d['iterint']/2], d['freq']/d['freq_orig'][0]), data0.data, d['sizex'], d['sizey'], d['res'], d['sigma_image'])

    # IF WE FOUND CANDIDATES, MAKE THEIR CANDIDATE PLOTS
    if len(candints) > 0:
#        spectra = []
        goodcandints = []
        lmarr = [];  lm2arr = []
        snr2arr = []
        for i in xrange(len(candints)):
            # phase shift to get dynamic spectrum
            peakl, peakm = n.where(ims[i] == ims[i].max())      # assumes new style u->u and v->v gridding

            l1 = (float((d['sizex'])/d['res'])/2. - peakl[0])/d['sizex']
            m1 = (float((d['sizey'])/d['res'])/2. - peakm[0])/d['sizey']
            if d['secondaryfilter'] == 'specmod':    # filter by spectral modulation
                # return spectrogram per pol
#                minint = max(candints[i]-twindow, 0)
#                maxint = min(candints[i]+twindow, len(data0))
                data0_snip = data0[candints[i]].copy()         # get candidate integration
                lib.phaseshift_threaded(data0_snip[None,:,:,:], d, l1, m1, u[candints[i]], v[candints[i]])
#                snipint = min(candints[i],twindow)       # correct for edge effects
#                print i, candints[i], minint, maxint, peakl, peakm, snipint, data0_snip.mean()
                bfspec = data0_snip.mean(axis=0).mean(axis=1).real   # mean over bl and pol for known int
#                bflc = data0_snip.mean(axis=3).mean(axis=2).mean(axis=1).real   # mean over ch, bl and pol
#                snrlc = bflc[snipint] / bflc[range(0,snipint-1)+range(snipint+2,twindow)].std()    # lc snr of event. more accurate than img snr
                sm = n.sqrt( ((bfspec**2).mean() - bfspec.mean()**2) / bfspec.mean()**2 )
                if sm < n.sqrt(d['nchan']/snr[i]**2 + d['specmodfilter']):
                    print 'Got one!  Int=%d, DM=%d, dt=%d, SNR_im=%.1f @ (%d,%d), SM=%.1f < %.1f, so keeping candidate.' % (d['nskip']+d['itercount']+candints[i]*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], peakl, peakm, sm, n.sqrt(d['nchan']/snr[i]**2 + d['specmodfilter']))
#                spectra.append(data0_snip.mean(axis=1))         # get real part of spectrogram of candidate
                    goodcandints.append(i)
                    lmarr.append( (l1, m1) )
                else:
                    print 'Almost... Int=%d, DM=%d, dt=%d, SNR_im=%.1f @ (%d,%d), SM=%.1f > %.1f, so rejecting candidate.' % (d['nskip']+d['itercount']+candints[i]*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], peakl, peakm, sm, n.sqrt(d['nchan']/snr[i]**2 + d['specmodfilter']))

            elif d['secondaryfilter'] == 'fullim':     # filter with an image of all data
                im2 = qimg.imgonefullxy(n.outer(u[candints[i]], d['freq']/d['freq_orig'][0]), n.outer(v[candints[i]], d['freq']/d['freq_orig'][0]), data0.data[candints[i]], d['full_sizex'], d['full_sizey'], d['res'])
                snr2 = im2.max()/im2.std()
                peakl2, peakm2 = n.where(im2 == im2.max())      # assumes new style u->u and v->v gridding
                l2 = (float((d['full_sizex'])/d['res'])/2. - peakl2[0])/d['full_sizex']
                m2 = (float((d['full_sizey'])/d['res'])/2. - peakm2[0])/d['full_sizey']

                if snr2 > d['sigma_image']:
                    print 'Got one!  Int=%d, DM=%d, dt=%d: SNR_im=%.1f @ (%.2e,%.2e) and SNR2=%.1f @ (%.2e, %.2e), so keeping candidate.' % (d['nskip']+d['itercount']+candints[i]*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], l1, m1, snr2, l2, m2)
                    goodcandints.append(i)
                    lmarr.append( (l1, m1) )
                    lm2arr.append( (l2, m2) )
                    snr2arr.append(snr2)
                else:
                    print 'Almost...  Int=%d, DM=%d, dt=%d: SNR_im=%.1f @ (%.2e,%.2e) and SNR2=%.1f @ (%.2e, %.2e), so rejecting candidate.' % (d['nskip']+d['itercount']+candints[i]*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], l1, m1, snr2, l2, m2)

        if d['secondaryfilter'] == 'specmod':    # filter by spectral modulation
            return [(beamnum, dtind, d['nskip']+d['itercount']+candints[goodcandints[i]]*d['dtarr'][dtind], dmind, snr[goodcandints[i]], lmarr[i]) for i in xrange(len(goodcandints))]   # smaller data returned            
        elif d['secondaryfilter'] == 'fullim':     # filter with an image of all data
            return [(beamnum, dtind, d['nskip']+d['itercount']+candints[goodcandints[i]]*d['dtarr'][dtind], dmind, snr[goodcandints[i]], lmarr[i], snr2arr[i], lm2arr[i]) for i in xrange(len(goodcandints))]   # smaller data returned

#        return [(beamnum, dtind, d['nskip']+d['itercount']+candints[goodcandints[i]]*d['dtarr'][dtind], dmind, snr[goodcandints[i]], ims[goodcandints[i]], spectra[i]) for i in xrange(len(goodcandints))]           # return data coods (delta_t, int, dm) and properties (snr, image, spectrum*pol)
    else:
        return 0
#        return ( n.empty( (0,7) ) )

def time_filter(data0, d, width, show=0):
        """ Replaces data array with filtered version via convolution in time. Note that this has trouble with zeroed data.
        kernel specifies the convolution kernel. 'm' for mexican hat (a.k.a. ricker, effectively does bg subtraction), 'g' for gaussian. 't' for a tophat. 'b' is a tophat with bg subtraction (or square 'm'). 'w' is a tophat with width that varies with channel, as kept in 'twidth[dmind]'.
        width is the kernel width with length nchan. should be tuned to expected pulse width in each channel.
        bgwindow is used by 'b' only.
        An alternate design for this method would be to make a new data array after filtering, so this can be repeated for many assumed widths without reading data in anew. That would require more memory, so going with repalcement for now.
        """
        kernel = d['filtershape']
        bgwindow = d['bgwindow']

        if not isinstance(width, types.ListType):
            width = [width] * len(d['chans'])

        # time filter by convolution. functions have different normlizations. m has central peak integral=1 and total is 0. others integrate to 1, so they don't do bg subtraction.
        kernelset = {}  # optionally could make set of kernels. one per width needed. (used only by 'w' for now).

        if kernel == 't':
            print 'Applying tophat time filter.'
            for w in n.unique(width):
                kernel = n.zeros(len(data0))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(ceil(w/2.)))
                kernel[onrange] = 1.
                kernelset[w] = kernel/n.where(kernel>0, kernel, 0).sum()         # normalize to have peak integral=1, thus outside integral=-1.
        elif kernel == 'b':
            print 'Applying tophat time filter with bg subtraction (square mexican hat) total width=%d.' % (bgwindow)
            for w in n.unique(width):
                kernel = n.zeros(len(data0))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(ceil(w/2.)))
                kernel[onrange] = 1.
                offrange = range(len(kernel)/2 - (bgwindow+w)/2, len(kernel)/2-w/2) + range(len(kernel)/2 + int(ceil(w/2.)), len(kernel)/2 + int(ceil((w+bgwindow)/2.)))
                kernel[offrange] = -1.
                posnorm = n.where(kernel>0, kernel, 0).sum()           # find normalization of positive
                negnorm = n.abs(n.where(kernel<0, kernel, 0).sum())    # find normalization of negative
                kernelset[w] = n.where(kernel>0, kernel/posnorm, kernel/negnorm)    # pos and neg both sum to 1/-1, so total integral=0
        elif kernel == 'g':
            print 'Applying gaussian time filter. Note that effective width is much larger than equivalent tophat width.'
            for w in n.unique(width):
                kernel = signal.gaussian(len(data0), w)     # gaussian. peak not quite at 1 for widths less than 3, so it is later renormalized.
                kernelset[w] = kernel / (w * n.sqrt(2*n.pi))           # normalize to pdf, not peak of 1.
        elif kernel == 'w':
            print 'Applying tophat time filter that varies with channel.'
            for w in n.unique(width):
                kernel = n.zeros(len(data0))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(ceil(w/2.)))
                kernel[onrange] = 1.
                kernelset[w] = kernel/n.where(kernel>0, kernel, 0).sum()         # normalize to have peak integral=1, thus outside integral=-1.
        elif kernel == None:
            print 'Applying no time filter.'
            return data0

        if show:
            for kernel in kernelset.values():
                p.plot(kernel,'.')
            p.title('Time filter kernel')
            p.show()

        # take ffts (in time)
        datafft = fft.fft(data0, axis=0)
#        kernelsetfft = {}
#        for w in n.unique(width):
#            kernelsetfft[w] = fft.fft(n.roll(kernelset[w], len(data0)/2))   # seemingly need to shift kernel to have peak centered near first bin if convolving complex array (but not for real array?)

        # **take first kernel. assumes single width in hacky way**
        kernelsetfft = fft.fft(n.roll(kernelset[kernelset.keys()[0]], len(data0)/2))   # seemingly need to shift kernel to have peak centered near first bin if convolving complex array (but not for real array?)

        # filter by product in fourier space
#        for i in range(d['nbl']):    # **can't find matrix product I need, so iterating over nbl, chans, npol**
#            for j in range(len(d['chans'])):
#                for k in range(d['npol']):
#                    datafft[:,i,j,k] = datafft[:,i,j,k]*kernelsetfft[width[j]]    # index fft kernel by twidth

        datafft = datafft * kernelsetfft[:,None,None,None]

        # ifft to restore time series
#        return n.ma.masked_array(fft.ifft(datafft, axis=0), self.flags[:self.nints,:, self.chans,:] == 0)
        return n.array(fft.ifft(datafft, axis=0))

def specmod(data0, d, ii):
    """Calculate spectral modulation for given track.
    Spectral modulation is basically the standard deviation of a spectrum. 
    This helps quantify whether the flux is located in a narrow number of channels or across all channels.
    Broadband signal has small modulation (<sqrt(nchan)/SNR) while RFI has larger values.
    See Spitler et al 2012 for details.
    """

    bfspec = data0[ii].mean(axis=0).mean(axis=1).real   # mean over bl and pol
    sm = n.sqrt( ((bfspec**2).mean() - bfspec.mean()**2) / bfspec.mean()**2 )

    return sm

def readprep(d):
    """ Prepare to read data
    """

    filename = d['filename']; spw = d['spw']; iterint = d['iterint']; datacol = d['datacol']; selectpol = d['selectpol']
    scan = d['scan']; nints = d['nints']; nskip = d['nskip']

    # read metadata either from pickle or ms file
    pklname = string.join(filename.split('.')[:-1], '.') + '_init.pkl'
    if os.path.exists(pklname):
        print 'Pickle of initializing info found. Loading...'
        pkl = open(pklname, 'r')
        try:
            (d['npol_orig'], d['nbl'], d['blarr'], d['inttime'], spwinfo, scansummary) = pickle.load(pkl)
        except EOFError:
            print 'Bad pickle file. Exiting...'
            return 1
        scanlist = sorted(scansummary.keys())
        starttime_mjd = scansummary[scanlist[scan]]['0']['BeginTime']
    else:
        print 'No pickle of initializing info found. Making anew...'
        pkl = open(pklname, 'wb')
        ms.open(filename)
        spwinfo = ms.getspectralwindowinfo()
        scansummary = ms.getscansummary()
        ms.selectinit(datadescid=0)  # reset select params for later data selection
        selection = {'uvdist': [1., 1e10]}    # exclude auto-corrs
        ms.select(items = selection)
        ms.selectpolarization(selectpol)

        scanlist = sorted(scansummary.keys())
        starttime_mjd = scansummary[scanlist[scan]]['0']['BeginTime']
        d['inttime'] = scansummary[scanlist[scan]]['0']['IntegrationTime']
        print 'Initializing integration time (s):', d['inttime']

        ms.iterinit(['TIME'], iterint*d['inttime'])
        ms.iterorigin()
        da = ms.getdata([datacol, 'axis_info'], ifraxis=True)
        ms.close()

        d['nbl'] = da[datacol].shape[2]
        bls = da['axis_info']['ifr_axis']['ifr_shortname']
        d['blarr'] = n.array([[int(bls[i].split('-')[0]),int(bls[i].split('-')[1])] for i in xrange(len(bls))])
#        d['npol'] = len(selectpol)
        d['npol_orig'] = da[datacol].shape[0]
        print 'Initializing %d polarizations' % (d['npol'])

        pickle.dump((d['npol_orig'], d['nbl'], d['blarr'], d['inttime'], spwinfo, scansummary), pkl)
        pkl.close()

    # set ants
    if len(d['excludeants']):
        print 'Excluding ant(s) %s' % d['excludeants']
    antlist = list(n.unique(d['blarr']))
    d['ants'] = [ant for ant in range(len(antlist)) if antlist[ant] not in d['excludeants']]
    d['blarr'] = n.array( [(ant1,ant2) for (ant1,ant2) in d['blarr'] if ((ant1 not in d['excludeants']) and (ant2 not in d['excludeants']))] )
    d['nbl'] = len(d['blarr'])

    d['nants'] = len(n.unique(d['blarr']))
    print 'Initializing nants:', d['nants']
    print 'Initializing nbl:', d['nbl']

    # define list of spw keys (may not be in order!)
    freqs = []
    for i in spwinfo.keys():
        freqs.append(spwinfo[i]['Chan1Freq'])
    d['spwlist'] = n.array(sorted(zip(freqs, spwinfo.keys())))[:,1][spw].astype(int)  # spwlist defines order of spw to iterate in freq order

    d['freq_orig'] = n.array([])
    for spw in d['spwlist']:
        nch = spwinfo[str(spw)]['NumChan']
        ch0 = spwinfo[str(spw)]['Chan1Freq']
        chw = spwinfo[str(spw)]['ChanWidth']
        d['freq_orig'] = n.concatenate( (d['freq_orig'], (ch0 + chw * n.arange(nch)) * 1e-9) ).astype('float32')

    d['freq'] = d['freq_orig'][d['chans']]
    d['nchan'] = len(d['chans'])
    print 'Initializing nchan:', d['nchan']

    # set requested time range based on given parameters
    timeskip = d['inttime']*nskip
    starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]
    stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+(timeskip+(nints+1)*d['inttime'])/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]  # nints+1 to be avoid buffer running out and stalling iteration
    print 'First integration of scan:', qa.time(qa.quantity(starttime_mjd,'d'),form=['ymd'],prec=9)[0]
    print
    print 'Reading scan', str(scanlist[scan]) ,'for times', qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['hms'], prec=9)[0], 'to', qa.time(qa.quantity(starttime_mjd+(timeskip+(nints+1)*d['inttime'])/(24.*60*60), 'd'), form=['hms'], prec=9)[0]

    # read data into data structure
    ms.open(filename)
    if len(d['spwlist']) == 1:
        ms.selectinit(datadescid=d['spwlist'][0])
    else:
        ms.selectinit(datadescid=0, reset=True)    # reset includes spw in iteration over time
    selection = {'time': [starttime, stoptime], 'uvdist': [1., 1e10], 'antenna1': d['ants'], 'antenna2': d['ants']}    # exclude auto-corrs
    ms.select(items = selection)
    ms.selectpolarization(selectpol)
    ms.iterinit(['TIME'], iterint*d['inttime'], 0, adddefaultsortcolumns=False)   # read with a bit of padding to get at least nints
    iterstatus = ms.iterorigin()
    d['itercount1'] = 0
    d['l0'] = 0.; d['m0'] = 0.

    # find full res/size and set actual res/size
    d['full_res'] = n.round(25./(3e-1/d['freq'][len(d['freq'])/2])/2).astype('int')    # full field of view. assumes freq in GHz
    #set actual res/size
    if d['res'] == 0:
        d['res'] = d['full_res']

    da = ms.getdata(['u','v','w'])
    uu = n.outer(da['u'], d['freq']).flatten() * (1e9/3e8)
    vv = n.outer(da['v'], d['freq']).flatten() * (1e9/3e8)
    # **this may let vis slip out of bounds. should really define grid out to 2*max(abs(u)) and 2*max(abs(v)). in practice, very few are lost.**
    powers = n.fromfunction(lambda i,j: 2**i*3**j, (12,8), dtype='int')   # power array for 2**i * 3**j
    rangex = n.round(uu.max() - uu.min()).astype('int')
    rangey = n.round(vv.max() - vv.min()).astype('int')
    largerx = n.where(powers-rangex/d['res'] > 0, powers, powers[-1,-1])
    p2x, p3x = n.where(largerx == largerx.min())
    largery = n.where(powers-rangey/d['res'] > 0, powers, powers[-1,-1])
    p2y, p3y = n.where(largery == largery.min())
    d['full_sizex'] = ((2**p2x * 3**p3x)*d['res'])[0]
    d['full_sizey'] = ((2**p2y * 3**p3y)*d['res'])[0]
    print 'Ideal uvgrid size=(%d,%d) for res=%d' % (d['full_sizex'], d['full_sizey'], d['res'])

    if d['size'] == 0:
        d['sizex'] = d['full_sizex']
        d['sizey'] = d['full_sizey']
        print 'Using uvgrid size=(%d,%d) (2**(%d,%d)*3**(%d,%d) = (%d,%d)) and res=%d' % (d['sizex'], d['sizey'], p2x, p2y, p3x, p3y, 2**p2x*3**p3x, 2**p2y*3**p3y, d['res'])
    else:
        d['sizex'] = d['size']
        d['sizey'] = d['size']
        print 'Using uvgrid size=(%d,%d) and res=%d' % (d['sizex'], d['sizey'], d['res'])

    d['size'] = max(d['sizex'], d['sizey'])
    print 'Image memory usage for %d threads is %d GB' % (d['nthreads'], 8 * d['sizex']/d['res'] * d['sizey']/d['res'] * d['iterint'] * d['nthreads']/1024**3)

    return iterstatus

def readiter(d):
    """ Iterates over ms.
    Returns everything needed for analysis as tuple.
    """

    da = ms.getdata([d['datacol'],'axis_info','u','v','w','flag','data_desc_id'], ifraxis=True)
#    spws = n.unique(da['data_desc_id'])    # spw in use
#    good = n.where((da['data_desc_id']) == spws[0])[0]   # take first spw
    good = n.where((da['data_desc_id']) == d['spwlist'][0])[0]   # take first spw
    time0 = da['axis_info']['time_axis']['MJDseconds'][good]
    data0 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]

    if d['telcalfile']:    # apply telcal solutions
        if len(d['spwlist']) > 1:
            spwbin = d['spwlist'][0]
        else:
            spwbin = 0
        chanfreq = da['axis_info']['freq_axis']['chan_freq'][:,spwbin]
        sols = applytelcal.solutions(d['telcalfile'], chanfreq)
        for i in range(len(d['selectpol'])):
            try:
                sols.setselection(d['telcalcalibrator'], time0[0]/(24*3600), d['selectpol'][i], verbose=0)   # chooses solutions closest in time that match pol and source name
                sols.apply(data0, d['blarr'], i)
                print 'Applied cal for spw %d and pol %s' % (spwbin, d['selectpol'][i])
            except:
                pass

    flag0 = n.transpose(da['flag'], axes=[3,2,1,0])[good]
    u0 = da['u'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)    # uvw are in m, so divide by wavelength of first chan to set in lambda
    v0 = da['v'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)
    w0 = da['w'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)
    if len(d['spwlist']) > 1:
        for spw in d['spwlist'][1:]:
            good = n.where((da['data_desc_id']) == spw)[0]
            data1 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]
            if d['telcalfile']:    # apply telcal solutions
                chanfreq = da['axis_info']['freq_axis']['chan_freq'][:,spw]
                sols = applytelcal.solutions(d['telcalfile'], chanfreq)
                for i in range(len(d['selectpol'])):
                    try:
                        sols.setselection(d['telcalcalibrator'], time0[0]/(24*3600), d['selectpol'][i], verbose=0)   # chooses solutions closest in time that match pol and source name
                        sols.apply(data1, d['blarr'], i)
                        print 'Applied cal for spw %d and pol %s' % (spw, d['selectpol'][i])
                    except:
                        pass

            data0 = n.concatenate( (data0, data1), axis=2 )
            flag0 = n.concatenate( (flag0, n.transpose(da['flag'], axes=[3,2,1,0])[good]), axis=2 )

    del da
    data0 = data0[:,:,d['chans'],:] * n.invert(flag0[:,:,d['chans'],:])   # flag==1 means bad data (for vla)

    if d['gainfile']:
        sols = applycals2.solutions(d['gainfile'], flagants=d['flagantsol'])
        sols.parsebp(d['bpfile'])
#        sols.setselection(time0[0]/(24*3600.), d['freq']*1e9, d['spw'], d['selectpol'])
        sols.setselection(time0[0]/(24*3600.), d['freq']*1e9)  # only dualpol, 2sb mode implemented
        sols.apply(data0, d['blarr'])

    d['iterstatus1'] = ms.iternext()
    return data0.astype('complex64'), u0.astype('float32'), v0.astype('float32'), w0.astype('float32'), time0.astype('float32')

def dataprep(d, dmind, dtind, usetrim=True):
    """ Takes most recent data read and dedisperses with white space. also adds previously trimmed data.
    data2 is next iteration of data of size iterint by ...
    usetrim is default behavior, but can be turned off to have large single-segment reading to reproduce cands.
    """

    dt = d['dtarr'][dtind]
    if d['datadelay'][dmind] >= dt:     # if doing dedispersion...
        data2 = n.concatenate( (n.zeros( (d['datadelay'][dmind], d['nbl'], d['nchan'], d['npol']), dtype='complex64'), data), axis=0)  # prepend with zeros of length maximal dm delay
        lib.dedisperse_resample(data2, d['freq'], d['inttime'], d['dmarr'][dmind], dt, verbose=0)        # dedisperses data.

        if usetrim:
            for i in xrange(len(datatrim[dmind][dtind])):
                data2[i] = data2[i] + datatrim[dmind][dtind][i]
        datatrim[dmind][dtind][:] = data2[d['iterint']/dt: d['iterint']/dt + len(datatrim[dmind][dtind])]
        return n.ma.masked_array(data2[:d['iterint']/dt], data2[:d['iterint']/dt] == 0j)
    else:                     # if no dedispersion
        data2 = data.copy()
        lib.dedisperse_resample(data2, d['freq'], d['inttime'], d['dmarr'][dmind], dt, verbose=0)        # only resample data
        return n.ma.masked_array(data2[:d['iterint']/dt], data2[:d['iterint']/dt] == 0j)

def readloop(d, eproc, emove):
    """ Data generating stage of parallel data function.
    data is either read into 'data' buffer, when ready
    this keeps data reading bottleneck to 1x the read time.
    """

    # now start main work of readloop
    iterint = d['iterint']; nbl = d['nbl']; nchan = d['nchan']; npol = d['npol']
#    data1_mem = mp.sharedctypes.RawArray(ctypes.c_float, (d['iterint']*d['nbl']*d['nchan']*d['npol']*2))    # x2 to store complex values in single array
    datacal_mem = mp.Array(ctypes.c_float, (iterint*nbl*nchan*len(d['selectpol'])*2))    # x2 to store complex values in single array
    datacal = numpyview(datacal_mem, 'complex64', (iterint, nbl, nchan, len(d['selectpol'])))

    datacap, ucap, vcap, wcap, timecap = readiter(d)    # read "cap", a hack to make sure any single iteration has enough integrations (artifact of irregular inttime)
    print 'Read first iteration with shape', datacap.shape

    while 1:
#            name = mp.current_process().name
#            print '%s: filling buffer' % name
        datanext, unext, vnext, wnext, timenext = readiter(d)
        print 'Read next %d ints from iter %d' % (len(datanext), d['itercount1']+iterint)

        datanext = n.vstack((datacap,datanext))
        unext = n.vstack((ucap,unext))
        vnext = n.vstack((vcap,vnext))
        wnext = n.vstack((wcap,wnext))
        timenext = n.concatenate((timecap,timenext))
        if ((len(datanext) < iterint) and d['iterstatus1']):  # read once more if data buffer is too small. don't read if no data! iterator gets confused.
            datanext2, unext2, vnext2, wnext2, timenext2 = readiter(d)
            print 'Read another %d ints for iter %d' % (len(datanext2), d['itercount1']+iterint)
            datanext = n.vstack((datanext,datanext2))
            unext = n.vstack((unext,unext2))
            vnext = n.vstack((vnext,vnext2))
            wnext = n.vstack((wnext,wnext2))
            timenext = n.concatenate((timenext,timenext2))
            del datanext2, unext2, vnext2, wnext2, timenext2    # clean up
        # select just the next iteration's worth of data and metadata. leave rest for next iteration's buffer cap.
        if len(datanext) >= iterint:
            datacal[:] = datanext[:iterint]
            datacap = datanext[iterint:]   # save rest for next iteration
            u1 = unext[:iterint]
            ucap = unext[iterint:]
            v1 = vnext[:iterint]
            vcap = vnext[iterint:]
            w1 = wnext[:iterint]
            wcap = wnext[iterint:]
            time1 = timenext[:iterint]
            timecap = timenext[iterint:]

# optionally can insert transient here
#            lib.phaseshift(data1, d, n.radians(0.1), n.radians(0.), u, v)    # phase shifts data in place
#            data1[100] = data1[100] + 10+0j
#            lib.phaseshift(data1, d, n.radians(0.), n.radians(0.1), u, v)    # phase shifts data in place
# flag data before moving into place

            # bg subtract in time
            if d['filtershape']:
                if d['filtershape'] == 'z':   # 'z' means do zero-mean subtraction in time
                    pass
                else:                         # otherwise do fft convolution
                    datacal = time_filter(datacal, d, 1)     # assumes pulse width of 1 integration

            # flag data
            if (d['flagmode'] == 'standard'):
                lib.dataflag(datacal, d, 2.5, convergence=0.05, mode='badch')
                lib.dataflag(datacal, d, 3., mode='badap')
                lib.dataflag(datacal, d, 4., convergence=0.1, mode='blstd')
                lib.dataflag(datacal, d, 4., mode='ring')
            else:
                print 'No real-time flagging.'

            if d['filtershape'] == 'z':
                print 'Subtracting mean visibility in time...'
                lib.meantsub(datacal)

            # write noise pkl with: itercount, noiseperbl, zerofrac, imstd_midtdm0
            noiseperbl = estimate_noiseperbl(datacal)
            if d['savecands'] and n.any(datacal[d['iterint']/2]):
                imstd = qimg.imgonefullxy(n.outer(u1[d['iterint']/2], d['freq']/d['freq_orig'][0]), n.outer(v1[d['iterint']/2], d['freq']/d['freq_orig'][0]), datacal[d['iterint']/2], d['sizex'], d['sizey'], d['res']).std()
                zerofrac = float(len(n.where(datacal == 0j)[0]))/datacal.size
                noisefile = 'noise_' + string.join(d['candsfile'].split('_')[1:-1], '_') + '.pkl'
                pkl = open(noisefile,'a')
                pickle.dump( (d['itercount1'], noiseperbl, zerofrac, imstd), pkl )
                pkl.close()

            # after cal and flagging, can optionally average to Stokes I to save memory
            # do this after measuring noise, etc to keep zero counting correct in imaging
            if 'lowmem' in d['searchtype']:
                datacal[...,0] = datacal.sum(axis=3)

            # emove is THE MOVE EVENT THAT WAITS FOR PROCESSOR TO TELL IT TO GO
            # wait for signal to move everything to processing buffers
            print 'Ready to move data into place for itercount ', d['itercount1']
            emove.wait()
            emove.clear()
            if 'lowmem' in d['searchtype']:
                data[...,0] = datacal[...,0]
            else:
                data[:] = datacal[:]
#            flag[:] = flag1[:]
            u[:] = u1[:]
            v[:] = v1[:]
            w[:] = w1[:]
            time[:] = time1[:]
            d['itercount'] = d['itercount1']
            d['iterstatus'] = d['iterstatus1']
            d['itercount1'] += iterint

            eproc.set()    # reading buffer filled, start processing
        # NOW MAKE SURE ALL ENDS GRACEFULLY
        else:
            print 'End of data (in buffer)'
            d['iterstatus'] = False    # to force processloop to end
            eproc.set()
            ms.iterend()
            ms.close()
            break
        if not d['iterstatus']:          # using iterstatus1 is more conservative here. trying to get around hangup on darwin.
            print 'End of data (iterator)'
            eproc.set()
            ms.iterend()
            ms.close()
            break

def readtriggerloop(d, eproc, emove):
    """ Defined purely to trigger readloop to continue without starting processloop
    """

    while 1:
        eproc.wait()
        eproc.clear()
        print 'Iterating readloop...'
        emove.set()

        if not d['iterstatus']:          # using iterstatus1 is more conservative here. trying to get around hangup on darwin.
            print 'End of data (iterator)'
            eproc.set()
            ms.iterend()
            ms.close()
            break

def processloop(d, eproc, emove):
    """ Processing stage of parallel data function. 
    Only processes from data. Assumes a "first in, first out" model, where 'data' defines next buffer to process.
    Event triggered by readloop when 'data' is filled.
    """

    while 1:
        eproc.wait()
        eproc.clear()
        print 'Processing for itercount %d. ' % (d['itercount'])

#        name = mp.current_process().name
#        print '%s: processing data' % name

# optionally can flag or insert transients here. done in readloop to improve parallelization
#        lib.phaseshift(data, d, n.radians(0.1), n.radians(0.), u, v)    # phase shifts data in place
#        data[100] = data[100] + 10+0j
#        lib.phaseshift(data, d, n.radians(0.), n.radians(0.1), u, v)    # phase shifts data in place
#        lib.dataflag(datacal, sigma=1000., mode='blstd')
        beamnum = 0
        resultlist = []

        # SUBMITTING THE LOOPS
        pool = mp.Pool(processes=d['nthreads'])      # reserve one for reading. also one for processloop?
        if n.any(data):
            for dmind in xrange(len(d['dmarr'])):
                print 'Processing DM = %d (max %d)' % (d['dmarr'][dmind], d['dmarr'][-1])
                for dtind in xrange(len(d['dtarr'])):
                    result = pool.apply_async(imgallloop, [d, dmind, dtind, beamnum])
                    resultlist.append(result)
        else:
            print 'Data for processing is zeros. Moving on...'

        # COLLECTING THE RESULTS
        candslist = []
        for i in xrange(len(resultlist)):
            results = resultlist[i].get()
            if results:
                for i in xrange(len(results)):
                    candslist.append(results[i])

        print 'Adding %d from itercount %d of %s. ' % (len(candslist), d['itercount'], d['filename'])

        # if the readloop has run out of data, close down processloop, else continue
        if not d['iterstatus']:
            pool.close()
            pool.join()
            if d['savecands']:
                save(d, candslist)
            emove.set()    # clear up any loose ends
            print 'End of processloop'
            break
        else:           # we're continuing, so signal data move, then save cands
            emove.set()  
            pool.close()
            pool.join()
            if d['savecands']:
                save(d, candslist)

def readloop2(d, eproc, emove):
    """ Profiles readloop 
    """
    cProfile.runctx('readloop(d, eproc, emove)', globals(), locals(), 'readloop.prof')

def processloop2(d, eproc, emove):
    """ Profiles processloop
    """
    cProfile.runctx('processloop(d, eproc, emove)', globals(), locals(), 'processloop.prof')

def calc_dmlist(dm_lo,dm_hi,t_samp,t_intr,b_chan,ctr_freq,n_chans,tolerance=1.25):
    """
    This procedure runs the HTRU-style calculation of DM trial steps.
    Input parameters:
     - Lowest DM desired
     - Highest DM desired
     - tsamp
     - intrinsic pulse width
     - bandwidth of single channel
     - center freq
     - n channels
     - tolerance of how much you're willing to smear out your signal (in units of ideal sample time)
    """
    dmarr = []

    dm = dm_lo
    while (dm <= dm_hi):
        dmarr.append(dm)
        old_dm = dm
        ch_fac = 8.3*b_chan/(ctr_freq*ctr_freq*ctr_freq)
        bw_fac = 8.3*b_chan*n_chans/4/(ctr_freq*ctr_freq*ctr_freq)
        t00 = n.sqrt(t_samp*t_samp + t_intr*t_intr + (dm*ch_fac)**2)
        tol_fac = tolerance*tolerance*t00*t00 - t_samp*t_samp - t_intr*t_intr
        new_dm = (bw_fac*bw_fac*dm + n.sqrt(-1.*(ch_fac*bw_fac*dm)**2. + ch_fac*ch_fac*tol_fac + bw_fac*bw_fac*tol_fac))/(ch_fac**2. + bw_fac**2)
        dm = new_dm

    return dmarr

### THIS IS THREAD IS THE "MAIN"
def pipe_thread(filename, nints=200, nskip=0, iterint=200, spw=[0], chans=range(5,59), dmarr=[0.], dtarr=[1], fwhmsurvey=0.5, fwhmfield=0.5, selectpol=['RR','LL'], scan=0, datacol='data', size=0, res=0, sigma_bisp=6.5, sigma_image=6.5, filtershape=None, secondaryfilter='fullim', specmodfilter=1.5, searchtype='imageall', telcalfile='', telcalcalibrator='', gainfile='', bpfile='', savecands=0, candsfile='', flagmode='standard', flagantsol=True, nthreads=1, wplanes=0, excludeants=[]):
    """ Threading for parallel data reading and processing.
    Either side can be faster than the other, since data are held for processing in shared buffer.
    size/res define uvgrid parameters. if either set to 0, then they are dynamically set to image full field of view and include all visibilities.
    searchtype can be 'readonly', '' to do little but setup, or any string to do image search, or include 'lowmem' for low memory version that sums polarizations.

    DESCRIPTION OF PARAMETERS:
    nints to datacol parameters define data to read
    size gives uv extent in N_wavelengths
    res chosen to be 50 to cover the full FOV of VLA
    sigma_'s tell what threshold to use for bispec or image
    filtershape etc. is about matched filtering for candidate detection. 'b' uses conv to subtract bgwindow, 'z' subtracts mean over all times in iterint.
    secondaryfilter defines how imaged candidates are filtered ('specmod' or 'fullim' are the options)
    specmodfilter IS A FUDGE FACTOR. In qimg, this factor tells you how much to tolerate spectral modulation deviance.
    searchtype tells what algorithm to do detection. List defined by Casey. Don't change this, it might break things.
    telcal thru bpfile --> options for calibration
    savecands is bool to save candidates or not.
    candsfile is the prefix used to name candidates.
    flagmode defines algorithm to do flagging. applies casa flags always.
    flagantsol --> uses CASA antenna flagging or not
    nthreads --> size of pool for multithreaded work.
    wplanes defines the number of w-planes for w-projection (0 means don't do w-projection)

    """

    # set up thread management and shared memory and metadata
    global data, datatrim, u, v, w, time

    mgr = mp.Manager()
    d = mgr.dict()
    eproc = mp.Event()      # event signalling to begin processing
    emove = mp.Event()      # event signalling to move data into processing buffers (data, flag, u, v, w, time)

    # define basic shared params
    d['filename'] = filename
    d['spw'] = spw
    d['datacol'] = datacol
    d['dmarr'] = dmarr
    d['dtarr'] = dtarr
    d['scan'] = scan
    d['nskip'] = nskip
    d['nints'] = nints       # total ints to iterate over
    d['iterint'] = iterint    # time step for msiter
    d['chans'] = chans
    d['nchan'] = len(chans)
    d['selectpol'] = selectpol
    if 'lowmem' in searchtype:
        print 'Running in \'lowmem\' mode. Reading pols %s, then summing after cal, flag, and filter. Flux scale not right if pols asymmetrically flagged.' % selectpol
        d['npol'] = 1
    else:
        d['npol'] = len(selectpol)
    d['filtershape'] = filtershape
    d['bgwindow'] = 10
    d['sigma_bisp'] = sigma_bisp
    d['sigma_image'] = sigma_image
    d['size'] = size
    d['sizex'] = size
    d['sizey'] = size
    d['res'] = res
    d['secondaryfilter'] = secondaryfilter
    d['specmodfilter'] = specmodfilter     # fudge factor for spectral modulation. 1==ideal, 0==do not apply, >1==non-ideal broad-band signal
    d['searchtype'] = searchtype
    d['delaycenters'] = calc_hexcenters(fwhmsurvey, fwhmfield)
    d['telcalfile'] = telcalfile               # telcal file produced by online system
    d['telcalcalibrator'] = telcalcalibrator
    d['gainfile'] = gainfile
    d['bpfile'] = bpfile
    d['savecands'] = savecands
    d['excludeants'] = excludeants
    d['candsfile'] = candsfile
    d['flagmode'] = flagmode
    d['flagantsol'] = flagantsol
    d['nthreads'] = nthreads
    d['wplanes'] = wplanes     # flag to turn on/off wproj. later overwritten with wplane inv conv kernel

    # define basic data state
    print 'Preparing to read...'
    d['iterstatus'] = readprep(d)
#    d['datadelay'] = n.array([[lib.calc_delay(d['freq'], d['inttime']*d['dtarr'][i], d['dmarr'][j]).max() for i in range(len(d['dtarr']))] for j in range(len(d['dmarr']))])  # keep track of delay shift as array indexed with [dmind][dtind]
    d['datadelay'] = n.array([lib.calc_delay(d['freq'], d['inttime'], d['dmarr'][i]).max() for i in range(len(d['dmarr']))])  # keep track of delay shift as array indexed with [dmind][dtind]

    # time stamp and candidate save file
    tt = timestamp.localtime()
    d['starttime'] = tt
    print 'Start time: %s_%s_%s:%s:%s:%s' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
    # define candidate file
    if d['savecands']:
        if not d['candsfile']:
            timestring = '%s_%s_%s:%s:%s:%s' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
            d['candsfile'] = 'cands_'+filename[:-3]+'.pkl'
#            d['candsfile'] = 'cands_'+timestring+'.pkl'
        picklabledict = d.copy()
        pkl = open(d['candsfile'], 'wb')
        pickle.dump(picklabledict, pkl)
        pkl.close()

    # create shared data arrays
    print 'Preparing shared memory objects...'
#    data_mem = {}
#    data = {}
#    for dmind in xrange(len(d['dmarr'])):
#        data_mem[dmind] = mp.Array(ctypes.c_float, ((d['iterint']/d['resamplearr'][dmind])*d['nbl']*d['nchan']*d['npol']*2))    # x2 to store complex values in single array
#        data[dmind] = numpyview(data_mem[dmind], 'complex64', ((d['iterint']/d['resamplearr'][dmind]), d['nbl'], d['nchan'], d['npol']))
#        data[dmind][:] = n.zeros(((d['iterint']/d['resamplearr'][dmind]), d['nbl'], d['nchan'], d['npol']))
    data_mem = mp.Array(ctypes.c_float, (d['iterint']*d['nbl']*d['nchan']*d['npol']*2))    # x2 to store complex values in single array
    data = numpyview(data_mem, 'complex64', (d['iterint'], d['nbl'], d['nchan'], d['npol']))
    data[:] = n.zeros((d['iterint'], d['nbl'], d['nchan'], d['npol']))

    datatrim = {}; datatrim_mem = {}
    totalnint = iterint   # start counting size of memory in integrations
    for dmind in xrange(len(d['dmarr'])):    # save the trimmings!
        datatrim[dmind] = {}; datatrim_mem[dmind] = {}
        for dtind in xrange(len(d['dtarr'])):
            dt = d['dtarr'][dtind]
            if d['datadelay'][dmind] >= dt:
                datatrim_mem[dmind][dtind] = mp.Array(ctypes.c_float, ((d['datadelay'][dmind]/dt)*d['nbl']*d['nchan']*d['npol']*2))    # x2 to store complex values in single array
                datatrim[dmind][dtind] = numpyview(datatrim_mem[dmind][dtind], 'complex64', ((d['datadelay'][dmind]/dt), d['nbl'], d['nchan'], d['npol']))
                datatrim[dmind][dtind][:] = n.zeros(((d['datadelay'][dmind]/dt), d['nbl'], d['nchan'], d['npol']), dtype='complex64')
                totalnint += d['datadelay'][dmind]/dt
            else:
                datatrim[dmind][dtind] = n.array([])

    print 'Visibility memory usage is %d GB' % (8*(totalnint * d['nbl'] * d['nchan'] * d['npol'])/1024**3)  # factor of 2?

    # later need to update these too
#    flag_mem = mp.Array(ctypes.c_bool, iterint*d['nbl']*d['nchan']*d['npol'])
    u_mem = mp.Array(ctypes.c_float, iterint*d['nbl'])
    v_mem = mp.Array(ctypes.c_float, iterint*d['nbl'])
    w_mem = mp.Array(ctypes.c_float, iterint*d['nbl'])
    time_mem = mp.Array(ctypes.c_float, iterint)
    # new way is to convert later
#    flag = numpyview(flag_mem, 'bool', (iterint, d['nbl'], d['nchan'], d['npol']))
    u = numpyview(u_mem, 'float32', (iterint, d['nbl']))
    v = numpyview(v_mem, 'float32', (iterint, d['nbl']))
    w = numpyview(w_mem, 'float32', (iterint, d['nbl']))
    time = numpyview(time_mem, 'float32', (iterint))

    print 'Starting processing and reading loops...'
    try:
        if searchtype:
            if searchtype == 'readonly':
                pread = mp.Process(target=readloop, args=(d,eproc,emove))
                pread.start()
                pproc = mp.Process(target=readtriggerloop, args=(d, eproc,emove))
                pproc.start()

                # trigger events to allow moving data to working area
                # This initial set makes it so the read loop bypasses the emove event the first time through.
                emove.set()

                # wait for threads to end (when read iteration runs out of data)
                pread.join()
                pproc.join()
            else:
                # start processes
                pread = mp.Process(target=readloop, args=(d,eproc,emove))
                pread.start()
                pproc = mp.Process(target=processloop, args=(d,eproc,emove))
                pproc.start()

                # trigger events to allow moving data to working area
                # This initial set makes it so the read loop bypasses the emove event the first time through.
                emove.set()

                # wait for threads to end (when read iteration runs out of data)
                pread.join()
                pproc.join()
        else:
            print 'Not starting read and process threads...'
    except KeyboardInterrupt:
        print 'Ctrl-C received. Shutting down threads...'
        pread.terminate()
        pproc.terminate()
        pread.join()
        pproc.join()

    return d.copy()
