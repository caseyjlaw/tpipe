import numpy as n
import os, glob

# optional import for casapy-free CASA
#import casautil
#tb = casautil.tools.table()

class solutions():
    """ Container for CASA caltable(s).
    Provides tools for applying to data of shape (nints, nbl, nch, npol).
    Initialize class based on input file(s) and selection criteria.
    Optional flux scale gain file can be given. Should be gain file applied to source with setjy applied.
    """

    def __init__(self, gainfile, flagants=True):
        """ Initialize with a table of CASA gain solutions. Can later add BP.
        """

        self.parsegain(gainfile)
        self.flagants = flagants

    def parsegain(self, gainfile):
        """Takes .g1 CASA cal table and places values in numpy arrays.
        """

        tb.open(gainfile)
        mjd = tb.getcol('TIME')/(24*3600)     # mjd days, as for telcal
        spw = tb.getcol('SPECTRAL_WINDOW_ID')
        gain = tb.getcol('CPARAM')    # dimensions of (npol, 1?, ntimes*nants)
        snr = tb.getcol('SNR')
        flagged = tb.getcol('FLAG')
        tb.close()
        tb.open(gainfile + '/ANTENNA')
        antname = tb.getcol('NAME')   # ant number in order written to gain file
        antnum = n.array([int(antname[i][2:]) for i in range(len(antname))])
        tb.close()

        # # need to find parent data MS to get some metadata
        # mslist = glob.glob(gainfile[:-3] + '*.ms')
        # try:
        #     msfile = mslist[0]
        #     print 'Found parent data MS %s' % msfile
        # except IndexError:
        #     print 'Could not find parent data MS for metadata...'

        # tb.open(msfile + '/ANTENNA')
        # antname = tb.getcol('NAME')      # one name per ant
        # tb.close()
        # tb.open(msfile + '/SPECTRAL_WINDOW')
        # reffreq = 1e-6*(tb.getcol('REF_FREQUENCY')+tb.getcol('TOTAL_BANDWIDTH')/2)   # similar to telcal "skyfreq"
        # specname = tb.getcol('NAME')
        # tb.close()
        # tb.open(msfile + '/SOURCE')
        # source = [name for name in tb.getcol('NAME') if 'J' in name][0]          # should return single cal name **hack**
        # tb.close()
        # nsol = len(gain[0,0])

        # ifid0R = specname[0][7] + '-' + specname[0][8]       # one value
        # ifid0L = specname[0][9] + '-' + specname[0][10]       # one value
        # ifid1R = specname[1][7] + '-' + specname[1][8]       # one value
        # ifid1L = specname[1][9] + '-' + specname[1][10]       # one value

        # # paste R,L end to end, so first loop over time, then spw, then pol
        # mjd = n.concatenate( (time, time), axis=0)
        # ifid = [ifid0R]*(nsol/2) + [ifid1R]*(nsol/2) + [ifid0L]*(nsol/2) + [ifid1L]*(nsol/2)   # first quarter is spw0,pol0, then spw1,pol0, ...
        # skyfreq = n.concatenate( (reffreq[0]*n.ones(nsol/2), reffreq[1]*n.ones(nsol/2), reffreq[0]*n.ones(nsol/2), reffreq[1]*n.ones(nsol/2)), axis=0)
        # gain = n.concatenate( (gain[0,0],gain[1,0]), axis=0)
        # amp = n.abs(gain)
        # phase = n.degrees(n.angle(gain))
        # source = [source]*nsol*2
        # flagged = n.concatenate( (flag[0,0],flag[1,0]), axis=0)
                   
        nants = len(n.unique(antnum))
        nspw = len(n.unique(spw))
        self.spwlist = n.unique(spw)
        npol = len(gain)

        # merge times less than 2s
        nsol = 0
        newmjd = [n.unique(mjd)[0]]
        skip = []
        for i in range(1, len(n.unique(mjd))):
            if 24*3600*(n.unique(mjd)[i] - n.unique(mjd)[i-1]) < 30.:
                skip.append(n.unique(mjd)[i])
                continue
            else:
                newmjd.append(n.unique(mjd)[i])
        
        self.uniquemjd = n.array(newmjd)
        nsol = len(self.uniquemjd)

        print 'Parsed gain table solutions for %d solutions (skipping %d), %d ants, %d spw, and %d pols' % (nsol, len(skip), nants, nspw, npol)
        print 'Unique solution times', self.uniquemjd

        self.gain = n.zeros( (nsol, nants, nspw, npol), dtype='complex' )
        flags = n.zeros( (nsol, nants, nspw, npol), dtype='complex' )
        for sol in range(nsol):
            for ant in range(nants):
                for spw in range(nspw):
                    for pol in range(npol):
                        self.gain[sol, ant, spw, pol] = gain[pol,0,spw*nsol*nants+sol*nants+ant]
                        flags[sol, ant, spw, pol] = flagged[pol,0,spw*nsol*nants+sol*nants+ant]
        self.gain = n.ma.masked_array(self.gain, flags)

#        gain = n.concatenate( (n.concatenate( (gain[0,0,:nants*nsol].reshape(nsol,nants,1,1), gain[1,0,:nants*nsol].reshape(nsol,nants,1,1)), axis=3), n.concatenate( (gain[0,0,nants*nsol:].reshape(nsol,nants,1,1), gain[1,0,nants*nsol:].reshape(nsol,nants,1,1)), axis=3)), axis=2)
#        flagged = n.concatenate( (n.concatenate( (flagged[0,0,:nants*nsol].reshape(nsol,nants,1,1), flagged[1,0,:nants*nsol].reshape(nsol,nants,1,1)), axis=3), n.concatenate( (flagged[0,0,nants*nsol:].reshape(nsol,nants,1,1), flagged[1,0,nants*nsol:].reshape(nsol,nants,1,1)), axis=3)), axis=2)
#        self.gain = n.ma.masked_array(gain, flagged == True)        

        self.mjd = n.array(mjd); self.antnum = antnum

        # make another version of ants array
#        self.antnum = n.concatenate( (antnum, antnum), axis=0)
#        self.amp = n.array(amp); self.phase = n.array(phase)
#        self.antname = n.concatenate( (antname[antnum], antname[antnum]), axis=0)
#        self.complete = n.arange(len(self.mjd))

        # for consistency with telcal
        #self.ifid = n.array(ifid); self.skyfreq = n.array(skyfreq); self.source = n.array(source)

    def parsebp(self, bpfile, debug=False):
        """ Takes bp CASA cal table and places values in numpy arrays.
        Assumes two or fewer spw. :\
        Assumes one bp solution per file.
        """

        # bandpass. taking liberally from Corder et al's analysisutilities
        ([polyMode, polyType, nPolyAmp, nPolyPhase, scaleFactor, nRows, nSpws, nUniqueTimesBP, uniqueTimesBP,
          nPolarizations, frequencyLimits, increments, frequenciesGHz, polynomialPhase,
          polynomialAmplitude, timesBP, antennasBP, cal_desc_idBP, spwBP]) = openBpolyFile(bpfile, debug)

        # index iterates over antennas, then times/sources (solution sets). each index has 2x npoly, which are 2 pols
        polynomialAmplitude = n.array(polynomialAmplitude); polynomialPhase = n.array(polynomialPhase)
        polynomialAmplitude[:,0] = 0.; polynomialAmplitude[:,nPolyAmp] = 0.
        polynomialPhase[:,0] = 0.; polynomialPhase[nPolyPhase] = 0.
        ampSolR, ampSolL = calcChebyshev(polynomialAmplitude, frequencyLimits, n.array(frequenciesGHz)*1e+9)
        phaseSolR, phaseSolL = calcChebyshev(polynomialPhase, frequencyLimits, n.array(frequenciesGHz)*1e+9)

        nants = len(n.unique(antennasBP))
        self.bptimes = n.array(timesBP)
        ptsperspec = 1000
        npol = 2
        print 'Parsed bp solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations)
        self.bandpass = n.zeros( (nants, nSpws*ptsperspec, npol), dtype='complex')
        for spw in range(nSpws):
            ampSolR[spw*nants:(spw+1)*nants] += 1 - ampSolR[spw*nants:(spw+1)*nants].mean()     # renormalize mean over ants (per spw) == 1
            ampSolL[spw*nants:(spw+1)*nants] += 1 - ampSolL[spw*nants:(spw+1)*nants].mean()
            for ant in range(nants):
                self.bandpass[ant, spw*ptsperspec:(spw+1)*ptsperspec, 0] = ampSolR[ant+spw*nants] * n.exp(1j*phaseSolR[ant+spw*nants])
                self.bandpass[ant, spw*ptsperspec:(spw+1)*ptsperspec, 1] = ampSolL[ant+spw*nants] * n.exp(1j*phaseSolL[ant+spw*nants])

        self.bpfreq = n.zeros( (nSpws*ptsperspec) )
        for spw in range(nSpws):
            self.bpfreq[spw*ptsperspec:(spw+1)*ptsperspec] = 1e9 * frequenciesGHz[nants*spw]

#        bpSolR0 = ampSolR[:nants] * n.exp(1j*phaseSolR[:nants])
#        bpSolR1 = ampSolR[nants:] * n.exp(1j*phaseSolR[nants:])
#        bpSolL0 = ampSolL[:nants] * n.exp(1j*phaseSolL[:nants])
#        bpSolL1 = ampSolL[nants:] * n.exp(1j*phaseSolL[nants:])

        # structure close to tpipe data structure (nant, freq, pol). note that freq is oversampled to 1000 bins.
#        self.bandpass = n.concatenate( (n.concatenate( (bpSolR0[:,:,None], bpSolR1[:,:,None]), axis=1), n.concatenate( (bpSolL0[:,:,None], bpSolL1[:,:,None]), axis=1)), axis=2)
#        self.bpfreq = 1e9*n.concatenate( (frequenciesGHz[0], frequenciesGHz[nants]), axis=0)    # freq values at bp bins
#        print 'Parsed bp table solutions for %d solutions, %d ants, %d spw, and %d pols' % (nUniqueTimesBP, nants, nSpws, nPolarizations)

    def setselection(self, time, freqs, spws=[0,1], pols=[0,1], verbose=0):
        """ Set select parameter that defines time, spw, and pol solutions to apply.
        time defines the time to find solutions near in mjd.
        freqs defines frequencies to select bandpass solution
        spws is list of min/max indices to be used (e.g., [0,1])
        pols is index of polarizations.
        pols/spws not yet implemented beyond 2sb, 2pol.
        """

        # spw and pols selection not yet implemented beyond 2/2
        self.spws = spws
        self.pols = pols

        # select by smallest time distance for source
        mjddist = n.abs(time - self.uniquemjd)
        closestgain = n.where(mjddist == mjddist.min())[0][0]

        print 'Using gain solution at MJD %.5f, separated by %d min ' % (self.uniquemjd[closestgain], mjddist[closestgain]*24*60)
        self.gain = self.gain[closestgain,:,spws[0]:spws[1]+1,pols[0]:pols[1]+1]

        if hasattr(self, 'bandpass'):
            bins = [n.where(n.min(n.abs(self.bpfreq-selfreq)) == n.abs(self.bpfreq-selfreq))[0][0] for selfreq in freqs]
            self.bandpass = self.bandpass[:,bins,pols[0]:pols[1]+1]
            self.freqs = freqs
            if verbose:
                print 'Using solution at BP bins: ', bins

    def calc_flag(self, sig=3.0):
        """ Calculates antennas to flag, based on bad gain and bp solutions.
        """
 
        if len(self.gain.shape) == 4:
            gamp = n.abs(self.gain).mean(axis=0)   # mean gain amp for each ant over time
        elif len(self.gain.shape) == 3:
            gamp = n.abs(self.gain)   # gain amp for selected time

#        badgain = n.where(gamp < gamp.mean() - sig*gamp.std())
        badgain = n.where( (gamp < n.median(gamp) - sig*gamp.std()) | gamp.mask)
        print 'Flagging low/bad gains for ant/spw/pol:', self.antnum[badgain[0]], badgain[1], badgain[2]

        badants = badgain
        return badants

    def apply(self, data, blarr):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        """

        ant1ind = [n.where(ant1 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]
        ant2ind = [n.where(ant2 == n.unique(blarr))[0][0] for (ant1,ant2) in blarr]

        # flag bad ants
        if self.flagants:
            badants = self.calc_flag()
        else:
            badants = n.array([[]])

        # apply gain correction
        if hasattr(self, 'bandpass'):
            corr = n.ones_like(data)
            flag = n.ones_like(data).astype('int')
            chans_uncal = range(len(self.freqs))
            for spw in range(len(self.gain[0])):
                chsize = n.round(self.bpfreq[1]-self.bpfreq[0], 0)
                ww = n.where( (self.freqs >= self.bpfreq[spw*1000]) & (self.freqs <= self.bpfreq[(spw+1)*1000-1]+chsize) )[0]
                if len(ww) == 0:
                    print 'Gain solution frequencies not found in data for spw %d.' % (self.spws[spw])
                firstch = ww[0]
                lastch = ww[-1]+1
                for ch in ww:
                    chans_uncal.remove(ch)
                print 'Combining gain sol from spw=%d with BW chans from %d-%d' % (self.spws[spw], firstch, lastch)
                for badant in n.transpose(badants):
                    if badant[1] == spw:
                        badbl = n.where((badant[0] == n.array(ant1ind)) | (badant[0] == n.array(ant2ind)))[0]
                        flag[:, badbl, firstch:lastch, badant[2]] = 0

                corr1 = self.gain[ant1ind, spw, :][None, :, None, :] * self.bandpass[ant1ind, firstch:lastch, :][None, :, :, :]
                corr2 = (self.gain[ant2ind, spw, :][None, :, None, :] * self.bandpass[ant2ind, firstch:lastch, :][None, :, :, :]).conj()

                corr[:, :, firstch:lastch, :] = corr1 * corr2
            if len(chans_uncal):
                print 'Setting data without bp solution to zero for chan range %d-%d.' % (chans_uncal[0], chans_uncal[-1])
                flag[:, :, chans_uncal,:] = 0
            data[:] *= flag/corr
        else:
            for spw in range(len(self.gain[0,0])):
                pass

def openBpolyFile(caltable, debug=False):
#    mytb = au.createCasaTool(tbtool)    # from analysisutilities by corder
    tb.open(caltable)
    desc = tb.getdesc()
    if ('POLY_MODE' in desc):
        polyMode = tb.getcol('POLY_MODE')
        polyType = tb.getcol('POLY_TYPE')
        scaleFactor = tb.getcol('SCALE_FACTOR')
        antenna1 = tb.getcol('ANTENNA1')
        times = tb.getcol('TIME')
        cal_desc_id = tb.getcol('CAL_DESC_ID')
        nRows = len(polyType)
        for pType in polyType:
            if (pType != 'CHEBYSHEV'):
                print "I do not recognized polynomial type = %s" % (pType)
                return
        # Here we assume that all spws have been solved with the same mode
        uniqueTimesBP = n.unique(tb.getcol('TIME'))
        nUniqueTimesBP = len(uniqueTimesBP)
        if (nUniqueTimesBP >= 2):
            if debug:
                print "Multiple BP sols found with times differing by %s seconds. Using first." % (str(uniqueTimesBP-uniqueTimesBP[0]))
            nUniqueTimesBP = 1
            uniqueTimesBP = uniqueTimesBP[0]
        mystring = ''
        nPolyAmp = tb.getcol('N_POLY_AMP')
        nPolyPhase = tb.getcol('N_POLY_PHASE')
        frequencyLimits = tb.getcol('VALID_DOMAIN')
        increments = 0.001*(frequencyLimits[1,:]-frequencyLimits[0,:])
        frequenciesGHz = []
        for i in range(len(frequencyLimits[0])):
           freqs = (1e-9)*n.arange(frequencyLimits[0,i],frequencyLimits[1,i],increments[i])       # **for some reason this is nch-1 long?**
           frequenciesGHz.append(freqs)
        polynomialAmplitude = []
        polynomialPhase = []
        for i in range(len(polyMode)):
            polynomialAmplitude.append([1])
            polynomialPhase.append([0])
            if (polyMode[i] == 'A&P' or polyMode[i] == 'A'):
                polynomialAmplitude[i]  = tb.getcell('POLY_COEFF_AMP',i)[0][0][0]
            if (polyMode[i] == 'A&P' or polyMode[i] == 'P'):
                polynomialPhase[i] = tb.getcell('POLY_COEFF_PHASE',i)[0][0][0]
  
        tb.close()
        tb.open(caltable+'/CAL_DESC')
        nSpws = len(tb.getcol('NUM_SPW'))
        spws = tb.getcol('SPECTRAL_WINDOW_ID')
        spwBP = []
        for c in cal_desc_id:
            spwBP.append(spws[0][c])
        tb.close()
        nPolarizations = len(polynomialAmplitude[0]) / nPolyAmp[0]
        if debug:
            mystring += '%.3f, ' % (uniqueTimesBP/(24*3600))
            print 'BP solution has unique time(s) %s and %d pols' % (mystring, nPolarizations)
        
        # This value is overridden by the new function doPolarizations in ValueMapping.
        # print "Inferring %d polarizations from size of polynomial array" % (nPolarizations)
        return([polyMode, polyType, nPolyAmp, nPolyPhase, scaleFactor, nRows, nSpws, nUniqueTimesBP,
                uniqueTimesBP, nPolarizations, frequencyLimits, increments, frequenciesGHz,
                polynomialPhase, polynomialAmplitude, times, antenna1, cal_desc_id, spwBP])
    else:
        tb.close()
        return([])
   # end of openBpolyFile()

def calcChebyshev(coeffs, validDomain, freqs):
    """
    Given a set of coefficients,
    this method evaluates a Chebyshev approximation.
    Used for CASA bandpass reading.
    input coeffs and freqs are numpy arrays
    """

    domain = (validDomain[1] - validDomain[0])[0]
    bins = -1 + 2* n.array([ (freqs[i]-validDomain[0,i])/domain for i in range(len(freqs))])
    ncoeffs = len(coeffs[0])/2
    rr = n.array([n.polynomial.chebval(bins[i], coeffs[i,:ncoeffs]) for i in range(len(coeffs))])
    ll = n.array([n.polynomial.chebval(bins[i], coeffs[i,ncoeffs:]) for i in range(len(coeffs))])

    return rr,ll
