import numpy as n

class solutions():
    """ Container for telcal solutions. Parses .GN files and provides tools for applying to data of shape (nints, nbl, nch, npol)
    Initialize class based on telcalfile and selection criteria.
    solnum is iteration of solution (0-based), pol is 0/1 for R/L.
    freqs is array of channel frequencies in Hz. Should be something like tpipe.freq*1e9.
    """

    def __init__(self, telcalfile, freqs=[1.4e9,1.401e9]):
        self.freqs = freqs
        self.chansize = freqs[1]-freqs[0]
        self.parseGN(telcalfile)
        print 'Read telcalfile %s' % telcalfile

    def setselection(self, calname, time, polstr, verbose=0):
        """ Set select parameter that defines spectral window, time, or any other selection.
        calname defines the name of the calibrator to use. if blank, uses only the time selection.
        time defines the time to find solutions near for given calname. it is in mjd.
        polstr is either 'RR' or 'LL', where (A,C) == (R,L), it seems.
        """

        self.select = self.complete   # use only complete solution sets (set during parse)

        if calname:
            nameselect = []
            for ss in n.unique(self.source[self.select]):
                if calname in ss:
                    nameselect = n.where(self.source[self.select] == ss)   # define selection for name
                    self.select = self.select[nameselect[0]]       # update overall selection
                    if verbose:
                        print 'Selection down to %d solutions with %s' % (len(self.select), calname)
            if len(nameselect) == 0:
                print 'Calibrator name %s not found. Ignoring.' % (calname)

        # select freq
        freqselect = n.where( n.around(1e6*self.skyfreq[self.select],-6) == n.around(self.freqs[len(self.freqs)/2],-6) )   # define selection for time
        if len(freqselect[0]) == 0:
            raise StandardError('No complete set of telcal solutions at that frequency.')
        self.select = self.select[freqselect[0]]    # update overall selection
        if verbose:
            print 'Frequency selection cut down to %d solutions' % (len(self.select))

        # select pol
        ifids = self.ifid[self.select]
        for pp in n.unique(ifids):
            if (('A' in pp or 'B' in pp) and ((polstr == 'RR') or (polstr == 'XX'))):
                polselect = n.where(ifids == pp)
            elif (('C' in pp or 'D' in pp) and ((polstr == 'LL') or (polstr == 'YY'))):
                polselect = n.where(ifids == pp)

        self.select = self.select[polselect[0]]    # update overall selection

        # select by smallest time distance for source
        mjddist = n.abs(time - n.unique(self.mjd[self.select]))
        closest = n.where(mjddist == mjddist.min())
        timeselect = n.where(self.mjd[self.select] == n.unique(self.mjd[self.select])[closest])   # define selection for time
        self.select = self.select[timeselect[0]]    # update overall selection
        if verbose:
            print 'Selection down to %d solutions separated from given time by %d minutes' % (len(self.select), mjddist[closest]*24*60)

        if verbose:
            print 'Selected solutions: ', self.select
            print 'MJD: ', n.unique(self.mjd[self.select])
            print 'Mid frequency (MHz),', n.unique(self.skyfreq[self.select])
            print 'IFID: ', n.unique(self.ifid[self.select])
            print 'Source: ', n.unique(self.source[self.select])
            print 'Ants: ', n.unique(self.antname[self.select])

    def parseGN(self, telcalfile):
        """Takes .GN telcal file and places values in numpy arrays.
        """

        skip = 3   # skip first three header lines
        MJD = 0; UTC = 1; LSTD = 2; LSTS = 3; IFID = 4; SKYFREQ = 5; ANT = 6; AMP = 7; PHASE = 8
        RESIDUAL = 9; DELAY = 10; FLAGGED = 11; ZEROED = 12; HA = 13; AZ = 14; EL = 15
        SOURCE = 16
        #FLAGREASON = 17

        mjd = []; utc = []; lstd = []; lsts = []; ifid = []; skyfreq = []; 
        antname = []; amp = []; phase = []; residual = []; delay = []; 
        flagged = []; zeroed = []; ha = []; az = []; el = []; source = []
        #flagreason = []

        i = 0
        for line in open(telcalfile,'r'):

            fields = line.split()
            if i < skip:
                i += 1
                continue

            if ('NO_ANTSOL_SOLUTIONS_FOUND' in line) or ('ERROR' in line):
                continue

            mjd.append(float(fields[MJD])); utc.append(fields[UTC]); lstd.append(float(fields[LSTD])); lsts.append(fields[LSTS])
            ifid.append(fields[IFID]); skyfreq.append(float(fields[SKYFREQ])); antname.append(fields[ANT])
            amp.append(float(fields[AMP])); phase.append(float(fields[PHASE])); residual.append(float(fields[RESIDUAL]))
            delay.append(float(fields[DELAY])); flagged.append('true' == (fields[FLAGGED]))
            zeroed.append('true' == (fields[ZEROED])); ha.append(float(fields[HA])); az.append(float(fields[AZ]))
            el.append(float(fields[EL])); source.append(fields[SOURCE])
#            flagreason.append('')  # 18th field not yet implemented

        self.mjd = n.array(mjd); self.utc = n.array(utc); self.lstd = n.array(lstd); self.lsts = n.array(lsts)
        self.ifid = n.array(ifid); self.skyfreq = n.array(skyfreq); self.antname = n.array(antname); self.amp = n.array(amp) 
        self.phase = n.array(phase); self.residual = n.array(residual); self.delay = n.array(delay)
        self.flagged = n.array(flagged); self.zeroed = n.array(zeroed); self.ha = n.array(ha); self.az = n.array(az)
        self.el = n.array(el); self.source = n.array(source); 
        #self.flagreason = n.array(flagreason)

        # purify list to keep only complete solution sets
#        uu = n.unique(self.mjd)
#        uu2 = n.concatenate( (uu, [uu[-1] + (uu[-1]-uu[-2])]) )  # add rightmost bin
#        count,bin = n.histogram(self.mjd, bins=uu2)
#        goodmjd = bin[n.where(count == count.max())]
#        complete = n.array([], dtype='int')
#        for mjd in goodmjd:
#            complete = n.concatenate( (complete, n.where(mjd == self.mjd)[0]) )
#        self.complete = n.array(complete)
        self.complete = n.arange(len(self.mjd))

        # make another version of ants array
        antnum = []
        for aa in self.antname:
            antnum.append(int(aa[2:]))    # cuts the 'ea' from start of antenna string to get integer
        self.antnum = n.array(antnum)

    def calcgain(self, ant1, ant2):
        """ Calculates the complex gain product (g1*g2) for a pair of antennas.
        """

        ind1 = n.where(ant1 == self.antnum[self.select])
        ind2 = n.where(ant2 == self.antnum[self.select])
        g1 = self.amp[self.select][ind1]*n.exp(1j*n.radians(self.phase[self.select][ind1]))
        g2 = self.amp[self.select][ind2]*n.exp(-1j*n.radians(self.phase[self.select][ind2]))
        if len(g1*g2) > 0:
            invg1g2 = 1/(g1*g2)
            invg1g2[n.where( (g1 == 0j) | (g2 == 0j) )] = 0.
            return invg1g2
        else:
            return n.array([0])

    def calcdelay(self, ant1, ant2):
        """ Calculates the relative delay (d1-d2) for a pair of antennas in ns.
        """

        ind1 = n.where(ant1 == self.antnum[self.select])
        ind2 = n.where(ant2 == self.antnum[self.select])
        d1 = self.delay[self.select][ind1]
        d2 = self.delay[self.select][ind2]
        if len(d1-d2) > 0:
            return d1-d2
        else:
            return n.array([0])

    def apply(self, data, blarr, pol):
        """ Applies calibration solution to data array. Assumes structure of (nint, nbl, nch, npol).
        blarr is array of size 2xnbl that gives pairs of antennas in each baseline (a la tpipe.blarr).
        pol is an index to apply solution (0/1)
        """

        # define freq structure to apply delay solution
        nch = data.shape[2]
        chanref = nch/2    # reference channel at center
        freqarr = self.chansize*(n.arange(nch) - chanref)   # relative frequency

        for i in range(len(blarr)):
            ant1, ant2 = blarr[i]  # ant numbers (1-based)

            # apply gain correction
            invg1g2 = self.calcgain(ant1, ant2)
            data[:,i,:,pol] = data[:,i,:,pol] * invg1g2[0]

            # apply delay correction
            d1d2 = self.calcdelay(ant1, ant2)
            delayrot = 2*n.pi*(d1d2 * 1e-9)*freqarr      # phase to rotate across band
            data[:,i,:,pol] = data[:,i,:,pol] * n.exp(-1j*delayrot[None, :])     # do rotation
