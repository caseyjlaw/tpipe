#! /usr/bin/env python

"""
tpipe.py --- read and visualize visibility data to search for transients
Generalization of evlavis, etc.
Can read MS or Miriad formatted data. Looks for any of the following:
- CASA and pyrap for MS reading
- miriad-python for Miriad reading
- aipy for imaging numpy array data
"""

import sys, string, os, shutil
from os.path import join
import pickle
import numpy as n
import pylab as p

# set up libraries for reading and imaging visibility data
try:
    # for simplicity, we should use pyrap for reading MS
    import pyrap
    print 'Imported pyrap'
except ImportError:
    try:
        # as a backup, casa stuff can be imported if running casapy
        from casa import ms
        from casa import quanta as qa
        print 'Imported CASA'
    except ImportError:
        print 'No CASA or pyrap available. Can\'t read MS data.'

try:
    # miriad-python can be used to read miriad format data
    from mirtask import util
    from mirexec import TaskInvert, TaskClean, TaskRestore, TaskImFit, TaskCgDisp, TaskImStat, TaskUVFit
    import miriad
    print 'Imported miriad-python'
except ImportError:
    print 'No miriad-python available. Can\'t read miriad data.'

try:
    # try aipy for imaging numpy array data
    import aipy
    print 'Imported aipy...'
except ImportError:
    print 'No aipy available. Can\'t image in Python.'


class reader:
    """ Master class with basic functions.
    params defines various tunable parameters for reading data and running pipelines. 
    Not all params are useful for all pipelines.
    """

    # parameters used by various subclasses
    # Note that each parameter must also be listed in set_params method in order to get set
    params = {
        'default' : {
            'ants' : range(0,29),          # antenna set to look for (only works for ms data)
            'chans': n.array(range(5,59)),   # channels to read
            'dmarr' : [44.,88.],      # dm values to use for dedispersion (only for some subclasses)
            'pulsewidth' : 0.0,      # width of pulse in time (seconds)
            'approxuvw' : True,      # flag to make template visibility file to speed up writing of dm track data
            'pathout': './'         # place to put output files
            }
        }

    def set_params(self, version='default', key='', value=''):
        """ Method called by __init__ in subclasses. This sets all parameters needed elsewhere.
        Can optionally set up range of defaults called with name
        """

        # either set parameter in master self.params dictionary or set parameters to dictionary values
        if key != '':
            self.params[version][key] = value
        else:
            self.pathout = self.params[version]['pathout']
            self.chans = self.params[version]['chans']
            self.dmarr = self.params[version]['dmarr']
            self.pulsewidth = self.params[version]['pulsewidth']
            self.approxuvw = self.params[version]['approxuvw']
            self.ants = self.params[version]['ants']

    def show_params(self, version='default'):
        """ Print parameters of pipeline.
        """
        
        return self.params[version]

    def spec(self, ind=[], save=0):
        """ Plot spectrogram for phase center by taking mean over baselines and polarizations.
        Optionally can zoom in on small range in time with ind parameter.
        """

        reltime = self.reltime
        bf = self.dataph

        print 'Data mean, std: %f, %f' % (self.dataph.mean(), self.dataph.std())
        (vmin, vmax) = sigma_clip(bf.ravel())

        p.figure(1)
        p.clf()
        ax = p.axes()
        ax.set_position([0.2,0.2,0.7,0.7])
        if len(ind) > 0:
            for i in ind:
                p.subplot(len(ind),1,list(ind).index(i)+1)
                intmin = n.max([0,i-50])
                intmax = n.min([len(self.reltime),i+50])
                im = p.imshow(n.rot90(bf[intmin:intmax]), aspect='auto', origin='upper', interpolation='nearest', extent=(intmin,intmax,0,len(self.chans)), vmin=vmin, vmax=vmax)
                p.subplot(len(ind),1,1)
        else:
            im = p.imshow(n.rot90(bf), aspect='auto', origin='upper', interpolation='nearest', extent=(0,len(self.reltime),0,len(self.chans)), vmin=vmin, vmax=vmax)
        p.title(str(self.scan) + ' scan, ' +str(self.nskip/self.nbl) + ' nskip, candidates ' + str(ind))

        cb = p.colorbar(im)
        cb.set_label('Flux Density (Jy)',fontsize=12,fontweight="bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 30))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        p.yticks(n.arange(0,len(self.chans),4), (self.chans[(n.arange(0,len(self.chans), 4))]))
        p.xlabel('Time (integration number)',fontsize=12,fontweight="bold")
        p.ylabel('Frequency Channel',fontsize=12,fontweight="bold")
        if save:
            savename = self.file.split('.')[:-1]
            savename.append(str(self.scan) + '_' + str(self.nskip/self.nbl) + '_spec.png')
            savename = string.join(savename,'.')
            print 'Saving file as ', savename
            p.savefig(self.pathout+savename)

    def drops(self, chan=0, pol=0, show=1):
        """ Displays info on missing baselines by looking for zeros in data array.
        """

        nints = float(len(self.reltime))
        bllen = []

        if self.data_type == 'mir':
            bls = self.preamble[:,4]
            for bl in n.unique(bls):
                bllen.append(n.shape(n.where(bls == bl))[1])
        elif self.data_type == 'ms':
            for i in range(len(self.blarr)):
                bllen.append(len(n.where(self.data[:,i,chan,pol] != 0.00)[0]))

        bllen = n.array(bllen)

        if show:
            p.clf()
            for i in range(self.nbl):
                p.text(self.blarr[i,0], self.blarr[i,1], s=str(100*(bllen[i]/nints - 1)), horizontalalignment='center', verticalalignment='center', fontsize=9)
            p.axis((0,self.nants+1,0,self.nants+1))
            p.plot([0,self.nants+1],[0,self.nants+1],'b--')
#            p.xticks(int(self.blarr[:,0]), self.blarr[:,0])
#            p.yticks(int(self.blarr[:,1]), self.blarr[:,1])
            p.xlabel('Ant 1')
            p.ylabel('Ant 2')
            p.title('Drop fraction for chan %d, pol %d' % (chan, pol))
#            p.show()

        return self.blarr,bllen

    def imagetrack(self, track, size=48000, res=500, clean=True):
        """ Image a track returned by tracksub
        """

        tr = track.mean(axis=1)
        fov = n.degrees(1./res)*3600.  # field of view in arcseconds
        p.clf()

        # make image
        ai = aipy.img.Img(size=size, res=res)
        ai.put( (self.u[tbin],self.v[tbin],self.w[tbin]), tr)
        image = ai.image(center = (size/res/2, size/res/2))
        image_final = image

        # optionally clean image
        if clean:
            beam = ai.bm_image()
            beamgain = aipy.img.beam_gain(beam[0])
            (clean, dd) = aipy.deconv.clean(image, beam[0], verbose=True, gain=0.01, tol=1e-4)  # light cleaning
            kernel = n.where(beam[0] >= 0.4*beam[0].max(), beam[0], 0.)  # take only peak (gaussian part) pixels of beam image
            restored = aipy.img.convolve2d(clean, kernel)
            image_restored = (restored + dd['res']).real/beamgain
            image_final = image_restored

        ax = p.imshow(image_final, aspect='auto', origin='upper', interpolation='nearest', extent=[-fov/2, fov/2, -fov/2, fov/2])
        p.colorbar()
        p.xlabel('Offset (arcsec)')
        p.ylabel('Offset (arcsec)')

        peak = n.where(n.max(image_final) == image_final)
        print 'Image peak of %e at (%d,%d)' % (n.max(image_final), peak[0][0], peak[1][0])
        print 'Peak/RMS = %e' % (image_final.max()/image_final.std())
        return image_final

    def phaseshift(self, l, m):
        """ Function to apply phase shift to (l,m) coordinates of data array.
        Should return new data array.
        This should be used before .mean(axis=bl) step is done to produce a new spectrogram.
        """

        newdata = n.zeros(shape=self.data.shape, dtype='complex')
        ang = lambda l,m,u,v,freq: l*n.outer(u,freq/freq.mean()) + m*n.outer(v,freq/freq.mean())  # operates on single time of u,v

        print 'Shifting phase center by (l,m) = (%e,%e) = (%e,%e) arcsec' % (l, m, n.degrees(l)*3600, n.degrees(m)*3600)

        for int in range(len(newdata)):
            for pol in range(self.npol):
                newdata[int,:,:,pol] = self.data[int,:,:,pol] * n.exp(-2j*n.pi*ang(l, m, self.u[int], self.v[int], self.freq))
    
        self.data = newdata
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real  # multi-pol
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'New dataph min, max:'
        print self.min, self.max

    def make_triples(self, amin=0, amax=0):
        """ Calculates and returns data indexes (i,j,k) for all closed triples.
        amin and amax define range of antennas (with index, in order). only used if nonzero.
        """

        if amax == 0:
            amax = self.nants
        blarr = self.blarr

        # first make triples indexes in antenna numbering
        anttrips = []

        for i in self.ants[amin:amax+1]:
            for j in self.ants[list(self.ants).index(i)+1:amax+1]:
                for k in self.ants[list(self.ants).index(j)+1:amax+1]:
                    anttrips.append([i,j,k])

        # next return data indexes for triples
        bltrips = []
        for (ant1, ant2, ant3) in anttrips:
            try:
                bl1 = n.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant2) )[0][0]
                bl2 = n.where( (blarr[:,0] == ant2) & (blarr[:,1] == ant3) )[0][0]
                bl3 = n.where( (blarr[:,0] == ant1) & (blarr[:,1] == ant3) )[0][0]
                bltrips.append([bl1, bl2, bl3])
            except IndexError:
                continue

        return n.array(bltrips)



class mirreader(reader):
    """ Class for reading Miriad format data with miriad-python
    """

    def read(self, file, nints, nskip, nocal, nopass):
        """ Reads in Miriad data using miriad-python.
        Seems to have some small time (~1 integration) errors in light curves and spectrograms, 
        as compared to CASA-read data.
        """

        self.file = file
        vis = miriad.VisData(self.file,)

        # read data into python arrays
        i = 0
        for inp, preamble, data, flags in vis.readLowlevel ('dsl3', False, nocal=True, nopass=True):
            # Loop to skip some data and read shifted data into original data arrays
            if i == 0:
                # get few general variables
                self.nants0 = inp.getScalar ('nants', 0)
                self.inttime0 = inp.getScalar ('inttime', 10.0)
                self.nspect0 = inp.getScalar ('nspect', 0)
                self.nwide0 = inp.getScalar ('nwide', 0)
                self.sdf0 = inp.getScalar ('sdf', self.nspect0)
                self.nschan0 = inp.getScalar ('nschan', self.nspect0)
                self.ischan0 = inp.getScalar ('ischan', self.nspect0)
                self.sfreq0 = inp.getScalar ('sfreq', self.nspect0)
                self.restfreq0 = inp.getScalar ('restfreq', self.nspect0)
                self.pol0 = inp.getScalar ('pol')

                self.sfreq = self.sfreq0
                self.sdf = self.sdf0
                self.npol = 1
                self.nchan = len(data)
                print 'Initializing nchan:', self.nchan
                bls = []

            # build complete list of baselines
            bls.append(preamble[4])

            # end here. assume at least one instance of each bl occurs before ~three integrations
            if len(bls) == 3*len(n.unique(bls)):
                blarr = []
                for bl in n.unique(bls):
                    blarr.append(util.decodeBaseline (bl))
                self.blarr = n.array(blarr)
                bldict = dict( zip(n.unique(bls), n.arange(len(blarr))) )
                break

            i = i+1

        # Initialize more stuff...
        self.freq = self.sfreq + self.sdf * self.chans
        self.pulsewidth = self.pulsewidth * n.ones(len(self.chans)) # pulse width of crab and m31 candidates

        # good baselines
        self.nbl = len(self.blarr)
        print 'Initializing nbl:', self.nbl
        self.ants = n.unique(self.blarr)
        self.nants = len(self.ants)
        print 'Initializing nants:', self.nants
        self.nskip = int(nskip*self.nbl)    # number of iterations to skip (for reading in different parts of buffer)
        nskip = int(self.nskip)

        # define data arrays
        da = n.zeros((nints,self.nbl,self.nchan),dtype='complex64')
        fl = n.zeros((nints,self.nbl,self.nchan),dtype='bool')
        pr = n.zeros((nints*self.nbl,5),dtype='float64')

        print
        # go back and read data into arrays
        i = 0
        for inp, preamble, data, flags in vis.readLowlevel ('dsl3', False, nocal=nocal, nopass=nopass):
            # Loop to skip some data and read shifted data into original data arrays

            if i < nskip:
                i = i+1
                continue 

            # assumes ints in order, but may skip. after nbl iterations, it fills next row, regardless of number filled.
            if (i-nskip) < nints*self.nbl:
                da[(i-nskip)//self.nbl,bldict[preamble[4]]] = data
                fl[(i-nskip)//self.nbl,bldict[preamble[4]]] = flags
                pr[i-nskip] = preamble
            else:
                break     # stop at nints

            if not (i % (self.nbl*100)):
                print 'Read spectrum ', str(i)

            i = i+1

        # build final data structures
#        good = n.where ( (self.blarr[:,0] != 5) & (self.blarr[:,1] != 5) & (self.blarr[:,0] != 10) & (self.blarr[:,1] != 10) )[0] # remove bad ants?
        self.rawdata = n.expand_dims(da, 3)  # hack to get superfluous pol axis
        self.flags = n.expand_dims(fl, 3)
        self.data = self.rawdata[:,:,self.chans,:] # [:,good,:,:]  # remove bad ants?
#        self.blarr = self.blarr[good]  # remove bad ants?
        self.preamble = pr
        self.u = (pr[:,0] * self.freq.mean()).reshape(nints, self.nbl)
        self.v = (pr[:,1] * self.freq.mean()).reshape(nints, self.nbl)
        self.w = (pr[:,2] * self.freq.mean()).reshape(nints, self.nbl)
        # could add uvw, too... preamble index 0,1,2 in units of ns
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real  #dataph is summed and detected to form TP beam at phase center, multi-pol
        time = self.preamble[::self.nbl,3]
        self.reltime = 24*3600*(time - time[0])      # relative time array in seconds. evla times change...?

        # print summary info
        print
        print 'Data read!\n'
        print 'Shape of raw data, flagged, time:'
        print self.rawdata.shape, self.data.shape, self.reltime.shape
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'Dataph min, max:'
        print self.min, self.max


class msreader(reader):
    """ Class for reading MS data with either CASA or (eventually) pyrap
    """

    def read(self, file, nints, nskip, spw, selectpol, scan, datacol):
        """ Reads in Measurement Set data using CASA.
        spw is list of subbands. zero-based.
        Scan is zero-based selection based on scan order, not actual scan number.
        """
        self.file = file
        self.scan = scan

        # get spw info. either load pickled version (if found) or make new one
        pklname = string.join(file.split('.')[:-1], '.') + '_init.pkl'
#        pklname = pklname.split('/')[-1]  # hack to remove path and write locally
        if os.path.exists(pklname):
            print 'Pickle of initializing info found. Loading...'
            pkl = open(pklname, 'r')
            try:
                (self.npol_orig, self.nbl, self.blarr, self.inttime, self.inttime0, spwinfo, scansummary) = pickle.load(pkl)
            except EOFError:
                print 'Bad pickle file. Exiting...'
                return 1
            scanlist = scansummary['summary'].keys()
            starttime_mjd = scansummary['summary'][scanlist[scan]]['0']['BeginTime']
            self.nskip = int(nskip*self.nbl)    # number of iterations to skip (for reading in different parts of buffer)
            self.npol = len(selectpol)
        else:
            print 'No pickle of initializing info found. Making anew...'
            pkl = open(pklname, 'wb')
            ms.open(self.file)
            spwinfo = ms.getspectralwindowinfo()
            scansummary = ms.getscansummary()
            scanlist = scansummary['summary'].keys()

            starttime_mjd = scansummary['summary'][scanlist[scan]]['0']['BeginTime']
            starttime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0/(24.*60*60),'d'),form=['ymd'], prec=9), 's'))
            stoptime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0.5/(24.*60*60), 'd'), form=['ymd'], prec=9), 's'))
            ms.selectinit(datadescid=0)  # initialize to initialize params
            selection = {'time': [starttime0, stoptime0]}
            ms.select(items = selection)
            da = ms.getdata([datacol,'axis_info'], ifraxis=True)
            ms.close()

            self.npol_orig = da[datacol].shape[0]
            self.nbl = da[datacol].shape[2]
            print 'Initializing nbl:', self.nbl

            # good baselines
            bls = da['axis_info']['ifr_axis']['ifr_shortname']
            self.blarr = n.array([[int(bls[i].split('-')[0]),int(bls[i].split('-')[1])] for i in range(len(bls))])
            self.nskip = int(nskip*self.nbl)    # number of iterations to skip (for reading in different parts of buffer)

            # set integration time
            ti0 = da['axis_info']['time_axis']['MJDseconds']
#            self.inttime = n.mean([ti0[i+1] - ti0[i] for i in range(len(ti0)-1)])
            self.inttime = scansummary['summary'][scanlist[scan]]['0']['IntegrationTime']
            self.inttime0 = self.inttime
            print 'Initializing integration time (s):', self.inttime

            pickle.dump((self.npol_orig, self.nbl, self.blarr, self.inttime, self.inttime0, spwinfo, scansummary), pkl)
        pkl.close()

        self.nants = len(n.unique(self.blarr))
        self.nants0 = len(n.unique(self.blarr))
        print 'Initializing nants:', self.nants
        self.npol = len(selectpol)
        print 'Initializing %d of %d polarizations' % (self.npol, self.npol_orig)

        # set desired spw
        if (len(spw) == 1) & (spw[0] == -1):
            spwlist = range(len(spwinfo['spwInfo']))
        else:
            spwlist = spw

        freq = n.array([])
        for spw in spwlist:
            nch = spwinfo['spwInfo'][str(spw)]['NumChan']
            ch0 = spwinfo['spwInfo'][str(spw)]['Chan1Freq']
            chw = spwinfo['spwInfo'][str(spw)]['ChanWidth']
            freq = n.concatenate( (freq, (ch0 + chw * n.arange(nch)) * 1e-9) )

        self.freq = freq[self.chans]
        self.nchan = len(self.freq)
        print 'Initializing nchan:', self.nchan

        # set requested time range based on given parameters
        timeskip = self.inttime*nskip
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['ymd'], prec=9), 's'))
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+(timeskip+nints*self.inttime)/(24.*60*60), 'd'), form=['ymd'], prec=9), 's'))
        print 'First integration of scan:', qa.time(qa.quantity(starttime_mjd,'d'),form=['ymd'],prec=9)
        print
        print 'Reading scan', str(scanlist[scan]) ,'for times', qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['hms'], prec=9), 'to', qa.time(qa.quantity(starttime_mjd+(timeskip+nints*self.inttime)/(24.*60*60), 'd'), form=['hms'], prec=9)

        # read data into data structure
        ms.open(self.file)
        ms.selectinit(datadescid=spwlist[0])  # reset select params for later data selection
        selection = {'time': [starttime, stoptime], 'antenna1': self.ants, 'antenna2': self.ants}
        ms.select(items = selection)
        print 'Reading %s column, SB %d, polarization %s...' % (datacol, spwlist[0], selectpol)
        ms.selectpolarization(selectpol)
        da = ms.getdata([datacol,'axis_info','u','v','w'], ifraxis=True)
        u = da['u']; v = da['v']; w = da['w']
        if da == {}:
            print 'No data found.'
            return 1
        newda = n.transpose(da[datacol], axes=[3,2,1,0])  # if using multi-pol data.
        if len(spwlist) > 1:
            for spw in spwlist[1:]:
                ms.selectinit(datadescid=spw)  # reset select params for later data selection
                ms.select(items = selection)
                print 'Reading %s column, SB %d, polarization %s...' % (datacol, spw, selectpol)
                ms.selectpolarization(selectpol)
                da = ms.getdata([datacol,'axis_info'], ifraxis=True)
                newda = n.concatenate( (newda, n.transpose(da[datacol], axes=[3,2,1,0])), axis=2 )
        ms.close()

        # Initialize more stuff...
        self.nschan0 = self.nchan
        self.pulsewidth = self.pulsewidth * n.ones(self.nchan) # pulse width of crab and m31 candidates

        # set variables for later writing data **some hacks here**
        self.nspect0 = 1
        self.nwide0 = 0
        self.sdf0 = da['axis_info']['freq_axis']['resolution'][0][0] * 1e-9
        self.sdf = self.sdf0
        self.ischan0 = 1
        self.sfreq0 = da['axis_info']['freq_axis']['chan_freq'][0][0] * 1e-9
        self.sfreq = self.sfreq0
        self.restfreq0 = 0.0
        self.pol0 = -1 # assumes single pol?

#        self.rawdata = newda[len(newda)/2:]  # hack to remove autos
        self.u = u.transpose() * (-self.freq.mean()*1e9/3e8)  # uvw are in m on ground. scale by -wavelenth to get projected lamba uvw (as in miriad?)
        self.v = v.transpose() * (-self.freq.mean()*1e9/3e8)
        self.w = w.transpose() * (-self.freq.mean()*1e9/3e8)
        self.rawdata = newda
        self.data = self.rawdata[:,:,self.chans]
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real  # multi-pol
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'Shape of rawdata, data:'
        print self.rawdata.shape, self.data.shape
        print 'Dataph min, max:'
        print self.min, self.max

        # set integration time and time axis
        ti = da['axis_info']['time_axis']['MJDseconds']
        self.reltime = ti - ti[0]


class pipe_msint(msreader):
    """ Create pipeline object for reading in MS data and doing integration-based analysis without dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    spw is list of spectral windows to read from MS.
    selectpol is list of polarization product names for reading from MS
    scan is zero-based selection of scan for reading from MS. It is based on scan order, not actual scan number.
    datacol is the name of the data column name to read from the MS.
    """

    def __init__(self, file, version='default', nints=1000, nskip=0, spw=[-1], selectpol=['RR','LL'], scan=0, datacol='data'):
        self.set_params(version=version)

        self.read(file=file, nints=nints, nskip=nskip, spw=spw, selectpol=selectpol, scan=scan, datacol=datacol)

        self.track0 = self.track(0.)
        self.twidth = 0
        for k in self.track0[1]:
            self.twidth = max(self.twidth, len(n.where(n.array(self.track0[1]) == k)[0]))

    def track(self, t0 = 0., show=0):
        """ Takes time offset from first integration in seconds.
        t0 defined at first (unflagged) channel.
        Returns an array of (timebin, channel) to select from the data array.
        """

        reltime = self.reltime
        chans = self.chans
        tint = self.inttime     # hack! actually grabs in window of 2*tint. avoids problems with time alignment.

        # calculate pulse time and duration
        pulset = t0
        pulsedt = self.pulsewidth[0]   # dtime in seconds. just take one channel, since there is no freq dep

        timebin = []
        chanbin = []

        ontime = n.where(((pulset + pulsedt) >= reltime - tint/2.) & (pulset <= reltime + tint/2.))
        for ch in range(len(chans)):
            timebin = n.concatenate((timebin, ontime[0]))
            chanbin = n.concatenate((chanbin, (ch * n.ones(len(ontime[0]), dtype=int))))

        track = (list(timebin), list(chanbin))

        if show:
            p.plot(track[0], track[1], 'w*')

        return track

    def tracksub(self, tbin, bgwindow = 0):
        """ Creates a background-subtracted set of visibilities.
        For a given track (i.e., an integration number) and bg window, tracksub subtractes a background in time and returns an array with new data.
        """

        data = self.data
        track_t,track_c = self.track0  # get track time and channel arrays
        trackon = (list(n.array(track_t)+tbin), track_c)   # create new track during integration of interest
        twidth = self.twidth
        dataon = data[trackon[0], :, trackon[1]]

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(-bgwindow/2 - twidth + tbin, - twidth + tbin) + range(twidth + tbin, twidth + bgwindow/2 + tbin)
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track_t)+k), track_c)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track_t)+k), trackoff[1] + track_c)

            dataoff = data[trackoff[0], :, trackoff[1]]

        datadiffarr = n.zeros((self.nchan, self.nbl, self.npol),dtype='complex')

        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            meanon = dataon[indon].mean(axis=0)

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                meanoff = dataoff[indoff].mean(axis=0)
                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])

    def make_bispectra(self, bgwindow=4):
        """
        Makes numpy array of bispectra for each integration. Subtracts visibilities in time in bgwindow.
        """


        bisp = lambda d, ij, jk, ki: d[:,ij] * d[:,jk] * n.conj(d[:,ki])    # bispectrum for pol data
#        bisp = lambda d, ij, jk, ki: n.complex(d[ij] * d[jk] * n.conj(d[ki]))  # without pol axis

        triples = self.make_triples()

        self.bispectra = n.zeros((len(self.data), len(triples)), dtype='complex')
        truearr = n.ones( (self.npol, self.nbl, self.nchan))
        falsearr = n.zeros( (self.npol, self.nbl, self.nchan))

        for i in range(bgwindow/2, len(self.data)-(bgwindow+1)):
            diff = self.tracksub(i, bgwindow=bgwindow)

            if len(n.shape(diff)) == 1:    # no track
                continue

            weightarr = n.where(diff != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack

            try:
                diffmean = n.average(diff, axis=2, weights=weightarr)
            except ZeroDivisionError:
                diffmean = n.mean(diff, axis=2)    # if all zeros, just make mean # bit of a hack

            for trip in range(len(triples)):
                ij, jk, ki = triples[trip]
                self.bispectra[i, trip] = bisp(diffmean, ij, jk, ki).mean(axis=0)  # Stokes I bispectrum. should be ok for multipol data...

    def detect_bispectra(self, sigma=5., tol=1.3, show=0, save=0):
        """Function to search for a transient in a bispectrum lightcurve.
        Designed to be used by bisplc function or easily fed the output of that function.
        sigma gives the threshold for SNR_bisp (apparent). 
        tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
        Returns the SNR and integration number of any candidate events.
        """

        ba = self.bispectra

        ntr = lambda num: num*(num-1)*(num-2)/6.  # theoretical number of triples
#        ntr = lambda num: n.mean([len(n.where(ba[i] != 0j)[0]) for i in range(len(ba))])   # consider possibility of zeros in data and take mean number of good triples over all times

        # using s=S/Q
#        mu = lambda s: s/(1+s)  # for independent bispectra, as in kulkarni 1989
        mu = lambda s: 1.  # for bispectra at high S/N from visibilities?
        sigbQ3 = lambda s: n.sqrt((1 + 3*mu(s)**2) + 3*(1 + mu(s)**2)*s**2 + 3*s**4)  # from kulkarni 1989, normalized by Q**3, also rogers et al 1995
        s = lambda basnr, nants: (2.*basnr/n.sqrt(ntr(nants)))**(1/3.)

        # measure SNR_bl==Q from sigma clipped times with normal mean and std of bispectra. put into time,dm order
        bamean = ba.real.mean(axis=1)
        bastd = ba.real.std(axis=1)

        (meanmin,meanmax) = sigma_clip(bamean)  # remove rfi
        (stdmin,stdmax) = sigma_clip(bastd)  # remove rfi
        clipped = n.where((bamean > meanmin) & (bamean < meanmax) & (bastd > stdmin) & (bastd < stdmax) & (bamean != 0.0))[0]  # remove rf

        bameanstd = ba[clipped].real.mean(axis=1).std()
        basnr = bamean/bameanstd
        Q = ((bameanstd/2.)*n.sqrt(ntr(self.nants)))**(1/3.)
#        Q = n.median( bastd[clipped]**(1/3.) )              # alternate for Q
#        Q = sigt0toQ(bameanstd, self.nants)              # alternate for Q
        print 'Noise per baseline (system units), Q =', Q

        # detect
        cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, self.nants))) & (basnr > sigma) )[0]  # define compact sources with good snr

        # plot snrb lc and expected snr vs. sigb relation
        if show or save:
            p.figure(1)
            ax = p.axes()
            p.subplot(211)
            p.title(str(self.nskip/self.nbl)+' nskip, ' + str(len(cands))+' candidates', transform = ax.transAxes)
            p.plot(basnr, 'b.')
            if len(cands) > 0:
                p.plot(cands, basnr[cands], 'r*')
                p.ylim(-2*basnr[cands].max(),2*basnr[cands].max())
            p.xlabel('Integration')
            p.ylabel('SNR$_{bisp}$')
            p.subplot(212)
            p.plot(bastd/Q**3, basnr, 'b.')

            # plot reference theory lines
            smax = s(basnr.max(), self.nants)
            sarr = smax*n.arange(0,51)/50.
            p.plot(sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k')
            p.plot(tol*sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k--')
            p.plot(bastd[cands]/Q**3, basnr[cands], 'r*')

            if len(cands) > 0:
                p.axis([0, tol*sigbQ3(s(basnr[cands].max(), self.nants)), -0.5*basnr[cands].max(), 1.1*basnr[cands].max()])

                # show spectral modulation next to each point
                for candint in cands:
                    sm = n.single(round(self.specmod(candint),1))
                    p.text(bastd[candint]/Q**3, basnr[candint], str(sm), horizontalalignment='right', verticalalignment='bottom')
            p.xlabel('$\sigma_b/Q^3$')
            p.ylabel('SNR$_{bisp}$')
            if save:
                savename = self.file.split('.')[:-1]
                savename.append(str(self.nskip/self.nbl) + '_' + '.bisplc.png')
                savename = string.join(savename,'.')
                p.savefig(pathout+savename)
            else:
                pass

        return basnr[cands], bastd[cands], cands

    def specmod(self, tbin, bgwindow=4):
        """Calculate spectral modulation for given track.
        Spectral modulation is basically the standard deviation of a spectrum. 
        This helps quantify whether the flux is located in a narrow number of channels or across all channels.
        Narrow RFI has large (>5) modulation, while spectrally broad emission has low modulation.
        See Spitler et al 2012 for details.
        """

        diff = self.tracksub(tbin, bgwindow=bgwindow)
        bfspec = diff.mean(axis=0).real  # should be ok for multipol data...
        sm = n.sqrt( ((bfspec**2).mean() - bfspec.mean()**2) / bfspec.mean()**2 )

        return sm


class pipe_msdisp(msreader):
    """ Create pipeline object for reading in MS data and doing integration-based analysis with dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    spw is list of spectral windows to read from MS.
    selectpol is list of polarization product names for reading from MS
    scan is zero-based selection of scan for reading from MS. It is based on scan order, not actual scan number.
    datacol is the name of the data column name to read from the MS.
    """

    def __init__(self, file, version='default', nints=1000, nskip=0, spw=[-1], selectpol=['RR','LL'], scan=0, datacol='data'):
        self.set_params(version=version)

        self.read(file=file, nints=nints, nskip=nskip, spw=spw, selectpol=selectpol, scan=scan, datacol=datacol)

        # set up ur tracks (lol)
        self.dmtrack0 = {}
        self.twidths = {}
        for dmbin in range(len(self.dmarr)):
            self.dmtrack0[dmbin] = self.dmtrack(self.dmarr[dmbin],0)
            self.twidths[dmbin] = 0
            for k in self.dmtrack0[dmbin][1]:
                self.twidths[dmbin] = max(self.twidths[dmbin], len(n.where(n.array(self.dmtrack0[dmbin][1]) == k)[0]))

    def dmtrack(self, dm = 0., t0 = 0., show=0):
        """ Takes dispersion measure in pc/cm3 and time offset from first integration in seconds.
        t0 defined at first (unflagged) channel. Need to correct by flight time from there to freq=0 for true time.
        Returns an array of (timebin, channel) to select from the data array.
        """

        reltime = self.reltime
        chans = self.chans
        tint = self.inttime   # hack! actually grabs in window of 2*tint. avoids problems with time alignment.

        # given freq, dm, dfreq, calculate pulse time and duration
        pulset_firstchan = 4.2e-3 * dm * self.freq[len(self.chans)-1]**(-2)   # used to start dmtrack at highest-freq unflagged channel
        pulset = 4.2e-3 * dm * self.freq**(-2) + t0 - pulset_firstchan  # time in seconds
        pulsedt = n.sqrt( (8.3e-6 * dm * (1000*self.sdf) * self.freq**(-3))**2 + self.pulsewidth**2)   # dtime in seconds

        timebin = []
        chanbin = []

        for ch in range(len(chans)):
            ontime = n.where(((pulset[ch] + pulsedt[ch]) >= reltime - tint/2.) & (pulset[ch] <= reltime + tint/2.))
            timebin = n.concatenate((timebin, ontime[0]))
            chanbin = n.concatenate((chanbin, (ch * n.ones(len(ontime[0]), dtype=int))))

        track = (list(timebin), list(chanbin))

        if show:
            p.plot(track[0], track[1], 'w*')

        return track

    def tracksub(self, dmbin, tbin, bgwindow = 0):
        """ Creates a background-subtracted set of visibilities.
        For a given track (i.e., an integration number) and bg window, tracksub subtractes a background in time and returns an array with new data.
        Uses ur track for each dm, then shifts by tint. Faster than using n.where to find good integrations for each trial, but assumes int-aligned pulse.
        """

        data = self.data
        track0,track1 = self.dmtrack0[dmbin]
        trackon = (list(n.array(track0)+tbin), track1)
        twidth = self.twidths[dmbin]
        dataon = data[trackon[0], :, trackon[1]]

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(-bgwindow/2 - twidth + tbin, - twidth + tbin) + range(twidth + tbin, twidth + bgwindow/2 + tbin)
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track0)+k), track1)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track0)+k), trackoff[1] + track1)

            dataoff = data[trackoff[0], :, trackoff[1]]

        datadiffarr = n.zeros((self.nchan, self.nbl, self.npol),dtype='complex')
        
        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            meanon = dataon[indon].mean(axis=0)

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                meanoff = dataoff[indoff].mean(axis=0)
                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])

    def make_bispectra(self, bgwindow=4, show=0, save=0):
        """ Steps in Bispectrum Transient Detection Algorithm

1) Collect visibility spectra for some length of time. In Python, I read data into an array with a shape of (n_int, n_chan, n_bl).

2) Prepare visibility spectra to create bispectra. Optionally, one can form dedispersed spectra for each baseline. A simpler start (and probably more relevant for LOFAR) would be to instead select a single integration from the data array described above. Either way, this step changes the data shape to (n_chan, n_bl).

3) Subtract visibilities in time. If the sky has many (or complex) sources, the bispectrum is hard to interpret. Subtracting neighboring visibilities in time (or a rolling mean, like v_t2 - (v_t1+v_t3)/2) removes most constant emission. The only trick is that this assumes that the array has not rotated much and that gain and other effects have not changed. This should preserve the data shape as (n_chan, n_bl).

4) Calculate mean visibility for each baseline. After subtracting in time, one can measure the mean visibility across the band. This reduces the shape to (n_bl).

5) Form a bispectrum for every closed triple in the array. There are a total of n_a * (n_a-1) * (n_a-2) / 6 possible closed triples in the array, where n_a is the number of antennas. One way to form all bispectra is to iterate over antenna indices like this:
for i in range(0, len(n_a)-2):
  for j in range(i, len(n_a)-1):
    for k in range(k, len(n_a)):
      bl1, bl2, bl3 = ant2bl(i, j, k)
      bisp = vis[bl1] * vis[bl2] * vis[bl3]

As you can see, this loop needs a function to convert antenna triples to baseline triples (I call it "ant2bl" here). That is, for antennas (i, j, k), you need (bl_ij, bl_jk, bl_ki). Note that the order of the last baseline is flipped; this is a way of showing that the way you "close" a loop is by tracing a single line around all three baselines. This step changes the basic data product from a shape of (n_bl) to (n_tr). 

6) Search the set of bispectra for sign of a source. Each bispectrum is complex, but if there is a point source in the (differenced) data, all bispectra will respond in the same way. This happens regardless of the location in the field of view.
The mean of all bispectra will scale with the source brightness to the third power, since it is formed from the product of three visibilities. Oddly, the standard deviation of the bispectra will *also* change with the source brightness, due to something called "self noise". The standard deviation of bispectra in the real-imaginary plane should be sqrt(3) S^2 sigma_bl, where S is the source brightness and sigma_bl is the noise on an individual baseline.
In practice, this search involves plotting the mean bispectrum versus time and searching for large deviations. At the same time, a plot of mean versus standard deviation of bispectra will show whether any significant deviation obeys the expected self-noise scaling. That scaling is only valid for a single point source in the field of view, which is what you expect for a fast transient. Any other behavior would be either noise-like or caused by RFI. In particular, RFI will look like a transient, but since it does not often look like a point source, it can be rejected in the plot of mean vs. standard deviation of bispectra. This is a point that I've demonstrated on a small scale, but would needs more testing, since RFI is so varied.
        """

        bisp = lambda d, ij, jk, ki: d[:,ij] * d[:,jk] * n.conj(d[:,ki])    # bispectrum for pol data
#        bisp = lambda d, ij, jk, ki: n.complex(d[ij] * d[jk] * n.conj(d[ki]))   # without pol axis

        triples = self.make_triples()

        # set up arrays for bispectrum and for weighting data (ignoring zeros)
        self.bispectra = n.zeros((len(self.dmarr), len(self.data), len(triples)), dtype='complex')
        truearr = n.ones( (self.npol, self.nbl, self.nchan))
        falsearr = n.zeros( (self.npol, self.nbl, self.nchan))

        # iterate over dm trials and integrations
        for d in range(len(self.dmarr)):
            twidth = n.round(self.twidths[d])
            dmwidth = int(n.round(n.max(self.dmtrack0[d][0]) - n.min(self.dmtrack0[d][0])))

            for i in range((bgwindow+twidth)/2, len(self.data)-( (bgwindow+twidth)/2+dmwidth+1 )):   # dmwidth avoided at end, others are split on front and back side of time iteration
                diff = self.tracksub(d, i, bgwindow=bgwindow)

                if len(n.shape(diff)) == 1:    # no track
                    continue

                weightarr = n.where(diff != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack

                try:
                    diffmean = n.average(diff, axis=2, weights=weightarr)
                except ZeroDivisionError:
                    diffmean = n.mean(diff, axis=2)    # if all zeros, just make mean # bit of a hack

                for trip in range(len(triples)):
                    ij, jk, ki = triples[trip]
                    self.bispectra[d, i, trip] = bisp(diffmean, ij, jk, ki).mean(axis=0)  # Stokes I bispectrum. should be ok for multipol data...

    def detect_bispectra(self, sigma=5., tol=1.3, show=0, save=0):
        """ Function to detect source in bispectra
        sigma gives the threshold for SNR_bisp (apparent). 
        tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
        """

        ba = self.bispectra

        ntr = lambda num: num*(num-1)*(num-2)/6.   # assuming all triples are present
#        ntr = lambda bispectra: n.mean([len(n.where(bispectra[0][i] != 0j)[0]) for i in range(len(bispectra[0]))])   # consider possibility of zeros in data and take mean number of good triples over all times. assumes first dm trial is reasonable estimate.

        # using s=S/Q
        mu = lambda s: 1.  # for bispectra formed from visibilities
        sigbQ3 = lambda s: n.sqrt((1 + 3*mu(s)**2) + 3*(1 + mu(s)**2)*s**2 + 3*s**4)  # from kulkarni 1989, normalized by Q**3, also rogers et al 1995
        s = lambda basnr, nants: (2.*basnr/n.sqrt(ntr(nants)))**(1/3.)  # see rogers et al. 1995 for factor of 2

        # measure SNR_bl==Q from sigma clipped times with normal mean and std of bispectra. put into time,dm order
        bamean = ba.real.mean(axis=2).transpose()
        bastd = ba.real.std(axis=2).transpose()

        bameanstd = []
        for dmind in range(len(self.dmarr)):
            (meanmin,meanmax) = sigma_clip(bamean[:, dmind])  # remove rfi to estimate noise-like parts
            (stdmin,stdmax) = sigma_clip(bastd[:, dmind])
            clipped = n.where((bamean[:, dmind] > meanmin) & (bamean[:, dmind] < meanmax) & (bastd[:, dmind] > stdmin) & (bastd[:, dmind] < stdmax) & (bamean[:, dmind] != 0.0))[0]  # remove rfi and zeros
            bameanstd.append(ba[dmind][clipped].real.mean(axis=1).std())

        bameanstd = n.array(bameanstd)
        basnr = bamean/bameanstd    # = S**3/(Q**3 / n.sqrt(n_tr)) = s**3 * n.sqrt(n_tr)
        Q = ((bameanstd/2.)*n.sqrt(ntr(self.nants)))**(1/3.)
#        Q = n.median( bastd[clipped]**(1/3.) )              # alternate for Q
#        Q = sigt0toQ(bameanstd, self.nants)              # alternate for Q
        print 'Noise per baseline (system units), Q per DM trial =', Q

        # detect
        cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, self.nants))) & (basnr > sigma) )  # get compact sources with high snr

        # plot snrb lc and expected snr vs. sigb relation
        if show or save:
            for dmbin in range(len(self.dmarr)):
                cands_dm = cands[0][n.where(cands[1] == dmbin)[0]]  # find candidates for this dmbin
                p.figure(range(len(self.dmarr)).index(dmbin)+1)
                ax = p.axes()
                p.subplot(211)
                p.title(str(self.scan) + ' scan, ' + str(self.nskip/self.nbl) + ' nskip, ' + str(dmbin) + ' dmbin, ' + str(len(cands_dm))+' candidates', transform = ax.transAxes)
                p.plot(basnr[:,dmbin], 'b.')
                if len(cands_dm) > 0:
                    p.plot(cands_dm, basnr[cands_dm,dmbin], 'r*')
                    p.ylim(-2*basnr[cands_dm,dmbin].max(),2*basnr[cands_dm,dmbin].max())
                p.xlabel('Integration')
                p.ylabel('SNR_b')
                p.subplot(212)
                p.plot(bastd[:,dmbin]/Q[dmbin]**3, basnr[:,dmbin], 'b.')

                # plot reference theory lines
                smax = s(basnr[:,dmbin].max(), self.nants)
                sarr = smax*n.arange(0,51)/50.
                p.plot(sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k')
                p.plot(tol*sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k--')
                p.plot(bastd[cands_dm,dmbin]/Q[dmbin]**3, basnr[cands_dm,dmbin], 'r*')

                if len(cands_dm) > 0:
                    p.axis([0, tol*sigbQ3(s(basnr[cands_dm,dmbin].max(), self.nants)), -0.5*basnr[cands_dm,dmbin].max(), 1.1*basnr[cands_dm,dmbin].max()])

                    # show spectral modulation next to each point
                    for candint in cands_dm:
                        sm = n.single(round(self.specmod(dmbin,candint),1))
                        p.text(bastd[candint,dmbin]/Q[dmbin]**3, basnr[candint,dmbin], str(sm), horizontalalignment='right', verticalalignment='bottom')
                p.xlabel('sigma_b/Q^3')
                p.ylabel('SNR_b')
                if save:
                    savename = self.file.split('.')[:-1]
                    savename.append(str(self.scan) + '_' + str(self.nskip/self.nbl) + '_' + str(dmbin) + '_bisp.png')
                    savename = string.join(savename,'.')
                    p.savefig(self.pathout+savename)
                else:
                    pass

        return basnr[cands], bastd[cands], cands

    def specmod(self, dmbin, tbin, bgwindow=4):
        """Calculate spectral modulation for given dmtrack.
        Narrow RFI has large (>5) modulation, while spectrally broad emission has low modulation.
        See Spitler et al 2012 for details.
        """

#        smarr = n.zeros(len(self.dataph))  # uncomment to do specmod lightcurve
#        for int in range(len(self.dataph)-bgwindow):
        diff = self.tracksub(dmbin, tbin, bgwindow=bgwindow)
        bfspec = diff.mean(axis=0).real  # should be ok for multipol data...
        sm = n.sqrt( ((bfspec**2).mean() - bfspec.mean()**2) / bfspec.mean()**2 )

        return sm

    def make_bflc(self):
        """ Integrates data at dmtrack for each pair of elements in dmarr, time.
        Not threaded.  Uses dmthread directly.
        Stores mean of detected signal after dmtrack, effectively forming beam at phase center.
        Probably ok for multipol data...
        """

        bfarr = n.zeros((len(self.dmarr),len(self.reltime)), dtype='float64')

        for i in range(len(self.dmarr)):
            for j in range(len(self.reltime)):
                dmtrack = self.dmtrack(dm=self.dmarr[i], t0=self.reltime[j])
                if ((dmtrack[1][0] == 0) & (dmtrack[1][len(dmtrack[1])-1] == len(self.chans)-1)):   # use only tracks that span whole band
                    bfarr[i,j] = ((((self.data).mean(axis=1))[dmtrack[0],dmtrack[1]]).mean()).real    # use real part to detect on axis, but keep gaussian dis'n
            print 'dedispersed for ', self.dmarr[i]

        self.bfarr = bfarr

    def plot_bflc(self, peaks, save=0):
        """ Plot beamformed lightcurve.
        Calculates rms noise in dmt0 space, then plots circles for each significant point
        save=1 means plot to file.
        peaks is the array of candidates returned by peakbflc
        """

        arr = self.bfarr
        tbuffer = 7  # number of extra iterations to trim from edge of dmt0 plot

        # Trim data down to where dmt0 array is nonzero
        arreq0 = n.where(arr == 0)
        trimt = arreq0[1].min()
        arr = arr[:,:trimt - tbuffer]
        reltime = self.reltime[:trimt - tbuffer]
        print 'bfarr/time trimmed to new shape:  ',n.shape(arr), n.shape(reltime)

        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean)/std
        peakmax = n.where(arr == arr.max())

        # Plot
#        p.clf()
        ax = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
        p.colorbar()

        if len(peaks[0]) > 0:
            print 'Peak of %f at DM=%f, t0=%f' % (arr.max(), self.dmarr[peakmax[0][0]], reltime[peakmax[1][0]])

            for i in range(len(peaks[1])):
                ax = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                p.axis((min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                p.plot([reltime[peaks[1][i]]], [self.dmarr[peaks[0][i]]], 'o', markersize=2*arr[peaks[0][i],peaks[1][i]], markerfacecolor='white', markeredgecolor='blue', alpha=0.5)

        p.xlabel('Time (s)')
        p.ylabel('DM (pc/cm3)')
        p.title('Summed Spectra in DM-t0 space')
        if save:
            savename = self.file.split('.')[:-1]
            savename.append(str(self.nskip/self.nbl) + '_bf.png')
            savename = string.join(savename,'.')
            p.savefig(self.pathout+savename)

    def detect_bflc(self, sig=5.):
        """ Method to find peaks in dedispersed data (in dmt0 space).
        Clips noise, also.
        returns peaks array indices for use in plotbflc (also array values)
        """

        arr = self.bfarr
        reltime = self.reltime
        tbuffer = 7  # number of extra iterations to trim from edge of dmt0 plot

        # Trim data down to where dmt0 array is nonzero
        arreq0 = n.where(arr == 0)
        trimt = arreq0[1].min()
        arr = arr[:,:trimt - tbuffer]
        reltime = reltime[:trimt - tbuffer]
        print 'bfarr/time trimmed to new shape:  ',n.shape(arr), n.shape(reltime)

        # single iteration of sigma clip to find mean and std, skipping zeros
        mean = arr.mean()
        std = arr.std()
        print 'initial mean, std:  ', mean, std
        min,max = sigma_clip(arr.flatten())
        cliparr = n.where((arr < max) & (arr > min))
        mean = arr[cliparr].mean()
        std = arr[cliparr].std()
        print 'final mean, sig, std:  ', mean, sig, std

        # Recast arr as significance array
        arr = (arr-mean)/std   # for real valued trial output (gaussian dis'n)

        # Detect peaks
        peaks = n.where(arr > sig)
        peakmax = n.where(arr == arr.max())
        print 'peaks:  ', peaks

        return peaks,arr[peaks]


class pipe_mirint(mirreader):
    """ Create pipeline object for reading in Miriad data and doing integration-based analysis without dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    """

    def __init__(self, file, version='default', nints=1000, nskip=0, nocal=False, nopass=False):
        self.set_params(version=version)

        self.read(file=file, nints=nints, nskip=nskip, nocal=nocal, nopass=nopass)

        self.track0 = [n.zeros(len(self.chans)), self.chans]  # note that this may lose some sensitivity at half-int steps

    def tracksub(self, tbin, bgwindow = 0):
        """ Creates a background-subtracted set of visibilities.
        For a given track (i.e., an integration number) and bg window, tracksub subtractes a background in time and returns an array with new data.
        """

        data = self.data
        track_t,track_c = self.track0  # get track time and channel arrays
        trackon = (list(n.array(track_t)+tbin), track_c)   # create new track during integration of interest
        dataon = data[trackon[0], :, trackon[1]]

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(-bgwindow/2 + tbin, tbin) + range(tbin, bgwindow/2 + tbin)
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track_t)+k), track_c)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track_t)+k), trackoff[1] + track_c)

            dataoff = data[trackoff[0], :, trackoff[1]]

        datadiffarr = n.zeros((self.nchan, self.nbl, self.npol),dtype='complex')

        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            meanon = dataon[indon].mean(axis=0)

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                meanoff = dataoff[indoff].mean(axis=0)
                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])


class pipe_mirdisp(mirreader):
    """ Create pipeline object for reading in Miriad data and doing integration-based analysis without dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    """

    def __init__(self, file, version='default', nints=1000, nskip=0, nocal=False, nopass=False):
        self.set_params(version=version)

        self.read(file=file, nints=nints, nskip=nskip, nocal=nocal, nopass=nopass)

        # set up ur tracks (lol)
        self.dmtrack0 = {}
        self.twidths = {}
        for dmbin in range(len(self.dmarr)):
            self.dmtrack0[dmbin] = self.dmtrack(self.dmarr[dmbin],0)
            self.twidths[dmbin] = 0
            for k in self.dmtrack0[dmbin][1]:
                self.twidths[dmbin] = max(self.twidths[dmbin], len(n.where(n.array(self.dmtrack0[dmbin][1]) == k)[0]))

    def tracksub(self, dmbin, tbin, bgwindow = 0):
        """ Creates a background-subtracted set of visibilities.
        For a given track (i.e., an integration number) and bg window, tracksub subtractes a background in time and returns an array with new data.
        Uses ur track for each dm, then shifts by tint. Faster than using n.where to find good integrations for each trial, but assumes int-aligned pulse.
        """

        data = self.data
        track0,track1 = self.dmtrack0[dmbin]
        trackon = (list(n.array(track0)+tbin), track1)
        twidth = self.twidths[dmbin]
        dataon = data[trackon[0], :, trackon[1]]

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(-bgwindow/2 - twidth + tbin, - twidth + tbin) + range(twidth + tbin, twidth + bgwindow/2 + tbin)
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track0)+k), track1)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track0)+k), trackoff[1] + track1)

            dataoff = data[trackoff[0], :, trackoff[1]]

        datadiffarr = n.zeros((self.nchan, self.nbl, self.npol),dtype='complex')
        
        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            meanon = dataon[indon].mean(axis=0)

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                meanoff = dataoff[indoff].mean(axis=0)
                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])

    def writetrack(self, dmbin, tbin, tshift=0, bgwindow=0, show=0, pol=0):
        """ Writes data from track out as miriad visibility file.
        Alternative to writetrack that uses stored, approximate preamble used from start of pulse, not middle.
        Optional background subtraction bl-by-bl over bgwindow integrations. Note that this is bgwindow *dmtracks* so width is bgwindow+track width
        """

        # create bgsub data
        datadiffarr = self.tracksub(dmbin, tbin, bgwindow=bgwindow)
        if n.shape(datadiffarr) == n.shape([0]):    # if track doesn't cross band, ignore this iteration
            return 0

        data = n.zeros(self.nchan, dtype='complex64')  # default data array. gets overwritten.
        data0 = n.zeros(self.nchan, dtype='complex64')  # zero data array for flagged bls
        flags = n.zeros(self.nchan, dtype='bool')

        # define output visibility file names
        outname = string.join(self.file.split('.')[:-1], '.') + '.' + str(self.nskip/self.nbl) + '-' + 'dm' + str(dmbin) + 't' + str(tbin) + '.mir'
        print outname
        vis = miriad.VisData(self.file,)

        int0 = int((tbin + tshift) * self.nbl)
        flags0 = []
        i = 0
        for inp, preamble, data, flags in vis.readLowlevel ('dsl3', False, nocal=True, nopass=True):
            if i == 0:
                # prep for temp output vis file
                shutil.rmtree(outname, ignore_errors=True)
                out = miriad.VisData(outname)
                dOut = out.open ('c')

                # set variables
                dOut.setPreambleType ('uvw', 'time', 'baseline')
                dOut.writeVarInt ('nants', self.nants0)
                dOut.writeVarFloat ('inttime', self.inttime0)
                dOut.writeVarInt ('nspect', self.nspect0)
                dOut.writeVarDouble ('sdf', self.sdf0)
                dOut.writeVarInt ('nwide', self.nwide0)
                dOut.writeVarInt ('nschan', self.nschan0)
                dOut.writeVarInt ('ischan', self.ischan0)
                dOut.writeVarDouble ('sfreq', self.sfreq0)
                dOut.writeVarDouble ('restfreq', self.restfreq0)
                dOut.writeVarInt ('pol', self.pol0)
#                inp.copyHeader (dOut, 'history')
                inp.initVarsAsInput (' ') # ???
                inp.copyLineVars (dOut)
            if i < self.nbl:
                flags0.append(flags.copy())
                i = i+1
            else:
                break

        l = 0
        for i in range(len(flags0)):  # iterate over baselines
            # write out track, if not flagged
            if n.any(flags0[i]):
                k = 0
                for j in range(self.nchan):
                    if j in self.chans:
                        data[j] = datadiffarr[pol, l, k]
#                        flags[j] = flags0[i][j]
                        k = k+1
                    else:
                        data[j] = 0 + 0j
#                        flags[j] = False
                l = l+1
            else:
                data = data0
#                flags = n.zeros(self.nchan, dtype='bool')

            dOut.write (self.preamble[int0 + i], data, flags0[i])

        dOut.close ()
        return 1

    def writetrack2(self, dmbin, tbin, tshift=0, bgwindow=0, show=0, pol=0):
        """ Writes data from track out as miriad visibility file.
        Alternative to writetrack that uses stored, approximate preamble used from start of pulse, not middle.
        Optional background subtraction bl-by-bl over bgwindow integrations. Note that this is bgwindow *dmtracks* so width is bgwindow+track width
        """

        # create bgsub data
        datadiffarr = self.tracksub(dmbin, tbin, bgwindow=bgwindow)
        if n.shape(datadiffarr) == n.shape([0]):    # if track doesn't cross band, ignore this iteration
            return 0

        data = n.zeros(self.nchan, dtype='complex64')  # default data array. gets overwritten.
        data0 = n.zeros(self.nchan, dtype='complex64')  # zero data array for flagged bls
        flags = n.zeros(self.nchan, dtype='bool')

        # define output visibility file names
        outname = string.join(self.file.split('.')[:-1], '.') + '.' + str(self.nskip/self.nbl) + '-' + 'dm' + str(dmbin) + 't' + str(tbin) + '.mir'
        print outname
        vis = miriad.VisData(self.file,)

        int0 = int((tbin + tshift) * self.nbl)
        flags0 = []
        i = 0
        for inp, preamble, data, flags in vis.readLowlevel ('dsl3', False, nocal=True, nopass=True):
            if i == 0:
                # prep for temp output vis file
                shutil.rmtree(outname, ignore_errors=True)
                out = miriad.VisData(outname)
                dOut = out.open ('c')

                # set variables
                dOut.setPreambleType ('uvw', 'time', 'baseline')
                dOut.writeVarInt ('nants', self.nants0)
                dOut.writeVarFloat ('inttime', self.inttime0)
                dOut.writeVarInt ('nspect', self.nspect0)
                dOut.writeVarDouble ('sdf', self.sdf0)
                dOut.writeVarInt ('nwide', self.nwide0)
                dOut.writeVarInt ('nschan', self.nschan0)
                dOut.writeVarInt ('ischan', self.ischan0)
                dOut.writeVarDouble ('sfreq', self.sfreq0)
                dOut.writeVarDouble ('restfreq', self.restfreq0)
                dOut.writeVarInt ('pol', self.pol0)
#                inp.copyHeader (dOut, 'history')
                inp.initVarsAsInput (' ') # ???
                inp.copyLineVars (dOut)
            if i < self.nbl:
                flags0.append(flags.copy())
                i = i+1
            else:
                break

        l = 0
        for i in range(len(flags0)):  # iterate over baselines
            # write out track, if not flagged
            if n.any(flags0[i]):
                k = 0
                for j in range(self.nchan):
                    if j in self.chans:
                        data[j] = datadiffarr[pol, l, k]
#                        flags[j] = flags0[i][j]
                        k = k+1
                    else:
                        data[j] = 0 + 0j
#                        flags[j] = False
                l = l+1
            else:
                data = data0
#                flags = n.zeros(self.nchan, dtype='bool')

            dOut.write (self.preamble[int0 + i], data, flags0[i])

        dOut.close ()
        return 1


def sigma_clip(arr,sigma=3):
    """ Function takes 1d array of values and returns the sigma-clipped min and max scaled by value "sigma".
    """

    cliparr = range(len(arr))  # initialize
    arr = n.append(arr,[1])    # append superfluous item to trigger loop
    while len(cliparr) != len(arr):
        arr = arr[cliparr]
        mean = arr.mean()
        std = arr.std()
        cliparr = n.where((arr < mean + sigma*std) & (arr > mean - sigma*std))[0]
    return mean - sigma*std, mean + sigma*std


