#! /usr/bin/env python

"""
tpipe.py --- read and visualize visibility data to search for transients
Generalization of evlavis, etc.
Can read MS or Miriad formatted data. Will try to import the following:
- CASA and pyrap for MS reading
- miriad-python for Miriad reading
- aipy for imaging numpy array data
"""

import sys, string, os, shutil, types
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


class Reader:
    """ Master class with basic functions.
    self.params defines various tunable parameters for reading data and running pipelines. 
    A "profile" is a set of params that can be hardwired for easy access.
    Can also set parameters giving them as keyword arguments (e.g., "chans=n.array(range(100,110))")
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def set_params(self, profile='default', **kargs):
        """ Method called by __init__ in subclasses. This sets all parameters needed elsewhere.
        Can optionally use a profile which is a set of params.
        Also can set key directly as keyword=value pair.
        """

        # parameters used by various subclasses
        # each set is indexed by a name, called a profile
        # Note that each parameter must also be listed in set_params method in order to get set
        self.profile = profile
        self.params = {
            'default' : {
                'chans': n.array(range(5,59)),   # channels to read
                'dmarr' : [44.,88.],      # dm values to use for dedispersion (only for some subclasses)
                'pulsewidth' : 0.0,      # width of pulse in time (seconds)
                'approxuvw' : True,      # flag to make template visibility file to speed up writing of dm track data
                'pathout': './',         # place to put output files
                'beam_params': [0]         # flag=0 or list of parameters for twodgaussian parameter definition
                },
            'vlacrab' : {
                'chans': n.array(range(5,59)),   # channels to read
                'dmarr' : [29.,58.],      # dm values to use for dedispersion (only for some subclasses)
                'pulsewidth' : 0.0,      # width of pulse in time (seconds)
                'approxuvw' : True,      # flag to make template visibility file to speed up writing of dm track data
                'pathout': './',         # place to put output files
                'beam_params': [0]         # flag=0 or list of parameters for twodgaussian parameter definition
                },
            'psa' : {
                'chans': n.array(range(140,150)),   # channels to read
                'dmarr' : [0.],      # dm values to use for dedispersion (only for some subclasses)
                'pulsewidth' : 0.0,      # width of pulse in time (seconds)
                'approxuvw' : True,      # flag to make template visibility file to speed up writing of dm track data
                'pathout': './',         # place to put output files
                'beam_params': [0]         # flag=0 or list of parameters for twodgaussian parameter definition
                },
            'pocob0329' : {
                'chans': n.array(range(5,59)),   # channels to read
                'dmarr' : [0, 13.4, 26.8, 40.2, 53.5],      # dm values to use for dedispersion (only for some subclasses)
                'pulsewidth' : 0.005,      # width of pulse in time (seconds)
                'approxuvw' : True,      # flag to make template visibility file to speed up writing of dm track data
                'pathout': './',         # place to put output files
                'beam_params': [0]         # flag=0 or list of parameters for twodgaussian parameter definition
                }
            }

        # may further modify parameters manually
        if len(kargs) > 0:
            for key in kargs:
                if key in self.params[profile].keys():
                    self.params[profile][key] = kargs[key]
                else:
                    print '%s not a standard key. Will not be used.' % (key)
                    
        self.pathout = self.params[profile]['pathout']
        self.chans = self.params[profile]['chans']
        self.dmarr = self.params[profile]['dmarr']
        self.pulsewidth = self.params[profile]['pulsewidth'] * n.ones(len(self.chans))
        self.approxuvw = self.params[profile]['approxuvw']
        self.beam_params = self.params[profile]['beam_params']

    def show_params(self):
        """ Print parameters of pipeline that can be modified upon creation.
        """
        
        return self.params[self.profile]

    def spec(self, ind=[], save=0):
        """ Plot spectrogram for phase center by taking mean over baselines and polarizations.
        Optionally can zoom in on small range in time with ind parameter.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        reltime = self.reltime
        bf = self.dataph

        print 'Data mean, std: %f, %f' % (self.dataph.mean(), self.dataph.std())
        (vmin, vmax) = sigma_clip(bf.data.ravel())
        if ( (not(vmin >= 0)) & (not(vmin <= 0)) ):
            print 'sigma_clip returning NaNs. Using data (min,max).'
            vmin = bf.ravel().min()
            vmax = bf.ravel().max()

        p.figure()
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
        p.title(str(self.nskip/self.nbl) + ' nskip, candidates ' + str(ind))

        cb = p.colorbar(im)
        cb.set_label('Flux Density (Jy)',fontsize=12,fontweight="bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.spines['left'].set_position(('outward', 30))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        p.yticks(n.arange(0,len(self.chans),4), (self.chans[(n.arange(0,len(self.chans), 4))]))
        p.xlabel('Time (integration)',fontsize=12,fontweight="bold")
        p.ylabel('Frequency (channel)',fontsize=12,fontweight="bold")
        if save:
            if save == 1:
                savename = self.file.split('.')[:-1]
                savename.append(str(self.scan) + '_' + str(self.nskip/self.nbl) + '_spec.png')
                savename = string.join(savename,'.')
            elif type(save) == type('hi'):
                savename = save
            print 'Saving file as ', savename
            p.savefig(self.pathout+savename)

    def drops(self, data_type='ms', chan=0, pol=0, show=1):
        """ Displays info on missing baselines by looking for zeros in data array.
        data_type is needed to understand how to grab baseline info. options are 'ms' and 'mir'.
        """

        nints = self.nints
        bllen = []

        if data_type == 'mir':
            bls = self.preamble[:,4]
            for bl in n.unique(bls):
                bllen.append(n.shape(n.where(bls == bl))[1])
        elif data_type == 'ms':
            for i in xrange(len(self.blarr)):
                bllen.append(len(n.where(self.data[:,i,chan,pol] != 0.00)[0]))

        bllen = n.array(bllen)

        if show:
            p.clf()
            for i in xrange(self.nbl):
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

    def imagetrack(self, trackdata, i=0, pol='i', size=48000, res=500, clean=True, gain=0.01, tol=1e-4, newbeam=0, save=0, show=0):
        """ Use apiy to image trackdata returned by tracksub of dimensions (npol, nbl, nchan).
        int is used to select uvw coordinates for track. default is first int.
        pol can be 'i' for a Stokes I image (mean over pol dimension) or a pol index.
        default params size and res are good for 1.4 GHz VLA, C-config image.
        clean determines if image is cleaned and beam corrected. gain/tol are cleaning params.
        newbeam forces the calculation of a new beam for restoring the cleaned image.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        # take mean over frequency => introduces delay beam
        truearr = n.ones( n.shape(trackdata) )
        falsearr = 1e-5*n.ones( n.shape(trackdata) )   # need to set to small number so n.average doesn't go NaN
        weightarr = n.where(trackdata != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack                        
        trackdata = n.average(trackdata, axis=2, weights=weightarr)
#        trackdata = trackdata.mean(axis=2)  # alternately can include zeros
        
        if ((pol == 'i') | (pol == 'I')):
            if len(trackdata) == 2:
                print 'Making Stokes I image as mean of two pols...'
            else:
                print 'Making Stokes I image as mean over all pols. Hope that\'s ok...'
            td = trackdata.mean(axis=0)
        elif type(pol) == type(0):
            print 'Making image of pol %d' % (pol)
            td = trackdata[pol]

        fov = n.degrees(1./res)*3600.  # field of view in arcseconds
        p.clf()

        # make image
        ai = aipy.img.Img(size=size, res=res)
        uvw_new, td_new = ai.append_hermitian( (self.u[i],self.v[i],self.w[i]), td)
        ai.put(uvw_new, td_new)
        image = ai.image(center = (size/res/2, size/res/2))
        image_final = image

        # optionally clean image
        if clean:
            print 'Cleaning image...'
            beam = ai.bm_image()
            beamgain = aipy.img.beam_gain(beam[0])
            (clean, dd) = aipy.deconv.clean(image, beam[0], verbose=True, gain=gain, tol=tol)

            try:
                import gaussfitter
                if (len(self.beam_params) == 1) | (newbeam == 1) :
                    print 'Restoring image with new fit to beam shape...'
                    beam_centered = ai.bm_image(center=(size/res/2, size/res/2))
                    peak = n.where(beam_centered[0] >= 0.1*beam_centered[0].max(), beam_centered[0], 0.)
                    self.beam_params = gaussfitter.gaussfit(peak)
                kernel = n.roll(n.roll(gaussfitter.twodgaussian(self.beam_params, shape=n.shape(beam[0])), size/res/2, axis=0), size/res/2, axis=1)   # fit to beam at center, then roll to corners for later convolution step
            except ImportError:
                print 'Restoring image with peak of beam...'
                kernel = n.where(beam[0] >= 0.4*beam[0].max(), beam[0], 0.)  # take only peak (gaussian part) pixels of beam image

            restored = aipy.img.convolve2d(clean, kernel)
            image_restored = (restored + dd['res']).real/beamgain
            image_final = image_restored

        if show or save:
            ax = p.axes()
            ax.set_position([0.2,0.2,0.7,0.7])
#            im = p.imshow(image_final, aspect='auto', origin='upper', interpolation='nearest', extent=[-fov/2, fov/2, -fov/2, fov/2])
            im = p.imshow(image_final, aspect='auto', origin='lower', interpolation='nearest', extent=[fov/2, -fov/2, -fov/2, fov/2])
            cb = p.colorbar(im)
            cb.set_label('Flux Density (Jy)',fontsize=12,fontweight="bold")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 20))
            ax.spines['left'].set_position(('outward', 30))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            p.xlabel('RA/l Offset (arcsec)',fontsize=12,fontweight="bold")
            p.ylabel('Dec/m Offset (arcsec)',fontsize=12,fontweight="bold")

            peak = n.where(n.max(image_final) == image_final)
            print 'Image peak of %e at (%d,%d)' % (n.max(image_final), peak[0][0], peak[1][0])
            print 'Peak/RMS = %e' % (image_final.max()/image_final[n.where(image_final <= 0.9*image_final.max())].std())   # estimate std without image peak

            if save:
                if save == 1:
                    savename = self.file.split('.')[:-1]
                    savename.append(str(self.nskip/self.nbl) + '_im.png')
                    savename = string.join(savename,'.')
                elif type(save) == type('hi'):
                    savename = save
                print 'Saving file as ', savename
                p.savefig(self.pathout+savename)

        return image_final

    def phaseshift(self, dl=0, dm=0, im=[[0]], size=0):
        """ Function to apply phase shift (l,m) coordinates of data array, by (dl, dm).
        If dl,dm are arrays, will try to apply the given shift for each integration separately (courtesy DLK)
        If instead a 2d-array image, im, is given, phase center is shifted to image peak. Needs size to know image scale.
        Sets data and dataph arrays to new values.
        """

        ang = lambda dl,dm,u,v,freq: (dl*n.outer(u,freq/freq.mean()) + dm*n.outer(v,freq/freq.mean()))  # operates on single time of u,v

        if ((len(im) != 1) & (size != 0)):
            y,x = n.where(im == im.max())
            length = len(im)
            dl = (length/2 - x[0]) * 1./size
            dm = (y[0] - length/2) * 1./size
            print 'Shifting phase center to image peak: (dl,dm) = (%e,%e) = (%e,%e) arcsec' % (dl, dm, n.degrees(dl)*3600, n.degrees(dm)*3600)
        elif isinstance(dl,n.ndarray) and isinstance(dm,n.ndarray):
            if not len(dl) == self.nints:
                raise ValueError('dl is an array but its length (%d) does not match the number of integrations (%d)' % (len(dl),self.nints))
            
        elif ((dl != 0) | (dm != 0)):
            print 'Shifting phase center by given (dl,dm) = (%e,%e) = (%e,%e) arcsec' % (dl, dm, n.degrees(dl)*3600, n.degrees(dm)*3600)
        else:
            raise ValueError('Need to give either dl or dm, or im and size.')

        for i in xrange(self.nints):
            for pol in xrange(self.npol):
                if isinstance(dl,n.ndarray):
                    self.data[i,:,:,pol] = self.data[i,:,:,pol] * n.exp(-2j*n.pi*ang(dl[i], dm[i],
                                                                                     self.u[i], self.v[i], self.freq))
                else:
                    self.data[i,:,:,pol] = self.data[i,:,:,pol] * n.exp(-2j*n.pi*ang(dl, dm, self.u[i], self.v[i], self.freq))
    
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


class MiriadReader(Reader):
    """ Class for reading Miriad format data with miriad-python.
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def read(self, file, nints, nskip, nocal, nopass):
        """ Reads in Miriad data using miriad-python.
        """

        self.file = file
        self.nints = nints
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

            # end here. assume at least one instance of each bl occurs before ~six integrations (accommodates MWA)
            if len(bls) == 6*len(n.unique(bls)):
                blarr = []
                for bl in n.unique(bls):
                    blarr.append(util.decodeBaseline (bl))
                self.blarr = n.array(blarr)
                bldict = dict( zip(n.unique(bls), n.arange(len(blarr))) )
                break

            i = i+1

        # Initialize more stuff...
        self.freq_orig = self.sfreq + self.sdf * n.arange(self.nchan)
        self.freq = self.freq_orig[self.chans]

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
        u = n.zeros((nints,self.nbl),dtype='float64')
        v = n.zeros((nints,self.nbl),dtype='float64')
        w = n.zeros((nints,self.nbl),dtype='float64')
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
                # uvw stored in preamble index 0,1,2 in units of ns
                u[(i-nskip)//self.nbl,bldict[preamble[4]]] = preamble[0] * self.freq.mean()
                v[(i-nskip)//self.nbl,bldict[preamble[4]]] = preamble[1] * self.freq.mean()
                w[(i-nskip)//self.nbl,bldict[preamble[4]]] = preamble[2] * self.freq.mean()
            else:
                break     # stop at nints

            if not (i % (self.nbl*100)):
                print 'Read spectrum ', str(i)

            i = i+1

        # Assumes miriad files store uvw in ns. Corrects by mean frequency of channels in use.
        self.u = u
        self.v = v
        self.w = w

        # build final data structures
        self.rawdata = n.expand_dims(da, 3)  # hack to get superfluous pol axis
        self.flags = n.expand_dims(fl, 3)
        self.preamble = pr

        time = self.preamble[::self.nbl,3]
        self.reltime = 24*3600*(time - time[0])      # relative time array in seconds. evla times change...?
        self.inttime = n.array([self.reltime[i+1] - self.reltime[i] for i in xrange(len(self.reltime)/5,len(self.reltime)-1)]).mean()

        # print summary info
        print
        print 'Shape of raw data, time:'
        print self.rawdata.shape, self.reltime.shape

    def writetrack(self, dmbin, tbin, tshift=0, bgwindow=0, show=0, pol=0):
        """ **Not tested recently** Writes data from track out as miriad visibility file.
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
        for i in xrange(len(flags0)):  # iterate over baselines
            # write out track, if not flagged
            if n.any(flags0[i]):
                k = 0
                for j in xrange(self.nchan):
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
        """ **Not tested recently** Writes data from track out as miriad visibility file.
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
        for i in xrange(len(flags0)):  # iterate over baselines
            # write out track, if not flagged
            if n.any(flags0[i]):
                k = 0
                for j in xrange(self.nchan):
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


class MSReader(Reader):
    """ Class for reading MS data with either CASA. (Will eventually use pyrap.)
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def read(self, file, nints, nskip, spw, selectpol, scan, datacol):
        """ Reads in Measurement Set data using CASA.
        spw is list of subbands. zero-based.
        Scan is zero-based selection based on scan order, not actual scan number.
        selectpol is list of polarization strings (e.g., ['RR','LL'])
        """
        self.file = file
        self.scan = scan
        self.nints = nints

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
# old way, casa 3.3?
#            scanlist = scansummary['summary'].keys()
#            starttime_mjd = scansummary['summary'][scanlist[scan]]['0']['BeginTime']
# new way, casa 4.0?
            scanlist = scansummary.keys()
            starttime_mjd = scansummary[scanlist[scan]]['0']['BeginTime']
            self.nskip = int(nskip*self.nbl)    # number of iterations to skip (for reading in different parts of buffer)
            self.npol = len(selectpol)
        else:
            print 'No pickle of initializing info found. Making anew...'
            pkl = open(pklname, 'wb')
            ms.open(self.file)
            spwinfo = ms.getspectralwindowinfo()
            scansummary = ms.getscansummary()

# original (general version)
#            scanlist = scansummary['summary'].keys()
#            starttime_mjd = scansummary['summary'][scanlist[scan]]['0']['BeginTime']
#            starttime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0/(24.*60*60),'d'),form=['ymd'], prec=9), 's'))
#            stoptime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0.5/(24.*60*60), 'd'), form=['ymd'], prec=9), 's'))

# for casa 4.0 (?) and later
            scanlist = scansummary.keys()
            starttime_mjd = scansummary[scanlist[scan]]['0']['BeginTime']
            starttime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]
            stoptime0 = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+0.5/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]

            ms.selectinit(datadescid=0)  # initialize to initialize params
            selection = {'time': [starttime0, stoptime0]}
            ms.select(items = selection)
            da = ms.getdata([datacol, 'axis_info'], ifraxis=True)
            ms.close()

            self.npol_orig = da[datacol].shape[0]
            self.nbl = da[datacol].shape[2]
            print 'Initializing nbl:', self.nbl

            # good baselines
            bls = da['axis_info']['ifr_axis']['ifr_shortname']
            self.blarr = n.array([[int(bls[i].split('-')[0]),int(bls[i].split('-')[1])] for i in xrange(len(bls))])
            self.nskip = int(nskip*self.nbl)    # number of iterations to skip (for reading in different parts of buffer)

            # set integration time
            ti0 = da['axis_info']['time_axis']['MJDseconds']
#            self.inttime = scansummary['summary'][scanlist[scan]]['0']['IntegrationTime']  # general way
            self.inttime = scansummary[scanlist[scan]]['0']['IntegrationTime']   # subset way, or casa 4.0 way?
            self.inttime0 = self.inttime
            print 'Initializing integration time (s):', self.inttime

            pickle.dump((self.npol_orig, self.nbl, self.blarr, self.inttime, self.inttime0, spwinfo, scansummary), pkl)
        pkl.close()

        self.ants = n.unique(self.blarr)
        self.nants = len(n.unique(self.blarr))
        self.nants0 = len(n.unique(self.blarr))
        print 'Initializing nants:', self.nants
        self.npol = len(selectpol)
        print 'Initializing %d of %d polarizations' % (self.npol, self.npol_orig)

        # set desired spw
        if (len(spw) == 1) & (spw[0] == -1):
#            spwlist = spwinfo['spwInfo'].keys()    # old way
            spwlist = spwinfo.keys()    # new way
        else:
            spwlist = spw

        self.freq_orig = n.array([])
        for spw in spwlist:
# new way
            nch = spwinfo[str(spw)]['NumChan']
            ch0 = spwinfo[str(spw)]['Chan1Freq']
            chw = spwinfo[str(spw)]['ChanWidth']
            self.freq_orig = n.concatenate( (self.freq_orig, (ch0 + chw * n.arange(nch)) * 1e-9) )
# old way
#            nch = spwinfo['spwInfo'][str(spw)]['NumChan']
#            ch0 = spwinfo['spwInfo'][str(spw)]['Chan1Freq']
#            chw = spwinfo['spwInfo'][str(spw)]['ChanWidth']

        self.freq = self.freq_orig[self.chans]
        self.nchan = len(self.freq)
        print 'Initializing nchan:', self.nchan

        # set requested time range based on given parameters
        timeskip = self.inttime*nskip
# new way        
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+(timeskip+nints*self.inttime)/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]
        print 'First integration of scan:', qa.time(qa.quantity(starttime_mjd,'d'),form=['ymd'],prec=9)[0]
        print
# new way
        print 'Reading scan', str(scanlist[scan]) ,'for times', qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['hms'], prec=9)[0], 'to', qa.time(qa.quantity(starttime_mjd+(timeskip+nints*self.inttime)/(24.*60*60), 'd'), form=['hms'], prec=9)[0]

        # read data into data structure
        ms.open(self.file)
        ms.selectinit(datadescid=spwlist[0])  # reset select params for later data selection
        selection = {'time': [starttime, stoptime]}
        ms.select(items = selection)
        print 'Reading %s column, SB %d, polarization %s...' % (datacol, spwlist[0], selectpol)
        ms.selectpolarization(selectpol)
        da = ms.getdata([datacol,'axis_info','u','v','w','flag'], ifraxis=True)
        u = da['u']; v = da['v']; w = da['w']
        if da == {}:
            print 'No data found.'
            return 1
        newda = n.transpose(da[datacol], axes=[3,2,1,0])  # if using multi-pol data.
        flags = n.transpose(da['flag'], axes=[3,2,1,0])
        if len(spwlist) > 1:
            for spw in spwlist[1:]:
                ms.selectinit(datadescid=spw)  # reset select params for later data selection
                ms.select(items = selection)
                print 'Reading %s column, SB %d, polarization %s...' % (datacol, spw, selectpol)
                ms.selectpolarization(selectpol)
                da = ms.getdata([datacol,'axis_info','flag'], ifraxis=True)
                newda = n.concatenate( (newda, n.transpose(da[datacol], axes=[3,2,1,0])), axis=2 )
                flags = n.concatenate( (flags, n.transpose(da['flag'], axes=[3,2,1,0])), axis=2 )
        ms.close()

        # Initialize more stuff...
        self.nschan0 = self.nchan

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

        # Assumes MS files store uvw in meters. Corrects by mean frequency of channels in use.
        self.u = u.transpose() * self.freq.mean() * (1e9/3e8)
        self.v = v.transpose() * self.freq.mean() * (1e9/3e8)
        self.w = w.transpose() * self.freq.mean() * (1e9/3e8)

        # set integration time and time axis
        ti = da['axis_info']['time_axis']['MJDseconds']
        self.reltime = ti - ti[0]

        self.rawdata = newda
        self.flags = flags
        print 'Shape of raw data, time:'
        print self.rawdata.shape, self.reltime.shape


class SimulationReader(Reader):
    """ Class for simulating visibility data for transients analysis.
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def simulate(self, nints, inttime, chans, freq, bw, array='vla10'):
        """ Simulates data
        array is the name of array config, nints is number of ints, inttime is integration duration, chans, freq, bw (in GHz) as normal.
        array can be 'vla_d' and 'vla10', the latter is the first 10 of vla_d.
        """

        self.file = 'sim'
        self.chans = chans
        self.nints = nints
        self.nchan = len(chans)
        print 'Initializing nchan:', self.nchan
        self.sfreq = freq    # in GHz
        self.sdf = bw/self.nchan
        self.npol = 1
        self.freq_orig = self.sfreq + self.sdf * n.arange(self.nchan)
        self.freq = self.freq_orig[self.chans]
        self.inttime = inttime   # in seconds
        self.reltime = inttime*n.arange(nints)

        # antennas and baselines
        vla_d = 1e3*n.array([[ 0.00305045,  0.03486681],  [ 0.00893224,  0.10209601],  [ 0.01674565,  0.19140365],  [ 0.02615514,  0.29895461],  [ 0.03696303,  0.42248936],  [ 0.04903413,  0.56046269],  [ 0.06226816,  0.7117283 ],  [ 0.07658673,  0.87539034],  [ 0.09192633,  1.05072281],  [ 0.02867032, -0.02007518],  [ 0.08395162, -0.05878355],  [ 0.1573876 , -0.11020398],  [ 0.24582472, -0.17212832],  [ 0.347405  , -0.2432556 ],  [ 0.46085786, -0.32269615],  [ 0.58524071, -0.40978995],  [ 0.71981691, -0.50402122],  [ 0.86398948, -0.60497195],  [-0.03172077, -0.01479164],  [-0.09288386, -0.04331245],  [-0.17413325, -0.08119967],  [-0.27197986, -0.12682629],  [-0.38436803, -0.17923376],  [-0.509892  , -0.23776654],  [-0.64750886, -0.30193834],  [-0.79640364, -0.37136912],  [-0.95591581, -0.44575086]])

        if array == 'vla_d':
            antloc = vla_d
        elif array == 'vla10':
            antloc = vla_d[5:15]    # 5:15 choses inner part  of two arms
        self.nants = len(antloc)
        print 'Initializing nants:', self.nants
        blarr = []; u = []; v = []; w = []
        for i in range(1, self.nants+1):
            for j in range(i, self.nants+1):
                blarr.append([i,j])
                u.append(antloc[i-1][0] - antloc[j-1][0])   # in meters (like MS, fwiw)
                v.append(antloc[i-1][1] - antloc[j-1][1])
                w.append(0.)
        self.blarr = n.array(blarr)

        self.nbl = len(self.blarr)
        self.u = n.zeros((nints,self.nbl),dtype='float64')
        self.v = n.zeros((nints,self.nbl),dtype='float64')
        self.w = n.zeros((nints,self.nbl),dtype='float64')
        print 'Initializing nbl:', self.nbl
        self.ants = n.unique(self.blarr)
        self.nskip = 0

        # no earth rotation yet
        for i in range(nints):
            self.u[i] = n.array(u) * self.freq.mean() * (1e9/3e8)
            self.v[i] = n.array(v) * self.freq.mean() * (1e9/3e8)
            self.w[i] = n.array(w) * self.freq.mean() * (1e9/3e8)

        # simulate data
        self.rawdata = n.zeros((nints,self.nbl,self.nchan,self.npol),dtype='complex64')
        self.flags = n.zeros((nints,self.nbl,self.nchan,self.npol),dtype='bool')
        self.rawdata.real = n.sqrt(self.nchan) * n.random.randn(nints,self.nbl,self.nchan,self.npol)   # normal width=1 after channel mean
        self.rawdata.imag = n.sqrt(self.nchan) * n.random.randn(nints,self.nbl,self.nchan,self.npol)

        # print summary info
        print
        print 'Shape of raw data, time:'
        print self.rawdata.shape, self.reltime.shape

    def add_transient(self, dl, dm, s, i):
        """ Add a transient to an integration.
        dl, dm are relative direction cosines (location) of transient, s is brightness, and i is integration.
        """

        ang = lambda dl,dm,u,v,freq: (dl*n.outer(u,freq/freq.mean()) + dm*n.outer(v,freq/freq.mean()))  # operates on single time of u,v
        for pol in range(self.npol):
            self.data[i,:,:,pol] = self.data[i,:,:,pol] + s * n.exp(-2j*n.pi*ang(dl, dm, self.u[i], self.v[i], self.freq))

        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real  #dataph is summed and detected to form TP beam at phase center, multi-pol


class ProcessByIntegration():
    """ Class defines methods for pipeline processing for integration-based (no dispersion) transients searches.
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def prep(self):
        """ Sets up tracks used to select data in time. Setting them early helped speed up dedispersion done elsewhere.
        """
        print
        print 'Filtering rawdata to data as masked array...'
# using 0 as flag
#        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.rawdata[:self.nints,:, self.chans,:] == 0j)
# using standard flags
        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.flags[:self.nints,:, self.chans,:] == 0)
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real   #dataph is summed and detected to form TP beam at phase center, multi-pol
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'Shape of data:'
        print self.data.shape
        print 'Dataph min, max:'
        print self.min, self.max

        self.freq = self.freq_orig[self.chans]

        self.track0 = self.track(0.)
        self.twidth = 0
        for k in self.track0[1]:
            self.twidth = max(self.twidth, len(n.where(n.array(self.track0[1]) == k)[0]))

        print 'Track width in time: %d. Iteration could step by %d/2.' % (self.twidth, self.twidth)

    def track(self, t0 = 0., show=0):
        """ Takes time offset from first integration in seconds.
        t0 defined at first (unflagged) channel.
        Returns an array of (timebin, channel) to select from the data array.
        """

        reltime = self.reltime
        chans = self.chans
        tint = self.inttime

        # calculate pulse time and duration
        pulset = t0
        pulsedt = self.pulsewidth[0]   # dtime in seconds. just take one channel, since there is no freq dep

        timebin = []
        chanbin = []

        ontime = n.where(((pulset + pulsedt) >= reltime - tint/2.) & (pulset <= reltime + tint/2.))
        for ch in xrange(len(chans)):
            timebin = n.concatenate((timebin, ontime[0]))
            chanbin = n.concatenate((chanbin, (ch * n.ones(len(ontime[0]), dtype='int'))))

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
        truearron = n.ones( n.shape(dataon) )
        falsearron = 1e-5*n.ones( n.shape(dataon) )  # small weight to keep n.average from giving NaN

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(tbin -(bgwindow/2+twidth)+1, tbin-twidth+1) + range(tbin + twidth, tbin + (twidth+bgwindow/2))
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track_t)+k), track_c)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track_t)+k), list(trackoff[1]) + list(track_c))

            dataoff = data[trackoff[0], :, trackoff[1]]
            truearroff = n.ones( n.shape(dataoff) )
            falsearroff = 1e-5*n.ones( n.shape(dataoff) )  # small weight to keep n.average from giving NaN

        datadiffarr = n.zeros((len(self.chans), self.nbl, self.npol),dtype='complex')

        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            weightarr = n.where(dataon[indon] != 0j, truearron[indon], falsearron[indon])
            meanon = n.average(dataon[indon], axis=0, weights=weightarr)
#            meanon = dataon[indon].mean(axis=0)    # include all zeros

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                weightarr = n.where(dataoff[indoff] != 0j, truearroff[indoff], falsearroff[indoff])
                meanoff = n.average(dataoff[indoff], axis=0, weights=weightarr)
#                meanoff = dataoff[indoff].mean(axis=0)   # include all zeros

                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])

    def make_bispectra(self, bgwindow=4):
        """ Makes numpy array of bispectra for each integration. Subtracts visibilities in time in bgwindow.

        Steps in Bispectrum Transient Detection Algorithm

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
#        bisp = lambda d, ij, jk, ki: n.complex(d[ij] * d[jk] * n.conj(d[ki]))  # without pol axis

        triples = self.make_triples()
        meanbl = self.data.mean(axis=3).mean(axis=2).mean(axis=0)  # find non-zero bls
        self.triples = triples[(meanbl[triples][:,0] != 0j) & (meanbl[triples][:,1] != 0j) & (meanbl[triples][:,2] != 0j)]   # take triples with non-zero bls

        self.bispectra = n.zeros((len(self.data), len(self.triples)), dtype='complex')
        truearr = n.ones( (self.npol, self.nbl, len(self.chans)))
        falsearr = n.zeros( (self.npol, self.nbl, len(self.chans)))

        for i in xrange((bgwindow/2)+self.twidth, len(self.data)-( (bgwindow/2)+self.twidth )):
#        for i in xrange((bgwindow/2)+self.twidth, len(self.data)-( (bgwindow/2)+self.twidth ), max(1,self.twidth)):  # leaves gaps in data
            diff = self.tracksub(i, bgwindow=bgwindow)

            if len(n.shape(diff)) == 1:    # no track
                continue

            weightarr = n.where(diff != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack

            try:
                diffmean = n.average(diff, axis=2, weights=weightarr)
            except ZeroDivisionError:
                diffmean = n.mean(diff, axis=2)    # if all zeros, just make mean # bit of a hack

            for trip in xrange(len(self.triples)):
                ij, jk, ki = self.triples[trip]
                self.bispectra[i, trip] = bisp(diffmean, ij, jk, ki).mean(axis=0)  # Stokes I bispectrum. Note we are averaging after forming bispectrum, so not technically a Stokes I bispectrum.

    def detect_bispectra(self, sigma=5., tol=1.3, Q=0, show=0, save=0):
        """Function to search for a transient in a bispectrum lightcurve.
        Designed to be used by bisplc function or easily fed the output of that function.
        sigma gives the threshold for SNR_bisp (apparent). 
        tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
        Q is noise per baseline and can be input. Otherwise estimated from data.
        Returns the SNR and integration number of any candidate events.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """
        try:
            ba = self.bispectra
        except AttributeError:
            print 'Need to make bispectra first.'
            return

#        ntr = lambda num: num*(num-1)*(num-2)/6  # theoretical number of triples
        ntr = lambda num: len(self.triples)  # consider possibility of zeros in data and take mean number of good triples over all times

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
        if Q:
            print 'Using given Q =', Q
        else:
            Q = ((bameanstd/2.)*n.sqrt(ntr(self.nants)))**(1/3.)
        #        Q = n.median( bastd[clipped]**(1/3.) )              # alternate for Q
            print 'Estimating noise per baseline from data. Q =', Q
        self.Q = Q

        # detect
        cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, self.nants))) & (basnr > sigma) )[0]  # define compact sources with good snr

        # plot snrb lc and expected snr vs. sigb relation
        if show or save:
            p.figure()
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
                if save == 1:
                    savename = self.file.split('.')[:-1]
                    savename.append(str(self.nskip/self.nbl) + '_bisp.png')
                    savename = string.join(savename,'.')
                elif type(save) == type('hi'):
                    savename = save
                print 'Saving file as ', savename
                p.savefig(self.pathout+savename)

        return basnr[cands], bastd[cands], zip(cands[0],cands[1])

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

    def make_phasedbeam(self):
        """Like that of dispersion-based classes, but integration-based.
        Not yet implemented.
        """
        raise NotImplementedError('For now, you could instead used dispersion code with dmarr=[0.]...')
    
    def detect_phasedbeam(self):
        """Like that of dispersion-based classes, but integration-based.
        Not yet implemented.
        """
        raise NotImplementedError('For now, you could instead used dispersion code with dmarr=[0.]...')


class ProcessByDispersion():
    """ Class defines methods for pipeline processing for dispersion-based transients searches.
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def prep(self):
        """ Sets up tracks used to speed up dedispersion code.
        """
        print
        print 'Filtering rawdata to data as masked array...'
# using 0 as flag
#        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.rawdata[:self.nints,:, self.chans,:] == 0j)
# using standard flags
        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.flags[:self.nints,:, self.chans,:] == 0)
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real   #dataph is summed and detected to form TP beam at phase center, multi-pol
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'Shape of data:'
        print self.data.shape
        print 'Dataph min, max:'
        print self.min, self.max

        self.freq = self.freq_orig[self.chans]

        # set up ur tracks (lol)
        self.dmtrack0 = {}
        self.twidths = {}
        for dmbin in xrange(len(self.dmarr)):
            self.dmtrack0[dmbin] = self.dmtrack(self.dmarr[dmbin],0)  # track crosses high-freq channel in first integration
            self.twidths[dmbin] = 0
            for k in self.dmtrack0[dmbin][1]:
                self.twidths[dmbin] = max(self.twidths[dmbin], len(n.where(n.array(self.dmtrack0[dmbin][1]) == k)[0]))

        print 'Track width in time: '
        for dmbin in self.twidths:
            print 'DM=%.1f, twidth=%d. Iteration could step by %d/2.' % (self.dmarr[dmbin], self.twidths[dmbin], self.twidths[dmbin])

    def dmtrack(self, dm = 0., t0 = 0., show=0):
        """ Takes dispersion measure in pc/cm3 and time offset from first integration in seconds.
        t0 defined at first (unflagged) channel. Need to correct by flight time from there to freq=0 for true time.
        Returns an array of (timebin, channel) to select from the data array.
        """

        reltime = self.reltime
        chans = self.chans
        tint = self.inttime

        # given freq, dm, dfreq, calculate pulse time and duration
        pulset_firstchan = 4.2e-3 * dm * self.freq[len(self.chans)-1]**(-2)   # used to start dmtrack at highest-freq unflagged channel
        pulset_midchan = 4.2e-3 * dm * self.freq[len(self.chans)/2]**(-2)   # used to start dmtrack at highest-freq unflagged channel. fails to find bright j0628 pulse
        pulset = 4.2e-3 * dm * self.freq**(-2) + t0 - pulset_firstchan  # time in seconds referenced to some frequency (first, mid, last)
        pulsedt = n.sqrt( (8.3e-6 * dm * (1000*self.sdf) * self.freq**(-3))**2 + self.pulsewidth**2)   # dtime in seconds

        timebin = []
        chanbin = []

        for ch in xrange(len(chans)):
            ontime = n.where(((pulset[ch] + pulsedt[ch]) >= reltime - tint/2.) & (pulset[ch] <= reltime + tint/2.))
            timebin = n.concatenate((timebin, ontime[0]))
            chanbin = n.concatenate((chanbin, (ch * n.ones(len(ontime[0]), dtype='int'))))

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
        truearron = n.ones( n.shape(dataon) )
        falsearron = 1e-5*n.ones( n.shape(dataon) )  # small weight to keep n.average from giving NaN

        # set up bg track
        if bgwindow:
            # measure max width of pulse (to avoid in bgsub)
            bgrange = range(tbin -(bgwindow/2+twidth)+1, tbin-twidth+1) + range(tbin + twidth, tbin + (twidth+bgwindow/2))
            for k in bgrange:     # build up super track for background subtraction
                if bgrange.index(k) == 0:   # first time through
                    trackoff = (list(n.array(track0)+k), track1)
                else:    # then extend arrays by next iterations
                    trackoff = (trackoff[0] + list(n.array(track0)+k), list(trackoff[1]) + list(track1))

            dataoff = data[trackoff[0], :, trackoff[1]]
            truearroff = n.ones( n.shape(dataoff) )
            falsearroff = 1e-5*n.ones( n.shape(dataoff) )  # small weight to keep n.average from giving NaN

        datadiffarr = n.zeros((len(self.chans), self.nbl, self.npol),dtype='complex')
        
        # compress time axis, then subtract on and off tracks
        for ch in n.unique(trackon[1]):
            indon = n.where(trackon[1] == ch)
            weightarr = n.where(dataon[indon] != 0j, truearron[indon], falsearron[indon])
            meanon = n.average(dataon[indon], axis=0, weights=weightarr)
#            meanon = dataon[indon].mean(axis=0)    # include all zeros

            if bgwindow:
                indoff = n.where(trackoff[1] == ch)
                weightarr = n.where(dataoff[indoff] != 0j, truearroff[indoff], falsearroff[indoff])
                meanoff = n.average(dataoff[indoff], axis=0, weights=weightarr)
#                meanoff = dataoff[indoff].mean(axis=0)   # include all zeros

                datadiffarr[ch] = meanon - meanoff
                zeros = n.where( (meanon == 0j) | (meanoff == 0j) )  # find baselines and pols with zeros for meanon or meanoff
                datadiffarr[ch][zeros] = 0j    # set missing data to zero # hack! but could be ok if we can ignore zeros later...
            else:
                datadiffarr[ch] = meanon

        return n.transpose(datadiffarr, axes=[2,1,0])

    def make_bispectra(self, bgwindow=4):
        """ Makes numpy array of bispectra for each integration. Subtracts visibilities in time in bgwindow.

        Steps in Bispectrum Transient Detection Algorithm

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
        meanbl = self.data.mean(axis=2).mean(axis=0)   # find bls with no zeros in either pol to ignore in triples
        self.triples = triples[n.all(meanbl[triples][:,0] != 0j, axis=1) & n.all(meanbl[triples][:,1] != 0j, axis=1) & n.all(meanbl[triples][:,2] != 0j, axis=1)]   # only take triples if both pols are good. may be smaller than set for an individual pol

        # set up arrays for bispectrum and for weighting data (ignoring zeros)
        bispectra = n.zeros((len(self.dmarr), len(self.data), len(self.triples)), dtype='complex')
        truearr = n.ones( (self.npol, self.nbl, len(self.chans)))
        falsearr = n.zeros( (self.npol, self.nbl, len(self.chans)))

        # iterate over dm trials and integrations
        for d in xrange(len(self.dmarr)):
            twidth = n.round(self.twidths[d])
            dmwidth = int(n.round(n.max(self.dmtrack0[d][0]) - n.min(self.dmtrack0[d][0])))

            for i in xrange((bgwindow/2)+twidth, len(self.data)-( (bgwindow/2)+twidth+dmwidth )):   # dmwidth avoided at end, others are split on front and back side of time iteration
#            for i in xrange((bgwindow/2)+twidth, len(self.data)-( (bgwindow/2)+twidth+dmwidth ), max(1,twidth/2)):   # can step by twidth/2, but messes up data products
                diff = self.tracksub(d, i, bgwindow=bgwindow)

                if len(n.shape(diff)) == 1:    # no track
                    continue

# **need to redo for self.flags**
                weightarr = n.where(diff != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack
                try:
                    diffmean = n.average(diff, axis=2, weights=weightarr)
                except ZeroDivisionError:
                    diffmean = n.mean(diff, axis=2)    # if all zeros, just make mean # bit of a hack

                for trip in xrange(len(self.triples)):
                    ij, jk, ki = self.triples[trip]
                    bispectra[d, i, trip] = bisp(diffmean, ij, jk, ki).mean(axis=0)    # Stokes I bispectrum. Note we are averaging after forming bispectrum, so not technically a Stokes I bispectrum.
            print 'dedispersed for ', self.dmarr[d]
        self.bispectra = n.ma.masked_array(bispectra, bispectra == 0j)

    def detect_bispectra(self, sigma=5., tol=1.3, Q=0, show=0, save=0):
        """ Function to detect transient in bispectra
        sigma gives the threshold for SNR_bisp (apparent). 
        tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
        Q is noise per baseline and can be input. Otherwise estimated from data.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        try:
            ba = self.bispectra
        except AttributeError:
            print 'Need to make bispectra first.'
            return

#        ntr = lambda num: num*(num-1)*(num-2)/6   # assuming all triples are present
        ntr = lambda num: len(self.triples)  # consider possibility of zeros in data and take mean number of good triples over all times

        # using s=S/Q
        mu = lambda s: 1.  # for bispectra formed from visibilities
        sigbQ3 = lambda s: n.sqrt((1 + 3*mu(s)**2) + 3*(1 + mu(s)**2)*s**2 + 3*s**4)  # from kulkarni 1989, normalized by Q**3, also rogers et al 1995
        s = lambda basnr, nants: (2.*basnr/n.sqrt(ntr(nants)))**(1/3.)  # see rogers et al. 1995 for factor of 2

        # measure SNR_bl==Q from sigma clipped times with normal mean and std of bispectra. put into time,dm order
        bamean = ba.real.mean(axis=2).transpose()
        bastd = ba.real.std(axis=2).transpose()

        bameanstd = []
        for dmind in xrange(len(self.dmarr)):
            (meanmin,meanmax) = sigma_clip(bamean[:, dmind])  # remove rfi to estimate noise-like parts
            (stdmin,stdmax) = sigma_clip(bastd[:, dmind])
            clipped = n.where((bamean[:, dmind] > meanmin) & (bamean[:, dmind] < meanmax) & (bastd[:, dmind] > stdmin) & (bastd[:, dmind] < stdmax) & (bamean[:, dmind] != 0.0))[0]  # remove rfi and zeros
            bameanstd.append(ba[dmind][clipped].real.mean(axis=1).std())

        bameanstd = n.array(bameanstd)
        basnr = bamean/bameanstd    # = S**3/(Q**3 / n.sqrt(n_tr)) = s**3 * n.sqrt(n_tr)
        if Q:
            print 'Using given Q =', Q
        else:
            Q = ((bameanstd/2.)*n.sqrt(ntr(self.nants)))**(1/3.)
        #        Q = n.median( bastd[clipped]**(1/3.) )              # alternate for Q
            print 'Estimating noise per baseline from data. Q (per DM) =', Q
        self.Q = Q

        # detect
        cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, self.nants))) & (basnr > sigma) )  # get compact sources with high snr

        # plot snrb lc and expected snr vs. sigb relation
        if show or save:
            for dmbin in xrange(len(self.dmarr)):
                cands_dm = cands[0][n.where(cands[1] == dmbin)[0]]  # find candidates for this dmbin
                p.figure(range(len(self.dmarr)).index(dmbin)+1)
                ax = p.axes()
                p.subplot(211)
                p.title(str(self.nskip/self.nbl) + ' nskip, ' + str(dmbin) + ' dmbin, ' + str(len(cands_dm))+' candidates', transform = ax.transAxes)
                p.plot(basnr[:,dmbin], 'b.')
                if len(cands_dm) > 0:
                    p.plot(cands_dm, basnr[cands_dm,dmbin], 'r*')
                    p.ylim(-2*basnr[cands_dm,dmbin].max(),2*basnr[cands_dm,dmbin].max())
                p.xlabel('Integration',fontsize=12,fontweight="bold")
                p.ylabel('SNR_b',fontsize=12,fontweight="bold")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['left'].set_position(('outward', 30))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                p.subplot(212)
                p.plot(bastd[:,dmbin]/Q[dmbin]**3, basnr[:,dmbin], 'b.')

                # plot reference theory lines
                smax = s(basnr[:,dmbin].max(), self.nants)
                sarr = smax*n.arange(0,101)/100.
                p.plot(sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k')
                p.plot(tol*sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k--')
                p.plot(bastd[cands_dm,dmbin]/Q[dmbin]**3, basnr[cands_dm,dmbin], 'r*')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['left'].set_position(('outward', 30))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

                if len(cands_dm) > 0:
                    p.axis([0, tol*sigbQ3(s(basnr[cands_dm,dmbin].max(), self.nants)), -0.5*basnr[cands_dm,dmbin].max(), 1.1*basnr[cands_dm,dmbin].max()])

                    # show spectral modulation next to each point
                    for candint in cands_dm:
                        sm = n.single(round(self.specmod(dmbin,candint),1))
                        p.text(bastd[candint,dmbin]/Q[dmbin]**3, basnr[candint,dmbin], str(sm), horizontalalignment='right', verticalalignment='bottom')
                p.xlabel('sigma_b/Q^3',fontsize=12,fontweight="bold")
                p.ylabel('SNR_b',fontsize=12,fontweight="bold")
                if save:
                    if save == 1:
                        savename = self.file.split('.')[:-1]
                        savename.append(str(self.nskip/self.nbl) + '_' + str(dmbin) + '_bisp.png')
                        savename = string.join(savename,'.')
                    elif type(save) == type('hi'):
                        savename = save
                    print 'Saving file as ', savename
                    p.savefig(self.pathout+savename)

        return basnr[cands], bastd[cands], zip(cands[0],cands[1])

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

    def make_phasedbeam(self):
        """ Integrates data at dmtrack for each pair of elements in dmarr, time.
        Not threaded.  Uses dmthread directly.
        Stores mean of detected signal after dmtrack, effectively forming beam at phase center.
        Ignores zeros in any bl, freq, time.
        """

        self.phasedbeam = n.zeros((len(self.dmarr),len(self.reltime)), dtype='float64')

        for i in xrange(len(self.dmarr)):
            for j in xrange(len(self.reltime)):   
#            for j in xrange(0, len(self.reltime), max(1,self.twidths[i]/2)):   # can also step by twidth/2, but leaves gaps in data products
                dmtrack = self.dmtrack(dm=self.dmarr[i], t0=self.reltime[j])
                if ((dmtrack[1][0] == 0) & (dmtrack[1][len(dmtrack[1])-1] == len(self.chans)-1)):   # use only tracks that span whole band
                    truearr = n.ones( (len(dmtrack[0]), self.nbl, self.npol))
                    falsearr = n.zeros( (len(dmtrack[0]), self.nbl, self.npol))
                    selection = self.data[dmtrack[0], :, dmtrack[1], :]
                    weightarr = n.where(selection != 0j, truearr, falsearr)  # ignore zeros in mean across channels # bit of a hack
                    try:
                        self.phasedbeam[i,j] = n.average(selection, weights=weightarr).real
                    except ZeroDivisionError:
                        self.phasedbeam[i,j] = n.mean(selection).real    # if all zeros, just make mean # bit of a hack
            print 'dedispersed for ', self.dmarr[i]

    def detect_phasedbeam(self, sig=5., show=1, save=0, clipplot=1):
        """ Method to find transients in dedispersed data (in dmt0 space).
        Clips noise then does sigma threshold.
        returns array of candidates transients.
        Optionally plots beamformed lightcurve.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        try:
            arr = self.phasedbeam
        except AttributeError:
            print 'Need to make phasedbeam first.'
            return

        reltime = self.reltime

        # single iteration of sigma clip to find mean and std, skipping zeros
        mean = arr.mean()
        std = arr.std()
        print 'initial mean, std:  ', mean, std
        amin,amax = sigma_clip(arr.flatten())
        clipped = arr[n.where((arr < amax) & (arr > amin) & (arr != 0.))]
        mean = clipped.mean()
        std = clipped.std()
        print 'final mean, sig, std:  ', mean, sig, std

        # Recast arr as significance array
        arr_snr = (arr-mean)/std   # for real valued trial output, gaussian dis'n, zero mean

        # Detect peaks
        peaks = n.where(arr_snr > sig)
        peakmax = n.where(arr_snr == arr_snr.max())
        print 'peaks:  ', peaks

        # Plot
        if show:
            p.clf()
            ax = p.axes()
            ax.set_position([0.2,0.2,0.7,0.7])
            if clipplot:
                im = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)), vmin=amin, vmax=amax)
            else:
                im = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
            cb = p.colorbar(im)
            cb.set_label('Flux Density (Jy)',fontsize=12,fontweight="bold")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 20))
            ax.spines['left'].set_position(('outward', 30))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            if len(peaks[0]) > 0:
                print 'Peak of %f at DM=%f, t0=%f' % (arr.max(), self.dmarr[peakmax[0][0]], reltime[peakmax[1][0]])

                for i in xrange(len(peaks[1])):
                    ax = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                    p.axis((min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                    p.plot([reltime[peaks[1][i]]], [self.dmarr[peaks[0][i]]], 'o', markersize=2*arr_snr[peaks[0][i],peaks[1][i]], markerfacecolor='white', markeredgecolor='blue', alpha=0.5)

            p.xlabel('Time (s)', fontsize=12, fontweight='bold')
            p.ylabel('DM (pc/cm3)', fontsize=12, fontweight='bold')
            if save:
                if save == 1:
                    savename = self.file.split('.')[:-1]
                    savename.append(str(self.scan) + '_' + str(self.nskip/self.nbl) + '_disp.png')
                    savename = string.join(savename,'.')
                elif type(save) == type('hi'):
                    savename = save
                print 'Saving file as ', savename
                p.savefig(self.pathout+savename)

        return peaks,arr[peaks],arr_snr[peaks]


class ProcessByDispersion2():
    """ Class defines methods for pipeline processing for dispersion-based transients searches.
    Has several optimizations (esp for speed), including:
    -- dedisperses using data.roll()
    -- time windowing and bgsubtraction with mexican hat filter convolution
    -- visibility interpolation in frequency
    May want to produce data objects for each dm and filter dt...?
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate class directly. Use \'pipe\' subclasses.')

    def prep(self, deleteraw=False):
        """ Sets up tracks used to speed up dedispersion code.
        Has the option to delete raw data and flags to save memory.
        """

        print
        print 'Filtering rawdata to data as masked array...'
# using 0 as flag
#        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.rawdata[:self.nints,:, self.chans,:] == 0j)
# using standard flags
        self.data = n.ma.masked_array(self.rawdata[:self.nints,:, self.chans,:], self.flags[:self.nints,:, self.chans,:] == 0)
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real   #dataph is summed and detected to form TP beam at phase center, multi-pol
        self.min = self.dataph.min()
        self.max = self.dataph.max()
        print 'Shape of data:'
        print self.data.shape
        print 'Dataph min, max:'
        print self.min, self.max

        if deleteraw:
            del self.rawdata
            del self.flags

        self.freq = self.freq_orig[self.chans]

        # set up ur tracks (lol)
        self.dmtrack0 = {}
        self.twidths = {}
        self.delay = {}
        for dmbin in xrange(len(self.dmarr)):
            self.dmtrack0[dmbin] = self.dmtrack(self.dmarr[dmbin],0)  # track crosses high-freq channel in first integration
            (trackt, trackc) = self.dmtrack0[dmbin]
# old way
#            self.twidths[dmbin] = [len(n.where(trackc == (chan-self.chans[0]))[0]) for chan in self.chans]    # width of track for each unflagged channel
#            self.delay[dmbin] = [n.int(trackt[n.where(trackc == (chan-self.chans[0]))[0][0]]) for chan in self.chans]  # integration delay for each unflagged channel of a given dm.
# new way
            self.twidths[dmbin] = [len(n.where(n.array(trackc) == chan)[0]) for chan in range(len(self.chans))]    # width of track for each unflagged channel
            self.delay[dmbin] = [n.int(trackt[n.where(n.array(trackc) == chan)[0][0]]) for chan in range(len(self.chans))]  # integration delay for each unflagged channel of a given dm.

        print 'Track width in time: '
        for dmbin in self.twidths:
            print 'DM=%.1f, max(twidth)=%d. Iteration could step by %d/2.' % (self.dmarr[dmbin], max(self.twidths[dmbin]), max(self.twidths[dmbin]))

    def dmtrack(self, dm = 0., t0 = 0., show=0):
        """ Takes dispersion measure in pc/cm3 and time offset from first integration in seconds.
        t0 defined at first (unflagged) channel. Need to correct by flight time from there to freq=0 for true time.
        Returns an array of (timebin, channel) to select from the data array.
        """

        reltime = self.reltime
        chans = self.chans
        tint = self.inttime

        # given freq, dm, dfreq, calculate pulse time and duration 
        pulset_firstchan = 4.2e-3 * dm * self.freq[len(self.chans)-1]**(-2)   # used to start dmtrack at highest-freq unflagged channel
        pulset_midchan = 4.2e-3 * dm * self.freq[len(self.chans)/2]**(-2)   # used to start dmtrack at highest-freq unflagged channel. fails to find bright j0628 pulse
        pulset = 4.2e-3 * dm * self.freq**(-2) + t0 - pulset_firstchan  # time in seconds referenced to some frequency (first, mid, last)
        pulsedt = n.sqrt( (8.3e-6 * dm * (1000*self.sdf) * self.freq**(-3))**2 + self.pulsewidth**2)   # dtime in seconds

        timebin = []
        chanbin = []

        for ch in xrange(len(chans)):
            ontime = n.where(((pulset[ch] + pulsedt[ch]) >= reltime - tint/2.) & (pulset[ch] <= reltime + tint/2.))
            timebin = n.concatenate((timebin, ontime[0]))
            chanbin = n.concatenate((chanbin, (ch * n.ones(len(ontime[0]), dtype='int'))))

        track = (list(timebin), list(chanbin))

        if show:
            p.plot(track[0], track[1], 'w*')

        return track

    def time_filter(self, width, kernel='t', bgwindow=4, show=0):
        """ Replaces data array with filtered version via convolution in time. Note that this has trouble with zeroed data.
        kernel specifies the convolution kernel. 'm' for mexican hat (a.k.a. ricker, effectively does bg subtraction), 'g' for gaussian. 't' for a tophat. 'b' is a tophat with bg subtraction (or square 'm'). 'w' is a tophat with width that varies with channel, as kept in 'self.twidth[dmbin]'.
        width is the kernel width with length nchan. should be tuned to expected pulse width in each channel.
        bgwindow is used by 'b' only.
        An alternate design for this method would be to make a new data array after filtering, so this can be repeated for many assumed widths without reading data in anew. That would require more memory, so going with repalcement for now.
        """

        print 'Applying fft time filter. Assumes no missing data in time.'

        if type(width) != types.ListType:
            width = [width] * len(self.chans)

        # time filter by convolution. functions have different normlizations. m has central peak integral=1 and total is 0. others integrate to 1, so they don't do bg subtraction.
        kernelset = {}  # optionally could make set of kernels. one per width needed. (used only by 'w' for now).

        if kernel == 'm':
            from scipy import signal
            print 'Applying mexican hat filter. Note that effective width is somewhat larger than equivalent tophat width.'
            for w in n.unique(width):
                kernel = signal.wavelets.ricker(len(self.data), w)     # mexican hat (ricker) function can have given width and integral=0, so good for smoothing in time and doing bg-subtraction at same time! width of averaging is tied to width of bgsub though...
                kernelset[w] = kernel/n.where(kernel>0, kernel, 0).sum()         # normalize to have peak integral=1, thus outside integral=-1.
        elif kernel == 't':
            import math
            print 'Applying tophat filter.'
            for w in n.unique(width):
                kernel = n.zeros(len(self.data))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(math.ceil(w/2.)))
                kernel[onrange] = 1.
                kernelset[w] = kernel/n.where(kernel>0, kernel, 0).sum()         # normalize to have peak integral=1, thus outside integral=-1.
        elif kernel == 'b':
            import math
            print 'Applying tophat filter with bg subtraction (square mexican hat).'
            for w in n.unique(width):
                kernel = n.zeros(len(self.data))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(math.ceil(w/2.)))
                kernel[onrange] = 1.
                offrange = range(len(kernel)/2 - (bgwindow/2+w)+1, len(kernel)/2-w+1) + range(len(kernel)/2 + w, len(kernel)/2 + (w+bgwindow/2))
                offrange = range(len(kernel)/2 - (bgwindow+w)/2, len(kernel)/2-w/2) + range(len(kernel)/2 + int(math.ceil(w/2.)), len(kernel)/2 + int(math.ceil((w+bgwindow)/2.)))
                kernel[offrange] = -1.
                posnorm = n.where(kernel>0, kernel, 0).sum()           # find normalization of positive
                negnorm = n.abs(n.where(kernel<0, kernel, 0).sum())    # find normalization of negative
                kernelset[w] = n.where(kernel>0, kernel/posnorm, kernel/negnorm)    # pos and neg both sum to 1/-1, so total integral=0
        elif kernel == 'g':
            from scipy import signal
            print 'Applying gaussian filter. Note that effective width is much larger than equivalent tophat width.'
            for w in n.unique(width):
                kernel = signal.gaussian(len(self.data), w)     # gaussian. peak not quite at 1 for widths less than 3, so it is later renormalized.
                kernelset[w] = kernel / (w * n.sqrt(2*n.pi))           # normalize to pdf, not peak of 1.
        elif kernel == 'w':
            import math
            print 'Applying tophat filter that varies with channel.'
            for w in n.unique(width):
                kernel = n.zeros(len(self.data))                    # tophat.
                onrange = range(len(kernel)/2 - w/2, len(kernel)/2 + int(math.ceil(w/2.)))
                kernel[onrange] = 1.
                kernelset[w] = kernel/n.where(kernel>0, kernel, 0).sum()         # normalize to have peak integral=1, thus outside integral=-1.

        if show:
            for kernel in kernelset.values():
                p.plot(kernel,'.')
            p.title('Time filter kernel')
            p.show()

        # take ffts (in time)
        datafft = n.fft.fft(self.data, axis=0)
        kernelsetfft = {}
        for w in n.unique(width):
            kernelsetfft[w] = n.fft.fft(n.roll(kernelset[w], len(self.data)/2))   # seemingly need to shift kernel to have peak centered near first bin if convolving complex array (but not for real array?)

        # filter by product in fourier space
        for i in range(self.nbl):    # **can't find matrix product I need, so iterating over nbl, chans, npol**
            for j in range(len(self.chans)):
                for k in range(self.npol):
                    datafft[:,i,j,k] = datafft[:,i,j,k]*kernelsetfft[width[j]]    # index fft kernel by twidth

        # ifft to restore time series
        self.data = n.ma.masked_array(n.fft.ifft(datafft, axis=0), self.flags[:self.nints,:, self.chans,:] == 0)
        self.dataph = (self.data.mean(axis=3).mean(axis=1)).real

    def dedisperse(self, dmbin):
        """ Creates dedispersed visibilities integrated over frequency.
        Uses ur track for each dm, then shifts by tint. Faster than using n.where to find good integrations for each trial, but assumes int-aligned pulse.
        """

        dddata = self.data.copy()
        twidth = self.twidths[dmbin]
        delay = self.delay[dmbin]

        # dedisperse by rolling time axis for each channel
        for i in xrange(len(self.chans)):
            dddata[:,:,i,:] = n.roll(self.data[:,:,i,:], -delay[i], axis=0)

        return dddata

    def time_mean(self, width):
        """ Tophat mean of width for each channel (as in self.twidths[dmbin])
        """
        import math

        for i in range(len(self.data)):
            for j in range(len(self.chans)):
                self.data[i,:,j,:] = self.data[i - width[j]/2 : i + int(math.ceil(width[j]/2.)), :, j, :].mean(axis=0)

    def tracksub(self, dmbin, tbin):
        """ Simplistic reproduction of tracksub used in older version of this class.
        Does not have time integration.
        """

        dddata = self.dedisperse(dmbin)
        return n.rollaxis(dddata[tbin], 2)

    def spectralInterpolate(self, Y, axis=0, maxturns=1, turnIncrement=0.125, weightNorm=2):
        """ Function to interpolate visibilities across the bandwidth using an FFT.
        From LANL colleagues Scott vd Wiel and Earl Lawrence.
        Y is array of visibilities with one axis of frequency. Dimensions are 1d, 3d, or 4d (self.data-like)
        axis defines spectral axis over which iterpolation is done. assumptions for input data: for 0, array in freq, for 2, is self.data structure.
        maxturns is the number of phase wraps expected across band. This affects sensitivity of interpolation, since it effectively is a search space.
        """
	
	if maxturns==0:
            return(Y.mean(axis=axis))

	# 1:  DFT
	# Number of visibilities
	nr = Y.shape[axis]
        t0 = nr/2
	# Total length for fft
	nt = nr/turnIncrement
	# Half the number of Fourier frequencies considered
	nf = min(nt/2, maxturns/turnIncrement)

	if nf==nt/2:
            freqID = range(int(nt))
	else:
            freqID = range(int(nf+1));
            freqID.extend(range(int(nt-nf),int(nt)))

	# Fourier frequencies and rotation vector
	omega = n.array(freqID)/nt
	rotation = n.exp(2*n.pi*omega*t0*1j)

	# Some stuff that creates F
	F = n.fft.fft(Y, n=int(nt), axis=axis)
	F = F.take(freqID, axis=axis)
	F = n.sqrt(2)*F/nr

	# 2:  Calculate weights	
        # 3:  Rotate FFT per frequency, weight, and sum across frequencies
	W = n.abs(F)
        if axis == 0:                              # assume single visibility array
            W = W/W.max()
            W = W**weightNorm
            W = W/W.sum()
            Yhat = (rotation*(W*F)).sum(axis=axis)
        elif axis == 2:                               # assume structure of self.data
            W = W/W.max(axis=axis)[:, :, n.newaxis]
            W = W**weightNorm
            W = W/W.sum(axis=axis)[:, :, n.newaxis]
            Yhat = ((W*F)*rotation[n.newaxis , n.newaxis, :, n.newaxis]).sum(axis=axis)

	return Yhat

    def make_bispectra(self, stokes='postbisp', maxturns=0):
        """ Makes numpy array of bispectra for each integration.
        stokes defines how polarizations are used (assumes two polarizations): 
          'postbisp' means form bispectra for each pol, then average pols. 
          'prebisp' means average visibility pols, then form bispectra.
          'noavg' means calc for both stokes and store in bispectrum array
          an index (0,1) means take that index from pol axis.
          maxturns determines how to take visibility mean across channels. if > 0, does spectral interpolation (and loses sensitivity!).
        """

        bisp = lambda d: d[:,:,0] * d[:,:,1] * n.conj(d[:,:,2])    # bispectrum for data referenced by triple (data[:,triples])

        # set up triples and arrays for bispectrum considering flagged baselines (only having zeros).
        triples = self.make_triples()
        meanbl = self.data.mean(axis=2).mean(axis=0)   # find bls with no zeros in either pol to ignore in triples
        self.triples = triples[n.all(meanbl[triples][:,0] != 0j, axis=1) & n.all(meanbl[triples][:,1] != 0j, axis=1) & n.all(meanbl[triples][:,2] != 0j, axis=1) == True]   # only take triples if both pols are good. may be smaller than set for an individual pol

        # need to select path based on how polarization is handled. assumes only dual-pol data.
        print 'Bispectrum made for stokes =', stokes
        if ( (stokes == 'postbisp') | (stokes == 'prebisp') | (stokes == 'noavg') ):      # case of combining two stokes
            bispectra = n.zeros((len(self.dmarr), len(self.data), len(self.triples)), dtype='complex')
        elif (type(stokes) == types.IntType):          # case of using single pol
            if stokes >= self.npol:
                raise IndexError, 'Stokes parameter larger than number of pols in data.'
            bispectra = n.zeros((len(self.dmarr), len(self.data), len(self.triples)), dtype='complex')
        elif stokes == 'noavg':
            bispectra = n.zeros((len(self.dmarr), len(self.data), len(self.triples), self.npol), dtype='complex')

        # iterate over dm trials
        for dmbin in xrange(len(self.dmarr)):

            if maxturns == 0:
                dddata = self.dedisperse(dmbin).mean(axis=2)   # average over channels
            elif maxturns > 0:
                dddata = self.spectralInterpolate(self.dedisperse(dmbin), axis=2, maxturns=maxturns)   # interpolate over channels using fft

            if stokes == 'prebisp':
                dddata = dddata.mean(axis=2)
                bispectra[dmbin] = bisp(dddata[:, self.triples])
            elif stokes == 'postbisp':
                bispectra[dmbin] = bisp(dddata[:, self.triples]).mean(axis=2)
            elif stokes == 'noavg':
                bispectra[dmbin] = bisp(dddata[:, self.triples])
            elif type(stokes) == type(0):
                bispectra[dmbin] = bisp(dddata[:, self.triples, stokes])

            print 'dedispersed for ', self.dmarr[dmbin]
        self.bispectra = n.ma.masked_array(bispectra, bispectra == 0j)

    def detect_bispectra(self, sigma=5., tol=1.3, Q=0, show=0, save=0):
        """ Function to detect transient in bispectra
        sigma gives the threshold for SNR_bisp (apparent). 
        tol gives the amount of tolerance in the sigma_b cut for point-like sources (rfi filter).
        Q is noise per baseline and can be input. Otherwise estimated from data.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        try:
            ba = self.bispectra
        except AttributeError:
            print 'Need to make bispectra first.'
            return

#        ntr = lambda num: num*(num-1)*(num-2)/6   # assuming all triples are present
        ntr = lambda num: len(self.triples)  # assume only good triples are present and use array size as input for noise estimate

        # using s=S/Q
        mu = lambda s: 1.  # for bispectra formed from visibilities
        sigbQ3 = lambda s: n.sqrt((1 + 3*mu(s)**2) + 3*(1 + mu(s)**2)*s**2 + 3*s**4)  # from kulkarni 1989, normalized by Q**3, also rogers et al 1995
        s = lambda basnr, nants: (2.*basnr/n.sqrt(ntr(nants)))**(1/3.)  # see rogers et al. 1995 for factor of 2

        # measure SNR_bl==Q from sigma clipped times with normal mean and std of bispectra. put into time,dm order
        bamean = ba.real.mean(axis=2).transpose()
        bastd = ba.real.std(axis=2).transpose()

        bameanstd = []
        for dmind in xrange(len(self.dmarr)):
            (meanmin,meanmax) = sigma_clip(bamean[:, dmind])  # remove rfi to estimate noise-like parts
            (stdmin,stdmax) = sigma_clip(bastd[:, dmind])
            clipped = n.where((bamean[:, dmind] > meanmin) & (bamean[:, dmind] < meanmax) & (bastd[:, dmind] > stdmin) & (bastd[:, dmind] < stdmax) & (bamean[:, dmind] != 0.0))[0]  # remove rfi and zeros
            bameanstd.append(ba[dmind][clipped].real.mean(axis=1).std())

        bameanstd = n.array(bameanstd)
        basnr = bamean/bameanstd    # = S**3/(Q**3 / n.sqrt(n_tr)) = s**3 * n.sqrt(n_tr)
        if Q:
            print 'Using given Q =', Q
        else:
            Q = ((bameanstd/2.)*n.sqrt(ntr(self.nants)))**(1/3.)
        #        Q = n.median( bastd[clipped]**(1/3.) )              # alternate for Q
            print 'Estimating noise per baseline from data. Q (per DM) =', Q
        self.Q = Q

        # detect
        cands = n.where( (bastd/Q**3 < tol*sigbQ3(s(basnr, self.nants))) & (basnr > sigma) )  # get compact sources with high snr

        # plot snrb lc and expected snr vs. sigb relation
        if show or save:
            for dmbin in xrange(len(self.dmarr)):
                cands_dm = cands[0][n.where(cands[1] == dmbin)[0]]  # find candidates for this dmbin
                p.figure(range(len(self.dmarr)).index(dmbin)+1)
                ax = p.axes()
                p.subplot(211)
                p.title(str(self.nskip/self.nbl) + ' nskip, ' + str(dmbin) + ' dmbin, ' + str(len(cands_dm))+' candidates', transform = ax.transAxes)
                p.plot(basnr[:,dmbin], 'b.')
                if len(cands_dm) > 0:
                    p.plot(cands_dm, basnr[cands_dm,dmbin], 'r*')
                    p.ylim(-2*basnr[cands_dm,dmbin].max(),2*basnr[cands_dm,dmbin].max())
                p.xlabel('Integration',fontsize=12,fontweight="bold")
                p.ylabel('SNR_b',fontsize=12,fontweight="bold")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['left'].set_position(('outward', 30))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                p.subplot(212)
                p.plot(bastd[:,dmbin]/Q[dmbin]**3, basnr[:,dmbin], 'b.')

                # plot reference theory lines
                smax = s(basnr[:,dmbin].max(), self.nants)
                sarr = smax*n.arange(0,101)/100.
                p.plot(sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k')
                p.plot(tol*sigbQ3(sarr), 1/2.*sarr**3*n.sqrt(ntr(self.nants)), 'k--')
                p.plot(bastd[cands_dm,dmbin]/Q[dmbin]**3, basnr[cands_dm,dmbin], 'r*')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['left'].set_position(('outward', 30))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

                if len(cands_dm) > 0:
                    p.axis([0, tol*sigbQ3(s(basnr[cands_dm,dmbin].max(), self.nants)), -0.5*basnr[cands_dm,dmbin].max(), 1.1*basnr[cands_dm,dmbin].max()])

                    # show spectral modulation next to each point
                    for candint in cands_dm:
                        sm = n.single(round(self.specmod(dmbin,candint),1))
                        p.text(bastd[candint,dmbin]/Q[dmbin]**3, basnr[candint,dmbin], str(sm), horizontalalignment='right', verticalalignment='bottom')
                p.xlabel('sigma_b/Q^3',fontsize=12,fontweight="bold")
                p.ylabel('SNR_b',fontsize=12,fontweight="bold")
                if save:
                    if save == 1:
                        savename = self.file.split('.')[:-1]
                        savename.append(str(self.nskip/self.nbl) + '_' + str(dmbin) + '_bisp.png')
                        savename = string.join(savename,'.')
                    elif type(save) == type('hi'):
                        savename = save
                    print 'Saving file as ', savename
                    p.savefig(self.pathout+savename)

        return basnr[cands], bastd[cands], zip(cands[0],cands[1])

    def specmod(self, dmbin, tbin, bgwindow=4):
        """Calculate spectral modulation for given dmtrack.
        Narrow RFI has large (>5) modulation, while spectrally broad emission has low modulation.
        See Spitler et al 2012 for details.
        """

#        smarr = n.zeros(len(self.dataph))  # uncomment to do specmod lightcurve
#        for int in range(len(self.dataph)-bgwindow):
        bfspec = self.dedisperse(dmbin)[tbin].mean(axis=0).real
        sm = n.sqrt( ((bfspec**2).mean() - bfspec.mean()**2) / bfspec.mean()**2 )

        return sm

    def make_phasedbeam(self):
        """ Integrates data at dmtrack for each pair of elements in dmarr, time.
        Not threaded.  Uses dmthread directly.
        Stores mean of detected signal after dmtrack, effectively forming beam at phase center.
        Ignores zeros in any bl, freq, time.
        """

        self.phasedbeam = n.zeros((len(self.dmarr),len(self.reltime)), dtype='float64')

        for i in xrange(len(self.dmarr)):
            self.phasedbeam[i] = self.dedisperse(dmbin=i).mean(axis=3).mean(axis=2).mean(axis=1).real               # dedisperse and mean
            print 'dedispersed for ', self.dmarr[i]

    def detect_phasedbeam(self, sig=5., show=1, save=0, clipplot=1):
        """ Method to find transients in dedispersed data (in dmt0 space).
        Clips noise then does sigma threshold.
        returns array of candidates transients.
        Optionally plots beamformed lightcurve.
        save=0 is no saving, save=1 is save with default name, save=<string>.png uses custom name (must include .png). 
        """

        try:
            arr = self.phasedbeam
        except AttributeError:
            print 'Need to make phasedbeam first.'
            return

        reltime = self.reltime

        # single iteration of sigma clip to find mean and std, skipping zeros
        mean = arr.mean()
        std = arr.std()
        print 'initial mean, std:  ', mean, std
        amin,amax = sigma_clip(arr.flatten())
        clipped = arr[n.where((arr < amax) & (arr > amin) & (arr != 0.))]
        mean = clipped.mean()
        std = clipped.std()
        print 'final mean, sig, std:  ', mean, sig, std

        # Recast arr as significance array
        arr_snr = (arr-mean)/std   # for real valued trial output, gaussian dis'n, zero mean

        # Detect peaks
        peaks = n.where(arr_snr > sig)
        peakmax = n.where(arr_snr == arr_snr.max())
        print 'peaks:  ', peaks

        # Plot
        if show:
            p.clf()
            ax = p.axes()
            ax.set_position([0.2,0.2,0.7,0.7])
            if clipplot:
                im = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)), vmin=amin, vmax=amax)
            else:
                im = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
            cb = p.colorbar(im)
            cb.set_label('Flux Density (Jy)',fontsize=12,fontweight="bold")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 20))
            ax.spines['left'].set_position(('outward', 30))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            if len(peaks[0]) > 0:
                print 'Peak of %f at DM=%f, t0=%f' % (arr.max(), self.dmarr[peakmax[0][0]], reltime[peakmax[1][0]])

                for i in xrange(len(peaks[1])):
                    ax = p.imshow(arr, aspect='auto', origin='lower', interpolation='nearest', extent=(min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                    p.axis((min(reltime),max(reltime),min(self.dmarr),max(self.dmarr)))
                    p.plot([reltime[peaks[1][i]]], [self.dmarr[peaks[0][i]]], 'o', markersize=2*arr_snr[peaks[0][i],peaks[1][i]], markerfacecolor='white', markeredgecolor='blue', alpha=0.5)

            p.xlabel('Time (s)', fontsize=12, fontweight='bold')
            p.ylabel('DM (pc/cm3)', fontsize=12, fontweight='bold')
            if save:
                if save == 1:
                    savename = self.file.split('.')[:-1]
                    savename.append(str(self.scan) + '_' + str(self.nskip/self.nbl) + '_disp.png')
                    savename = string.join(savename,'.')
                elif type(save) == type('hi'):
                    savename = save
                print 'Saving file as ', savename
                p.savefig(self.pathout+savename)

        return peaks,arr[peaks],arr_snr[peaks]


class pipe_msint(MSReader, ProcessByIntegration):
    """ Create pipeline object for reading in MS data and doing integration-based analysis without dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    spw is list of spectral windows to read from MS.
    selectpol is list of polarization product names for reading from MS
    scan is zero-based selection of scan for reading from MS. It is based on scan order, not actual scan number.
    datacol is the name of the data column name to read from the MS.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, spw=[-1], selectpol=['RR','LL'], scan=0, datacol='data', **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, spw=spw, selectpol=selectpol, scan=scan, datacol=datacol)
        self.prep()


class pipe_msdisp(MSReader, ProcessByDispersion):
    """ Create pipeline object for reading in MS data and doing dispersion-based analysis
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    spw is list of spectral windows to read from MS.
    selectpol is list of polarization product names for reading from MS
    scan is zero-based selection of scan for reading from MS. It is based on scan order, not actual scan number.
    datacol is the name of the data column name to read from the MS.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, spw=[-1], selectpol=['RR','LL'], scan=0, datacol='data', **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, spw=spw, selectpol=selectpol, scan=scan, datacol=datacol)
        self.prep()


class pipe_msdisp2(MSReader, ProcessByDispersion2):
    """ Create pipeline object for reading in MS data and doing dispersion-based analysis
    This version uses optimized code for dedispersion, which also has some syntax changes.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    spw is list of spectral windows to read from MS.
    selectpol is list of polarization product names for reading from MS
    scan is zero-based selection of scan for reading from MS. It is based on scan order, not actual scan number.
    datacol is the name of the data column name to read from the MS.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, spw=[-1], selectpol=['RR','LL'], scan=0, datacol='data', **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, spw=spw, selectpol=selectpol, scan=scan, datacol=datacol)
        self.prep()


class pipe_mirint(MiriadReader, ProcessByIntegration):
    """ Create pipeline object for reading in Miriad data and doing integration-based analysis without dedispersion.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, nocal=False, nopass=False, **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, nocal=nocal, nopass=nopass)
        self.prep()


class pipe_mirdisp(MiriadReader, ProcessByDispersion):
    """ Create pipeline object for reading in Miriad data and doing dispersion-based analysis.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, nocal=False, nopass=False, **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, nocal=nocal, nopass=nopass)
        self.prep()


class pipe_mirdisp2(MiriadReader, ProcessByDispersion2):
    """ Create pipeline object for reading in Miriad data and doing dispersion-based analysis.
    This version uses optimized code for dedispersion, which also has some syntax changes.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, file, profile='default', nints=1024, nskip=0, nocal=False, nopass=False, **kargs):
        self.set_params(profile=profile, **kargs)
        self.read(file=file, nints=nints, nskip=nskip, nocal=nocal, nopass=nopass)
        self.prep()


class pipe_simdisp2(SimulationReader, ProcessByDispersion2):
    """ Create pipeline object for simulating data and doing dispersion-based analysis.
    This version uses optimized code for dedispersion, which also has some syntax changes.
    nints is the number of integrations to read.
    nskip is the number of integrations to skip before reading.
    nocal,nopass are options for applying calibration while reading Miriad data.
    Can also set some parameters as key=value pairs.
    """

    def __init__(self, profile='default', nints=256, inttime=0.001, chans=n.arange(64), freq=1.4, bw=0.128, array='vla10', **kargs):
        self.set_params(profile=profile, chans=chans, **kargs)
        self.simulate(nints, inttime, chans, freq, bw, array)
        self.prep()


def sigma_clip(arr,sigma=3):
    """ Function takes 1d array of values and returns the sigma-clipped min and max scaled by value "sigma".
    """

    cliparr = range(len(arr))  # initialize
    arr = n.append(arr,[1])    # append superfluous item to trigger loop
    while len(cliparr) != len(arr):
        arr = arr[cliparr]
        mean = arr.mean()
        std = arr.std()
        cliparr = n.where((arr < mean + sigma*std) & (arr > mean - sigma*std) & (arr != 0) )[0]
#        print 'Clipping %d from array of length %d' % (len(arr) - len(cliparr), len(arr))
    return mean - sigma*std, mean + sigma*std
