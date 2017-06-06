tpipe
=====

Casey Law
claw@astro.berkeley.edu

Introduction
----

Python library for analysis of radio interferometry data for finding dispersed (fast) transients.

Uses CASA to import visibility data from Measurement Sets and build Python numpy arrays for analysis. 
Applies CASA calibration tables on the fly. Multi-threaded mode does reading/processing in parallel.
Searches multiple DMs and timescales using multiple cores via the multiprocessing library.
Searches can be done with accelerated FFT imaging or bispectrum algorithms.

Requirements
----

-- python (probably version >2.6..?)
-- numpy, matplotlib
-- CASA 4.0 or higher

Optional (requires some hacking):
-- patchelf (to get casapy-free CASA)
-- pyFFTW (for accelerated fft)

Citation
----
If you use tpipe, please support open software by citing the record on the [Astrophysics Source Code Library](ascl.net) at http://ascl.net/1603.012. In AASTeX, you can do this like so:
```
\software{..., tpipe \citep{2016ascl.soft03012L}, ...}
```

Usage
----

For baseline mode, start a casapy session:
```
$ casapy
CASA> import leanpipedt
CASA> d = leanpipedt.pipe_thread(filename='data.ms', scan=0, nskip=0, iterint=100, nints=100, spw=[0], chans=range(5,60), selectpol=['RR','LL'], searchtype='imgall', filtershape='z', secondaryfilter='fullim', dmarr=[0,100], dtarr=[1,2], size=512*58, res=58, nthreads=8, gainfile='cal.g1', bpfile='cal.b1')
```

This command will read the first 100 interations from scan 0 (first scan) of 'data.ms'. It will look for the first spectral window and save channels 5,59 (inclusive) for two, orthogonal, circular polarizations. The data will have calibration applied from the gain and bp files. A zero-mean will be subtracted, data dedispersed, and resampled for DM=0 and 100 pc/cm3 and time widths of 1 and 2 integrations. The image search will use a uv grid cell size of 58 lambda and an image size of 512 pixels square (appropriate 2 pixels per beam for L-band, VLA images; covers twice the FWHM).

Files
----

-- leanpipedt.py: master script that defines search pipeline.
-- applycals2.py and applytelcal.py: script to parse CASA calibration tables (gain and bp) or telcalfile. Called by leanpipedt.py.
-- leanpipedt_cython.pyx: Cython-accelerated utility functions, including dedispersion of visibilities.
-- qimg_cython.pyx: Cython-accelerated imaging functions.
-- setup.py: script to compile Cython into shared-object libraries.
-- tpipe.py: deprecated version of search script (class-based structure, includes Miriad format data support).

Build Instructions
----

1) Install CASA

2) (optional) Build casapy-free CASA (not possible on OSX yet)
This step builds python modules to import CASA into any Python session (no "casapy" session needed).
This requires fixing some links to libraries with patchelf.
Full instructions at http://newton.cx/~peter/2014/02/casa-in-python-without-casapy. 
See also the casapatch.sh script, which must be edited and run to build new CASA python modules.
Once complete, you should be able to simply type "import casac" and get access to CASA's ms and table tools.
Finally, download "casautil.py" at https://github.com/pkgw/pwpy/blob/master/intflib/casautil.py.
To use this, you will need to uncomment references to "casautil" and the "ms" and "tb" definitions in leanpipedt.py and applycals2.py.

3) Install Cython
See http://cython.org to build Cython, an optimizing static compiler for Python.

(Optional 4) Install pyFFTW
Get pyFFTW at https://github.com/hgomersall/pyFFTW. Note that there is an unsupported library with similar name at https://pypi.python.org/pypi/PyFFTW3/0.2.1.
This supposedly requires Cython 0.15, FFTW 3.3, Python 2.7, Numpy 1.6. Although I've made it work with Python 2.6.
FFTW is an optimized FFT library and can boost performance by a factor of 2. Can be tricky to do this stage, so skipping is ok for many use cases.
To use this, you will need to change commented lines in leanpipedt.py and qimg_cython.pyx to redefine "fft" and add references to pyfftw.

5) Build accelerated functions
Edit "setup.py" file to define how Cython will compile (basically setting filename/function to be compiled).
Then for both "leanpipedt_cython.pyx" and "qimg_cython.pyx", type:
```
> python setup.py build_ext --inplace
```

This will produce "leanpipedt_cython.so" and "qimg_cython.so", which will get imported by leanpipedt.py.
