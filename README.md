tpipe
=====

Casey Law
claw@astro.berkeley.edu

Introduction:

Python script for analysis of radio interferometry data and building transient detection pipelines.

Uses variety of libraries for importing visibility data and building Python numpy arrays for manipulation. 
Supported data formats include Miriad and Measurement Sets. Uses CASA, miriad-python, or (soon) pyrap.
Focus is on building pipelines for searching for radio transients in visibilities.

Requirements:
-- python (probably version >2.5..?)
-- numpy, matplotlib
-- CASA 4.0 or higher (to read MS)
-- Miriad (to read Miriad format data)
-- aipy (to image data)

The script defines a master class and a set of subclasses for different data formats and types of analysis.
Aside from the two major data formats, the two types of analysis are either:
-- integration based: the transient can be indexed by an integration (but may be longer than just one), or
-- dispersion based: the transient can be indexed by time and dispersion measure.

Broadly, these classes are defined by the integration time of the visibilities, since dispersion is not 
typically detectable on time scales longer than about 1 second.


Usage:

If you have data in MS format, you must currently run this script from within casapy. For example:
$ casapy
... starting casapy ...
casapy> import tpipe
casapy> obs = tpipe.pipe_msint(file='data.ms', nints=100, nskip=0)     # read data
casapy> obs.spec()    # plots Stokes I spectrogram at phase center
casapy> obs.make_bispetra(bgwindow=4)     # makes Stokes I bispectra after subtracting 4-int bg in time
casapy> candidates = obs.detect_bispectra(sigma=5)    # find candidate transients according to mean(b) and sigma(b)

For data in Miriad format, you can run tpipe from any Python environment with access to miriad-python libraries.
In that case, the "pipe_mirint" and "pipe_mirdisp" classes are where to start.

