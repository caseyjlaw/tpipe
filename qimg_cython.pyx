import numpy as n
cimport numpy as n
cimport cython
# can choose between numpy and pyfftw
#from numpy import fft
import pyfftw
import pyfftw.interfaces.NUMPY_fft as fft

CTYPE = n.long
ctypedef n.long_t CTYPE_t
DTYPE = n.complex64
ctypedef n.complex64_t DTYPE_t

cpdef imgall(n.ndarray[n.float32_t, ndim=1] u, n.ndarray[n.float32_t, ndim=1] v, n.ndarray[DTYPE_t, ndim=2] data, unsigned int size, unsigned int res):
    # Images all integratons with single gridding of u,v. Input data has dimensions of time,bl
    # Ignores uv points off the grid

    # initial definitions
    cdef unsigned int ndim = n.round(1.*size/res)
    cdef unsigned int t
    cdef unsigned int i
    cdef int cellu
    cdef int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len(data),ndim,ndim), dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=1] uu = n.round(-v/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=1] vv = n.round(u/res).astype(n.int)

    # add uv data to grid
    for i in xrange(len(u)):
        cellu = uu[i]
        cellv = vv[i]
        if ( (n.abs(cellu) < ndim) & (n.abs(cellv) < ndim) ):
            for t in xrange(len(data)):
                grid[t, cellu, cellv] = data[t, i] + grid[t, cellu, cellv]

    ims = []
    for t in xrange(len(grid)):
        im = fft.ifft2(grid[t]).real.astype(n.float32)
        ims.append(n.roll(n.roll(im, ndim/2, axis=0), ndim/2, axis=1))

    print 'Pixel size %.1f\", Field size %.1f\"' % (3600*n.degrees(2./size), 3600*n.degrees(1./res))
    return ims

cpdef imgonefull(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=3] data, unsigned int size, unsigned int res):
    # Same as imgallfull, but takes one int of data
    # Ignores uv points off the grid

    # initial definitions
    cdef unsigned int ndim = n.round(1.*size/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef int cellu
    cdef int cellv
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros( (ndim,ndim), dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(-v/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(u/res).astype(n.int)

    # add uv data to grid
    for i in xrange(len0):
        for j in xrange(len1):
            cellu = uu[i,j]
            cellv = vv[i,j]
            if ( (n.abs(cellu) < ndim) & (n.abs(cellv) < ndim) ):
                for p in xrange(len2):
                    grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]

    im = n.roll(n.roll(fft.ifft2(grid).real.astype(n.float32), ndim/2, axis=0), ndim/2, axis=1)

    print 'Pixel size %.1f\", Field size %.1f\"' % (3600*n.degrees(2./size), 3600*n.degrees(1./res))
    return im

cpdef beamonefullxy(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=3] data, unsigned int sizex, unsigned int sizey, unsigned int res):
    # Same as imgonefullxy, but returns dirty beam
    # Ignores uv points off the grid
    # flips xy gridding! im on visibility flux scale!
    # on flux scale (counts nonzero data)

    # initial definitions
    cdef unsigned int ndimx = n.round(1.*sizex/res).astype(n.int)
    cdef unsigned int ndimy = n.round(1.*sizey/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef int cellu
    cdef int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros( (ndimx,ndimy), dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndimx/2, n.abs(vv) < ndimy/2)
    uu = n.mod(uu, ndimx)
    vv = n.mod(vv, ndimy)

    # add uv data to grid
    for i in xrange(len0):
        for j in xrange(len1):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for p in xrange(len2):
                    if data[i,j,p] != 0j:
                        grid[cellu, cellv] = 1 + grid[cellu, cellv] 
                        nonzeros = nonzeros + 1

    im = fft.ifft2(grid).real*int(ndimx*ndimy)/float(nonzeros)
    im = recenter(im, (ndimx/2,ndimy/2))

    print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(ndimx*ndimy)/float(nonzeros))
    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./sizex), 3600*n.degrees(2./sizey), 3600*n.degrees(1./res))
    return im

cpdef imgonefullxy(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=3] data, unsigned int sizex, unsigned int sizey, unsigned int res):
    # Same as imgallfullxy, but takes one int of data
    # Ignores uv points off the grid
    # flips xy gridding! im on visibility flux scale!
    # on flux scale (counts nonzero data)

    # initial definitions
    cdef unsigned int ndimx = n.round(1.*sizex/res).astype(n.int)
    cdef unsigned int ndimy = n.round(1.*sizey/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef int cellu
    cdef int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=2] grid = n.zeros( (ndimx,ndimy), dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndimx/2, n.abs(vv) < ndimy/2)
    uu = n.mod(uu, ndimx)
    vv = n.mod(vv, ndimy)

    # add uv data to grid
    for i in xrange(len0):
        for j in xrange(len1):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for p in xrange(len2):
                    grid[cellu, cellv] = data[i,j,p] + grid[cellu, cellv]
                    if data[i,j,p] != 0j:
                        nonzeros = nonzeros + 1

    im = fft.ifft2(grid).real*int(ndimx*ndimy)/float(nonzeros)
    im = recenter(im, (ndimx/2,ndimy/2))

    print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(ndimx*ndimy)/float(nonzeros))
    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./sizex), 3600*n.degrees(2./sizey), 3600*n.degrees(1./res))
    return im

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfull(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int size, unsigned int res):
    # Same as imgall, but takes uv for each bl,chan and full 4d data.
    # Defines uvgrid filter before loop
    # Now does not roll images to have center pixel. Must be done outside.

    # initial definitions
    cdef unsigned int ndim = n.round(1.*size/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndim,ndim), dtype='complex64')
#    cdef grid = pyfftw.n_byte_align_empty( (len0,ndim,ndim), 16, dtype='complex64')

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(-v/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(u/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndim/2, n.abs(vv) < ndim/2)
    uu = n.mod(uu, ndim)
    vv = n.mod(vv, ndim)

    # add uv data to grid
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    pyfftw.interfaces.cache.enable()
    for t in xrange(len0):
        grid[t] = fft.ifft2(grid[t])
# too slow
#    fft_obj = pyfftw.builders.fft2(grid, overwrite_input=True)
# also too slow
#    fft_obj = pyfftw.FFTW(grid, grid, direction='FFTW_BACKWARD', axes=(1,2), flags=['FFTW_PATIENT','FFTW_DESTROY_INPUT'])
#    grid = fft_obj()

#    print 'Pixel size %.1f\", Field size %.1f\"' % (3600*n.degrees(2./size), 3600*n.degrees(1./res))
#    return grid.real
    return grid.real * ndim * ndim / ok.sum()      # im on visibility flux scale

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilter(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int size, unsigned int res, float thresh):
    # Same as imgallfull, but returns only candidates and rolls images
    # Defines uvgrid filter before loop

    # initial definitions
    cdef unsigned int ndim = n.round(1.*size/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndim,ndim), dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndim/2, n.abs(vv) < ndim/2)
    uu = n.mod(uu, ndim)
    vv = n.mod(vv, ndim)

    # add uv data to grid
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    pyfftw.interfaces.cache.enable()
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        im = fft.ifft2(grid[t]).real
        snr = im.max()/im.std()
        if ((snr > thresh) & n.any(data[t,:,len2/2:,:])):
            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im*int(ndim*ndim)/float(ok.sum()), (ndim/2,ndim/2)))             # sets im to visibility flux scale

    print 'Detected %d candidates with at least half the band.' % len(candints)

#    print 'Pixel size %.1f\", Field size %.1f\"' % (3600*n.degrees(2./size), 3600*n.degrees(1./res))
    return candims,candsnrs,candints

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilterxy(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int sizex, unsigned int sizey, unsigned int res, float thresh):
    # Same as imgallfull, but returns only candidates and rolls images
    # Defines uvgrid filter before loop
    # flips xy gridding!

    # initial definitions
    cdef unsigned int ndimx = n.round(1.*sizex/res).astype(n.int)
    cdef unsigned int ndimy = n.round(1.*sizey/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndimx,ndimy), dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndimx/2, n.abs(vv) < ndimy/2)
    uu = n.mod(uu, ndimx)
    vv = n.mod(vv, ndimy)

    # add uv data to grid
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    pyfftw.interfaces.cache.enable()
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        im = fft.ifft2(grid[t]).real
        snr = im.max()/im.std()
        if ((snr > thresh) & n.any(data[t,:,len2/3:,:])):
            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im, (ndimx/2,ndimy/2)))

    print 'Detected %d candidates with at least third the band.' % len(candints)
#    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./sizex), 3600*n.degrees(2./sizey), 3600*n.degrees(1./res))
    return candims,candsnrs,candints

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilterxy2(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int sizex, unsigned int sizey, unsigned int res, float thresh):
    # Same as imgallfull, but returns only candidates and rolls images
    # Defines uvgrid filter before loop
    # flips xy gridding!
    # counts nonzero data and properly normalizes fft to be on flux scale

    # initial definitions
    cdef unsigned int ndimx = n.round(1.*sizex/res).astype(n.int)
    cdef unsigned int ndimy = n.round(1.*sizey/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef unsigned int nonzeros = 0
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndimx,ndimy), dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndimx/2, n.abs(vv) < ndimy/2)
    uu = n.mod(uu, ndimx)
    vv = n.mod(vv, ndimy)

    # calculate number of nonzero vis to normalize fft
    for i in xrange(len1):
        for j in xrange(len2):
            for p in xrange(len3):
                if data[len0/2,i,j,p] != 0j:
                    nonzeros = nonzeros + 1

    # add uv data to grid
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    pyfftw.interfaces.cache.enable()
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        im = fft.ifft2(grid[t]).real*int(ndimx*ndimy)/float(nonzeros)
        snr = im.max()/im.std()
        if ((snr > thresh) & n.any(data[t,:,len2/3:,:])):
            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im, (ndimx/2,ndimy/2)))    # gives image on vis sum normalization

    print 'Detected %d candidates with at least third the band.' % len(candints)
    print 'Gridded %.3f of data. Scaling fft by = %.1f' % (float(ok.sum())/ok.size, int(ndimx*ndimy)/float(nonzeros))
#    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./sizex), 3600*n.degrees(2./sizey), 3600*n.degrees(1./res))
    return candims,candsnrs,candints

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef imgallfullfilterminxy(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int sizex, unsigned int sizey, unsigned int res, float thresh):
    # Same as imgallfull, but returns only candidates and rolls images
    # Defines uvgrid filter before loop
    # flips xy gridding!

    # initial definitions
    cdef unsigned int ndimx = n.round(1.*sizex/res).astype(n.int)
    cdef unsigned int ndimy = n.round(1.*sizey/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef unsigned int cellu
    cdef unsigned int cellv
    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndimx,ndimy), dtype='complex64')
    cdef float snr

    # put uv data on grid
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(v/res).astype(n.int)

    ok = n.logical_and(n.abs(uu) < ndimx/2, n.abs(vv) < ndimy/2)
    uu = n.mod(uu, ndimx)
    vv = n.mod(vv, ndimy)

    # add uv data to grid
    for i in xrange(len1):
        for j in xrange(len2):
            if ok[i,j]:
                cellu = uu[i,j]
                cellv = vv[i,j]
                for t in xrange(len0):
                    for p in xrange(len3):
                        grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]

    # make images and filter based on threshold
    pyfftw.interfaces.cache.enable()
    candints = []; candims = []; candsnrs = []
    for t in xrange(len0):
        im = fft.ifft2(grid[t]).real
        snr = im.min()/im.std()
        if ((snr < thresh) & n.any(data[t,:,len2/3:,:])):
            candints.append(t)
            candsnrs.append(snr)
            candims.append(recenter(im, (ndimx/2,ndimy/2)))

    print 'Detected %d candidates with at least third the band.' % len(candints)
#    print 'Pixel sizes (%.1f\", %.1f\"), Field size %.1f\"' % (3600*n.degrees(2./sizex), 3600*n.degrees(2./sizey), 3600*n.degrees(1./res))
    return candims,candsnrs,candints

cpdef imgallfullw(n.ndarray[n.float32_t, ndim=2] u, n.ndarray[n.float32_t, ndim=2] v, n.ndarray[DTYPE_t, ndim=4] data, unsigned int size, unsigned int res, n.ndarray[n.long_t, ndim=1] order, uvkers):

    # initial definitions
    cdef unsigned int ndim = n.round(1.*size/res).astype(n.int)
    shape = n.shape(data)
    cdef unsigned int len0 = shape[0]
    cdef unsigned int len1 = shape[1]
    cdef unsigned int len2 = shape[2]
    cdef unsigned int len3 = shape[3]
    cdef unsigned int t
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int p
    cdef int cellu
    cdef int cellv
    cdef int keru
    cdef int kerv
    cdef unsigned int nvis
    cdef unsigned int blr0
    cdef unsigned int blr1

    cdef n.ndarray[DTYPE_t, ndim=3] grid = n.zeros((len0,ndim,ndim), dtype='complex64')
#    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(-v/res).astype(n.int)
#    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(u/res).astype(n.int)
    cdef n.ndarray[CTYPE_t, ndim=2] uu = n.round(-v.take(order, axis=0)/res).astype(n.int)   # small perf hit here
    cdef n.ndarray[CTYPE_t, ndim=2] vv = n.round(u.take(order, axis=0)/res).astype(n.int)
    data = data.take(order, axis=1)
    cdef int ksize
    cdef n.ndarray[DTYPE_t, ndim=2] uvker
    cdef n.ndarray[CTYPE_t, ndim=1] kerind = n.zeros( (len1), dtype='int')

    for i in xrange(len1):
        for ind in range(len(uvkers)):
            blr0 = uvkers[ind][0][0]
            blr1 = uvkers[ind][0][1]
            if ( (i >= blr0) & (i < blr1) ):
                kerind[i] = ind
                break

    # put uv data on grid
    for i in xrange(len1):
        uvker = uvkers[kerind[i]][1]
        ksize = len(uvker)
        for j in xrange(len2):
            cellu = uu[i,j]
            cellv = vv[i,j]
            if ( (n.abs(cellu) < (ndim-ksize)/2) & (n.abs(cellv) < (ndim-ksize)/2) ):
                for p in xrange(len3):
                    if data[len0/2,i,j,p] != 0j:
                        nvis = nvis + 1
#                print grid[:, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1].shape, uvker[None,:,:].shape, data[:,i,j].mean(axis=1)[:,None,None].shape, grid[:, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1].shape, cellu, cellv
#                grid[:, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1] = uvker[None,:,:]*data[:,i,j].mean(axis=1)[:,None,None] + grid[:, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1]
                for t in xrange(len0):
                    for p in xrange(len3):
                        for keru in xrange(ksize):
                            for kerv in xrange(ksize):
                                grid[t, cellu+keru-ksize/2, cellv+kerv-ksize/2] = uvker[keru,kerv]*data[t,i,j,p] + grid[t, cellu+keru-ksize/2, cellv+kerv-ksize/2]
#                            grid[t, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1] = uvker*data[t,i,j,p,None,None] + grid[t, cellu-ksize/2:cellu+ksize/2+1, cellv-ksize/2:cellv+ksize/2+1]
#                            grid[t, cellu, cellv] = data[t,i,j,p] + grid[t, cellu, cellv]    # no conv gridding

    pyfftw.interfaces.cache.enable()
    for t in xrange(len0):
        grid[t] = fft.ifft2(grid[t])

#    print 'Pixel sizes %.1f\", Field size %.1f\"' % (3600*n.degrees(2./size), 3600*n.degrees(1./res))
    return grid.real * ndim * ndim / nvis

cpdef genuvkernels(unsigned int size, unsigned int res, w, wres, float thresh=0.01):
    cdef unsigned int ndim = size/res
    cdef unsigned int wind0 = 0
    cdef unsigned int wind1 = 0
    cdef unsigned int ksize
    cdef n.ndarray[DTYPE_t, ndim=2] uvker

    order = n.argsort(w)
    w = w.take(order, axis=0)

    # set up w planes
    blrs = []
    sqrt_w = n.sqrt(n.abs(w)) * n.sign(w)
    while 1:
        wind1 = sqrt_w.searchsorted(sqrt_w[wind0]+wres)
        blrs.append((wind0, wind1))
        wind0 = wind1
        if wind1 >= len(sqrt_w): break

    # Grab a chunk of uvw's that grid w to same point.
    uvkers = []
    for blr in blrs:
        avg_w = n.average(w[blr[0]:blr[1]])
        print 'Added %d/%d baselines for avg_w %.1f' % (len(range(blr[0],blr[1])), len(w), avg_w)

        # get image extent
        l, m = get_lm(size, res)
        lmker = genlmkernel(l, m, avg_w)

        # uv kernel from inv fft of lm kernel
        uvker = recenter(fft.ifft2(lmker), (ndim/2,ndim/2)).astype('complex64')

        # keep uvker above a fraction (thresh) of peak amp
        largey, largex = n.where(n.abs(uvker) > thresh*n.abs(uvker).max())
        ksize = max(largey.max()-largey.min(), largex.max()-largex.min())                # take range of high values to define kernel size
        uvker = uvker[ndim/2-ksize/2:ndim/2+ksize/2+1, ndim/2-ksize/2:ndim/2+ksize/2+1]
        uvkers.append((uvker/uvker.sum()).astype('complex64'))

    return order, zip(blrs, uvkers)

cpdef genlmkernel(l, m, w):
    sqrtn = n.sqrt(1 - l**2 - m**2).astype(n.complex64)
    G = n.exp(-2*n.pi*1j*w*(sqrtn - 1))
    G = G.filled(0)
    # Unscramble difference between fft(fft(G)) and G
    G[1:] = n.flipud(G[1:]).copy()
    G[:,1:] = n.fliplr(G[:,1:]).copy()
    return G / G.size

cpdef recenter(a, c):
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    if n.ma.isMA(a):
        a1 = n.ma.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = n.ma.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    else:
        a1 = n.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = n.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a2

cpdef get_lm(size, res, center=(0,0)):
    ndim = size/res
    m,l = n.indices((ndim,ndim))
    l,m = n.where(l > ndim/2, ndim-l, -l), n.where(m > ndim/2, m-ndim, m)
    l,m = l.astype(n.float32)/ndim/res, m.astype(n.float32)/ndim/res
    mask = n.where(l**2 + m**2 >= 1, 1, 0)
    l,m = n.ma.array(l, mask=mask), n.ma.array(m, mask=mask)
    return recenter(l, center), recenter(m, center)

