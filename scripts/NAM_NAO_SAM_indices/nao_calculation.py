
import numpy as np
import xarray as xr
import scipy.ndimage
import os

from numba import guvectorize
from dask.distributed import Client



def detrend(da):
    '''
        Remove linear trend from time series

        - make use of relation between correlation and linear regression coefficient
          beta = r(x,y) * s(y) / s(x)
        - should not influence time mean
        - not sure whether applying a vectorized universal function like below would be faster
    '''
    x = da['time']
    x = x.astype(np.double)
    x = x - x.mean(dim='time')
    cov = (da * x).mean(dim='time').compute()
    var = (x**2).mean(dim='time')
    trend = cov / var * x
    da = da - trend
    return da



@guvectorize(
    "(float64[:], float64[:], float64[:])",
    "(m), (n) -> (m)",
    forceobj=True
)
def vectorized_convolution(x,kernel,out):
    '''
        Vectorized convolution -> generalized NumPy universal function

        - mode='wrap' means that input is assumed being periodic
        - mode='mirror' means that input is extended by reflectinf about the center of the last pixel
    '''
    out[:] = scipy.ndimage.convolve(x,kernel,mode='wrap')



def lowpass(da,dim,n,valid=False):
    '''
        Convolution with triangular window

        -convolution in time space is multiplication in frequency space
        - no chunking along core dimension dim
    '''

    kernel = np.hstack([np.arange(1,np.ceil(n/2)+1),np.arange(np.floor(n/2),0,-1)])
    kernel /= kernel.sum()
    kernel = xr.DataArray(kernel,dims=('kernel'))

    filtered = xr.apply_ufunc(vectorized_convolution,
                              da,kernel,
                              input_core_dims=[[dim,],['kernel']],
                              output_core_dims=[[dim,],],
                              dask='parallelized',
                              output_dtypes=[da.dtype])

    # input is assumed to be periodic
    # remove beginning and end if unvalid
    if valid:
        valid_slice = slice((n - 1) // 2, -(n - 1) // 2)
        filtered = filtered.isel(time=valid_slice)

    return filtered



@guvectorize(
    "(float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:])",
    "(m,n), (k) -> (m,k), (n,k), (k)",
    forceobj=True
)
def vectorized_svd(X,dummy,U,VS,S2):
    '''
        Vectorized singular value decomposition  -> generalized NumPy universal function

        - X = U @ np.diag(S) @ VH
        - U is standardized
        - m is dimension of time, n is stacked dimension, k = min(m,n)
    '''
    u, s, vh = np.linalg.svd(X,full_matrices=False)
    u_std = np.std(u,axis=0)
    U[:,:] = u/u_std
    VS[:,:] = vh.transpose() * s * u_std
    S2[:] = s**2



def pca(anomalies,coords_to_stack):
    '''
        Principal component analysis using a sigular value decomposition algorithm

        - stack spatial dimensions given by coords_to_stack
        - all dimensions apart from time and allpoints are broadcasted over
        - no chunking along core-dimensions
    '''
    # apply area weighting
    # exclude poles for data on regular grid to avoid zero-devision
    weights = np.sqrt(np.cos(anomalies['lat'] * np.pi/180))
    anomalies = anomalies * weights


    # stack spatial dimensions
    stacked = anomalies.stack(allpoints=coords_to_stack)

    # singular value decomposition
    dummy = min(len(stacked.allpoints),len(stacked.time))
    dummy = xr.DataArray(np.zeros(dummy),dims=('number'))
    pc, eof_stacked, expl = xr.apply_ufunc(vectorized_svd,
                                           stacked,dummy,
                                           input_core_dims=[['time','allpoints'],['number']],
                                           output_core_dims=[['time','number'],
                                                             ['allpoints','number'],
                                                             ['number']],
                                           dask='parallelized',
                                           output_dtypes=3*[stacked.dtype])


    # ratio of explained variance
    expl = expl/expl.sum('number')

    eof = eof_stacked.unstack('allpoints')

    # invert area weighting
    eof = eof / weights

    ds = xr.Dataset({'pc':pc,'eof':eof,'expl':expl})

    return ds    







if __name__ == '__main__':



    # configure computing environment
    client = Client(n_workers=1,threads_per_worker=1,memory_limit='600GB',host=os.environ['HOSTNAME'])
    print(client)

    # choose area according to index
    #area = dict(lat=slice(80,20),lon=slice(-90,40)) # NAO, Hurrel based definition
    #area = dict(lat=slice(90,20)) # NAM
    area = dict(lat=slice(-20,-90)) # SAM


    # set window length for lowpass filter
    n = 30 # in days
    dt = 0.25 # time resolution in days
    n = int(n/dt) + 1


    # define list of reanalysis file names
    directory = '/work/FAC/FGSE/IDYST/ddomeise/default/DATA/ERA5/eth/plev/'
    files = [directory+f for f in os.listdir(directory) if (f.startswith('era5_an_geopot_reg2_6h_198') or
                                                            f.startswith('era5_an_geopot_reg2_6h_199') or
                                                            f.startswith('era5_an_geopot_reg2_6h_200') or
                                                            f.startswith('era5_an_geopot_reg2_6h_201'))]
    files.sort()

    da = xr.open_mfdataset(files,chunks={},combine='nested',concat_dim='time')['var129']

    print('\n REANALYSIS FOR CLIMATOLOGY AND PCA:')
    print(da)

    # compute climatology and store to disk
    #clim = da.groupby('time.dayofyear').mean('time').chunk(dict(dayofyear=-1))
    #clim = lowpass(clim,dim='dayofyear',n=n)

    #print('\n CLIMATOLOGY:')
    #print(clim)

    #clim = clim.compute()
    #xr.Dataset(dict(Z=clim)).to_netcdf('./reanalysis_climatology.nc')


    # select region
    da['lon'] = xr.where(da['lon']>180,da['lon']-360,da['lon'])
    da = da.sortby('lon')
    da = da.sel(**area)

    print('\n SELECTED AREA:')
    print(da)


    # compute anomalies  and apply lowpass filter
    clim = xr.open_dataset('./reanalysis_climatology.nc')['Z']
    clim['lon'] = xr.where(clim['lon']>180,clim['lon']-360,clim['lon'])
    clim = clim.sortby('lon')

    anomalies = da.groupby('time.dayofyear') - clim

    #anomalies = detrend(anomalies) # detrending whilst the summer is still included is problematic
    anomalies = lowpass(anomalies.chunk(dict(time=-1)),dim='time',n=n)
    anomalies = anomalies.where(anomalies['time.month'].isin([12,1,2]),drop=True)

    print('\n ANOMALIES:')
    print(anomalies)

    anomalies = anomalies.persist()


    # perform PCA and store to disk
    # (here, msl is only three dimensional
    # for a field with a fourth dimension ,e.g. level, this routines works just as fine, obtaining of NAO pattern for each level
    nao = pca(anomalies,('lat','lon'))
    nao = nao.isel(number=0)

    print('\n PCA:')
    print(nao)

    nao = nao.compute()
    nao.to_netcdf('./reanalysis_Z_winter_sam.nc')


