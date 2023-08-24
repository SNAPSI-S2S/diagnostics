"""
    This python script computes NAM, SAM or NAO indices for a specified set of SNAPSI data.
    Requires positional command line arguments for intake catalog.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import intake
import pathlib
import os


def select_domain(da,index_id):

    area = dict(nao=dict(lat=slice(20,80),lon=slice(-90,40)),
                nam=dict(lat=slice(20,90)),
                sam=dict(lat=slice(-90,-20)))
    
    da['lon'] = xr.where(da['lon']>180,da['lon']-360,da['lon'])
    da = da.sortby('lon')
    da = da.sortby('lat')
    da = da.sel(**area[index_id])
    
    return da


def regrid(da,ref):

    common_plev = da['plev'][da['plev'].isin(ref.plev)]
    da = da.interp(lat=ref.lat,lon=ref.lon)
    da = da.sel(plev=common_plev)
    ref = ref.sel(plev=common_plev)

    return da, ref


def project(sample,eof):
    
    weighted = eof * np.cos(np.radians(sample.lat))
    norm = (eof * weighted).sum(('lat','lon'))
    series = (weighted * sample).sum(('lat','lon'))
    series = series / norm

    return series


def check_sign_convention(index,pattern,index_id):

    if index_id in ['nao','nam']:
        sign = np.sign(pattern.sel(lat=35,method='nearest').mean('lon') - 
                       pattern.sel(lat=70,method='nearest').mean('lon'))
    elif index_id == 'sam':
        sign = np.sign(pattern.sel(lat=-35,method='nearest').mean('lon') - 
                       pattern.sel(lat=-70,method='nearest').mean('lon'))
    index = index * sign
    pattern = pattern * sign

    return index, pattern


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source_id "Model"')
    parser.add_argument('experiment', type=str, help='experiment_id')
    parser.add_argument('subexperiment', type=str, help='sub_experiment_id "Startdate"')
    parser.add_argument('--index',dest='index_id',nargs='?',
                        default='nao',const='nao',choices=['nao', 'nam', 'sam'],
                        help='choose nao, nam, or sam (default: %(default)s)')
    parser.add_argument('--catalog_path',dest='catalog_path',nargs='?',
                        default='/gws/nopw/j04/snapsi/test-snapsi-catalog.json',
                        const='/gws/nopw/j04/snapsi/test-snapsi-catalog.json',
                        help='intake catalog (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print('\n ARGUMENTS')
        print(args)


    # open processed datasets
    # these might require change, e.g., the leap year treatment in the climatology
    input_dir = '/gws/nopw/j04/snapsi/processed/ERA5/NAM_NAO_SAM_indices/'
    
    clim = xr.open_dataset(input_dir+'reanalysis_climatology.nc')['Z']
    clim = select_domain(clim,args.index_id)
    clim = clim / 9.81

    eof = xr.open_dataset(input_dir+'reanalysis_Z_winter_'+args.index_id+'.nc')['eof']
    eof = select_domain(eof,args.index_id)
    eof = eof / 9.81

    
    # open SNAPSI data sample using intake
    catalog = intake.open_esm_datastore(args.catalog_path)
    subset = catalog.search(variable_id = 'zg',
                            source_id = args.source,
                            experiment_id = args.experiment,
                            sub_experiment_id = args.subexperiment)
    
    sample = subset.to_dask(xarray_open_kwargs=dict(chunks=dict(plev=1,time=-1)))['zg']
    
    sample = select_domain(sample,args.index_id)

    
    # regrid processed variables to sample grid & choose shared pressure levels
    clim, sample = regrid(clim, sample)
    eof, _ = regrid(eof, sample)


    # subtract reanalysis climatology from SNAPSI data sample 
    anomalies = sample.groupby('time.dayofyear') - clim

    if args.verbose:
        print('\n ANOMALIES')
        print(anomalies)
        print('\n EOF')
        print(eof)

    # linear projection of sample on pattern
    index = project(anomalies,eof)
    index = index.compute()

    if args.verbose:
        print('\n INDEX')
        print(index)

    # sign convention demands negative anomalies over pole
    index, pattern = check_sign_convention(index,eof,args.index_id)


    # store output as netCDF

    output_dir = '/gws/nopw/j04/snapsi/processed/'+args.source+'/'+args.experiment+'/'+args.subexperiment+'/NAM_NAO_SAM_indices/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = args.source+'_'+args.experiment+'_'+args.subexperiment+'_winter_'+args.index_id+'.nc'
    xr.Dataset(dict(index=index,pattern=pattern)).to_netcdf(output_dir+filename)

    # plot ensemble mean
    ax = plt.axes()
    index.mean('member_id').plot(ax=ax)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_yscale('log')
    ax.set_title('Ensemble mean')

    # store image
    filename = args.source+'_'+args.experiment+'_'+args.subexperiment+'_winter_'+args.index_id+'_ensemble_mean.png'
    plt.savefig(output_dir+filename,bbox_inches='tight')
    plt.close()
    


    