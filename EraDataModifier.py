import xarray as xr
import numpy as np

variable = "sst"  # variables to be coarsened
hasLevels = False

# dictionary for conversions
varDict = {"sst": "sst", "mslp": "msl", "hgtmid": "z", "hgtup": "z", "uwndlow": "u", "uwndup": "u", "shummid": "q"}

# paths for original data and coarse data
variableFile = f'D:/Nikhil/variablefiles/{variable}Era.nc'
coarseFile = f'D:/Nikhil/variablefiles/{variable}EraLowRes.nc'

# open variable dataset, coarsen lat/lon, saved coarse data to coarseFile
print(f"Beginning {variable.upper()} file modification:")
dataset = xr.open_dataset(variableFile)
data = dataset[varDict[variable]]

if hasLevels:
    levels = data.level.astype(int)
    levelArrays = []
    for level in levels:
        levelData = data.sel(level=level)
        levelArrays.append(levelData.coarsen(latitude=8, longitude=8, boundary="trim").mean())
        print(int(level.to_numpy()))
    coarseData = np.sum(levelArrays, axis=0) / len(levelArrays)

    coarseData = xr.Dataset(
        data_vars={varDict[variable]: (["time", "latitude", "longitude"], coarseData)},
        coords=dict(
            time=(["time"], levelArrays[0].time.values),
            latitude=(["latitude"], levelArrays[0].latitude.values),
            longitude=(["longitude"], levelArrays[0].longitude.values)
        )
    )
    coarseData = coarseData[varDict[variable]]

else:
    coarseData = data.coarsen(latitude=8, longitude=8, boundary="trim").mean()

# same thing but for 2023 dataset
variableFile2023 = f'D:/Nikhil/variablefiles/{variable}2023Era.nc'
coarseFile2023 = f'D:/Nikhil/variablefiles/{variable}2023EraLowRes.nc'

dataset2023 = xr.open_dataset(variableFile2023)
print(dataset2023)
data2023 = dataset2023[varDict[variable]]

if hasLevels:
    levels = data2023.level.astype(int)
    levelArrays = []
    for level in levels:
        levelData2023 = data2023.sel(level=level)
        levelArrays.append(levelData2023.coarsen(latitude=8, longitude=8, boundary="trim").mean())
        print(int(level.to_numpy()))
    coarseData2023 = np.sum(levelArrays, axis=0) / len(levelArrays)
    print(coarseData2023.shape)

    coarseData2023 = xr.Dataset(
        data_vars={varDict[variable]: (["time", "expver", "latitude", "longitude"], coarseData2023)},
        coords=dict(
            time=(["time"], levelArrays[0].time.values),
            expver=(["expver"], levelArrays[0].expver.values),
            latitude=(["latitude"], levelArrays[0].latitude.values),
            longitude=(["longitude"], levelArrays[0].longitude.values)
        )
    )
    coarseData2023 = coarseData2023[varDict[variable]]

else:
    coarseData2023 = data2023.coarsen(latitude=8, longitude=8, boundary="trim").mean()

# merge coarseData with coarseData2023
mergedFilePath = 'D:/Nikhil/variablefiles/' + variable + 'EraModified.nc'

newData1 = coarseData2023.sel(expver=1)
monthlyMeans1 = np.nan_to_num(newData1.to_numpy())
monthlyMeans1 = np.mean(monthlyMeans1, axis=(1, 2))
newData1 = newData1.isel(time=slice(0, np.argmax(monthlyMeans1 == 0)))
print(newData1)

newData5 = coarseData2023.sel(expver=5)
newData5 = newData5.isel(time=slice(np.argmax(monthlyMeans1 == 0), len(monthlyMeans1)))
print(newData5)

coarseData2023 = xr.concat([newData1, newData5], dim="time")
coarseData2023 = coarseData2023.drop_vars("expver")

# append 2023 dataset to pre-2023 dataset and save to files
mergedDataset = xr.concat([coarseData, coarseData2023], dim="time")
mergedDataset.to_netcdf(mergedFilePath)
print(mergedDataset)
