import xarray as xr

varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/sstEraModified.nc'
varDataset = xr.open_dataset(varPath)
print(varDataset.time.to_numpy())
