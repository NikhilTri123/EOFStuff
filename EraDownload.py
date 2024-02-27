import cdsapi
import xarray as xr

variable = "sst"  # variables to be downloaded
years = range(2009, 2010)  # years to download
months = range(1, 13)  # months to download
path = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/{variable}Era.nc'  # path to save variable data to

# dictionary for conversions
eraDict = {"sst": "sea_surface_temperature",
           "mslp": "mean_sea_level_pressure",
           "hgtmid": ["geopotential", ["450", "500", "550"]],
           "hgtup": ["geopotential", ["150", "175", "200", "225", "250"]],
           "uwndlow": ["u_component_of_wind", ["800", "825", "850", "875", "900"]],
           "uwndup": ["u_component_of_wind", ["150", "175", "200", "225", "250"]],
           "shummid": ["specific_humidity", ["400", "450", "500", "550", "600", "650", "700"]]}

c = cdsapi.Client()
years = list(map(str, list(years)))
months = list(map(str, list(months)))

if variable in ['sst', 'mslp']:
    c.retrieve("reanalysis-era5-single-levels-monthly-means",
               {"product_type": "monthly_averaged_reanalysis",
                "variable": eraDict[variable],
                "year": years,
                "month": months,
                "time": "00:00",
                "format": "netcdf"}
               , path)

else:
    c.retrieve("reanalysis-era5-pressure-levels-monthly-means",
               {"product_type": "monthly_averaged_reanalysis",
                "variable": eraDict[variable][0],
                "pressure_level": eraDict[variable][1],
                "year": years,
                "month": months,
                "time": "00:00",
                "format": "netcdf"}
               , path)

dataset = xr.open_dataset(path)
print(dataset)
