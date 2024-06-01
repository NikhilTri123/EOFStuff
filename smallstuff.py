import xarray as xr
import numpy as np


def zscoreThing(data, calcMonths):
    data = data[:len(data) % 12 * -1]
    all_zscores = []
    for calcMonth in calcMonths:
        print(calcMonth)
        mask = data['time.month'] == int(calcMonth)
        month_data = data.sel(time=mask)

        allMeans = month_data.mean(dim='time')
        allStds = month_data.std(dim='time')

        for time in month_data.time:
            calcYearData = data.sel(time=time)
            zscoreMap = (calcYearData - allMeans) / allStds
            all_zscores.append(zscoreMap)
    all_zscores = np.array(all_zscores)

    return all_zscores


varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/sstEraModified.nc'
varDataset = xr.open_dataset(varPath).sst
zScores = zscoreThing(varDataset, range(1, 13))
print(zScores)
