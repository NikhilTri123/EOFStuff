import xarray as xr
import numpy as np


def getStormAce(ibMonths, region):
    ibtracsFile = 'D:/Nikhil/ibtracs' + region + '.nc'
    dataset = xr.open_dataset(ibtracsFile)

    storm = dataset.usa_wind
    time = dataset.iso_time
    nature = dataset.nature
    basin = dataset.basin
    season = dataset.season

    stormAce = []
    for i in range(season.to_numpy().tolist().index(1970), len(storm)):
        currStorm = storm[i].values
        currStorm = np.nan_to_num(currStorm)
        currStorm = np.trim_zeros(currStorm, 'b')
        if len(currStorm) == 0:
            continue

        currTime = time[i].values.astype(str)
        currTime = np.resize(currTime, len(currStorm))

        currNature = nature[i].values.astype(str)
        currNature = np.resize(currNature, len(currStorm))

        currBasin = basin[i].values.astype(str)
        currBasin = np.resize(currBasin, len(currStorm))

        currSeason = season[i].to_numpy().astype(str)

        currAce = 0
        for j in range(len(currTime)):
            if currTime[j][11:16] in ['00:00', '06:00', '12:00', '18:00'] and currNature[j] in ['TS', 'SS'] \
                    and currStorm[j] >= 34 and int((currTime[j])[5:7]) in ibMonths and currBasin[j] == region:
                currAce += int(currStorm[j]) * int(currStorm[j]) / 10000
        stormAce.append([int(currSeason.astype(float)), currAce])
    return stormAce


def getMonthAce(ibMonths, region):
    # get ACE for each storm that occurs in the chosen month(s)
    stormAce = getStormAce(ibMonths, region)
    stormYears = list(np.array(stormAce)[:, 0].astype(int))
    stormYears.append(2023)

    if region == 'EP':
        calcYears = range(1970, 2023)
    else:
        calcYears = range(1970, 2022)

    allAce = []
    for year in calcYears:
        # get list of storms for the given year
        seasonAce = stormAce[stormYears.index(year):]
        tempYears = stormYears[stormYears.index(year):]
        num = 0
        while int(tempYears[num]) == year:
            num += 1
        seasonAce = seasonAce[:num]

        # add up ACE for each storm in the given year
        seasonAce = np.sum(np.array(seasonAce)[:, 1].astype(float))
        allAce.append([year, np.round(seasonAce, 2)])

    return allAce
