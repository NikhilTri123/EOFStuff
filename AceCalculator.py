import numpy as np


def getHurdatData(filePath):
    # open, read, and close hurdat file
    f = open(filePath, mode='r')
    lines = f.readlines()
    f.close()

    allData = []
    stormName = 'AL011851'
    stormData = []
    for line in lines:
        # turn line into list
        line = line.replace(' ', '').split(',')[:-1]
        # if new storm, append previous stormName/stormData to allData and clear stormData
        if len(line) == 3:
            allData.append([stormName, stormData])
            stormName = line[0]
            stormData = []
        # add data to stormData
        else:
            stormData.append(line[:7])

    # append last stormName/stormData to allData and remove first (empty) element
    allData.append([stormName, stormData])
    allData = allData[1:]
    return allData[1:]


def getStormAce(hurData, hurMonths):
    # slice all data prior to 1970
    hurNames = [storm[0] for storm in hurData]
    hurData = hurData[hurNames.index('AL011970'):]

    # calculate ACE for all valid data points and add them up for each storm
    stormAce = []
    for storm in hurData:
        currAce = 0
        for stormData in storm[1]:
            if stormData[1] in ['0000', '0600', '1200', '1800'] and stormData[3] in ['TS', 'HU', 'SS'] \
                    and int(stormData[6]) >= 34 and int((stormData[0])[4:6]) in hurMonths:
                currAce += int(stormData[6]) * int(stormData[6]) / 10000
            if float(stormData[4][:-1]) < 13 and float(stormData[5][:-1]) > 87:
                break
        stormAce.append([storm[0], currAce])
    return stormAce


def getMonthAce(hurMonths):
    # get ACE for each storm that occurs in the chosen month(s)
    allHurdatData = getHurdatData('C:/Nikhil Stuff/Coding Stuff/hurdatdata.txt')
    stormAce = getStormAce(allHurdatData, hurMonths)

    # get names and years of all storms from 1970-present
    names = [storm[0] for storm in allHurdatData]
    names = names[names.index('AL011970'):]
    names.append('AL012023')
    stormYears = [storm[-4:] for storm in names]

    allAce = []
    for year in range(1970, 2023):
        # get list of storms for the given year
        seasonAce = stormAce[stormYears.index(str(year)):]
        tempYears = stormYears[stormYears.index(str(year)):]
        num = 0
        while int(tempYears[num]) == year:
            num += 1
        seasonAce = seasonAce[:num]

        # add up ACE for each storm in the given year
        seasonAce = np.sum(np.array(seasonAce)[:, 1].astype(float))
        allAce.append([year, np.round(seasonAce, 2)])

    return allAce
