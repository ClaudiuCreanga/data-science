# generate the list of dates when you didn't have strava installed yet
from datetime import date, datetime, timedelta

def perdelta(start, end):
    curr = start
    weekday = 1
    while curr < end:
        if weekday == 6:
            weekday += 1
            curr += timedelta(days=1)
            continue
        if weekday == 7:
            weekday = 1
            curr += timedelta(days=1)
            continue
        yield curr
        curr += timedelta(days=1)
        weekday += 1

dates = []
for result in perdelta(date(2018, 2, 5), date(2018, 4, 15)):
    dates.append(result)


f = open("strava/Morning_Ride.gpx",'r')
filedata = f.read()

for date in dates:
    newdata = filedata.replace("2018-04-19",str(date))
    n = open("strava/generated/" + str(date) + ".gpx",'w+')
    n.write(newdata)
    n.close()