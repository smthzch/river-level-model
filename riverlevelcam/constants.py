from dateutil import tz

DATAURL = "https://waterservices.usgs.gov/nwis/iv/?sites=0214642825&period=P3D&format=rdb"
IMGURL = "https://www2.usgs.gov/water/southatlantic/nc/rivercam/nc0214642825.jpg"
IMGDIR = "data/imgs/"
DBNAME = "data/data.db"
TZINFOZ = {
    "EST": tz.gettz("America/New_York"),
    "EDT": tz.gettz("America/New_York")
}
UTCTZ = tz.gettz("UTC")
