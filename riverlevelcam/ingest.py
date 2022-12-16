#%%
import json
import requests
import os
import sqlite3
from datetime import datetime
from dateutil import tz
from dateutil.parser import parse
from exif import Image

from riverlevelcam.constants import DATAURL, IMGURL, IMGDIR, DBNAME, TZINFOZ, UTCTZ

### Image Fetch
def save_image_as_date():
    r = requests.get(IMGURL)
    img = Image(r.content)
    imgdt = tz.resolve_imaginary(parse(img.datetime+' EST', tzinfos=TZINFOZ)).astimezone(UTCTZ)
    flnm = datetime.strftime(imgdt, IMGDIR + 'img_%Y%m%d_%H%M%SUTC.jpeg')
    with open(flnm, 'wb') as f:
        f.write(img.get_file())

### STREAMFLOW DATA FETCH
def createDB():
    sqlCreate = '''
    CREATE TABLE IF NOT EXISTS discharge
    (
        datetime real,
        datetext text,
        siteno text,
        stage real,
        discharge real,
        imgfile text
    )
    '''
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()
    c.execute(sqlCreate)
    conn.commit()
    conn.close()

def insertValues(datetime, datetext, siteno, stage, discharge, imgfile):
    conn = sqlite3.connect(DBNAME)
    sqlInsert = 'INSERT INTO discharge VALUES (?,?,?,?,?,?)'
    params = (datetime, datetext, siteno, stage, discharge, imgfile)
    c = conn.cursor()
    c.execute(sqlInsert, params)
    conn.commit()
    conn.close()

def imgsINdb():
    imgSQL = 'SELECT imgfile from discharge'
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()
    c.execute(imgSQL)
    return [im[0] for im in c.fetchall()]

def imgsNOTdb():
    imgs = os.listdir(IMGDIR)
    indb = imgsINdb()
    return [img for img in imgs if img not in indb]

def usgsData():
    #parses usgs return rdb data format
    #skips all # lines and determines columns with datetime, siteno, dis & stage
    r = requests.get(DATAURL)
    strdat = r.content.decode('utf-8')
    #dynamic column search structure
    cols = dict(
        agency = dict(name = 'agency_cd', value = 0),
        siteno = dict(name = 'site_no', value = None),
        datetime = dict(name = 'datetime', value = None, format ='%Y-%m-%d %H:%M'),
        tz = dict(name = 'tz_cd', value = None),
        discharge = dict(name = '239330_00060', value = None),
        #dcode = dict(name = '239330_00060_cd', value = None),
        stage =  dict(name = '90060_00065', value = None),
        #stcode = dict(name = '90060_00065_cd', value = None)
    )
    #data structure
    dat = dict(
        datetime = [],
        datetext = [],
        siteno = [],
        stage = [],
        discharge = []
    )
    #parse into data structure
    indata = False
    discharge_col = None
    stage_col = None
    lines = strdat.split('\n')
    for i in range(len(lines)):
        cells = lines[i].split('\t')
        if cells[0]==cols['agency']['name']:
            for ci, cell in enumerate(cells):
                if cell==cols['siteno']['name']:
                    cols['siteno']['value'] = ci
                elif cell==cols['datetime']['name']:
                    cols['datetime']['value'] = ci
                elif cell==cols['tz']['name']:
                    cols['tz']['value'] = ci
                elif cell==cols['discharge']['name']:
                    cols['discharge']['value'] = ci
                elif cell==cols['stage']['name']:
                    cols['stage']['value'] = ci
        if cells[0]=='USGS':
            dt = cells[cols['datetime']['value']] + cells[cols['tz']['value']]
            dt = parse(dt).astimezone(UTCTZ)
            dtu = dt.timestamp()
            dt = datetime.strftime(dt, '%Y-%m-%d %H:%M:%S UTC')
            dat['datetime'] += [dtu]
            dat['datetext'] += [dt]
            dat['siteno'] += [cells[cols['siteno']['value']]]
            dat['stage'] += [float(cells[cols['stage']['value']])]
            dat['discharge'] += [float(cells[cols['discharge']['value']])]
    return dat

def matchData(imgs, dat):
    augdat = []
    for img in imgs:
        #get date, find nearest data point
        dtu = datetime.strptime(img, 'img_%Y%m%d_%H%M%SUTC.jpeg')
        dtu = dtu.replace(tzinfo=UTCTZ).timestamp()
        closesti = -1
        closest = float('inf')
        for i in range(len(dat['datetime'])):
            absdiff = abs(dtu - dat['datetime'][i])
            if absdiff < closest:
                closesti = i
                closest = absdiff
        augdat += [dict(
            datetime = dat['datetime'][closesti],
            datetext = dat['datetext'][closesti],
            siteno = dat['siteno'][closesti],
            stage = dat['stage'][closesti],
            discharge =dat['discharge'][closesti],
            imgfile = img
        )]
    return augdat

def insertData(augdat):
    conn = sqlite3.connect(DBNAME)
    sqlInsert = 'INSERT INTO discharge VALUES (?,?,?,?,?,?)'
    params = [(
        dat['datetime'],
        dat['datetext'],
        dat['siteno'],
        dat['stage'],
        dat['discharge'],
        dat['imgfile']
    ) for dat in augdat]
    c = conn.cursor()
    c.executemany(sqlInsert, params)
    conn.commit()
    conn.close()
