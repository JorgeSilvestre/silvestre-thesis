import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import bson
import requests
import pymongo
from tqdm import tqdm

from params import mapping_flightPlan, mapping_flightData
from paths import *
import utils

from pyhive import hive
import pandas as pd
import sys

from pyhive import hive
import pandas as pd



## PARAMETERS

airport_orig = ['EHAM', 'EDDF', 'LIRF', 'LFPG', 'LGAV', 'EKCH', 'EGLL'] # 'LEMD',
airport_dest = ['LEMD']


def _configure_mongo_client() -> pymongo.MongoClient:
    client = pymongo.MongoClient(
        host='privata',
        port=27017,
        username='readBoeing',
        password='Hp43Jbmc',
        authSource='Boeing',
        maxPoolSize = 20,
        tls=True,
        tlsCAFile='./certs/brtecacert001.crt'
    )
    return client


def _configure_hive_client() -> hive.Connection:
    try:
        conn = hive.Connection(host="filisteo5.brte.boeing.es", port=10000, username="BRTE-jsilvestre" ,database="gold_zone")
        print ("Connected to Hive.")
    except ImportError as e:
        print ("cant connect to Hive: ", e)
        try:
            conn = hive.Connection(host="filisteo6.brte.boeing.es", port=10000, username="BRTE-jsilvestre" ,database="gold_zone")
            print ("Connected to Hive.")
        except ImportError as e:
            print ("Cant connect to Hive: ", e)
            sys.exit(1)
    return conn


def _serialize_datetime(obj) -> str:
    if isinstance(obj, datetime):
            return obj.isoformat()
    if isinstance(obj, bson.objectid.ObjectId):
        return str(obj)
    raise TypeError("Type not serializable")


def extract_NMFPLAN_mongo(date, client, airport_orig: list[str]=None, airport_dest: list[str]=None) -> None:
    sep_date = list(int(x) for x in date.split('-'))
    start = datetime(*sep_date,  0,  0,  0)
    end   = datetime(*sep_date, 23, 23, 59)

    query = {'ps:FlightPlanMessage.flightPlanData.structured.flightPlan.estimatedOffBlockTime' : { '$gte' : start, '$lte' : end }}
    if airport_orig:
        query.update({'ps:FlightPlanMessage.flightPlanData.structured.flightPlan.aerodromeOfDeparture.icaoId' : { '$in' : airport_orig }})
    if airport_dest:
        query.update({'ps:FlightPlanMessage.flightPlanData.structured.flightPlan.aerodromesOfDestination.aerodromeOfDestination.icaoId' : { '$in' : airport_dest }})

    filters = {y:1 for _, y in mapping_flightPlan.items()}
    cursor = client.Boeing.NMFPL.find(query, filters, max_time_ms = 300000)

    temp = []
    for d in tqdm(cursor, desc=f'{datetime.now().strftime("%H:%M:%S")} FPlan {date}'):
        temp.append(d)
        # if len(temp)==100: break

    dir = NM_JSON_FPLAN_PATH / f'flightDate={date}'
    if not dir.exists():
        dir.mkdir(parents=True)
    with open(dir / f'nm.fplan.{date}.json', 'w+', encoding='utf8') as file:
        json.dump(temp, file, default=_serialize_datetime) # , separators=['\n',':']


def extract_NMFDATA_mongo(date, client, num_threads=0) -> None:
    sep_date = list(int(x) for x in date.split('-'))
    start = datetime(*sep_date,  0,  0,  0) - timedelta(days=2) #, tzinfo=pytz.timezone('UTC'))
    end   = datetime(*sep_date, 23, 23, 59) + timedelta(days=7) #, tzinfo=pytz.timezone('UTC'))
    records_per_file = 100000

    with open(NM_JSON_FPLAN_PATH / f'flightDate={date}/nm.fplan.{date}.json', 'r', encoding='utf8') as file:
        data = json.load(file)
    fpIds = set()
    for i in data:
        elem = i['ps:FlightPlanMessage']['flightPlanData']['structured']['flightPlan'].get('ifplId', None)
        if elem:
            fpIds.add(elem)
        else:
            print('patata')
    fpIds = list(fpIds)

    dir = NM_JSON_FDATA_PATH / f'flightDate={date}'
    if not dir.exists():
        dir.mkdir(parents=True)

    if num_threads: # Paralelizado - carga todos los resultados en memoria
        def extract_fpid_fdata(fpId):
            temp = []
            query = {
                # 'ps:FlightDataMessage.flightData.flightId.keys.aerodromeOfDeparture' : {'$in' : airport_list},
                # 'ps:FlightDataMessage.flightData.flightId.keys.aerodromeOfDestination' : {'$in' : airport_list},
                'ps:FlightDataMessage.flightData.flightId.keys.estimatedOffBlockTime' : { '$gte' : start, '$lte' : end },
                'ps:FlightDataMessage.flightData.flightId.id' : fpId}
            filters = {y:1 for x,y in mapping_flightData.items()}
            cursor = client.Boeing.NMFDATA_Snappy.find(query, filters, max_time_ms = 300000)
            for d in cursor:
                temp.append(d)

            return temp

        # print(f'Ejecutando paralelamente con {num_threads} hilos...')
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(extract_fpid_fdata, fpIds),
                                desc=f'{datetime.now().strftime("%H:%M:%S")} FData {date} | Extracción'))

        results = [y for x in results for y in x]
        temp = []
        counter = 0
        for d in tqdm(results, desc=f'{datetime.now().strftime("%H:%M:%S")} FData {date} | Escritura', ncols=125):
            temp.append(d)
            if len(temp)>=records_per_file:
                with open(dir / f'nm.fdata.{date}.{counter:0>3}.json', 'w+', encoding='utf8') as file:
                    json.dump(temp, file, default=_serialize_datetime)
                counter += 1
                temp = []
        with open(dir / f'nm.fdata.{date}.{counter:0>3}.json', 'w+', encoding='utf8') as file:
            json.dump(temp, file, default=_serialize_datetime)
    else: # Secuencial
        query = {
            'ps:FlightDataMessage.flightData.flightId.keys.estimatedOffBlockTime' : { '$gte' : start, '$lte' : end },
            'ps:FlightDataMessage.flightData.flightId.id' : {'$in' : fpIds}}
        filters = {y:1 for _, y in mapping_flightData.items()}
        cursor = client.Boeing.NMFDATA_Snappy.find(query, filters, max_time_ms = 7200000)

        counter = 0
        temp = []
        for d in tqdm(cursor, desc=f'{datetime.now().strftime("%H:%M:%S")} FData {date}', ncols=125):
            temp.append(d)
            if len(temp)>=records_per_file:
                with open(dir / f'nm.fdata.{date}.{counter:0>3}.json', 'w+', encoding='utf8') as file:
                    json.dump(temp, file, default=_serialize_datetime)
                counter += 1
                temp = []
        with open(dir / f'nm.fdata.{date}.{counter:0>3}.json', 'w+', encoding='utf8') as file:
            json.dump(temp, file, default=_serialize_datetime)


def extract_OpenSky_flights(date, airport_orig: str = None, airport_dest: str = None) -> None:
    day_start = datetime.strptime(date, '%Y-%m-%d')
    results = []

    if airport_orig and airport_dest:
        print('No pueden fijarse los dos aeropuertos a la vez.')
    elif airport_orig or airport_dest:
        day_end = day_start + timedelta(hours=24)
        ts_start = int(day_start.timestamp())
        ts_end = int(day_end.timestamp())-1
        if airport_orig:
            query = f'https://opensky-network.org/api/flights/departure?airport={airport_orig}&begin={ts_start}&end={ts_end}'
        elif airport_dest:
            query = f'https://opensky-network.org/api/flights/arrival?airport={airport_dest}&begin={ts_start}&end={ts_end}'
        response = requests.get(query)
        if response.status_code == 200:
            results = response.json()
        else:
            print(f'WARNING: Status code {response.status_code}')
            print(query)
            print(response.content)
    else:
        cur_dt = day_start
        day_end = day_start + timedelta(hours=24)

        while cur_dt < day_end:
            ts_start = int(cur_dt.timestamp())
            cur_dt = cur_dt + timedelta(hours=2)
            ts_end = int(cur_dt.timestamp())-1
            query = f'https://opensky-network.org/api/flights/all?begin={ts_start}&end={ts_end}'
            response = requests.get(query)
            if response.status_code == 200:
                results.extend(response.json())
            else:
                print(f'WARNING: Status code {response.status_code}')
                print(response.content)
                break

    if results:
        dir = OPENSKY_RAW_FLIGHTS_PATH / f'flightDate={date}'
        if not dir.exists():
            dir.mkdir(parents=True)
        with open(dir / f'os.flight.{date}.json', 'w+', encoding='utf8') as file:
            json.dump(results, file)


def extract_OpenSky_vectors_gold(date, only_eu=True) -> None:
    conn = _configure_hive_client()
    cursor = conn.cursor()
    query = f"SELECT * FROM gold_zone.opensky WHERE part_date_utc = {date}"
    if only_eu:
        # query = f"SELECT * FROM gold_zone.opensky WHERE part_date_utc = '{date}' AND longitude > -20 AND longitude < 50 AND latitude > 20 LIMIT 300000"
        query = f"SELECT * FROM gold_zone.opensky WHERE part_date_utc = '{date}'"
        query += " AND longitude > -10 AND longitude < 40 AND latitude > 30 AND latitude < 70"
    cursor.execute(query)
    colnames = [x[0].split('.')[1] for x in cursor.description]
    
    folder = OPENSKY_RAW_VECTORS_PATH / f'flightDate={date}'
    if not folder.exists():
        folder.mkdir(parents=True)

    print()
    counter = 0
    while batch := cursor.fetchmany(2_000_000):
        print(f'{date} Batch: {counter:>2}, Length: {len(batch):>6}', end='\r')
        data = pd.DataFrame(batch, columns=colnames)
        data.to_parquet(folder / f'os.vectors.{date}.{counter:0>3}.parquet', index=False)
        del data
        counter += 1

    # #READ HIVE TABLE AND CREATE PANDAS DATAFRAME
    # df = pd.read_sql("SELECT * FROM gold_zone.ectl_swim_flight_data limit 10", conn)
    # ds.dataset("SELECT * FROM gold_zone.ectl_swim_flight_data limit 10")
    # print(df.head())


if __name__ == '__main__':
    date_start, date_end = '2022-07-11','2022-07-11'

    client = _configure_mongo_client()
    dates = utils.get_dates_between(date_start, date_end)

    dates = [x.strftime('%Y-%m-%d') for x in dates]

    # for date in dates:
    #     extract_NMFPLAN_mongo(date, client, airport_orig, airport_dest)

    # for date in dates:
    #     extract_NMFDATA_mongo(date, client, 6)

    # for date in dates:
    #     extract_OpenSky_flights(date, airport_dest=airport_dest[0])

    for date in dates:
        extract_OpenSky_vectors_gold(date)