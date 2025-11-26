import streamlit as st
from paths import *
import datetime
import re
import json
import pandas as pd
import plotly.express as px
import metrics
import time
from utils import get_dates_between

if __name__ == '__main__':
    pass

st.set_page_config(layout="wide", page_title='Dashboard L3 Metrics')

def load_data(date_start: datetime.date, date_end: datetime.date):
    dates = get_dates_between(
        date_start.strftime('%Y-%m-%d'), 
        date_end.strftime('%Y-%m-%d'))
    data_metrics = []
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        files = NM_TRAYS_METRICS_L3_PATH.glob( f'tray.{date}.*.json')
        for file in files:
            with open(file, 'r', encoding='utf8') as f:
                tray_metadata = json.load(f)
                tray_metadata['ifplId'] = re.search(f'tray\.{date}\.(\w+).json', file.name).group(1)
                tray_metadata['flightDate'] = date
            data_metrics.append(tray_metadata)
    for datum in data_metrics:
        del datum['completitude']
        del datum['gaps']
        del datum['segments']
        del datum['thresholds']
    data_metrics = pd.DataFrame.from_records(data_metrics)
    data_metrics['duration'] = data_metrics.duration//60

    data_flights = []
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        flights = pd.read_parquet(
            NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date}.parquet', 
            engine='pyarrow', dtype_backend='pyarrow')
        data_flights.append(flights)
    data_flights = pd.concat(data_flights)

    data = pd.merge(data_metrics, data_flights, on='ifplId')

    return data

top_columns = st.columns([2,1,1.5,2], gap='large')
with top_columns[1]:
    dates = NM_PARQUET_FLIGHTS_PATH.glob('*')
    dates = [re.search(r'nm.flights.([\d-]{10}).parquet', str(x))[1] for x in dates]
    # with columns_filter[0]:
    dates = st.date_input(
        label='Date',
        value=[datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
               datetime.datetime.strptime(dates[0], '%Y-%m-%d')],
        min_value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
        max_value=datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
    )

if len(dates)>1:
    data_metrics = load_data(*dates)
elif len(dates)==1:
    # data_metrics = load_data(dates[0],dates[0])
    data_metrics=None

if data_metrics is not None:
    with top_columns[2]:
        filter_cols = st.columns([3,1,3])
        with filter_cols[0]:
            origin = st.selectbox(':material/flight_takeoff: Origin', options=['All']+sorted(data_metrics.aerodromeOfDeparture.unique().tolist()))
        with filter_cols[1]:
            st.write(':material/arrow_forward:')
        with filter_cols[2]:
            destination = st.selectbox(':material/flight_land: Destination', options=['All']+sorted(data_metrics.aerodromeOfDestination.unique().tolist()))
    
    if origin != 'All':
        data_metrics = data_metrics[data_metrics.aerodromeOfDeparture==origin]
    if destination != 'All':
        data_metrics = data_metrics[data_metrics.aerodromeOfDestination==destination]

    with st.container(border=1):
        g_num_vectors = px.histogram(
            data_frame=data_metrics,
            x='num_vectors',
            nbins=data_metrics.num_vectors.max()//50+1,
            title='Number of vectors per flight')
        
        g_duration = px.histogram(
            data_frame=data_metrics,
            x='duration',
            nbins=data_metrics.duration.max()//5+1,
            title='Duration (in minutes) per flight')

        g_distance = px.histogram(
            data_frame=data_metrics,
            x='distance',
            nbins=int(data_metrics.distance.max()//1000+1),
            title='Distance (in Km) per flight')
        
        st.write('### Length of the trajectory')
        columns_content_1 = st.columns([1,1,1])
        columns_content_1[0].plotly_chart(g_num_vectors, use_container_width=True)
        columns_content_1[1].plotly_chart(g_duration, use_container_width=True)
        columns_content_1[2].plotly_chart(g_distance, use_container_width=True)

    with st.container(border=1):
        g_distance_to_origin = px.histogram(
            data_frame=data_metrics,
            x='distance_to_origin',
            nbins=int(data_metrics.distance_to_origin.max()//50+1),
            title='Frequency of distances to departure airport (in Km)')

        g_distance_to_destination = px.histogram(
            data_frame=data_metrics,
            x='distance_to_destination', 
            nbins=int(data_metrics.distance_to_destination.max()//50+1),
            title='Frequency of distances to destination airport (in Km)')

        g_missing_start = px.histogram(
            data_frame=data_metrics, barmode='group',
            x=['missing_start', 'missing_end'],
            title='Number of flights with missing start or end')
        
        st.write('### Completitude identification')
        columns_content_2 = st.columns([1,1,1])
        columns_content_2[0].plotly_chart(g_distance_to_origin, use_container_width=True)
        columns_content_2[1].plotly_chart(g_distance_to_destination, use_container_width=True)
        columns_content_2[2].plotly_chart(g_missing_start, use_container_width=True)

    with st.container(border=1):
        g_mean_granularity = px.histogram(
            data_frame=data_metrics,
            x='mean_granularity',
            nbins=int(data_metrics.mean_granularity.max()//1+1),
            title='Mean time difference between vectors for each flight')

        g_std_granularity = px.histogram(
            data_frame=data_metrics,
            x='std_granularity',
            nbins=int(data_metrics.std_granularity.max()//10+1),
            title='StD of the time difference between vectors for each flight')

        st.write('### Granularity')
        columns_content_3 = st.columns([1,1,1])
        columns_content_3[0].plotly_chart(g_mean_granularity, use_container_width=True)
        columns_content_3[1].plotly_chart(g_std_granularity, use_container_width=True)

    with st.container(border=1):
        g_num_segments = px.histogram(
            data_frame=data_metrics,
            x='num_segments',
            nbins=int(data_metrics.num_segments.max()//1+1),
            title='Number of segments per flight')

        g_continuity_ratio = px.histogram(
            data_frame=data_metrics,
            x=['continuity_ratio', 'discontinuity_ratio', 'gap_ratio'],
            range_x=(1,0),
            nbins=50,
            barmode='overlay',
            title='Continuity ratio for each flight')
        
        st.write('### Continuity')
        columns_content_4 = st.columns([1,1,1])
        columns_content_4[0].plotly_chart(g_num_segments, use_container_width=True)
        columns_content_4[1].plotly_chart(g_continuity_ratio, use_container_width=True)
