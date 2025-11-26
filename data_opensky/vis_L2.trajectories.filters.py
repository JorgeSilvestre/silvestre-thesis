import streamlit as st
from paths import *
import datetime
import re
import json
import pandas as pd
import plotly.express as px
import metrics

st.set_page_config(layout="wide", page_title='Dashboard L2 Filters')

dates = NM_PARQUET_FLIGHTS_PATH.glob('*')
dates = [re.search(r'nm.flights.([\d-]{10}).parquet', str(x))[1] for x in dates]

@st.cache_data
def load_data(date):
    data_flight = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    
    data_vectors = pd.read_parquet(NM_TRAJECTORIES_RAW_PATH / f'tray.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    data_vectors['time_position'] = pd.to_datetime(data_vectors.time_position, unit='s')
    data_vectors['last_contact'] = pd.to_datetime(data_vectors.last_contact, unit='s')
    data_vectors['timestamp'] = pd.to_datetime(data_vectors.timestamp, unit='s')
    data_vectors['vector_timestamp'] = pd.to_datetime(data_vectors.timestamp, unit='s')

    data_flight = data_flight[data_flight.ifplId.isin(data_vectors.ifplId.drop_duplicates())]

    data_flight['estimatedOffBlockTime'] = pd.to_datetime(data_flight.estimatedOffBlockTime, unit='s')
    data_flight['actualTakeOffTime'] = pd.to_datetime(data_flight.actualTakeOffTime, unit='s')
    data_flight['actualTimeOfArrival'] = pd.to_datetime(data_flight.actualTimeOfArrival, unit='s')
    data_flight['estimatedTakeOffTime'] = pd.to_datetime(data_flight.estimatedTakeOffTime, unit='s')
    data_flight['estimatedTimeOfArrival'] = pd.to_datetime(data_flight.estimatedTimeOfArrival, unit='s')
    data_flight['calculatedTakeOffTime'] = pd.to_datetime(data_flight.calculatedTakeOffTime, unit='s')
    data_flight['calculatedTimeOfArrival'] = pd.to_datetime(data_flight.calculatedTimeOfArrival, unit='s')
    
    data_metadata = []
    files = NM_TRAYS_METRICS_L2_PATH.glob( f'tray.{date}.*.json')
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            tray_metadata = json.load(f)
            tray_metadata['ifplId'] = re.search(f'tray\.{date}\.(\w+).json', file.name).group(1)
            tray_metadata['flightDate'] = date
        data_metadata.append(tray_metadata)
    for datum in data_metadata:
        del datum['completitude']
        del datum['gaps']
        del datum['segments']
        del datum['thresholds']
    data_metadata = pd.DataFrame.from_records(data_metadata)
    data_metadata['duration'] = data_metadata.duration//60

    data_flight = pd.merge(data_flight, data_metadata, on='ifplId')

    return data_flight, data_vectors

columns_content = st.columns([1,4])
with columns_content[0]:
    columns_filter = st.columns([1,1])
    with columns_filter[0]:
        date = st.date_input(
            label='Date',
            value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            min_value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            max_value=datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        )
    

data_flight, data_vectors = load_data(date)
with columns_content[0]:
    with st.expander('Filters'):
        filter_cols = st.columns([1,1])
        with filter_cols[0]:
            origin = st.selectbox(':material/flight_takeoff: Origin', options=['All']+sorted(data_flight.aerodromeOfDeparture.unique().tolist()))
        with filter_cols[1]:
            destination = st.selectbox(':material/flight_land: Destination', options=['All']+sorted(data_flight.aerodromeOfDestination.unique().tolist()))
        st.divider()
        missing_start = st.checkbox('Missing start', value=False)
        missing_end = st.checkbox('Missing end', value=False)
        is_complete = st.checkbox('Complete', value=False)
        num_vectors = st.slider('Num. vectors', 0, int(data_flight.num_vectors.max())+1, 
                                [0, int(data_flight.num_vectors.max())+1])
        duration = st.slider('Duration (minutes)', 0, int(data_flight.duration.max())+1, 
                            [0, int(data_flight.duration.max())+1])
        distance = st.slider('Total distance', 0, int(data_flight.distance.max())+1, 
                            [0, int(data_flight.distance.max())+1])
        segments = st.slider('Segments', 0, int(data_flight.num_segments.max()), 
                            [0, int(data_flight.num_segments.max())])
        gaps = st.slider('Gaps', 0, int(data_flight.num_gaps.max()), 
                            [0, int(data_flight.num_gaps.max())])
        continuity_ratio = st.slider('Continuity ratio', 0.0, 1.0, [0.0, 1.0])
        discontinuity_ratio = st.slider('Discontinuity ratio', 0.0, 1.0, [0.0, 1.0])
        gap_ratio = st.slider('Gap ratio', 0.0, 1.0, [0.0, 1.0])
        

# Apply filters
if origin != 'All':
    data_flight = data_flight[data_flight.aerodromeOfDeparture==origin]
if destination != 'All':
    data_flight = data_flight[data_flight.aerodromeOfDestination==destination]

data_flight = data_flight[
    (data_flight.num_vectors.between(num_vectors[0],num_vectors[1])) &
    (data_flight.duration.between(duration[0],duration[1])) &
    (data_flight.distance.between(distance[0],distance[1])) &
    (data_flight.num_segments.between(segments[0],segments[1])) &
    (data_flight.num_gaps.between(gaps[0],gaps[1])) &
    (data_flight.continuity_ratio.between(continuity_ratio[0],continuity_ratio[1])) &
    (data_flight.discontinuity_ratio.between(discontinuity_ratio[0],discontinuity_ratio[1])) &
    (data_flight.gap_ratio.between(gap_ratio[0],gap_ratio[1]))
]
if missing_start:
    data_flight = data_flight[data_flight.missing_start]
if missing_end:
    data_flight = data_flight[data_flight.missing_end]
if is_complete:
    data_flight = data_flight[~data_flight.missing_start & ~data_flight.missing_end]

if len(data_flight) == 0:
    with columns_content[1]:
        st.write('### There are no results for the current filters.')
else:
    flight_list = data_flight.ifplId.drop_duplicates().tolist()

    with columns_filter[1]:
        flight_index = st.number_input(label='Flight index', min_value=0, max_value=len(flight_list)-1)
    with columns_content[0]:
        flight_id = st.selectbox(label='Flight id', index=flight_index, options=flight_list)
        st.caption(body=f'{len(flight_list)} flights with the current filters')

    ### Left column
    with columns_content[0]:
        flight_infobox = st.expander(label='**🔎 Información del vuelo**')
        with flight_infobox:
            st.dataframe(data_flight[data_flight.ifplId == flight_id].transpose().astype(str), 
                        use_container_width=True, height=420, 
                        column_config = {'_index' : st.column_config.Column(width="small",)},
        )
        metrics_infobox = st.expander(label='**📈 Métricas de reordenación**')
        with metrics_infobox:
            with open(NM_TRAYS_METRICS_L2_PATH / f'tray.{date}.{flight_id}.json', 'r', encoding='utf8') as file:
                tray_metadata = json.load(file)
            with st.container(height=400, border=False):
                st.json(tray_metadata )

    tray_data = data_vectors[data_vectors.ifplId == flight_id].copy()
    # tray_data['category'] = '#16ba00'
    # tray_data['diffs'] = [0]+(tray_data.timestamp.iloc[1:].values-tray_data.timestamp.iloc[:-1].values).tolist()
    # tray_data.loc[tray_data.diffs>metrics.THRESHOLD_GAP_TIME,'category'] = '#ff1100'
    # tray_data.loc[tray_data.diffs.between(metrics.THRESHOLD_CONTINUITY, metrics.THRESHOLD_GAP_TIME),'category'] = '#e86c00'

    with columns_content[1]:
        columns_graphs = st.columns([1,1])

        with columns_graphs[0]:
            raw_map = px.scatter_mapbox(
                tray_data, lat='latitude', lon='longitude',
                mapbox_style='carto-positron', zoom=3.5,
                height=450,  
                hover_data={'timestamp':True, 
                            'altitude':':.2f',
                            'on_ground':True}, 
            )
            raw_map.update_layout(margin=dict(l=0, r=0, b=0, t=0)) 
            st.plotly_chart(raw_map, use_container_width=True)
        
        with columns_graphs[1]:
            fig = px.scatter_3d(tray_data, x='latitude', y='longitude', z='altitude', height=450,)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) 
            fig.update_traces( marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)
        
        st.write('**Vectores**')
        st.dataframe(tray_data, height=420, )

if __name__ == '__main__':
    pass