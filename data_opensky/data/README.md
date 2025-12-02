_Note: This folder is empty since data distribution is not allowed by our license. Below we describe the expected structure of the directory and the organization of the data._

# Storage description

This data folder contain several subfolders defining different levels of maturity of data:
- **Level 0 (L0)** corresponds with raw data, that is, data extracted directly from the data sources and placed in our storage. The raw data is stored to ensure traceability to the source itself, and protect us from any changes that can occur in the source repositories.
- **Level 1 (L1)** corresponds with cleaned data, where the data from the individual data sources have been processed and some of the problems that can be found in them have been addressed.
- **Level 2 (L2)** corresponds with integrated data, that is, the constructed 4D trajectories through the integration of the cleaned data.
- **Level 3 (L3)** contains the processed trajectories. The definition as 4D trajectories arises new problems in the data that are addressed prior to this level.

### L0 (Raw)

Contains the following data folders:
- airports: Single JSON file.
- nmFData: Single JSON files partitioned by day in subfolders.
- nmFPlan: Single JSON files partitioned by day in subfolders.
- openskyVectors: Multiple parquet files partitioned by day in subfolders.
- taf: Single parquet file partitioned by month in subfolders.

### L1 (Clean)

Contains the following data folders:
- airports: Single parquet file.
- nmFData: Single parquet files partitioned by day.
- nmFlights: Single parquet files partitioned by day.
- nmFPlan: Single parquet files partitioned by day.
- openskyVectors: Multiple parquet files partitioned by day in subfolders.
- taf: Single parquet file partitioned by month.

### L2 (Raw trajectories)

Contains the following data folders:
- nmTrajectories: 
    - Single parquet file partitioned by day.
    - Additional subfolders with a JSON file for each trajectory with its metadata.

### L3 (Clean trajectories)
Contains the following data folders:
- nmTrajectories: 
    - Single parquet file partitioned by day.
    - Additional subfolders with a JSON file for each trajectory with its metadata.