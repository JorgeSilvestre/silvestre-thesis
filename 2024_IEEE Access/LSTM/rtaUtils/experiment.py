import json
import os
from functools import reduce
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam, adamw_experimental

import data_loading
import data_preparation
import paths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

metrics = ['mse','rmse', 'mape', 'mean', 'sample_size'] #  'stdev',
report_columns = ['MSE', 'RMSE', 'MAPE', 'Mean', 'Sample'] # 'StDev',

DEFAULT_TIMES = (15, 30, 60, 90, 120, 150, 0)
DEFAULT_DISTANCES = (25, 45, 60, 100, 125, 250)
MIN_SAMPLE_SIZE = 5

ExperimentResult = NamedTuple('ExperimentResult', [('dataset',str),('feature',str),('time',str)] + [(x,str) for x in metrics])

class Experiment:
    """Base class for RNN experiments

    Implements necessary logic, except data formatting and model initialization:
    model loading, data loading (either from parquet or TF Dataset data), and model
    training and evaluation.

    Attributes:
        lookback: Length of the sliding window
        lookforward: Lenght of the predicted window
        shift: Number of examples to skip between input window and true window
            in the original sequence
        months: The months used to train and evaluate the model, in format 'YYYYMM'
        sampling: Sampling period used to downsample trajectory data
        airport: ICAO code of an airport, or * for all available airports
        batch_size: TF parameter. The amount of examples to be fed before weight update
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }

    """
    def __init__(self,
                 lookback: int,
                 lookforward: int,
                 shift: int,
                 months: str,
                 sampling: int,
                 airport: str,
                 batch_size: int,
                 features: dict):
        # Model parameters
        self.lookback = lookback
        self.lookforward = lookforward
        self.shift = shift
        self.batch_size = batch_size
        # Data parameters
        self.months = months
        self.airport = airport
        self.sampling = sampling
        # Features
        self.features = features
        self.numeric_feat = features.get('numeric', [])
        self.categoric_feat = features.get('categoric', [])
        self.objective_feat = features.get('objective', [])
        self.num_features = len(self.numeric_feat) + len(self.categoric_feat)
        # Auxiliar attributes
        self.model = None
        self.trained_epochs = 0
        self.results = {}
        # Paths
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

       

    def init_model(self):
        """Model initialization

        To be implemented in child classes."""
        raise NotImplementedError

    def _format_data(self):
        """Formatting data for use as input to the model

        To be implemented in child classes."""
        raise NotImplementedError


    ### Model loading ###########################

    def load_model(self, name: str = 'last'):
        """Loads a model checkpoint

        The number of trained epochs is updated accordingly.

        Args:
            name: Name of the model. Can be either 'last', 'best' or a custom file name
        """
        self.model = tf.keras.models.load_model(self.model_path / f'{name}.h5')
        self.encoders = joblib.load(self.model_path / f'encoder_{self.num_features}.joblib')
        self.scaler   = joblib.load(self.model_path / f'scaler_{self.num_features}.joblib')

        try:
            logs = pd.read_csv(self.model_path_log)

            if name == 'last':
                self.trained_epochs = logs.epoch.max() + 1
            elif name == 'best':
                self.trained_epochs = logs.val_loss.argmin() + 1
            else:
                self.trained_epochs = int(name.split('_')[0][2:])
            # print(f'The number of trained epochs has been updated to {self.trained_epochs}.')
        except FileNotFoundError:
            print('No logged data for the selected model.')
            self.trained_epochs = 0

        self._init_callbacks()


    def _init_callbacks(self):
        """Initializes callbacks for model training

        Four callbacks are defined:
        - Model checkpoint: saves a checkpoint after every epoch
        - Model checkpoint best: performs an additional save of the best epoch
          based on validation loss
        - Model checkpoint last: performs an additional save of the latest
          movel version (used to ease the model loading)
        - CSVLogger: logs the training results after each epoch
        """
        modelCheckpoint = ModelCheckpoint(
            self.model_path_save,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )

        modelCheckpointBest = ModelCheckpoint(
            self.model_path_best,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=True
        )

        modelCheckpointLast = ModelCheckpoint(
            self.model_path_last,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )

        csvLogger = CSVLogger(
            self.model_path_log, append=True
        )

        self.callbacks = [modelCheckpoint, modelCheckpointBest, modelCheckpointLast, csvLogger]


    ### Data management #########################

    def _load_data(self, dataset: str, from_parquet: bool, randomize: bool) -> tf.data.Dataset:
        """Helper function to load data from parquet

        Provides a common interface for the different origins of the data. TF Datasets have a faster
        loading and fit in memory independently of the dataset size, but take up a large amount of
        disk space. Parquet data is loaded and formatted on the fly, but they require to fit in memory.

        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            from_parquet: Boolean to indicate whether the data is loaded from TF Datasets or parquet files
            randomize: Boolean to indicate whether the data should be randomized
        """
        if from_parquet:
            data = self._load_data_from_parquet(dataset, randomize)
        else:
            data = self._load_data_from_dataset(dataset, randomize)

        return self._format_data(data)


    def _load_data_from_parquet(self, dataset: str, randomize: bool) -> tf.data.Dataset:
        """Helper function to load data from parquet files

        Loads and merges the parquet files corresponding to months indicated in self.months.
        Optionally it randomizes data (for instance, for training). Randomization is weighted
        according to month's cardinality to ensure an homogeneous distribution.

        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            randomize: Boolean to indicate whether the data should be randomized
        """
        data = data_loading.load_final_data(self.months, dataset, self.airport, self.sampling)
        
        if randomize:
            aps = sorted(data.aerodromeOfDeparture.unique())
            counts = data.aerodromeOfDeparture.value_counts()
            probs = [counts[ap]/len(data) for ap in aps]

            datasets = [data_preparation.get_windows(data[data.aerodromeOfDeparture == ap].copy(),
                                    self.lookback+self.lookforward+self.shift, self.encoders, self.scaler, self.features).shuffle(1000)
                        for ap in aps]

            dataset  = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
        else:
            dataset = data_preparation.get_windows(data.copy(),
                                    self.lookback+self.lookforward+self.shift, self.encoders, self.scaler, self.features)
        return dataset


    def _load_data_from_dataset(self, dataset: str, randomize: bool) -> tf.data.Dataset:
        """Helper function to load data from a TF dataset

        Loads and merges the TF Datasets corresponding to months indicated in self.months.
        Optionally it randomizes data (for instance, for training). Randomization is weighted
        according to month's cardinality to ensure an homogeneous distribution.

        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            randomize: Boolean to indicate whether the data should be randomized
        """
        path = paths.window_data_path / f'data{self.lookback}_s{self.sampling}/{dataset}'
        datasets = [tf.data.Dataset.load(str(ds))
                    for ds in sorted(path.glob(f'{self.months}-{self.airport}'))]

        if randomize:
            freqs = [ds.cardinality().numpy() for ds in datasets]
            probs = [x/sum(freqs) for x in freqs]
            dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
        else:
            dataset  = tf.data.Dataset.from_tensor_slices(datasets)
            dataset  = dataset.interleave(lambda x: x, cycle_length=1,
                                          num_parallel_calls=tf.data.AUTOTUNE)

        return dataset


    def get_y(self, dataset: tf.data.Dataset) -> np.array:
        """Helper function to retrieve the labels of the examples in a TF dataset

        Args:
            dataset: A TF Dataset of labeled windows
        """
        return np.array([i.numpy() for i in dataset.map(lambda x,y: y,
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)])


    ### Training process ########################

    def train(self, epochs: int, from_parquet: bool = False, add_callbacks: list = None):
        """Trains the model up to a given number of epochs

        Loads train and validation datasets and trains the model up to a given number of
        epochs. If the model have already been trained some epochs, the amount of trained
        epochs is taken into account

        Args:
            epochs: Target number of epochs to train the model
            from_parquet: Boolean to indicate whether the data is loaded from parquet files or TF Datasets
            add_callbacks: List of callbacks to be added to the model (optional)
        """
        train_dataset = self._load_data('train', from_parquet, randomize=True)
        val_dataset = self._load_data('val', from_parquet, randomize=False)

        try:
            logs = pd.read_csv(self.model_path_log)
            # Remove records from later checkpoints
            if self.trained_epochs != logs.shape[0]:
                logs[logs.epoch<self.trained_epochs].to_csv(self.model_path_log, index=False)
        except FileNotFoundError:
            pass

        h = self.model.fit(
                x=train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                validation_data=val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                epochs=epochs,
                verbose=1,
                callbacks=self.callbacks + (add_callbacks if add_callbacks else []),
                initial_epoch=self.trained_epochs)
        self.trained_epochs = pd.read_csv(self.model_path_log).epoch.max() + 1

        return h


    ### Prediction ##############################

    def predict_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        dataset = data_preparation.get_windows(data, self.lookback, self.encoders, self.scaler, self.features) # +self.lookforward
        dataset = self._format_data(dataset)
        predictions = self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose=0)
        predictions = predictions.reshape((-1,len(self.objective_feat)))
        unsc_predictions = self.scaler.inverse_transform(
            np.concatenate([np.zeros((predictions.shape[0], len(self.numeric_feat)+len(self.categoric_feat))),
                            predictions], axis=1)
            )[:,-len(self.objective_feat):]
        df = pd.DataFrame(unsc_predictions, columns=self.objective_feat)

        return df

    def predict_trajectory_acumulado(self, dataA: pd.DataFrame, lf: int) -> pd.DataFrame:
        if(self.lookforward>1):
            dataset = data_preparation.get_windows(dataA, self.lookback, self.encoders, self.scaler, self.features) 
            dataset = self._format_data(dataset)
            predictions = self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose=0)
            predictions = predictions.reshape((-1,len(self.objective_feat)))
            unsc_predictions = self.scaler.inverse_transform(
                np.concatenate([np.zeros((predictions.shape[0], len(self.numeric_feat)+len(self.categoric_feat))),
                                predictions], axis=1)
                )[:,-len(self.objective_feat):]
            df = pd.DataFrame(unsc_predictions, columns=self.objective_feat)
        else:
            data = dataA[['fpId','timestamp','latitude','longitude','altitude']]
            df = pd.DataFrame()
            for i in range(0,lf):
                dataset = data_preparation.get_windows(data, self.lookback, self.encoders, self.scaler, self.features)             
                dataset = self._format_data(dataset)
                
                predictions = self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose=0)
                predictions = predictions.reshape((-1,len(self.objective_feat)))
                unsc_predictions = self.scaler.inverse_transform(
                    np.concatenate([np.zeros((predictions.shape[0], len(self.numeric_feat)+len(self.categoric_feat))),
                                    predictions], axis=1)
                    )[:,-len(self.objective_feat):]
                df_A = pd.DataFrame(unsc_predictions, columns=self.objective_feat)           
                data= pd.concat([data.iloc[1:], data.iloc[-1:,:]],axis=0)                
                data.iloc[-1:, [2,3,4]] = df_A[['latitude','longitude','altitude']].values
                df=df.append(df_A)
        return df

    ### Evaluation process ######################

    def evaluate(self, dataset: str = 'test', from_parquet:bool = False, print_err: bool = True, original_scale: bool = True,smooth: bool = False) -> None:
        """Calculates global metrics for validation or test datasets

        Args:
            dataset: Dataset to be used. Can be 'test' or 'val'.
            from_parquet: Boolean to indicate whether the data is loaded from parquet files or TF Datasets
            print_err: Boolean to indicate if results information should be displayed on screen
            original_scale: Boolean to indicate whether the results must be shown in their original scale
                or the scaled-down to (0,1) values
        """
        if dataset not in ('test', 'val', 'train'):
            print('Dataset name is incorrect.')
            return

        data = self._load_data(dataset, from_parquet, randomize=False)
        self.results[f'{dataset} all'] =  [ExperimentResult(dataset=dataset, time='all', feature=k, **v)
                                    for k, v in self._evaluate_on_dataset(data, dataset, print_err, original_scale,smooth).items()]


    def _evaluate_on_dataset(self, dataset: tf.data.Dataset, name: str = None, print_err: bool = False, original_scale: bool = True, smooth: bool = False) -> dict:
        """Helper function to compare predicted and real values for a dataset

        Calculates the defined metrics using real and predicted values, and optionally displays
        the results on screen. Retrieves a dictionary with 'metric name:value' pairs.

        Args:
            dataset: A TF Dataset of labeled windows
            name: Name of the dataset that is being evaluated
            print_err: Boolean to indicate if results information should be displayed on screen
            original_scale: Boolean to indicate whether the results must be shown in their original scale
                or the scaled-down to (0,1) values
        """
        metrics_values = {}

        real_Y = self.get_y(dataset)
        
        if real_Y.shape[0] < MIN_SAMPLE_SIZE:
            return metrics_values
        if original_scale:
            # Flatten array to scale values
            real_Y = real_Y.reshape(-1,len(self.objective_feat))
            scaled = np.concatenate([np.zeros((real_Y.shape[0], self.num_features)),real_Y],axis=1)
            real_Y = self.scaler.inverse_transform(scaled)[:,-len(self.objective_feat):]

        # Restore original shape (num_examples, lookforward, num_objective_features)
        real_Y = real_Y.reshape((-1, self.lookforward, len(self.objective_feat)))
        df = pd.DataFrame()
        # Obtiene la forma del tensor de entrada
        input_shape = dataset.element_spec[0].shape[1]

        # Crea una lista vacía para almacenar los datos
        data_list = [[] for i in range(input_shape)]

        # Itera a través del conjunto de datos y almacena la información en la lista
        for x, _ in dataset:
            for i, value in enumerate(x.numpy()[0]):
                data_list[i].append(value)

        # Crea un dataframe de pandas a partir de la lista
        dataF = pd.DataFrame(data_list).T
        pred_Y = self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose = print_err)
    
        if original_scale:
            # Flatten array to scale values
            pred_Y = pred_Y.reshape(-1,len(self.objective_feat))
                   
            scaled = np.concatenate([np.zeros((pred_Y.shape[0], self.num_features)),pred_Y],axis=1)
            pred_Y = self.scaler.inverse_transform(scaled)[:,-len(self.objective_feat):]

        # Restore original shape (num_examples, lookforward, num_objective_features)
       
        pred_Y = pred_Y.reshape((-1, self.lookforward, len(self.objective_feat)))

        if smooth:
            pred_traspuesto = np.transpose(pred_Y, axes=(0,2,1))
            
            pred_smoothed = np.empty((0,3,10),float)
            for i in range(0,len(pred_traspuesto)):
                polinomio_coefs = np.polyfit(pred_traspuesto[i][1], pred_traspuesto[i][0],2)
                polinomio = np.poly1d(polinomio_coefs)
                               
                pred_smoothed = np.append(pred_smoothed,
                               np.expand_dims(np.concatenate((polinomio(pred_traspuesto[i][1]).reshape(1,-1),
                               pred_traspuesto[i][1].reshape(1,-1),pred_traspuesto[i][2].reshape(1,-1))),axis=0),axis=0)
                
            pred_Y = np.transpose(pred_smoothed, axes=(0,2,1))
              

        for idx, feat in enumerate(self.objective_feat):
            mse_calc = tf.keras.metrics.MeanSquaredError()
            mse_calc.update_state(real_Y[:,:,idx], pred_Y[:,:,idx])
            rmse_calc = tf.keras.metrics.RootMeanSquaredError()
            rmse_calc.update_state(real_Y[:,:,idx], pred_Y[:,:,idx])
            mape_calc = tf.keras.metrics.MeanAbsolutePercentageError()
            mape_calc.update_state(real_Y[:,:,idx], pred_Y[:,:,idx])
            mean_calc = tf.keras.metrics.Mean()
            mean_calc.update_state(real_Y[:,:,idx]-pred_Y[:,:,idx])

            mse   = mse_calc.result().numpy()
            rmse  = rmse_calc.result().numpy()
            mape  = mape_calc.result().numpy()
            mean  = mean_calc.result().numpy()
            sample_size = len(real_Y)

            if print_err:
                print(f'{feat + " ":=<39}')
                print(f'{str.title(name)+" set":18}| MSE:     {mse  :>10.9f}')
                print(f'{                    "":18}| RMSE:    {rmse :>10.3f}')
                print(f'{                    "":18}| Mean:    {mean :>10.3f}')
                print(f'{                    "":18}| MAPE:    {mape :>10.3f}')
                print(f'{                    "":18}| Muestra: {sample_size:>10,}')

            metrics = dict(
                mse=mse,
                rmse=rmse,
                mape=mape,
                mean=mean,
                sample_size=sample_size
            )
            metrics_values[feat] = metrics

        return metrics_values

    
    def evaluate_recursive(data: pd.DataFrame, num_steps_forward = 10, original_scale = True) -> pd.DataFrame:
        y_reals = []
        windows = []

        indices = (data.reset_index(drop=True).reset_index()
                      .groupby(['fpId'])
                      .agg({'index': [min,max]})
                      .reset_index())

        for cat in self.categoric_feat:
            data[cat] = self.encoders[cat].transform(data[cat])
        data = data[self.numeric_feat+self.categoric_feat+self.objective_feat]
        data = self.scaler.transform(data)

        for idx, (fpId, start_index, end_index) in indices.iterrows():
            tmp = data[start_index:end_index+1].copy()
            for i in range(tmp.shape[0]-self.lookback-num_steps_forward+1):
                segment = tmp[i:i+self.lookback+num_steps_forward,:]
                window = segment[:self.lookback, :self.num_features]
                label = segment[-num_steps_forward:, self.num_features:]

                y_reals.append(label)
                windows.append(window)

        windows = np.array(windows).reshape((-1,self.lookback,self.num_features))
        y_reals = np.array(y_reals)

        for i in range(num_steps_forward):
            y_pred = self.model.predict(windows.reshape((-1,self.lookback,self.num_features)), 
                                               batch_size=1024)
            y_pred = y_pred.reshape((-1,self.lookforward,self.num_features))
            windows = np.concatenate([windows, y_pred], axis=1)
            windows = np.delete(windows, obj=0, axis=1)
        y_preds = windows[:,-num_steps_forward:,:]

        if original_scale:
            y_reals = y_reals.reshape(-1,len(self.objective_feat))
            y_reals = np.concatenate([np.zeros((y_reals.shape[0], self.num_features)),y_reals],axis=1)
            y_reals = self.scaler.inverse_transform(y_reals)[:,-len(self.objective_feat):]
            y_reals = y_reals.reshape((-1, num_steps_forward, len(self.objective_feat)))

            y_preds = y_preds.reshape(-1,len(self.objective_feat))
            y_preds = np.concatenate([np.zeros((y_preds.shape[0], self.num_features)),y_preds],axis=1)
            y_preds = self.scaler.inverse_transform(y_preds)[:,-len(self.objective_feat):]
            y_preds = y_preds.reshape((-1, num_steps_forward, len(self.objective_feat)))

        per_feature = []
        for idx, feat in enumerate(self.objective_feat):
            mse_calc = tf.keras.metrics.MeanSquaredError()
            mse_calc.update_state(y_reals[:,:,idx], y_preds[:,:,idx])
            mse   = mse_calc.result().numpy()
            per_feature.append(mse)
        return per_feature


    def get_evaluation_results(self, mode:str = 'wide') -> pd.DataFrame:
        if not self.results:
            print('El modelo no ha sido evaluado aún.')
            return

        res = reduce(lambda x,y: x+y, list(self.results.values()))
        report_df_long = pd.DataFrame.from_records(res, columns=['dataset','feature','time',*report_columns])
        if mode == 'wide':
            report_df = report_df_long.pivot_table(index=['dataset','feature'], columns=['time'], values=report_columns)
            report_df.columns = [' '.join((str(y) for y in x))
                                for x in report_df.columns.to_flat_index()]
            report_df = report_df.reset_index()

            return report_df
        elif mode == 'long':
            return report_df_long


    def evaluate_at_times(self, dataset: str = 'test', times: tuple[int] = DEFAULT_TIMES):
        """Calculates at-time metrics for validation or test datasets

        Data is loaded and formatted on-the-fly from parquet files
        """
        dataframe = data_loading.load_final_data(self.months, dataset, self.airport, self.sampling)

        for idx, time in enumerate(times):
            print(f'{dataset}: {idx+1}/{len(times)} Evaluando a {time} minutos     ', end='\r')
            ds = data_preparation.get_windows_at_time(dataframe, time, self.lookback+self.lookforward+self.shift,
                                                        self.encoders, self.scaler, self.features)
            ds = self._format_data(ds)

            if ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
                self.results[f'{dataset} {time}'] =  [ExperimentResult(dataset=dataset, time=time, feature=k, **v)
                                for k, v in self._evaluate_on_dataset(ds).items()]
        print(f'{dataset}: Finalizado' + ' '*50)


    def evaluate_airports(self):
        """Calculates global and at-time metrics for each airport.

        Data is loaded and formatted on-the-fly from parquet files
        """
        test_data = data_loading.load_final_data(self.months, 'test', self.airport, self.sampling)
        test_airports = sorted(test_data.aerodromeOfDeparture.unique())

        for idx, ap in enumerate(test_airports):
            airport_data = test_data[test_data.aerodromeOfDeparture == ap].copy()

            print(f'({idx+1}/{len(test_airports)}) Evaluando {ap}' + ' '*30, end='\r')
            ap_ds = data_preparation.get_windows(airport_data.copy(), self.lookback+self.lookforward+self.shift,
                                                 self.encoders, self.scaler, self.features)

            ap_ds = self._format_data(ap_ds)
            try:
                self.results[f'{ap} all'] =  [ExperimentResult(dataset=ap, time='all', feature=k, **v)
                                    for k, v in self._evaluate_on_dataset(ap_ds).items()]
            except TypeError:
                pass

            for idx2, time in enumerate(DEFAULT_TIMES):
                print(f'({idx+1}/{len(test_airports)}) Evaluando {ap} a {time} minutos' + ' '*30, end='\r')

                ap_ds = data_preparation.get_windows_at_time(airport_data.copy(), time, self.lookback+self.lookforward+self.shift,
                                                             self.encoders, self.scaler, self.features)

                if ap_ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
                    ap_ds = self._format_data(ap_ds)
                    self.results[f'{ap} {time}'] =  [ExperimentResult(dataset=ap, time=time, feature=k, **v)
                                    for k, v in self._evaluate_on_dataset(ap_ds).items()]
        print(f'({idx+1}/{len(test_airports)})  Done.                        ')


class ExperimentVanilla(Experiment):
    """Experiments that use vanilla LSTM networks

    Implements data formatting and model initialization. Inherits from Experiment class.

    Attributes:
        lookback: Length of the sliding window
        sampling: Sampling period used to downsample trajectory data
        model_config: Sets differents configurations of the model, such as batch size,
            activation function, or number of LSTM units
        batch_size: TF parameter. The amount of examples to be fed before weight update
        months: The months used to train and evaluate the model, in format 'YYYYMM'
        airports: ICAO code of an airport, or * for all available airports
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
        model_type: Descriptive name of the model type. By default, 'LSTM'

    """
    def __init__(self,
                 lookback: int,
                 sampling: int,
                 model_config: dict,
                 months: str,
                 airport: str,
                 features: dict,
                 lookforward: int = 1,
                 shift: int = -1,
                 model_type: str = None):
        self.model_type = model_type if model_type else 'LSTM'
        # Model hyperparameters
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')
        self.loss_function = model_config.get('loss_function', 'mean_absolute_error')
        self.optimizer = model_config.get('optimizer', 'adam')
        # Paths
        self.model_path = paths.models_path / f'{self.model_type}_s{sampling}_lb{lookback}_u{self.n_units}'

        super().__init__(
            lookback,
            lookforward,
            shift,
            months,
            sampling,
            airport,
            model_config.get('batch_size', 128),
            features)

        self.init_model()
        self._write_config()


    def init_model(self, add_metrics = None):
        self.model = Sequential([
            LSTM(self.n_units,
                 activation=self.act_function,
                 input_shape=(self.lookback, self.num_features)),
  
            Dense((self.lookforward)*len(self.objective_feat)),
            tf.keras.layers.Reshape((self.lookforward, len(self.objective_feat)))
        ])
        self.model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer,
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()



    def _write_config(self):
        experiment_config = dict(
            model_type = self.model_type,

            num_units = self.n_units,
            activation_function = self.act_function,
            loss_function = self.loss_function,
            batch_size = self.batch_size,

            lookback = self.lookback,
            lookforward = self.lookforward,
            shift = self.shift,
            months = self.months,
            sampling = self.sampling,
            airport = self.airport,

            features = self.features
        )
        if not self.model_path.exists():
            self.model_path.mkdir()
        with open(self.model_path / 'experiment_config.json', 'w+') as output_file:
            json.dump(experiment_config, output_file)


    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Formatting data for use as input to the model

        Uses window data to construct valed examples for the recurrent model.
        """
        # return dataset.map(lambda x: (x[:,:-1], x[-1:,-1:]))
        return dataset.map(lambda x: (x[:self.lookback, :-len(self.objective_feat)],
                                      x[-self.lookforward:, -len(self.objective_feat):]))


class ExperimentTrajectory(Experiment):
    """Experiments that use vanilla LSTM networks

    Implements data formatting and model initialization. Inherits from Experiment class.

    Attributes:
        lookback: Length of the sliding window
        sampling: Sampling period used to downsample trajectory data
        model_config: Sets differents configurations of the model, such as batch size,
            activation function, or number of LSTM units
        batch_size: TF parameter. The amount of examples to be fed before weight update
        months: The months used to train and evaluate the model, in format 'YYYYMM'
        airports: ICAO code of an airport, or * for all available airports
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
        model_type: Descriptive name of the model type. By default, 'LSTM'

    """
    def __init__(self,
                 lookback: int,
                 sampling: int,
                 model_config: dict,
                 months: str,
                 airport: str,
                 features: dict,
                 lookforward: int = 1,
                 shift: int = 0,
                 model_type: str = None):
        self.model_type = model_type if model_type else 'LSTMtray'
        # Model hyperparameters
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')
        self.loss_function = model_config.get('loss_function', 'mean_absolute_error')
        self.optimizer = model_config.get('optimizer', 'adam')
        # Paths
        self.model_path = paths.models_path / f'{self.model_type}_s{sampling}_lb{lookback}_lf{lookforward}_u{self.n_units}'

        super().__init__(
            lookback,
            lookforward,
            shift,
            months,
            sampling,
            airport,
            model_config.get('batch_size', 128),
            features)

        #self.init_model()
      
        if not self.model_path.exists():
            self._write_config()


    def init_model(self, add_metrics = None):
        self.model = Sequential([
            LSTM(self.n_units,
                 activation=self.act_function,
                 # return_sequences=True,
                 input_shape=(self.lookback, self.num_features)),
            #Dense(self.n_units),
            #Dense(self.n_units),
            Dense((self.lookforward)*len(self.objective_feat)),
            tf.keras.layers.Reshape((self.lookforward, len(self.objective_feat)))
        ])
        self.model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer,
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))
        self.encoders = joblib.load(paths.utils_path / f'encoder_{self.num_features}.joblib')
        self.scaler   = joblib.load(paths.utils_path / f'scaler_{self.num_features}.joblib')

        joblib.dump(self.encoders, self.model_path / f'encoder_{self.num_features}.joblib') 
        joblib.dump(self.scaler,   self.model_path / f'scaler_{self.num_features}.joblib')

        self._init_callbacks()


    def _write_config(self):
        experiment_config = dict(
            model_type = self.model_type,
            num_units = self.n_units,
            activation_function = self.act_function,
            loss_function = self.loss_function,
            batch_size = self.batch_size,

            lookback = self.lookback,
            lookforward = self.lookforward,
            shift = self.shift,
            months = self.months,
            sampling = self.sampling,
            airport = self.airport,

            features = self.features
        )
        if not self.model_path.exists():
            self.model_path.mkdir()
        with open(self.model_path / 'experiment_config.json', 'w+') as output_file:
            json.dump(experiment_config, output_file)


    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Formatting data for use as input to the model

        Uses window data to construct valed examples for the recurrent model.
        """
        return dataset.map(lambda x: (x[:self.lookback, :-len(self.objective_feat)],
                                      tf.reshape(x[-self.lookforward:, -len(self.objective_feat):],
                                                  (self.lookforward, len(self.objective_feat)))))




def main():
    n_units      = 20
    act_function = 'relu'
    batch_size   = 128

    lookback     = 20
    sampling     = 60

    epochs       = 20

    model_config = dict(
        n_units=n_units,
        act_function=act_function,
        batch_size=batch_size,
    )
    months       = '20220[123]'
    airport      = '*'
    glob_text    = f'{months}-{airport}' # 202001-LEBL

    numeric_feat   = ['latitude', 'longitude', 'altitude',
                    'departureDelay', 'vspeed', 'speed',
                    'day_of_week', 'track', 'wind_dir_degrees',
                    'wind_speed_kt', 'visibility_statute_mi',
                    'max_temp', 'min_temp', 'clouds', 'hav_distance']
    categoric_feat = ['time_of_day', 'operator', 'aerodromeOfDeparture', 'sky_status']
    objective      = ['RTA']
    ts_features  = ['latitude', 'longitude', 'altitude', 'vspeed', 'speed', 'track', 'hav_distance']
    nts_features = ['departureDelay', 'day_of_week', 'wind_dir_degrees','wind_speed_kt',
                    'visibility_statute_mi', 'max_temp', 'min_temp', 'time_of_day', 'operator',
                    'aerodromeOfDeparture', 'sky_status', 'clouds']

    feat_dict = {
        'numeric':numeric_feat,
        'categoric':categoric_feat,
        'objective':objective,
        'ts':ts_features,
        'nts':nts_features
    }

    experimento = ExperimentVanilla(
        lookback=lookback,
        sampling=sampling,
        model_config=model_config,
        months=months,
        airport=airport,
        features=feat_dict
    )
    

    experimento.load_model()
   
    experimento._load_data(dataset='train',from_parquet=True,randomize=True)
    
    for k, v in experimento.results.items():
        print(f'{k:10}\t{v}')

if __name__ == '__main__':
    main()