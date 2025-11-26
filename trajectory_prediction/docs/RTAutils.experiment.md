# Experiment class

Clase base para la implementación de experimentos. Incluye toda la lógica necesaria para la adaptación de los datos, el entrenamiento y la evaluación de acuerdo a las métricas definidas. Las clases hijas implementan la lógica directamente dependiente del modelo, en particular su definición e inicialización, y la estructura de datos adecuada para las entradas y salidas del modelo.

![](assets/D.Experiment.excalidraw.png)

## Métodos principales
Implementan el flujo de trabajo básico para el entrenamiento y evaluación de los modelos. Son públicamente accesibles (no comienzan con \_ ).

### Constructor (\_\_init\_\_())
Define los atributos para los objetos de la clase..

Argumentos:
- **Lookback**: tamaño de la ventana utilizado para realizar una predicción.
- **Sampling**: indica el periodo de muestreo de los conjuntos de datos. Requiere que se hayan generado antes mediante [RTAutils.data_preparation.sample_data()](RTAutils.data_preparation). 
- **Lookforward**: número de timesteps a predecir en cada predicción.
- **Batch** size: número de ejemplos por batch (requerido por el modelo).
- **Months**: meses, en formato `YYYYMM` (admite patrones: `20220[123]` proporciona los datos correspondientes a los meses 202201, 202202 y 202203). Un valor `*`  utiliza los datos de todos los meses disponibles.
- **Airport**: permite indicar el identificador ICAO de un aeropuerto en concreto para cargar únicamente los datos de ese aeropuerto (tanto para entrenamiento como para evaluación). Un valor `*`  utiliza los datos de todos los aeropuertos disponibles.
- **Features**: diccionario con las listas de identificadores de las características.
	- Para modelos más complejos, como el encoder-decoder, se pueden definir listas adicionales (en el ejemplo, `ts_feat` denota las características que forman series temporales, y `nts_feat` aquellas que forman parte del contexto). Estas listas adicionales se recogen en el constructor de la clase hija.
	- ❗ Importante: El orden de las características debe ser el mismo que el utilizado para generar el *scaler* en [RTAutils.ML_utils](RTAutils.ML_utils). 
``` python
features = {
	numeric_feat = ['feat1','feat2', ...],
	categoric_feat = [...],
	objective = [...],
	ts_feat = [...],
	nts_feat = [...]
}
```

Define los siguientes atributos:
- Atributos correspondientes a todos los argumentos recibidos.
- Atributos para cada una de las listas de características recibidas en el argumento *features*.
- Atributo *num_features* que indica el número de características.
- Atributo *results* para almacenar los resultados de evaluación del modelo.
- Atributos *model* y *trained_epochs*, para almacenar el modelo y el número de épocas que se ha entrenado.
- Atributos *encoders* y *scaler*, que cargan los objetos correspondientes a partir del número de características indicado.

### train()
Carga los datos de entrenamiento y validación correspondientes a la configuración indicada al instanciar el objeto, y lanza el proceso de entrenamiento del modelo en Keras.

Argumentos:
- **epochs**: El número de épocas hasta el cual se entrenará el modelo, teniendo en cuenta el número de épocas que ya se ha entrenado. Si se indica `epochs=60`, y ya se ha entrenado 20, se entrenará durante 40 épocas más, en lugar de entrenarse 60 épocas más.
	- Esto es así para poder reanudar el entrenamiento en caso de haberlo parado, sin preocuparse de andar modificando este parámetro.
- **from_parquet**: Indica si los datos se cargarán desde los ficheros parquet (construyendo al vuelo las ventanas y los ejemplos) o se cargará de los datasets de TensorFlow almacenados en disco (requiere haberlos generado previamente mediante [RTAutils.data_preparation.generate_save_windows()](RTAutils.data_preparation)).
- **add_callbacks**: Permite agregar nuevos callbacks (como el de Weights and Biases) al modelo, además de los ya existentes por defecto.

### evaluate()
Carga los datos de validación y evaluación correspondientes a la configuración indicada al instanciar el objeto, y genera los resultados de evaluación para estos dos datasets.

Argumentos:
- **from_parquet**: Indica si los datos se cargarán desde los ficheros parquet (construyendo al vuelo las ventanas y los ejemplos) o se cargará de los datasets de TensorFlow almacenados en disco (requiere haberlos generado previamente mediante [RTAutils.data_preparation.generate_save_windows()](RTAutils.data_preparation)).
- **print_err**: Indica si se imprimirán por pantalla los resultados a medida que se generen.

### load_model()
Carga un checkpoint existente para el modelo configurado.

Argumentos:
- **name**: El nombre del fichero del checkpoint (se carga desde el directorio del modelo). Se puede indicar `best` para cargar el mejor modelo, o `last` para cargar el último checkpoint disponible. Por defecto tiene el valor `last`.

### evaluate_at_times() / evaluate_airports()
Métodos de evaluación más específicos del modelo. *evaluate_at_times()* obtiene, para cada trayectoria, la última ventana a determinados horizontes temporales y evalúa el modelo sobre ellas, permitiendo caracterizar el comportamiento del modelo a diferentes distancias temporales del final de la trayectoria. *evaluate_airports()* segmenta los datos del conjunto de evaluación de acuerdo al aeropuerto de origen, y genera resultados de evaluación específicos para cada uno de estos conjuntos.

No reciben parámetros, ya que los puntos de corte se definen como constante en la propia clase *Experiment*, y la evaluación por aeropuertos obtiene los diferentes aeropuertos directamente de los datos.

>[!todo] Reformulación
>El método evaluate_at_times() se puede flexibilizar agregando un parámetro para indicar los tiempos (que podría tener como valor por defecto la constante declarada en el fichero `experiment.py`), y permitiendo que se pueda aplicar tanto por distancia como por tiempo.

## Métodos en las subclases
Métodos específicos de las clases hijas que deben ser implementados para utilizar toda la funcionalidad (en la clase padre se declaran como métodos "abstractos").

### Constructor
Recibe todos los parámetros indicados en el constructor de la clase padre, además de otros propios del modelo implementado o el problema que se esté abordando.

Argumentos:
- Model_type: Descriptor del modelo implementado. Se utiliza para nombrar el directorio del modelo, entre otros fines.
- Model_config: Diccionario con todos los parámetros de configuración del modelo.
```python
model_config = {
	'n_units' : 10,
	'act_function' : 'tanh',
	'batch_size' : 128
} 
```

Define los siguientes atributos:
- Atributo model_type para el argumento correspondiente. 
- Atributos específicos del modelo, extraídos del diccionario `model_config`. El valor de `batch_size` se pasa al constructor de la clase padre, el resto se definen en la clase hija.
- Atributo de ruta del directorio del modelo, dentro del directorio `models`, en el que se almacenan los checkpoints y el log durante el entrenamiento.

### init_model()
Instancia y compila el modelo de acuerdo a lo establecido por Keras. Al compilar, se indican (sin parametrizarlo en la versión actual) la función de pérdida, el optimizador y las métricas adicionales que se utilizan durante el entrenamiento.

Argumentos:
- **Add_metrics**: Permite agregar métricas adicionales para describir el entrenamiento. Puede recibir tanto strings identificativos (`'mean_squared_error'`) como objetos de métricas concretos `tf.keras.metrics.MeanSquaredError()`, de la misma manera que se hace directamente en Keras.

### format_data()
Se utiliza la API dataset de Tensorflow para formatear los datos acorde al modelo definido.

Argumentos:
- **Dataset**: Objeto de tipo tensorflow.data.Dataset que contiene las *ventanas* generadas a partir de los datos de trayectorias originales. Cada elemento de este dataset tendrá unas dimensiones $(lookback+lookforward, num\_features+num\_objective\_features)$.

En general, se aplicará un mapeo que segmenta la ventana de forma adecuada para generar los elementos de entrada y las etiquetas adecuadas para el modelo, devolviendo pares de elementos $(inputs_{lookback \times num\_features}, outputs_{lookforward \times num\_objective\_features})$ para el caso de la predicción de trayectorias. 