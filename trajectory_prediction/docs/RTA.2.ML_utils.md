# ML_utils notebook

En este cuaderno se realiza la preparación de los datos para su uso en el entrenamiento de modelos. En particular, se filtran las columnas necesarias para la generación de características, se definen y exportan los mecanismos complementarios para la conversión de tipos de datos y su escalado, y la separación estratificada de los datos para asegurar la generación de datasets de entrenamiento, evaluación y validación homogéneos de acuerdo a unos ciertos criterios.

## Feature selection
- Simplemente define y clasifica características utilizadas por el modelo. Se dividen en numéricas y categóricas para el desarrollo y aplicación del encoder y el scaler correspondientes.
- Las características elegidas se utilizarán para proyectar las columnas pertinentes en los pasos posteriores.

## Encoder and scaler generation
- Permite configurar el mes (utilizando un patrón) y un aeropuerto concreto.
- Carga los datos correspondientes a los meses indicados (y el aeropuerto, si se define), proyectando las columnas del dataset de acuerdo a lo establecido en la selección de características.
- Primero, se ajusta el *encoder* con las características calificadas como categóricas.
	- Actualmente se hace con un LabelEncoder (transforma cada valor categórico en un entero), pero en algunos casos sería interesante usar *one-hot encoding* (características booleanas).
- Se transforman las características categóricas a numéricas usando el encoder entrenado.
- Una vez todas las características son numéricas, se ajusta el *scaler*.
	- Actualmente se utiliza un MinMaxScaler, que escala por separado cada una de las columnas, $f$ dentro del rango $[0,1]$ aplicando la fórmula:
	  $$v' = \frac{v - v^f_{min}}{v^f_{max} - v^f_{min}}$$
- Finalmente, ambos se serializan y guardan en disco con la nomenclatura `encoder_X.joblib` y `scaler_X.joblib`, donde X es el número de características.

>[!warning] Orden de las características
>- El orden en que se indican las características es importante para la aplicación del scaler, pues aplica el factor de escala por índice de columna, no por su nombre.
>- Hay que comprobar si se puede hacer (de forma fácil) que lo aplique por nombre de característica, es más seguro.
>- El encoder no se ve afectado por estas cuestiones (se aplica por nombre de característica).


## Stratified data separation
- Con este mecanismo se asegura una composición uniforme en los conjuntos de datos para el holdout.
	- Actualmente, se estratifica únicamente por el aeropuerto de origen (e implícitamente por el mes, ya que se particionan los datos).
	- La separación se hace en base a las trayectorias, no a los ejemplos. Es decir:
		- La proporción es la del número de trayectorias de cada clase de estratificación entre el total de trayectorias (no el número de vectores asociados).
		- Se reparten trayectorias completas, no vectores. Así, si una trayectoria se asigna al conjunto de train, todos sus vectores se utilizan para el entrenamiento.
- Se define una proporción de datos para train y para test (sobre el conjunto global), y una proporción de validation (sobre el conjunto de train).
	- También se podría modificar para asignar un porcentaje a cada parte y santas pascuas, aunque habría que cambiar un poquito el proceso de división de los datos.
- Se cargan los datos limpios (desde `data/clean`) para cada mes indicado.
- Previamente a la división, se excluyen las trayectorias con maniobras de holding con múltiples loops.
- Además, se calculan los índices de los extremos de cada trayectoria, así como su aeropuerto de origen.
- Utilizando estos datos, se separan primero los conjuntos de train y de test, y a partir del conjunto de train se separan los correspondientes al conjunto de validación.
- Los resultados se guardan en disco en el directorio `data/final`.

>[!todo] Refactorizar para flexibilizar los criterios de estratificación más allá del aeropuerto de origen.

