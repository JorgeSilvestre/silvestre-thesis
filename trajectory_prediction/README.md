# RTAutils

Este repositorio contiene el módulo y los cuadernos para empezar a trastear con la aplicación de las LSTM (y más allá) al problema de la predicción de trayectorias.

## Documentación disponible:
### Básica 
- [Project file structure](docs/RTA.0.Structure.md)
- [Experiment class description](docs/RTAutils.experiment.md)

### Adicional
- [Preprocessing notebook](docs/RTA.1.Preprocessing.md)
- [ML utils notebook](docs/RTA.2.ML_utils.md)
- [Data preparation notebook](docs/RTA.3.Data_preparation.md)


## Uso de los recursos:
- El repositorio incluye la estructura de directorios necesaria para el proyecto. Puede clonarse o descargarse y debería estar listo para su uso.
    - Solo hay que depositar los ficheros de datos en las carpetas correspondientes: `clean`, `final` y `sampled` tal cual te los pasé.
- Es necesario generar el scaler y los encoders adecuados a los datos y los conjuntos de características que elijas.
    - Para ello, se ejecutan los dos primeros apartados del cuaderno [ML utils](0.2.ML_utils.ipynb) (selección de características y generación de scaler y encoder), indicando las características a utilizar y los meses que se van a procesar (para procesar los datos que ya tienes, los meses se pueden dejar como están).
    - Los artefactos generados se cargarán automáticamente en el objeto `Experiment` correspondiente.
- Si se va a trabajar con datos muestreados, es necesario ejecutar el apartado *Downsample* del cuaderno [Data preparation](0.3.data_preparation.ipynb), indicando el período de muestreo y los meses. Los datos muestreados a 15 segundos ya los tienes :)

- Opcionalmente, se pueden pre-generar las ventanas que se utilizarán para el entrenamiento y la evaluación mediante la API Dataset de Tensorflow, y guardarlas en disco. Para ello, se utiliza el apartado correspondiente del cuaderno [Data preparation](0.3.data_preparation.ipynb).
    - Su principal interés es permitir trabajar con grandes volúmenes de datos que no caben completos en memoria. Además, acelera levemente las operaciones, ya que evitan la construcción de las ventanas al vuelo.
    - Por otro lado, ocupan un montón. Siendo pocos datos, es posible trabajar con los ficheros parquet (haciendo uso del parámetro `from_parquet` convenientemente).

## Entrenamiento y evaluación de modelos

Se proporciona un [cuaderno base](1.baseTrays.ipynb) adaptado al entrenamiento y evaluación de modelos de predicción de trayectorias. Incluye el flujo de trabajo básico:
- Definición de todos los parámetros del proceso.
- Instanciación e inicialización del modelo.
- Entrenamiento.
- Evaluación global y por tiempos.
- Un mecanismo de visualización de trayectorias individuales que falla más que una escopeta de feria.

Se incluye una clase `ExperimentTrajectory` que implementa una LSTM *vanilla* para predecir el siguiente timestep. Para experimentar con nuevos modelos, debe implementarse una clase hija de `Experiment` definiendo los dos métodos "abstractos" de la clase padre, con el siguiente patrón:

```python
class NombreClaseHija(Experiment):
    def __init__(self, ...):
        super().__init__(...)
        # Constructor de la clase hija
        pass

    def init_model(self, ...):
        # Aquí va la definición del modelo
        self.model = ...
        self.model.compile(...)
        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        # Adaptación de dataset de ventanas al formato requerido por el modelo
        pass
```

Puedes basarte en la clase `ExperimentTrajectory`, y preguntarme en caso de duda (●'◡'●).

## Avisos

> ⚠ **Únicamente un timestep** (de momento)
>
> En su versión actual, el módulo permite realizar experimentos con modelos LSTM para predecir un único timestep. Sin embargo, se puede usar cualquier conjunto de características, así que se pueden evaluar diferentes combinaciones para este problema.

> ⚠ **Scaler y encoder**
>
> El criterio de nombres para estos artefactos es muy simple: `<artefacto>_<num. características>`. Esto quiere decir que dos conjuntos de características diferentes, pero con el mismo número de características, harían que los artefactos de uno sobreescribieran al otro. Yo no me he visto en esta situación, pero si te diera problemas modifico el código para evitar que ocurra.

## Dependencias:

No es que se necesiten exactamente estas versiones, pero son las que tengo instaladas actualmente en el entorno. Las versiones anteriores pueden ser problemáticas principalmente con Tensorflow, porque en alguna actualización introdujeron oficialmente la API Dataset (antes era una feature experimental) y cambia su ubicación en la librería. Ante la duda, probar.

- tensorflow=2.9.1
- keras=2.11.0
- scikit-learn=1.2.0
- tqdm=4.64.1
- plotly=5.9.0
- pyarrow=7.0.0
- pandas=1.5.2
- numpy=1.22.3