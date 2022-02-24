# Super-resolution

Implementaciones de distintos algoritmos de aplicacion de super-resolution a imagenes con PyTorch.

## Instalación

1 - Obtener version de python>=3.9.

2 - Crear venv con el comando `python3 -v venv .venv`.

3 - Seleccionar el venv como source con `source .venv/bin/activate`.

4 - Instalar los requerimientos con `pip install -r requirements.txt`.

5 - Descargar los pesos de los modelos de drive para [ESRGAN](https://drive.google.com/drive/folders/11EDB_YuHQmcbUFm2GE0xcekvZ2b7YOT-) y [SRGAN](https://drive.google.com/drive/folders/12OG-KawSFFs6Pah89V4a_Td-VcwMBE5i) y colocarlos en *super-resolution/weights/*.

6 - Correr el dashboard desde la terminal (con el venv activado) con el comando `streamlit run dashboard.py`.

## Código

El repo se divide en 3 carpetas principales: **model** y **research**.

En **research** se encuentran los scipts utilizados para entrenar y probar los paquetes.

En **model** se tienen los scripts **viz&#46;py** y **inference&#46;py** que contienen funciones generales usadas en distintas partes y una carpeta por cada modelo GAN utilizado: **SRGAN** y **ESRGAN**. En cada una de ellas se encuentra el código de las redes generativas y discriminativas, asi como el codigo de inferencia.

Estos son resultados utilizando los pesos pre-entrenados de las distintas arquitecturas probadas:

![imagen](./images/SR-comparisson.jpeg)

## Entrenamiento y datasets

Se descargaron múltiples datasets de [Kaggle](https://www.kaggle.com/andrewmvd/animal-faces) y se reentrenaron ambas redes para un caso particular: imagenes de caras de animales.

El dataset de animales consiste en mas de 15000 imagenes de perros, gatos y animales salvajes.

Se realizaron distintos entrenamientos para encontrar la configuracion que producia mejores resultados.

Se tienen de esta forma dos sets de pesos, uno general descargado de internet entrenado con el dataset BSDS100 y otro para animales.


## Dashboard

Ademas de dichos resultados, se puede correr un dashboard hecho con streamlit para visualizar rapidamente resultados. Para ello, se necesita correr desde la linea de comandos `streamlit run dashboard.py` en la raiz del proyecto y estando con el venv activado.

Este dashboard se encuentra deployado ademas ([link](https://share.streamlit.io/patco96/super-resolution/main/dashboard.py)).