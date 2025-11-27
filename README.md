<h1 align="center">Sistema de Reconocimiento Facial con CNN en Tiempo Real</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
</p>

---

<h2>Descripcion del Proyecto</h2>

<p>
Este proyecto consiste en el desarrollo de una red neuronal convolucional (CNN) capaz de identificar y clasificar rostros humanos en tiempo real. El sistema es entrenado con un dataset personalizado que puede incluir rostros de famosos, familiares, amigos y, especialmente, imagenes del propio estudiante.
</p>

<p>
El objetivo principal es construir un sistema funcional que abarque desde la captura y preprocesamiento de imagenes hasta el entrenamiento del modelo y su ejecucion en tiempo real mediante webcam.
</p>

---

<h2>Caracteristicas Principales</h2>

<h3>Procesamiento de Imagenes</h3>
<ul>
  <li>Deteccion de rostros utilizando MTCNN.</li>
  <li>Alineacion y recorte automatico de las imagenes.</li>
  <li>Normalizacion previa al entrenamiento.</li>
</ul>

<h3>Entrenamiento del Modelo</h3>
<p>Se implementan dos enfoques:</p>
<ol>
  <li>Modelo CNN basico, ideal para comprender los fundamentos del reconocimiento facial.</li>
  <li>Transfer Learning utilizando modelos preentrenados como:
    <ul>
      <li>MobileNetV2</li>
      <li>EfficientNet</li>
    </ul>
  </li>
</ol>

<h3>Reconocimiento en Tiempo Real</h3>
<ul>
  <li>Deteccion de rostros desde la webcam.</li>
  <li>Clasificacion de la identidad en tiempo real.</li>
  <li>Visualizacion en pantalla con etiquetado.</li>
</ul>

---

<h2>Controles del Sistema</h2>

<table>
  <tr>
    <th>Tecla</th>
    <th>Accion</th>
  </tr>
  <tr>
    <td><code>Q</code></td>
    <td>Salir de la aplicacion</td>
  </tr>
  <tr>
    <td><code>S</code></td>
    <td>Guardar una captura de pantalla</td>
  </tr>
</table>

---

<h2>Tecnologias Implementadas</h2>

<table>
  <tr>
    <th>Tecnologia</th>
    <th>Uso</th>
  </tr>
  <tr>
    <td>TensorFlow / Keras</td>
    <td>Entrenamiento del modelo de deep learning</td>
  </tr>
  <tr>
    <td>OpenCV</td>
    <td>Procesamiento de imagenes y video</td>
  </tr>
  <tr>
    <td>MTCNN</td>
    <td>Deteccion precisa de rostros</td>
  </tr>
  <tr>
    <td>MobileNetV2</td>
    <td>Modelo base para transfer learning</td>
  </tr>
</table>

---

<h2>Configuracion del Proyecto</h2>

<p>Los parametros de entrenamiento pueden modificarse en el archivo: <code>2_src/utils/config.py</code></p>

<table>
  <tr>
    <th>Parametro</th>
    <th>Valor</th>
    <th>Descripcion</th>
  </tr>
  <tr>
    <td>IMAGE_SIZE</td>
    <td>160x160</td>
    <td>Tamaño de las imagenes</td>
  </tr>
  <tr>
    <td>BATCH_SIZE</td>
    <td>8</td>
    <td>Tamaño del batch</td>
  </tr>
  <tr>
    <td>EPOCHS</td>
    <td>200</td>
    <td>Numero de epocas</td>
  </tr>
  <tr>
    <td>LEARNING_RATE</td>
    <td>0.0001</td>
    <td>Tasa de aprendizaje</td>
  </tr>
</table>

---

<h2>Autores</h2>

<p>Proyecto desarrollado por:</p>
<ul>
  <li>Chiquete Martinez Karen Daniela</li>
  <li>Gavilanes Medina Tania Elizabeth</li>
  <li>Pacheco Reyes Kimberlyn</li>
</ul>

---

<h2>Recursos de Apoyo</h2>

<p>Este proyecto se elaboro con apoyo de distintos recursos y tutoriales sobre:</p>
<ul>
  <li>Redes Neuronales Convolucionales (CNN)</li>
  <li>Transfer Learning con TensorFlow/Keras</li>
  <li>Deteccion de rostros con OpenCV y MTCNN</li>
  <li>Implementacion de reconocimiento facial en tiempo real</li>
</ul>

<h3>Tutoriales de YouTube</h3>
<ul>
  <li><a href="https://youtu.be/Lo1jljIIP9k?si=MuVNbwseTBPoLBbh">Tutorial 1</a></li>
  <li><a href="https://youtu.be/eBYlOGRUsCw?si=MHtPn8a1hpEIEbZ7">Tutorial 2</a></li>
  <li><a href="https://youtu.be/LsdxvjLWkIY?si=pp-RWq9n8Nr2UrbY">Tutorial 3</a></li>
  <li><a href="https://youtu.be/taC5pMCm70U?si=2zDLTLsefgpn_yze">Tutorial 4</a></li>
</ul>
