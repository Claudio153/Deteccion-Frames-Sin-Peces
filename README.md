# 🧠 Proyecto de Clasificación de Frames de Salmones

Este repositorio contiene el desarrollo de un modelo de clasificación binaria basado en **MobileNetV3** para distinguir entre frames **con peces** y **vacíos** dentro de secuencias de video. El proyecto incluye el preprocesamiento de datos, el entrenamiento con **validación cruzada (K-Fold)** y la evaluación final en un conjunto de prueba independiente.

---

## 📂 Estructura del Repositorio

```
salmon-detector/
│
|
├── Preprocesamiento.ipynb   ← Data augmentation (offline) para clase "vacio"
├── Entrenamiento.ipynb      ← Entrenamiento + validación cruzada + métricas finales
│
├── pyproject.toml                 ← Dependencias gestionadas por UV
├── requirements.txt               ← Alternativa manual (opcional)
├── README.md                      ← Este archivo
└── 
```

> 🧩 **Importante:** El dataset es confidencial y no se incluye en este repositorio. Debe cargarse manualmente desde una fuente privada (por ejemplo, Google Drive).

---

## ⚙️ Instalación y Ejecución

> **Requisitos previos:** Tener instalado [Python 3.12](https://www.python.org/downloads/) y [UV](https://docs.astral.sh/uv/).

### 1️⃣ Clonar el repositorio

```bash
git clone [https://github.com/<tu_usuario>/salmon-detector.git](https://github.com/Claudio153/Deteccion-Frames-Sin-Peces.git)
cd Deteccion-Frames-Sin-Peces
```

### 2️⃣ Crear el entorno y resolver dependencias

```bash
uv sync
```

Esto:

* Crea automáticamente el entorno virtual `.venv`
* Instala todas las dependencias especificadas en `pyproject.toml`

### 3️⃣ Ejecutar el entorno de trabajo (JupyterLab)

```bash
uv run jupyter lab
```

Esto abrirá **JupyterLab** en tu navegador, desde donde podrás ejecutar los notebooks paso a paso.

---

## 🧠 Flujo de Ejecución del Proyecto

1. **Preprocesamiento (Preprocesamiento.ipynb):**

   * Aumenta la cantidad de imágenes de la clase `vacio` mediante técnicas de data augmentation offline.
   * Guarda los nuevos frames en el dataset local.

2. **Entrenamiento (Entrenamiento.ipynb):**

   * Entrena el modelo usando validación cruzada (5 folds estratificados).
   * Evalúa métricas de rendimiento: *Precision, Recall, F1-score y Accuracy*.
   * Guarda automáticamente el mejor modelo (`mobilenetv3_binaria_best.pt`).

3. **Evaluación final:**

   * Usa un conjunto de prueba externo (`datasets_test/`) para medir el desempeño final del modelo.
   * Genera la matriz de confusión y el reporte de clasificación.



## 🧩 Notas adicionales

* Se usa **MobileNetV3 Small** preentrenada en *ImageNet* y ajustada a 2 clases: `con_pez` y `vacio`.
* La función de pérdida aplica un **peso mayor a la clase `con_pez`**, priorizando evitar falsos negativos (frames con peces clasificados como vacíos).
* El entrenamiento emplea **Early Stopping** y **ReduceLROnPlateau** para optimizar la convergencia.

---

## 🧾 Comandos Útiles

| Acción                       | Comando              |
| ---------------------------- | -------------------- |
| Crear entorno y dependencias | `uv sync`            |
| Iniciar JupyterLab           | `uv run jupyter lab` |
| Limpiar entorno              | `uv clean`           |

---

## 🧑‍💻 Autor

**Claudio Díaz**
Proyecto académico – Universidad San Sebastián, 2025.
