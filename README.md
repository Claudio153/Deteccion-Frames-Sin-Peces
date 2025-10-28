# 🐟 Detección de Frames Vacíos en Secuencias de Video

Este proyecto implementa un modelo de **clasificación binaria** basado en **MobileNetV3 preentrenada en ImageNet**, cuyo objetivo es **identificar y eliminar frames vacíos** (sin peces) provenientes de secuencias de video de inspección.
El propósito es optimizar el procesamiento de datos, eliminando los frames que no aportan información útil.

---

## 🎯 Objetivo del proyecto

El objetivo principal es desarrollar un modelo de visión por computador que:

* Clasifique los frames en dos categorías: **con_pez** y **vacio**.
* Evite a toda costa eliminar frames con peces.
* Utilice **transfer learning** con la red **MobileNetV3-Small**.

---

## ⚙️ Etapas del proyecto

### 1️⃣ Preprocesamiento (`Preprocesamiento.ipynb`)

En esta etapa se:

* Cargan los datasets originales desde:

  ```
  datasets/con_pez/
  datasets/vacio/
  ```
* Se aplica **redimensionamiento** de imágenes a **224×224 píxeles**.
* Se realiza **data augmentation offline** sobre los frames vacíos (rotación leve, espejado, cambios de contraste, brillo y blur suave).
* Se balancea el dataset final seleccionando aleatoriamente un número similar de imágenes con peces.
* Finalmente, se divide el dataset en:

  ```
  datasets_final/
  ├── train/
  └── val/
  ```

  con una proporción **80% entrenamiento / 20% validación**.

---

### 2️⃣ Entrenamiento (`Entrenamiento.ipynb`)

Durante el entrenamiento:

* Se aplica **data augmentation al vuelo** con `torchvision.transforms`:

  * Recortes aleatorios (`RandomResizedCrop`)
  * Rotaciones leves (`RandomRotation`)
  * Flips horizontales
  * Cambios de color y contraste (`ColorJitter`)
  * Blur gaussiano
* Las imágenes se **normalizan** siguiendo las estadísticas de **ImageNet**.
* Se entrena una **MobileNetV3-Small** con fine-tuning:

  ```python
  model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  model.classifier[3] = nn.Linear(1024, 2)  # ['con_pez', 'vacio']
  ```
* Se utiliza **CrossEntropyLoss** con pesos ajustados para penalizar más los errores al clasificar frames con peces:

  ```python
  criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(DEVICE))
  ```
* Optimización con **Adam** y **scheduler**:

  ```python
  optimizer = optim.Adam(model.parameters(), lr=3e-4)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
  ```
* Se implementa **early stopping** automático basado en `val_acc`.

El modelo final se guarda como:

```
mobilenetv3_binaria_best.pt
```

---

## 📊 Evaluación

El modelo se evalúa sobre el conjunto de validación usando métricas de clasificación:

* **Matriz de confusión**
* **Precisión (Precision)**
* **Sensibilidad (Recall)**
* **F1-score**
* **Accuracy global**

Ejemplo de resultados:

```
=== MATRIZ DE CONFUSIÓN ===
[[231   2]
 [  0 150]]

=== MÉTRICAS POR CLASE ===
con_pez    → Precision: 1.000 | Recall: 0.991 | F1-score: 0.996
vacio      → Precision: 0.987 | Recall: 1.000 | F1-score: 0.993

=== ACCURACY GLOBAL ===
0.995
```

**Interpretación:**

* Alta precisión en “vacio” → elimina muy pocos frames con peces.
* Alto recall en “vacio” → detecta correctamente la mayoría de los frames vacíos.
* Excelente equilibrio general (F1 ≈ 0.99).

---

## 🧱 Estructura del proyecto

```
deteccion_frames_vacios/
│
├── Preprocesamiento.ipynb
├──Entrenamiento.ipynb
│
├── datasets/
│   ├── con_pez/
│   │    ├── 1_pez/
│   │    ├── 2_peces/
│   │    └── 3_peces/
│   │ 
│   └── vacio/
│
├── mobilenetv3_binaria_best.pt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💻 Instalación y ejecución

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/<tu_usuario>/deteccion_frames_vacios.git
cd deteccion_frames_vacios
```

### 2️⃣ Crear un entorno virtual (opcional)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux
.venv\Scripts\activate     # Windows
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecutar notebooks

```bash
jupyter notebook
```

---

## 📦 Dependencias principales (`requirements.txt`)

Estas librerías se extraen directamente de los notebooks:

```
torch==2.4.0
torchvision==0.19.0
numpy==2.1.1
pillow==10.4.0
matplotlib==3.9.2
scikit-learn==1.5.2
tqdm==4.66.5
```

---

## 🧠 Consideraciones finales

* El modelo prioriza **no eliminar frames con peces**.
* Se ajustó la función de pérdida para penalizar más los falsos vacíos.
* El entrenamiento y validación se realizaron con imágenes **balanceadas y aumentadas**.
* Se utilizó **precisión mixta (AMP)** para acelerar el entrenamiento en GPU.

---

## 📚 Licencia

Este proyecto se distribuye bajo licencia **MIT**, permitiendo su libre uso y modificación con fines académicos o de investigación.
