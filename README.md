# 🐟 Detección de Frames Vacíos en Secuencias de Video

Este proyecto implementa un modelo de **clasificación binaria** basado en **MobileNetV3 preentrenada en ImageNet**, cuyo objetivo es **identificar y eliminar frames vacíos** (sin peces) provenientes de secuencias de video de inspección.  
El propósito es optimizar el procesamiento de datos, eliminando los frames que no aportan información útil.

---

## 🎯 Objetivo del proyecto

El objetivo principal es desarrollar un modelo de visión por computador que:
- Clasifique los frames en dos categorías: **con_pez** y **vacio**.  
- Evite a toda costa eliminar frames con peces.  
- Utilice **transfer learning** con la red **MobileNetV3-Small**.

---

## ⚙️ Etapas del proyecto

### 1️⃣ Preprocesamiento (`Preprocesamiento.ipynb`)
En esta etapa se:
- Cargan los datasets originales desde:
  
  datasets/con_pez/
  
  datasets/vacio/
  
- Se aplica **redimensionamiento** de imágenes a **224×224 píxeles**.
- Se realiza **data augmentation offline** sobre los frames vacíos (rotación leve, espejado, cambios de contraste, brillo y blur suave).
- Se balancea el dataset final seleccionando aleatoriamente un número similar de imágenes con peces.
- Finalmente, se divide el dataset en:
  
    datasets_final/
  ├── train/
  └── val/
