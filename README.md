# P3. Redes Neuronales — Clasificación de Dígitos (0–9)

## Índice
1. [Notebook principal](./P3_Redes_Neuronales.ipynb)  
2. [Script de reconocimiento en tiempo real](./realtime_digits.py)  
3. [Video de demostración](./video_numeros.mp4)  
4. [README](./README.md)

---

## Descripción

Este proyecto desarrolla un sistema completo de clasificación de dígitos escritos a mano (0–9) utilizando **redes neuronales convolucionales (CNN)**.

Incluye todo el flujo de trabajo:

- **Importación y preprocesamiento de imágenes**  
  Las imágenes se convierten a escala de grises y se normalizan al rango \[0–1\].

- **Construcción y entrenamiento de múltiples modelos CNN** 
  Cada arquitectura se documenta, explicando capas, tamaños, parámetros y comportamiento.

- **Selección del mejor modelo** mediante desempeño en validación.

- **Entrenamiento final** usando *todo* el dataset de entrenamiento con la arquitectura ganadora.

- **Evaluación completa en test**, incluyendo métricas y matriz de confusión.

- **Uso en tiempo real** mediante cámara web

- **Video de demostración** donde se clasifican al menos 5 dígitos en diferentes formatos (pluma, lápiz, pintarrón, etc.).

---

## Objetivo general

Entrenar un modelo robusto que clasifique correctamente dígitos escritos a mano en diversos formatos, validarlo rigurosamente y demostrar su funcionamiento mediante un sistema de reconocimiento en tiempo real.

