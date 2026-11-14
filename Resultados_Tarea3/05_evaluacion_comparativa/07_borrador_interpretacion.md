# Evaluación comparativa descriptiva – Tarea 3

**Alcance:** monitoreo no supervisado de diez señales de sensores; sin etiquetas de fallas confirmadas.
La prueba se revisa descriptivamente, no se usa para ajustar parámetros o elegir umbrales de nuevo.

En el conjunto de prueba se analizaron **21,535 registros**.
- **Isolation Forest:** 948 alertas (4.40 %) y 764 episodios de SMP consecutivos.
- **K-means:** 962 alertas (4.47 %) y 627 episodios de SMP consecutivos.
- **DBSCAN:** 3,168 alertas (14.71 %) y 1,959 episodios de SMP consecutivos.

Los tres marcaron simultáneamente **581 registros** 
(al menos dos: **1,225**; al menos uno: **3,272**).
La coincidencia puede servir para priorizar inspecciones, pero no mide precisión diagnóstica.

## Coincidencia por pares (Jaccard)

- Isolation Forest / K-means: 604 coincidencias, Jaccard = 46.2 %.
- Isolation Forest / DBSCAN: 867 coincidencias, Jaccard = 26.7 %.
- K-means / DBSCAN: 916 coincidencias, Jaccard = 28.5 %.

## Explicabilidad SHAP

En la **muestra explicada** por el código 12, la mayor media de |SHAP| correspondió a Temp (0.025972).
Se trata de una explicación del modelo, no de una causa física ni de la importancia global validada.

## Advertencias metodológicas

- Isolation Forest y K-means usan un umbral fijado al percentil 99 del entrenamiento (alrededor de 1 %).
- DBSCAN seleccionó parámetros con una tasa orientativa de ruido en una muestra del entrenamiento; fuera de muestra emplea asignación aproximada al punto núcleo más cercano.
- Comparar tasas no demuestra cuál modelo es mejor: las políticas de alerta no están igualadas.
- La evaluación en la prueba ya ha sido inspeccionada. Cualquier ajuste posterior necesita nuevos datos retenidos para una comprobación independiente.
- No hay verdad terreno para calcular exactitud, precisión, sensibilidad, F1, PR-AUC ni falsos positivos reales.
- SMP es un contador de muestras: episodios y bloques no indican duración en segundos.
- Cambios estadísticos en las señales no permiten inferir averías sin mediciones operativas e inspecciones.
