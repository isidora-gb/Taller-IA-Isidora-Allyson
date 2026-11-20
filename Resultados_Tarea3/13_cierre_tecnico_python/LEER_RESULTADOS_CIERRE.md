# Cierre técnico de Python — Tarea 3

**Fuente analizada:** `C:\Users\isigb\OneDrive\Desktop\Taller-IA-Isidora-Allyson\Datos_corregidos\datos_multisensor_corregidos.csv`
**Filas en resultados del código 19:** 143,561
**Modalidad:** auditoría descriptiva. No se reentrenan modelos, no se modifican umbrales.

## Qué se verificó
Se contrastaron las banderas de los tres modelos entre los códigos 09–11 y el 19 por SMP y por partición; se analizaron tamaños y perfiles de grupos, silueta, desplazamientos entre cortes e imputaciones históricas.

## K-means y DBSCAN
- K elegido: **2**; silueta original en muestra de validación: **0.091209** (no es exactitud de fallas).
- Silueta diagnóstica sobre muestra de todas las particiones: **0.08465067724**; no se usa para escoger K.
- DBSCAN: No definida: DBSCAN nativo tiene solo 1 grupo no ruido (o no hay suficientes grupos).
- Los perfiles estandarizados de los clusters son descriptivos; K-means no asigna el significado «sano/falla» a los grupos.

## Cinco cortes y revisión del corte 4
Se calcularon diferencias de medias respecto del propio entrenamiento de cada corte, expresadas en desviaciones estándar del entrenamiento. No equivalen a límites físicos o a diagnósticos.
- Temp: -1.5280 DS (corte 4).
- Gi Z: +0.1864 DS (corte 4).
- Ma Y: -0.1661 DS (corte 4).
- Gi Y: -0.1205 DS (corte 4).
- Gi X: -0.1124 DS (corte 4).

## Procedencia y límites
- El CSV corregido ya puede contener dos imputaciones de la Tarea 2: el contador de nuevas imputaciones del prototipo no certifica las históricas.
- No existe una matriz de confusión válida sin etiquetas verificadas de averías.
- DBSCAN fuera de la muestra se asigna mediante proximidad a puntos núcleo (aproximación).
- SHAP del código 12 explica puntuación; TreeSHAP del 19 explica longitud de aislamiento. Sus magnitudes no se comparan.
- La prueba externa fue consultada en las comparaciones previas; cualquier reajuste exige una nueva evaluación independiente.
- SMP identifica posición de adquisición, no segundos.

## Estado automático de comprobaciones
- Errores detectados: 0.
- Advertencias / no verificado: 0.
- Ver `00_lista_comprobaciones.csv` y `02_consistencia_salidas_modelos.csv`.

## Archivos más importantes
- `04_variables_que_distinguen_clusters.csv`: ejes que más diferencian las medias de ambos clusters (en unidades comparables).
- `05_k_seleccion_silueta_original.csv`: siluetas de los K explorados.
- `07_cambios_variables_por_corte.csv`: desplazamiento por sensor y por corte.
- `09_diagnostico_corte_4_ordenado.csv`: ranking del corte 4.
- `10_auditoria_documentos_imputacion.csv` y `11_comprobacion_valores_terminales.csv`: trazabilidad parcial de dos imputaciones.
- `12_shap_importancias_separadas.csv`: sin mezclar tipos distintos de SHAP.
- `13_auditoria_preprocesamiento_entrenamiento.csv`: comprueba que mediana y escalado proceden solo del 70 % inicial.
- `14_auditoria_episodios_SMP.csv`: comprueba episodios sin unir saltos SMP.

Este documento presenta diagnósticos de calidad de datos y comportamiento de modelos; **no confirma fallas físicas**.