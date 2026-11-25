# Comparacion exploratoria de algoritmos | base total

- Mediciones alineadas por SMP: **143.561**.
- Senales usadas por los ajustes del codigo 16: 10 simultaneamente (segun su diseno).
- **No se reentrena ningun modelo** en el codigo 17; se comparan sus salidas guardadas.
- Se trata de resultados **dentro de muestra**, no de prueba independiente.
- SMP ordena adquisiciones, NO indica segundos ni intervalos regulares.
- Ruido por densidad, grupos K-Means y alertas IF son conceptos distintos.
- No hay fallas confirmadas: NO se calcula precision, recall ni exactitud diagnostica.

## Resumen por modelo

| Modelo | Grupos | Marcados ruido / alertas | % | Criterio |
|:---|---:|---:|---:|:---|
| K-Means | 2 | — | — | K-Means solo crea grupos |
| DBSCAN | 1 | 31.825 | 22.17% | Ruido por densidad |
| HDBSCAN | 2 | 143.111 | 99.69% | Ruido por densidad |
| Isolation Forest | — | 11.783 | 8.21% | Alerta IF (umbral automatico) |

## Concordancia entre criterios de interes

Indice Jaccard = cantidad marcada por ambos / cantidad marcada por al menos uno.
No es una metrica de acierto frente a fallas reales.

| Criterio A | Criterio B | Ambas marcas | Jaccard | % de A que tambien es B |
|:---|:---|---:|---:|---:|
| DBSCAN: ruido | HDBSCAN: ruido | 31.825 | 0.2224 | 100.00% |
| DBSCAN: ruido | IF: alerta | 11.376 | 0.3529 | 35.75% |
| HDBSCAN: ruido | IF: alerta | 11.783 | 0.0823 | 8.23% |

## SHAP: interpretacion de Isolation Forest

Senales de mayor importancia estimada en la muestra SHAP: Ma X (12.07%), Ac X (10.98%), Gi Y (10.77%).
SHAP explica **80** mediciones; NO se calcularon contribuciones individuales para las 143.561.
Una contribucion SHAP explica una prediccion del modelo; no demuestra causalidad ni falla de una pieza.

## Lectura tecnica sugerida

1. La cantidad de grupos NO determina la calidad del agrupamiento.
2. El porcentaje de ruido de HDBSCAN NO implica ese porcentaje de averias.
3. Que IF y DBSCAN marquen simultaneamente una medicion respalda concordancia estadistica, pero NO valida que exista una falla.
4. Sin etiquetas de fallas, registros de mantenimiento, datos nominales del equipo y frecuencia temporal, no se puede cuantificar desempeno de deteccion ni definir frecuencia de mantencion.
5. La seleccion de parametros en la base completa es exploratoria; conservar por separado la evaluacion cronologica 70/15/15.

## Advertencias de trazabilidad

- Los parametros del JSON de trazabilidad son globales y pudieron ser sobrescritos: no demuestran los parametros exactos de cada CSV.
- Los indicadores SSE y silueta proceden del barrido K y no demuestran que sea exactamente el mismo ajuste que las etiquetas CSV.
- Parametros HDBSCAN verificados con firma de etiquetas: min_cluster_size=100, min_samples=5, seleccion=leaf, escalado=estandar.

## Archivos generados

- `00_auditoria_insumos.csv`: archivos, columnas y muestras.
- `01_resultados_alineados_143561.csv`: etiquetas y puntuaciones unidas por SMP.
- `02_resumen_comparativo.csv`, `03_concordancia.csv`, `04_combinaciones.csv`,
  `05_evolucion_bloques_smp.csv`, `06_por_cluster_kmeans.csv`,
  `07_cuantiles_puntuacion_if.csv`: tablas de apoyo.
- Figuras 01 a 09: resumen, solapamiento, evolucion, perfiles y SHAP.

