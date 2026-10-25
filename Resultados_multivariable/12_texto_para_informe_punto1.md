# Taller IA 2 — Punto 1. Análisis exploratorio de datos

**Base:** `datos_multisensor_corregidos.csv`. **Registros:** 143,561. **Variables:** 10 en cuatro sensores.

## 1. Visualización y estadísticas

Los estadísticos descriptivos por variable (media, desviación estándar, mínimo, máximo,
mediana, cuartiles y MAD) están en `01_estadisticos_10_variables.csv`.
Los gráficos individuales de cada eje están en `Resultados_acelerometro`,
`Resultados_giroscopio`, `Resultados_magnetometro` y `Resultados_temperatura`.
Los gráficos 01–07 de esta carpeta complementan los análisis individuales.

**Validez física:** el CSV no trae fabricante, calibración, ubicación instrumental,
unidades verificadas para todos los sensores, rangos admisibles ni estado operativo
nominal. Por ello, no se declara que todas las mediciones sean físicamente normales
ni se aplican límites de otros aerogeneradores. Solo se ha verificado su consistencia
interna. Ver `02_validez_fisica_pendiente.csv`.

## 2. Nulos y corrección estructural

El Excel de origen tiene 352 filas con fragmentos concatenados y una última fila
incompleta. El código 00 recupera el bloque íntegro de cada fila afectada, mantiene
la última fila y documenta las imputaciones. En la última se imputan exclusivamente
MagneZ y temperatura con la media de las dos últimas observaciones previas.
`SMP original` final no se inventa: queda vacío y se usa un SMP continuo estimado.
Se conservan **143,561 registros**; hay **0 mediciones nulas**
en las 10 variables después del proceso. El origen SÍ contenía datos faltantes.
La auditoría está en `Datos_corregidos/registro_imputaciones.csv` y
`10_auditoria_nulos_integridad.csv`. Existen **351**
identificadores de muestra no representados en registros completos, que no se interpolan.

## 3. Criterios de valores atípicos

**6σ clásico:** media ±6 desviaciones estándar muestrales, por variable.
Los valores fuera se registran como sospechosos y se dejan vacíos solo en la copia
tratada. Recuentos por variable: Ac-X=0, Ac-Y=0, Ac-Z=0, Gi-X=0, Gi-Y=0, Gi-Z=6, Ma-X=0, Ma-Y=0, Ma-Z=0, Temp=0.

**Z modificado (NIST/Iglewicz-Hoaglin):** z=0,6745·(x−mediana)/MAD. Un valor
con |z| >3,5 es un posible atípico. El proyecto separa alertas (3,5 <|z|≤6)
de extremos estadísticos (|z| >6), límite este último elegido para explorar.
**No son criterios físicos de seguridad ni diagnósticos de falla.**

Acelerómetro: 133 alertas y 0 extremos; giroscopio:
238 y 6; magnetómetro: 1622 y 19;
temperatura: 0 y 0. Se identificaron 7
filas con anomalías en al menos dos sensores (no necesariamente fallas).

Fuente metodológica: https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

## 4. Relación entre variables y evolución en la adquisición

Se calculó correlación Pearson y Spearman para las diez variables, y Pearson
después de dejar vacíos valores >6σ en la copia (matrices 03–06 y gráficos 01–03).
El mayor valor absoluto Pearson entre pares distintos de variables fue
**r=-0.308** para **Gi-Y y Gi-Z**.
Entre sensores diferentes, el mayor |r| fue **-0.156** para **Gi-Z y Temp**.

Correlación no demuestra causalidad ni descarta relaciones no lineales o dependientes
de condiciones operativas. La evolución por bloques y el porcentaje de alertas por
SMP se presentan en los gráficos 04 y 07. Se incluyen dispersiones entre sensores
(gráfico 05) y comparación del primer y último cuartil de registros (tabla 11).

**Advertencia metodológica:** el contador SMP permite ordenar observaciones, pero
no equivale a segundos. Sin frecuencia de muestreo y su estabilidad comprobadas
no se puede estimar duración, velocidad de degradación ni periodicidad de
mantenimiento. Los saltos de SMP pueden afectar análisis de episodios.

## 5. Conclusión para el informe

Los datos corregidos permiten describir el comportamiento de las diez variables,
revisar su completitud y detectar desviaciones estadísticas. Tras la corrección de
las filas desplazadas, las anomalías extremas previamente observadas en temperatura
y acelerómetro desaparecen bajo el criterio robusto aplicado a esta base. El
magnetómetro concentra la mayor proporción de alertas y el giroscopio presenta
algunos valores extremos aislados; estos hallazgos aconsejan una **revisión dirigida
de las señales**, no una orden automática de mantenimiento.

Las correlaciones lineales observadas son bajas o moderadas y, por sí mismas, no
permiten inferir averías. La falta de especificaciones de sensores, escala
instrumental, frecuencia de muestreo y registros etiquetados de fallas impide
validar la seguridad física de las mediciones, fechar eventos o predecir la fecha
de mantenimiento. Se recomienda obtener la ficha técnica, validar unidades y
operación normal, y comparar las anomalías con inspecciones reales.

## Relación con los cinco requisitos del profesor

1. **Estadísticas y gráficos:** completo como análisis descriptivo; validez
   física condicionada a información técnica ausente.
2. **Nulos:** auditados antes/después y registradas las imputaciones.
3. **Atípicos:** definidos por dos criterios documentados, sin confundirlos.
4. **Relaciones y evolución:** matrices conjuntas + dispersión + SMP;
   evolución en tiempo real pendiente de frecuencia de adquisición verificada.
5. **Conclusión:** redactada aquí con alcances y limitaciones explícitos.

## Bibliografía metodológica y normativa

- NIST/SEMATECH, *Engineering Statistics Handbook – Detection of Outliers*:
  https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- IEC, *IEC 61400-1:2019+AMD1:2025*, requisitos generales de diseño:
  https://webstore.iec.ch/en/publication/111042

Las normas citadas no proporcionan en la vista pública límites aplicables directamente
al acelerómetro, giroscopio o magnetómetro de esta base.
