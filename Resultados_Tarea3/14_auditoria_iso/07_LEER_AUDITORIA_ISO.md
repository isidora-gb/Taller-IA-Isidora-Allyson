# Auditoría de aplicabilidad ISO — Tarea 3

**Fecha de generación:** 2026-11-22  
**Fuente de predicciones:** `C:\Users\isigb\OneDrive\Desktop\Taller-IA-Isidora-Allyson\Resultados_Tarea3\11_analisis_completo_143561\01_resultados_TODOS_143561_registros.csv`  
**Registros analizados:** 143,561  
**Modo:** DESCRIPTIVO; no se entrenaron modelos ni se alteraron umbrales.

## Base normativa (resúmenes públicos oficiales)

- **ISO 16079-1:2026**: orientación para elegir métodos de monitoreo en aerogeneradores. Fuente: https://www.iso.org/standard/88268.html
- **ISO 17359:2018**: orientación general para estructurar un programa de monitoreo de maquinaria. Fuente: https://www.iso.org/standard/71194.html
- **ISO 20816-21:2025**: evaluación de vibraciones en aerogeneradores de eje horizontal con generador de potencia nominal superior a 200 kW. Fuente: https://www.iso.org/standard/84280.html

**Alcance de este documento:** relación metodológica propuesta con los resúmenes oficiales disponibles. No se analizó íntegramente el texto normativo protegido y no se declara cumplimiento o certificación ISO.

## Hallazgos computacionales del histórico

- SMP: 351 saltos y 351 posiciones de adquisición ausentes. Esto NO representa segundos.
- Celdas físicas nulas en el CSV analizado: 0 (no permite deducir cuántas se imputaron históricamente).
- Isolation Forest: 2,533 registros señalados.
- K-means por distancia a centroide: 2,518 registros señalados.
- DBSCAN aproximado por puntos núcleo: 10,884 registros ruido.
- Consenso de al menos dos modelos: 3,299; consenso de tres: 1,464.

Estas cantidades reflejan las salidas del código 19 sobre TODOS los registros, incluyendo entrenamiento, validación y prueba. **No deben presentarse como precisión en prueba ni como cantidad de averías.**

## Evaluación de la aplicabilidad ISO 20816-21

- Estado: **NO_VERIFICABLE**.
- Justificación: Falta verificar eje horizontal y/o potencia nominal superior a 200 kW.
- No se convierten las lecturas de aceleración en niveles ISO de vibración ni se comparan con límites físicos; faltan datos instrumentales y de operación.

## Aplicación propuesta

1. Recibir mediciones y comprobar integridad, orden SMP y calidad de datos.
2. Aplicar los tres métodos entrenados según la metodología previa; distinguir DBSCAN aproximado.
3. Conservar puntuaciones, asignaciones a clusters, episodios y consenso, usando prioridades internas exploratorias (NO categorías de la norma).
4. Usar SHAP solo para explicar la puntuación del Isolation Forest en registros seleccionados; la explicabilidad no es causal.
5. Solicitar inspecciones e historial de operación/fallas antes de declarar averías, definir límites físicos o calcular métricas supervisadas.

## Información faltante para una evaluación física defendible

- Marca/modelo, potencia nominal, eje, transmisión y componentes instrumentados.
- Posiciones y montaje de sensores, magnitudes y unidades calibradas, frecuencia de muestreo.
- Condiciones operativas, régimen de funcionamiento, procedimientos de inspección.
- Correspondencia comprobable entre SMP, tiempo físico y registros de fallas.

## Interpretación obligatoria

**El sistema detecta anomalías estadísticas; no diagnostica fallas reales ni demuestra cumplimiento integral de ISO 16079-1, ISO 17359 o ISO 20816-21.** Los campos opcionales de metadatos se consideran DECLARACIONES del usuario; el programa no verifica fichas técnicas externamente.
