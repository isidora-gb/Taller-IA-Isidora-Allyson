# Optimizacion de Isolation Forest | lectura para informe

- Configuraciones con ambas semillas completadas: **16**.
- Se emplean todas las filas de entrenamiento (100.492) y las diez senales juntas.
- Validacion tiene 21.534 filas; prueba queda reservada hasta modo final.
- Los modelos difieren en arboles y submuestras internas por arbol; se fija max_features=1.0.
- Se usa contamination=auto solo como referencia; cuotas de 1/2/5% se calculan en entrenamiento.
- Barrido de perfil completo: True. Un perfil incompleto no autoriza elegir candidato.
- El ranking es exploratorio, en ausencia de etiquetas de falla. No indica precision o recall.

- Lider del ranking: `estandar_a500_m2048` (no equivale a mejor detector de fallas).
- Candidato que supera filtros orientativos: `estandar_a500_m2048`.

## Criterios y cautelas
- Spearman compara el orden de puntuaciones de DOS semillas en validacion.
- Jaccard top-1% y top-2% comparan los mismos registros mas inusuales sin aplicar umbral de falla.
- Jaccard de alertas auto es sensible a umbrales diferentes entre ajustes.
- La tasa de alertas NO es tasa de averias.
- El filtro de estabilidad es una convencion de esta investigacion, NO una norma ISO.
- No usar el conjunto de prueba para elegir parametros o umbrales.
- SMP no representa tiempo ni duracion.
