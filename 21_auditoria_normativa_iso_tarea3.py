
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMP = 'smp_continuo'
FLAGS = ['alerta_if', 'alerta_kmeans', 'ruido_dbscan_aproximado']
PARTES = ['entrenamiento', 'validacion', 'prueba']
SENSORES = {
    'AcelX1_SisGenEol': ('Acelerometro', 'X', 'm/s2 (reportada, no verificada)'),
    'AcelY1_SisGenEol': ('Acelerometro', 'Y', 'm/s2 (reportada, no verificada)'),
    'AcelZ1_SisGenEol': ('Acelerometro', 'Z', 'm/s2 (reportada, no verificada)'),
    'GiroX_SisGenEol': ('Giroscopio', 'X', 'grados/s (reportada, no verificada)'),
    'GiroY_SisGenEol': ('Giroscopio', 'Y', 'grados/s (reportada, no verificada)'),
    'GiroZ_SisGenEol': ('Giroscopio', 'Z', 'grados/s (reportada, no verificada)'),
    'MagneX_SisGenEol': ('Magnetometro', 'X', 'unidad no verificada'),
    'MagneY_SisGenEol': ('Magnetometro', 'Y', 'unidad no verificada'),
    'MagneZ_SisGenEol': ('Magnetometro', 'Z', 'unidad no verificada'),
    'Ter_SisGenEol': ('Temperatura', 'unica', 'grados C (segun descripcion de base)'),
}
REFERENCIAS = [
    {'norma':'ISO 16079-1:2026', 'titulo':'Condition monitoring and diagnostics of wind turbines — Part 1: General guidelines',
     'fuente':'https://www.iso.org/standard/88268.html',
     'alcance_publico':'Directrices generales para escoger metodos de monitoreo de condicion de componentes eolicos; no especifica sistemas informaticos de monitoreo.'},
    {'norma':'ISO 17359:2018', 'titulo':'Condition monitoring and diagnostics of machines — General guidelines',
     'fuente':'https://www.iso.org/standard/71194.html',
     'alcance_publico':'Directrices generales para establecer programas de monitoreo de condicion de maquinas.'},
    {'norma':'ISO 20816-21:2025', 'titulo':'Mechanical vibration — Measurement and evaluation of machine vibration — Part 21: Horizontal axis wind turbines',
     'fuente':'https://www.iso.org/standard/84280.html',
     'alcance_publico':'Evaluacion de vibracion de aerogeneradores de eje horizontal con potencia nominal de generador superior a 200 kW; distingue con/sin multiplicadora.'},
]
DEFECTO = {
    '_instrucciones':'Completar SOLO con documentacion tecnica o evidencia verificable. null = desconocido; no inventar valores.',
    'modelo_aerogenerador': None,
    'potencia_generador_kw': None,
    'eje_horizontal': None,
    'transmision': None,
    'ubicacion_sensores_documentada': None,
    'calibracion_documentada': None,
    'frecuencia_muestreo_hz': None,
    'magnitud_y_metodo_vibracion_compatibles_con_iso_20816_21': None,
    'unidades_fisicas_verificadas': None,
    'condiciones_operacion_documentadas': None,
    'historial_fallas_verificadas_disponible': None,
    'historial_inspecciones_verificadas_disponible': None,
    'sistema_temporal_vincula_SMP_con_eventos_reales': None,
    'documentacion_consultada': [],
}


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    pd.DataFrame(tabla).to_csv(ruta, sep=';', index=False, encoding='utf-8-sig', float_format='%.10g')


def leer_csv(ruta: Path) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f'No existe {ruta}. Ejecuta primero codigo 19 (modo normal).')
    with ruta.open('r', encoding='utf-8-sig') as f:
        encabezado = f.readline()
    sep = ';' if ';' in encabezado else ','
    return pd.read_csv(ruta, sep=sep, encoding='utf-8-sig', low_memory=False)


def texto_meta(v):
    if v is None or v == '':
        return 'NO DOCUMENTADO'
    if isinstance(v, bool):
        return 'SI' if v else 'NO'
    return str(v)


def bool_confirmado(v):
    return v is True


def iso20816_alcance(m: dict) -> tuple[str, str]:
    eje = m.get('eje_horizontal')
    potencia = m.get('potencia_generador_kw')
    if eje is False:
        return ('FUERA_DEL_ALCANCE_DECLARADO', 'El equipo se declara de eje no horizontal.')
    if potencia is not None:
        try:
            p = float(potencia)
            if p <= 0:
                return ('DATO_INVALIDO', 'La potencia declarada debe ser mayor que cero.')
            if p <= 200:
                return ('FUERA_DEL_ALCANCE_DECLARADO', 'La potencia declarada es de 200 kW o menor.')
        except (TypeError, ValueError):
            return ('DATO_INVALIDO', 'La potencia declarada no es numerica.')
    if eje is None or potencia is None:
        return ('NO_VERIFICABLE', 'Falta verificar eje horizontal y/o potencia nominal superior a 200 kW.')
    return ('ALCANCE_GENERAL_POSIBLE',
            'El tipo de eje y la potencia podrian corresponder, pero NO hay demostracion de mediciones compatibles ni conformidad ISO.')


def comprobar_columnas_y_orden(df: pd.DataFrame) -> pd.DataFrame:
    exigidas = [SMP, 'conjunto', *FLAGS, *SENSORES]
    faltan = [x for x in exigidas if x not in df.columns]
    if faltan:
        raise ValueError('Faltan columnas del codigo 19: '+', '.join(faltan))
    out = df.copy()
    for c in [SMP, *FLAGS, *SENSORES]:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    if out[SMP].isna().any() or ((out[SMP] % 1) != 0).any():
        raise ValueError('SMP continuo vacio o no entero.')
    if out[SMP].duplicated().any():
        raise ValueError('Hay SMP duplicados en la salida del codigo 19.')
    for c in FLAGS:
        if (~out[c].isin([0, 1])).any():
            raise ValueError(f'La columna {c} contiene valores fuera de 0/1.')
        out[c] = out[c].astype('int8')
    out[SMP] = out[SMP].astype('int64')
    out = out.sort_values(SMP, kind='mergesort').reset_index(drop=True)
    if out['conjunto'].isna().any() or not set(out['conjunto'].unique()).issubset(set(PARTES)):
        raise ValueError('Conjuntos incompatibles con las tres particiones conocidas.')
    return out


def construir_episodios(df: pd.DataFrame, mascara: pd.Series, nombre: str) -> pd.DataFrame:
    # Una separacion >1 SMP o cambio de conjunto ROMPE continuidad.
    d = df[[SMP, 'conjunto']].copy()
    d['activo'] = mascara.to_numpy(dtype=bool)
    nuevo = ((d[SMP].diff() != 1) |
             d['conjunto'].ne(d['conjunto'].shift()) |
             d['activo'].ne(d['activo'].shift()))
    d['bloque'] = nuevo.cumsum()
    eventos = d[d['activo']].groupby('bloque', sort=False).agg(
        conjunto=('conjunto', 'first'),
        smp_inicio=(SMP, 'min'), smp_fin=(SMP, 'max'),
        longitud_muestras=(SMP, 'size')).reset_index(drop=True)
    eventos.insert(0, 'tipo', nombre)
    eventos.insert(0, 'episodio', np.arange(1, len(eventos)+1))
    return eventos


def reporte_matriz_normativa(meta: dict) -> pd.DataFrame:
    s20816, detalle = iso20816_alcance(meta)
    filas = [
        {'norma': REFERENCIAS[0]['norma'], 'uso_propuesto':'Marco metodologico para elegir monitor de condicion',
         'estado_en_proyecto':'REFERENCIA_METODOLOGICA',
         'evidencia_actual':'Tres detectores no supervisados, indicadores y trazabilidad computacional',
         'informacion_pendiente':'Modos de falla, configuracion eolica, instrumentacion e inspecciones reales',
         'aclaracion':'No acredita conformidad, diagnostico o prognostico de fallas'},
        {'norma': REFERENCIAS[1]['norma'], 'uso_propuesto':'Referencia para organizar un programa de monitoreo de maquinaria',
         'estado_en_proyecto':'REFERENCIA_METODOLOGICA',
         'evidencia_actual':'Control de datos, scores, episodios, prioridades internas y revision propuesta',
         'informacion_pendiente':'Responsables, plan de inspeccion, procedimientos, umbrales fisicos y validacion',
         'aclaracion':'No se ha desplegado un programa operacional'},
        {'norma': REFERENCIAS[2]['norma'], 'uso_propuesto':'Referencia futura para medicion/evaluacion de vibracion',
         'estado_en_proyecto':s20816,
         'evidencia_actual':detalle,
         'informacion_pendiente':'Potencia >200 kW, eje, multiplicadora, ubicacion/montaje, unidades, magnitudes, condiciones y calibracion',
         'aclaracion':'No se aplican zonas/umbrales numericos ISO a la aceleracion registrada'},
    ]
    return pd.DataFrame(filas)


def figura_alertas(res: pd.DataFrame, carpeta: Path) -> None:
    modelos = [('if_pct','Isolation Forest'), ('kmeans_pct','K-means'), ('dbscan_pct','DBSCAN (aprox.)')]
    d = res[res.conjunto.isin(PARTES)].set_index('conjunto').reindex(PARTES)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = np.arange(len(PARTES))
    for i, (col, nombre) in enumerate(modelos):
        ax.bar(xs + (i-1)*.24, d[col].to_numpy(float), .22, label=nombre)
    ax.set_xticks(xs, ['Entrenamiento','Validacion','Prueba'])
    ax.set_ylabel('Registros marcados (%)')
    ax.set_title('Tasas de alertas por particion (NO son tasas de falla)')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(carpeta/'01_alertas_modelos_por_conjunto.png', dpi=190)
    plt.close(fig)


def figura_consenso(res: pd.DataFrame, carpeta: Path) -> None:
    d = res[res.conjunto.isin(PARTES)].set_index('conjunto').reindex(PARTES)
    colors = ['#9EABB5','#78A1C6','#D6A56C','#B86970']
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(PARTES))
    abajo = np.zeros(3)
    for i in range(4):
        values = d[f'votos_{i}'].to_numpy(int)
        ax.bar(x, values, bottom=abajo, label=f'{i} modelos', color=colors[i])
        abajo += values
    ax.set_xticks(x,['Entrenamiento','Validacion','Prueba'])
    ax.set_ylabel('Cantidad de registros')
    ax.set_title('Consenso entre modelos (criterio interno, no ISO)')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(carpeta/'02_consenso_por_conjunto.png', dpi=190)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description='Auditoria de aplicabilidad ISO y trazabilidad de anomalías (SIN diagnosticar fallas)')
    p.add_argument('--raiz', type=Path, default=Path(__file__).resolve().parent,
                   help='Carpeta del repositorio que contiene Resultados_Tarea3')
    p.add_argument('--metadatos', type=Path, default=None,
                   help='JSON opcional con datos REALES del equipo, comprobados documentalmente')
    p.add_argument('--salida', type=Path, default=None, help='Carpeta de salidas (por defecto Resultados_Tarea3/14_auditoria_iso)')
    args = p.parse_args()
    raiz = args.raiz.expanduser().resolve()
    entrada = raiz/'Resultados_Tarea3'/'11_analisis_completo_143561'/'01_resultados_TODOS_143561_registros.csv'
    salida = args.salida or (raiz/'Resultados_Tarea3'/'14_auditoria_iso')
    if not salida.is_absolute():
        salida = raiz/salida
    salida.mkdir(parents=True, exist_ok=True)
    meta = dict(DEFECTO)
    origen_meta = args.metadatos
    if origen_meta is not None:
        origen_meta = origen_meta if origen_meta.is_absolute() else raiz/origen_meta
        if not origen_meta.is_file():
            raise FileNotFoundError(f'No existe el JSON de metadatos: {origen_meta}')
        usuario = json.loads(origen_meta.read_text(encoding='utf-8-sig'))
        if not isinstance(usuario, dict):
            raise ValueError('El archivo de metadatos debe contener un objeto JSON.')
        meta.update(usuario)
    (salida/'99_metadatos_utilizados.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    df = comprobar_columnas_y_orden(leer_csv(entrada))
    n = len(df)
    if n < 1:
        raise ValueError('La entrada no contiene registros.')
    smp = df[SMP].to_numpy()
    posiciones_ausentes = int(np.maximum(0, np.diff(smp)-1).sum())
    saltos = int(np.sum(np.diff(smp)>1))
    nulos_fisicos = int(df[list(SENSORES)].isna().sum().sum())

    votos = df[FLAGS].sum(axis=1).astype('int8')
    # Clasificacion de PRIORIDAD OPERACIONAL PROPUESTA, NO ISO, no diagnostica falla.
    prioridades = np.select([votos == 3, votos == 2, votos == 1],
        ['revisar_3_modelos', 'revisar_2_modelos', 'alerta_1_modelo'], default='sin_alerta_modelos')
    bitacora = df[[SMP, 'conjunto', *FLAGS]].copy()
    bitacora['votos_modelos'] = votos
    bitacora['prioridad_interna_no_iso'] = prioridades
    bitacora['regla'] = 'consenso_exploratorio_no_diagnostico'
    exportar(bitacora, salida/'03_priorizacion_historica_NO_ISO.csv')

    resumen = []
    for conjunto in [*PARTES, 'total']:
        d = df if conjunto == 'total' else df[df['conjunto']==conjunto]
        v = votos.loc[d.index]
        cantidad = len(d)
        tmp = {'conjunto':conjunto,'registros':cantidad,
               'alertas_if':int(d[FLAGS[0]].sum()),
               'alertas_kmeans':int(d[FLAGS[1]].sum()),
               'ruido_dbscan_aprox':int(d[FLAGS[2]].sum()),
               'consenso_2_o_3':int((v>=2).sum()),
               'consenso_los_3':int((v==3).sum())}
        for i in range(4):
            tmp[f'votos_{i}'] = int((v==i).sum())
        for col, rescol in [('alertas_if','if_pct'),('alertas_kmeans','kmeans_pct'),('ruido_dbscan_aprox','dbscan_pct')]:
            tmp[rescol] = round(100*tmp[col]/cantidad,6) if cantidad else np.nan
        resumen.append(tmp)
    resumen = pd.DataFrame(resumen)
    exportar(resumen, salida/'02_resumen_alertas_y_concordancia.csv')
    figura_alertas(resumen,salida)
    figura_consenso(resumen,salida)

    ep = []
    for nombre, activa in [('isolation_forest',df[FLAGS[0]].eq(1)),
                          ('kmeans_distancia',df[FLAGS[1]].eq(1)),
                          ('dbscan_ruido_aproximado',df[FLAGS[2]].eq(1)),
                          ('consenso_2_o_3',votos.ge(2)),
                          ('consenso_3',votos.eq(3))]:
        ep.append(construir_episodios(df, activa, nombre))
    episodios = pd.concat(ep, ignore_index=True)
    exportar(episodios, salida/'04_episodios_SMP_sin_falsa_continuidad.csv')
    episodios_resumen = episodios.groupby(['tipo','conjunto'], as_index=False).agg(
        episodios=('episodio','size'), muestras_alerta=('longitud_muestras','sum'),
        longitud_maxima_muestras=('longitud_muestras','max'))
    exportar(episodios_resumen, salida/'04b_resumen_episodios.csv')

    s = []
    for campo,(tipo,eje,unidad) in SENSORES.items():
        col = df[campo]
        s.append({'variable':campo,'sensor':tipo,'eje':eje,'unidad_reportada':unidad,
                  'verificacion_de_unidades_declarada':'SI_DECLARADO' if bool_confirmado(meta.get('unidades_fisicas_verificadas')) else 'NO_VERIFICADO',
                  'verificacion_de_posicion_declarada':'SI_DECLARADO' if bool_confirmado(meta.get('ubicacion_sensores_documentada')) else 'NO_VERIFICADO',
                  'verificacion_de_calibracion_declarada':'SI_DECLARADO' if bool_confirmado(meta.get('calibracion_documentada')) else 'NO_VERIFICADO',
                  'registros':n, 'nulos_csv_corregido':int(col.isna().sum()),
                  'minimo_observado':float(col.min()) if col.notna().any() else np.nan,
                  'maximo_observado':float(col.max()) if col.notna().any() else np.nan,
                  'interpretacion':'Estadistica exploratoria; no son limites de operacion ISO'})
    exportar(pd.DataFrame(s), salida/'01_matriz_instrumental_10_senales.csv')
    matriz = reporte_matriz_normativa(meta)
    exportar(matriz, salida/'00_aplicabilidad_normas_iso.csv')
    exportar(pd.DataFrame(REFERENCIAS), salida/'00b_fuentes_ISO_oficiales.csv')

    chequeos = [
        {'area':'Integridad de datos','comprobacion':'Archivo 19 disponible', 'resultado':'DISPONIBLE', 'detalle':str(entrada)},
        {'area':'Calidad de datos','comprobacion':'Registros SMP únicos y ordenables','resultado':'COMPROBADO', 'detalle':f'{n} registros; {saltos} saltos; {posiciones_ausentes} posiciones ausentes'},
        {'area':'Calidad de datos','comprobacion':'Nulos físicos en salida corregida','resultado':'COMPROBADO','detalle':f'{nulos_fisicos} celdas nulas en las 10 señales; las imputaciones anteriores requieren auditoría propia'},
        {'area':'Modelos','comprobacion':'Salidas IF/K-means/DBSCAN','resultado':'DISPONIBLE','detalle':'Banderas existentes del código 19; DBSCAN aproximado fuera de muestra'},
        {'area':'Monitoreo','comprobacion':'Episodios con saltos SMP', 'resultado':'COMPROBADO','detalle':'No se unen muestras cuando SMP salta ni se cruzan particiones'},
        {'area':'Trazabilidad','comprobacion':'Origen de etiquetas de fallas','resultado':'NO_VERIFICADO','detalle':'La base no entrega fallas confirmadas ni fechas asociadas'},
        {'area':'Instrumentación','comprobacion':'Modelo, ubicación y calibración','resultado':'DOCUMENTACION_DECLARADA_SIN_AUDITORIA' if (meta.get('modelo_aerogenerador') and bool_confirmado(meta.get('ubicacion_sensores_documentada')) and bool_confirmado(meta.get('calibracion_documentada'))) else 'NO_VERIFICADO','detalle':'Se requieren documentos técnicos auditables del equipo y sensores'},
        {'area':'Instrumentación','comprobacion':'Frecuencia y cronología real','resultado':'DOCUMENTACION_DECLARADA_SIN_AUDITORIA' if (meta.get('frecuencia_muestreo_hz') and bool_confirmado(meta.get('sistema_temporal_vincula_SMP_con_eventos_reales'))) else 'NO_VERIFICADO','detalle':'SMP es índice, no segundos; una frecuencia declarada no demuestra sincronización por sí sola'},
        {'area':'ISO 20816-21','comprobacion':'Aplicabilidad física de evaluación vibratoria','resultado':iso20816_alcance(meta)[0],
         'detalle':iso20816_alcance(meta)[1]},
        {'area':'Diagnóstico','comprobacion':'Matriz de confusión frente a fallas','resultado':'NO_EVALUABLE','detalle':'Sin etiquetas reales verificadas no pueden calcularse FP/TP/FN/TN'},
        {'area':'Normas','comprobacion':'Conformidad integral ISO','resultado':'NO_EVALUADA','detalle':'Consulta de resúmenes oficiales; no se realizó auditoría de conformidad con textos completos'},
    ]
    revis20 = raiz/'Resultados_Tarea3'/'13_cierre_tecnico_python'/'00_lista_comprobaciones.csv'
    if revis20.is_file():
        x20 = leer_csv(revis20)
        if 'estado' in x20.columns:
            val = x20['estado'].astype(str).value_counts().to_dict()
            chequeos.append({'area':'Auditoria codigo 20','comprobacion':'Verificaciones computacionales del código 20',
                'resultado':'DISPONIBLE','detalle':str(val)+'; no equivale a conformidad ISO'})
    exportar(pd.DataFrame(chequeos), salida/'05_estado_trazabilidad_y_pendientes.csv')
    exportar(pd.DataFrame([{'registros':n,'smp_minimo':int(smp.min()),'smp_maximo':int(smp.max()),
                          'saltos_smp_mayores_a_1':saltos,'posiciones_faltantes_por_saltos':posiciones_ausentes,
                          'nulos_fisicos_csv_corregido':nulos_fisicos,
                          'tiempo_en_segundos':'NO DISPONIBLE SIN FRECUENCIA DE MUESTREO'}]),
             salida/'06_control_secuencia_y_completitud.csv')

    final = resumen.loc[resumen.conjunto=='total'].iloc[0]
    estado20816,detalle20816=iso20816_alcance(meta)
    documento = f'''# Auditoría de aplicabilidad ISO — Tarea 3

**Fecha de generación:** {date.today().isoformat()}  
**Fuente de predicciones:** `{entrada}`  
**Registros analizados:** {n:,}  
**Modo:** DESCRIPTIVO; no se entrenaron modelos ni se alteraron umbrales.

## Base normativa (resúmenes públicos oficiales)

- **ISO 16079-1:2026**: orientación para elegir métodos de monitoreo en aerogeneradores. Fuente: {REFERENCIAS[0]['fuente']}
- **ISO 17359:2018**: orientación general para estructurar un programa de monitoreo de maquinaria. Fuente: {REFERENCIAS[1]['fuente']}
- **ISO 20816-21:2025**: evaluación de vibraciones en aerogeneradores de eje horizontal con generador de potencia nominal superior a 200 kW. Fuente: {REFERENCIAS[2]['fuente']}

**Alcance de este documento:** relación metodológica propuesta con los resúmenes oficiales disponibles. No se analizó íntegramente el texto normativo protegido y no se declara cumplimiento o certificación ISO.

## Hallazgos computacionales del histórico

- SMP: {saltos} saltos y {posiciones_ausentes} posiciones de adquisición ausentes. Esto NO representa segundos.
- Celdas físicas nulas en el CSV analizado: {nulos_fisicos} (no permite deducir cuántas se imputaron históricamente).
- Isolation Forest: {int(final['alertas_if']):,} registros señalados.
- K-means por distancia a centroide: {int(final['alertas_kmeans']):,} registros señalados.
- DBSCAN aproximado por puntos núcleo: {int(final['ruido_dbscan_aprox']):,} registros ruido.
- Consenso de al menos dos modelos: {int(final['consenso_2_o_3']):,}; consenso de tres: {int(final['consenso_los_3']):,}.

Estas cantidades reflejan las salidas del código 19 sobre TODOS los registros, incluyendo entrenamiento, validación y prueba. **No deben presentarse como precisión en prueba ni como cantidad de averías.**

## Evaluación de la aplicabilidad ISO 20816-21

- Estado: **{estado20816}**.
- Justificación: {detalle20816}
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
'''
    (salida/'07_LEER_AUDITORIA_ISO.md').write_text(documento, encoding='utf-8')
    print('\nAUDITORIA ISO FINALIZADA — SIN REENTRENAR MODELOS')
    print('Registros:', f'{n:,}', '| SMP saltos:', saltos, '| nulos físicos:',nulos_fisicos)
    print('Alertas IF:',int(final['alertas_if']), '| K-means:',int(final['alertas_kmeans']),
          '| DBSCAN aproximado:',int(final['ruido_dbscan_aprox']))
    print('Consenso al menos 2:',int(final['consenso_2_o_3']), '| los 3:',int(final['consenso_los_3']))
    print('ISO 20816-21:',estado20816)
    print('Resultados:',salida)
    print('AVISO: No se certifica cumplimiento ISO ni se diagnostican fallas reales.')


if __name__ == '__main__':
    main()
