# -*- coding: utf-8 -*-
"""TAREA 3 | Aplicar todos los modelos YA ENTRENADOS a un CSV nuevo.

NO se entrenan modelos ni se recalibran umbrales. Se utilizan exactamente los
artefactos guardados por los scripts 08, 09, 10 y 11:
- Isolation Forest: puntuacion y alerta por umbral fijo.
- K-means: cluster, distancia al centroide y alerta por umbral fijo.
- DBSCAN: extension APROXIMADA por vecino nucleo mas cercano y eps fijo.
- SHAP: explica solamente puntuaciones seleccionadas de Isolation Forest.

Desde la carpeta del proyecto:
    python 16_aplicar_modelos_nuevos_datos.py --entrada "nuevas_mediciones.csv"

Si aun no existe un CSV genuinamente nuevo, probar sin inventar mediciones:
    python 16_aplicar_modelos_nuevos_datos.py --demo-prueba

El modo --demo-prueba reutiliza el conjunto historico de PRUEBA; no constituye
validacion independiente ni datos nuevos. El modo --entrada exige SMP continuos
posteriores a toda la base historica para evitar falsas demostraciones.

Dependencias: numpy, pandas, scikit-learn, joblib, matplotlib, shap.
ATENCION: cargar joblib SOLO de archivos confiables generados en este proyecto.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMP = 'smp_continuo'
SEMILLA = 42
CORTOS = {
    'AcelX1_SisGenEol': 'Ac X', 'AcelY1_SisGenEol': 'Ac Y',
    'AcelZ1_SisGenEol': 'Ac Z', 'GiroX_SisGenEol': 'Gi X',
    'GiroY_SisGenEol': 'Gi Y', 'GiroZ_SisGenEol': 'Gi Z',
    'MagneX_SisGenEol': 'Ma X', 'MagneY_SisGenEol': 'Ma Y',
    'MagneZ_SisGenEol': 'Ma Z', 'Ter_SisGenEol': 'Temp',
}


def exigir_archivo(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f'Falta: {path}\nEjecuta primero 08, 09, 10 y 11 en esta carpeta, '
            'o revisa las rutas de Resultados_Tarea3.'
        )
    return path


def cargar_json(path: Path) -> dict:
    return json.loads(exigir_archivo(path).read_text(encoding='utf-8'))


def cargar_artefactos(raiz: Path) -> dict:
    base = raiz / 'Resultados_Tarea3'
    prepdir = base / '00_preparacion'
    ifdir = base / '01_isolation_forest'
    kmdir = base / '02_kmeans'
    dbdir = base / '03_dbscan'
    # Los joblib pueden ejecutar codigo al cargarse: usar solo los propios.
    prep = joblib.load(exigir_archivo(prepdir / '00_preprocesamiento.joblib'))
    modelo_if = joblib.load(exigir_archivo(ifdir / '00_modelo_isolation_forest.joblib'))
    modelo_km = joblib.load(exigir_archivo(kmdir / '00_modelo_kmeans.joblib'))
    modelo_db = joblib.load(exigir_archivo(dbdir / '00_modelo_dbscan_muestra.joblib'))
    if_config = cargar_json(ifdir / '00_configuracion_modelo.json')
    km_config = cargar_json(kmdir / '00_configuracion_kmeans.json')

    variables = list(prep['variables'])
    vars_z = ['z_' + x for x in variables]
    if len(variables) != 10 or len(set(variables)) != 10:
        raise ValueError('El preprocesamiento no contiene 10 variables fisicas diferentes')
    if list(modelo_db['variables']) != vars_z:
        raise ValueError('El orden de columnas guardado en DBSCAN no coincide con el preprocesamiento')
    if int(modelo_if.n_features_in_) != 10 or int(modelo_km.n_features_in_) != 10:
        raise ValueError('Modelos entrenados con distinto numero de variables')
    if getattr(prep['escalador'], 'n_features_in_', -1) != 10:
        raise ValueError('El escalador no coincide con los modelos')
    for key in ('vecinos_core', 'etiquetas_nucleos', 'eps'):
        if key not in modelo_db:
            raise ValueError(f'El archivo de DBSCAN no tiene el campo {key}')
    umbral_if = float(if_config['umbral_puntuacion'])
    umbral_km = float(km_config['umbral_distancia'])
    eps = float(modelo_db['eps'])
    if not np.isfinite([umbral_if, umbral_km, eps]).all() or eps <= 0:
        raise ValueError('Umbrales/eps no validos en modelos guardados')
    return {
        'preprocesador': prep, 'iforest': modelo_if, 'kmeans': modelo_km,
        'dbscan': modelo_db, 'umbral_if': umbral_if,
        'umbral_km': umbral_km, 'variables': variables,
        'vars_z': vars_z, 'preparados': prepdir,
    }


def a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype('string').str.strip().str.replace(',', '.', regex=False),
        errors='coerce',
    ).astype(float).replace([np.inf, -np.inf], np.nan)


def leer_csv(path: Path) -> pd.DataFrame:
    exigir_archivo(path)
    # Los CSV anteriores eran separados por ';'. Permite tambien ',' si
    # viene de un programa de captura con formato anglosajon.
    with path.open('r', encoding='utf-8-sig') as fh:
        primera = fh.readline()
    sep = ';' if ';' in primera else ','
    return pd.read_csv(path, sep=sep, encoding='utf-8-sig', dtype='string')


def preparar_entrada(
    raiz: Path, ruta: Path | None, demo: bool, obj: dict
) -> tuple[pd.DataFrame, str, Path]:
    preparado = obj['preparados']
    division = pd.read_csv(
        exigir_archivo(preparado / '00_division_cronologica.csv'),
        sep=';', encoding='utf-8-sig', usecols=[SMP, 'conjunto']
    )
    max_historico = int(division[SMP].max())

    if demo:
        metadata = cargar_json(preparado / '00_metadatos.json')
        ruta_real = Path(metadata['archivo_fuente'])
        if not ruta_real.is_file():
            ruta_real = raiz / 'Datos_corregidos' / 'datos_multisensor_corregidos.csv'
        ruta = exigir_archivo(ruta_real)
        modo = 'demostracion_prueba_historica'
    else:
        assert ruta is not None
        ruta = ruta.expanduser()
        ruta = ruta if ruta.is_absolute() else raiz / ruta
        modo = 'nuevos_datos'

    df = leer_csv(ruta)
    columnas = [SMP, *obj['variables']]
    faltan = [v for v in columnas if v not in df.columns]
    if faltan:
        raise ValueError(
            f'Faltan columnas: {faltan}\nColumnas leidas: {list(df.columns)}\n'
            'El CSV debe tener smp_continuo y los 10 nombres originales de sensores.'
        )
    if df.empty:
        raise ValueError('El CSV esta vacio: no hay registros para aplicar los modelos')

    df = df[columnas].copy()
    df[SMP] = a_numero(df[SMP])
    if df[SMP].isna().any() or (df[SMP] % 1 != 0).any():
        raise ValueError('smp_continuo debe ser numerico entero y no tener ausentes')
    df[SMP] = df[SMP].astype('int64')
    if df[SMP].duplicated().any():
        raise ValueError('Hay SMP continuos duplicados en el CSV')
    for v in obj['variables']:
        df[v] = a_numero(df[v])
    df = df.sort_values(SMP, kind='mergesort').reset_index(drop=True)

    if demo:
        esperados = division.loc[division['conjunto'] == 'prueba', SMP].to_numpy(np.int64)
        df = df.loc[df[SMP].isin(esperados)].copy().reset_index(drop=True)
        if not np.array_equal(df[SMP].to_numpy(np.int64), esperados):
            raise ValueError('No fue posible reconstruir exactamente los registros de prueba historica')
    elif (df[SMP] <= max_historico).any():
        minimo = int(df[SMP].min())
        raise ValueError(
            f'Estos datos no son posteriores al entrenamiento/evaluacion historica. '
            f'SMP minimo del CSV = {minimo}, SMP historico final = {max_historico}. '
            'Para demostrar el sistema con la prueba historica usa --demo-prueba; '
            'para mediciones genuinamente nuevas proporciona SMP continuos posteriores.'
        )

    if df[obj['variables']].isna().all(axis=1).any():
        n = int(df[obj['variables']].isna().all(axis=1).sum())
        raise ValueError(
            f'Hay {n} filas sin ninguna medicion fisica; no se generan puntuaciones para ellas.'
        )
    return df, modo, ruta


def obtener_alertas(df: pd.DataFrame, obj: dict, tamano_lote: int) -> tuple[pd.DataFrame, np.ndarray]:
    variables = obj['variables']
    faltantes = df[variables].isna().sum(axis=1).to_numpy(int)
    x_original = df[variables].to_numpy(float)
    prep = obj['preprocesador']
    x = prep['escalador'].transform(prep['imputador'].transform(x_original))
    if x.shape[1] != 10 or not np.isfinite(x).all():
        raise ValueError('El preprocesamiento produjo valores invalidos')

    ifscore = -obj['iforest'].score_samples(x)
    ifalerta = ifscore >= obj['umbral_if']
    kmclusters = obj['kmeans'].predict(x)
    kmcentros = obj['kmeans'].cluster_centers_[kmclusters]
    kmdist = np.linalg.norm(x - kmcentros, axis=1)
    kmalerta = kmdist >= obj['umbral_km']

    db = obj['dbscan']
    dbdist = np.empty(len(x), dtype=float)
    dbclusters = np.empty(len(x), dtype=np.int64)
    for a in range(0, len(x), tamano_lote):
        b = min(a + tamano_lote, len(x))
        dist, indice = db['vecinos_core'].kneighbors(x[a:b], n_neighbors=1)
        dbdist[a:b] = dist[:, 0]
        dbclusters[a:b] = db['etiquetas_nucleos'][indice[:, 0]]
    dbalerta = dbdist > float(db['eps'])
    dbclusters[dbalerta] = -1
    votos = ifalerta.astype(int) + kmalerta.astype(int) + dbalerta.astype(int)
    prioridad = np.select(
        [votos == 3, ifalerta & (votos == 2), ifalerta & (votos == 1),
         ~ifalerta & (votos == 2), ~ifalerta & (votos == 1)],
        ['alta', 'media', 'revisar_if', 'revision_complementaria', 'alerta_aislada'],
        default='sin_alerta',
    )
    result = df.copy()
    result['nulos_entrada_imputados_para_modelo'] = faltantes
    result['puntaje_if'] = ifscore
    result['umbral_if'] = obj['umbral_if']
    result['alerta_if'] = ifalerta.astype(int)
    result['cluster_kmeans'] = kmclusters
    result['distancia_kmeans'] = kmdist
    result['umbral_kmeans'] = obj['umbral_km']
    result['alerta_kmeans'] = kmalerta.astype(int)
    result['cluster_dbscan_aproximado'] = dbclusters
    result['distancia_nucleo_dbscan'] = dbdist
    result['eps_dbscan'] = float(db['eps'])
    result['ruido_dbscan_aproximado'] = dbalerta.astype(int)
    result['votos_alerta'] = votos
    result['consenso_2_o_3'] = (votos >= 2).astype(int)
    result['prioridad_revision_propuesta'] = prioridad
    return result, x


def episodios(df: pd.DataFrame, columna: str, nombre: str) -> pd.DataFrame:
    flags = df[columna].to_numpy(bool)
    smp = df[SMP].to_numpy(np.int64)
    ini = np.flatnonzero(flags & np.r_[True, ~flags[:-1] | (np.diff(smp) != 1)])
    fin = np.flatnonzero(flags & np.r_[~flags[1:] | (np.diff(smp) != 1), True])
    salida = []
    for i, (a, b) in enumerate(zip(ini, fin), 1):
        salida.append({
            'algoritmo': nombre, 'episodio': i,
            'smp_inicial': int(smp[a]), 'smp_final': int(smp[b]),
            'longitud_en_muestras': int(b - a + 1),
            'puntaje_if_maximo': float(df['puntaje_if'].iloc[a:b + 1].max()),
        })
    return pd.DataFrame(salida, columns=[
        'algoritmo', 'episodio', 'smp_inicial', 'smp_final',
        'longitud_en_muestras', 'puntaje_if_maximo'
    ])


def explicar_shap(
    tabla: pd.DataFrame, x: np.ndarray, obj: dict,
    max_shap: int, fondo: int, carpeta: Path
) -> tuple[int, float | None]:
    if max_shap == 0:
        return 0, None
    ids = np.flatnonzero(tabla['alerta_if'].to_numpy(bool))
    if len(ids) == 0:
        pd.DataFrame(columns=[SMP, 'sin_alertas_if_para_explicar']).to_csv(
            carpeta / '04_shap_alertas.csv', sep=';', index=False, encoding='utf-8-sig'
        )
        return 0, None
    ranking = ids[np.argsort(-tabla['puntaje_if'].to_numpy()[ids])]
    seleccion = ranking[:max_shap]
    try:
        import shap
    except ImportError as ex:
        raise RuntimeError('Falta SHAP: ejecuta python -m pip install shap') from ex
    datos_fondo = pd.read_csv(
        exigir_archivo(obj['preparados'] / '01_entrenamiento_escalado.csv'),
        sep=';', encoding='utf-8-sig'
    )
    if list(datos_fondo.columns) != [SMP, *obj['vars_z']]:
        raise ValueError('Fondo SHAP no coincide con columnas originales de entrenamiento')
    rng = np.random.default_rng(SEMILLA)
    ids_bg = rng.choice(len(datos_fondo), size=min(fondo, len(datos_fondo)), replace=False)
    bg = datos_fondo.iloc[ids_bg][obj['vars_z']].to_numpy(float)

    def score(matriz: np.ndarray) -> np.ndarray:
        return -obj['iforest'].score_samples(np.asarray(matriz, dtype=float))

    mascara = shap.maskers.Independent(bg, max_samples=len(bg))
    explicador = shap.Explainer(
        score, mascara, algorithm='permutation',
        feature_names=[CORTOS.get(v, v) for v in obj['variables']], seed=SEMILLA
    )
    exp = explicador(x[seleccion], max_evals=2 * x.shape[1] + 1, batch_size=64, silent=True)
    valores = np.asarray(exp.values, float)
    bases = np.broadcast_to(np.asarray(exp.base_values, float).reshape(-1), len(seleccion))
    predicho = tabla['puntaje_if'].to_numpy()[seleccion]
    rec = bases + valores.sum(axis=1)
    error = np.abs(rec - predicho)
    etiquetas = [CORTOS.get(v, v) for v in obj['variables']]
    filas = tabla.iloc[seleccion][[SMP, 'puntaje_if', 'votos_alerta', 'prioridad_revision_propuesta']].copy()
    filas = filas.reset_index(drop=True)
    filas['base_shap'] = bases
    filas['reconstruccion_shap'] = rec
    filas['error_reconstruccion'] = error
    for j, nombre in enumerate(etiquetas):
        filas['shap_' + nombre.replace(' ', '_')] = valores[:, j]
    filas['tres_aportes_positivos_mayores'] = [
        ', '.join(etiquetas[j] for j in np.argsort(-v) if v[j] > 0)[:300]
        for v in valores
    ]
    filas.to_csv(carpeta / '04_shap_alertas.csv', sep=';', index=False,
                 encoding='utf-8-sig', float_format='%.10g')
    importancia = pd.DataFrame({
        'variable': etiquetas,
        'promedio_abs_shap_alertas_seleccionadas': np.abs(valores).mean(axis=0),
    }).sort_values('promedio_abs_shap_alertas_seleccionadas', ascending=False)
    importancia.to_csv(carpeta / '05_importancia_shap.csv', sep=';', index=False,
                       encoding='utf-8-sig', float_format='%.10g')
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    v = importancia.sort_values('promedio_abs_shap_alertas_seleccionadas')
    ax.barh(v['variable'], v['promedio_abs_shap_alertas_seleccionadas'], color='#3c729b')
    ax.set(xlabel='Promedio de |SHAP| (puntuacion IF)',
           title='SHAP | Solo alertas seleccionadas de Isolation Forest')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(carpeta / '04_importancia_shap.png', dpi=250)
    plt.close(fig)
    return int(len(seleccion)), float(error.max())


def graficar_alertas(tabla: pd.DataFrame, salida: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    x = tabla[SMP]
    y = tabla['puntaje_if']
    flag = tabla['alerta_if'].to_numpy(bool)
    ax.plot(x, y, linewidth=.9, color='#4a6477', label='Puntuacion IF')
    if flag.any():
        ax.scatter(x[flag], y[flag], s=12, color='#b43f59', zorder=4, label='Alerta IF')
    ax.axhline(tabla['umbral_if'].iloc[0], color='#2d3439', linestyle='--',
               label='Umbral de entrenamiento')
    ax.set(xlabel='SMP continuo (NO segundos)', ylabel='Puntuacion -score_samples',
           title='Aplicacion a mediciones: Isolation Forest')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida / '01_grafico_puntuacion_if.png', dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    etiquetas = ['Isolation Forest', 'K-means', 'DBSCAN aprox.', 'Consenso >=2']
    datos = [int(tabla[c].sum()) for c in (
        'alerta_if', 'alerta_kmeans', 'ruido_dbscan_aproximado', 'consenso_2_o_3'
    )]
    bars = ax.bar(etiquetas, datos, color=['#3d749b', '#9b5278', '#b28742', '#668965'])
    ax.bar_label(bars, padding=3)
    ax.set(ylabel='Registros señalados', title='Alertas de los modelos en este CSV')
    ax.tick_params(axis='x', rotation=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida / '02_comparacion_alertas.png', dpi=250)
    plt.close(fig)


def ejecutar(raiz: Path, entrada: Path | None, demo: bool, salida: Path,
             max_shap: int, fondo: int, tamano_lote: int) -> None:
    if max_shap < 0 or fondo < 1 or tamano_lote < 1:
        raise ValueError('max-shap >=0, fondo >=1, lote >=1 son obligatorios')
    modelos = cargar_artefactos(raiz)
    datos, modo, ruta_real = preparar_entrada(raiz, entrada, demo, modelos)
    tabla, x = obtener_alertas(datos, modelos, tamano_lote)
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / '01_resultados_por_medicion.csv', sep=';', index=False,
                 encoding='utf-8-sig', float_format='%.10g')
    resumen_episodios = {}
    especificacion = [
        ('alerta_if', 'isolation_forest', '02_episodios_isolation_forest.csv'),
        ('alerta_kmeans', 'kmeans', '02_episodios_kmeans.csv'),
        ('ruido_dbscan_aproximado', 'dbscan_aproximado', '02_episodios_dbscan.csv'),
        ('consenso_2_o_3', 'consenso_2_o_3', '02_episodios_consenso.csv'),
    ]
    for columna, nombre, archivo in especificacion:
        e = episodios(tabla, columna, nombre)
        e.to_csv(salida / archivo, sep=';', index=False, encoding='utf-8-sig')
        resumen_episodios[nombre] = len(e)
    graficar_alertas(tabla, salida)
    n_explicadas, error = explicar_shap(tabla, x, modelos, max_shap, fondo, salida)
    resumen = {
        'modo': modo, 'archivo_entrada': str(ruta_real.resolve()),
        'registros': int(len(tabla)), 'smp_min': int(tabla[SMP].min()),
        'smp_max': int(tabla[SMP].max()),
        'alertas_if': int(tabla['alerta_if'].sum()),
        'alertas_kmeans': int(tabla['alerta_kmeans'].sum()),
        'ruido_dbscan_aproximado': int(tabla['ruido_dbscan_aproximado'].sum()),
        'consenso_2_o_3': int(tabla['consenso_2_o_3'].sum()),
        'consenso_los_3': int((tabla['votos_alerta'] == 3).sum()),
        'episodios_if': resumen_episodios['isolation_forest'],
        'episodios_kmeans': resumen_episodios['kmeans'],
        'episodios_dbscan_aproximado': resumen_episodios['dbscan_aproximado'],
        'episodios_consenso': resumen_episodios['consenso_2_o_3'],
        'filas_con_nulos_nuevos_imputados': int((tabla['nulos_entrada_imputados_para_modelo'] > 0).sum()),
        'celdas_nuevas_imputadas': int(tabla['nulos_entrada_imputados_para_modelo'].sum()),
        'alertas_if_explicadas_shap': int(n_explicadas),
        'error_max_reconstruccion_shap': error,
    }
    pd.DataFrame([resumen]).to_csv(
        salida / '00_resumen_aplicacion.csv', sep=';', index=False, encoding='utf-8-sig'
    )
    limitaciones = {
        'modo': modo,
        'entrenamiento_nuevo': False,
        'recalibracion_umbrales': False,
        'filtro_nuevos': 'SMP continuo estrictamente posterior al ultimo historico',
        'isolation_forest': 'score_samples transformado a puntuacion positiva; umbral de entrenamiento',
        'kmeans': 'asignacion nativa a centroides + regla auxiliar de distancia',
        'dbscan': 'extension aproximada a nuevas filas, nucleo mas cercano dentro de eps; NO predict nativo',
        'shap': 'solo contribuciones al puntaje Isolation Forest de alertas seleccionadas, NO detector adicional',
        'episodios': 'filas alertadas consecutivas por SMP (NO tiempo fisico)',
        'imputaciones': 'solo nulos presentes en el CSV de entrada; NO cuenta reemplazos historicos de Tarea 2 ya incorporados',
        'evaluacion': ('demostracion retrospectiva de prueba ya examinada, NO validacion nueva'
                       if demo else 'aplicacion a registros declarados posteriores; sin etiquetas no se conoce precision'),
        'interpretacion': 'alertas estadisticas para revision humana; NO fallas confirmadas ni orden de mantenimiento',
        'seguridad': 'solo cargar archivos joblib de origen confiable',
    }
    (salida / '00_limitaciones_y_metodologia.json').write_text(
        json.dumps(limitaciones, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('\nAPLICACION DE LOS TRES MODELOS Y SHAP FINALIZADA')
    print(pd.DataFrame([resumen]).to_string(index=False))
    print(f'\nArchivos generados: {salida.resolve()}')
    print('No se ajustaron modelos ni umbrales. DBSCAN es aproximado; SHAP explica IF.')
    if demo:
        print('MODO DEMO: son registros historicos de prueba, NO mediciones nuevas.')
    else:
        print('MODO NUEVOS DATOS: SMP posteriores a todo el historial conocido.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--entrada', help='Ruta a CSV con mediciones nuevas y SMP continuo posterior')
    grupo.add_argument('--demo-prueba', action='store_true',
                        help='Usar la prueba historica SOLO para verificar funcionamiento')
    parser.add_argument('--salida', help='Carpeta donde guardar el analisis')
    parser.add_argument('--max-shap', type=int, default=20,
                        help='Cuantas alertas IF explicar (0 = sin SHAP)')
    parser.add_argument('--fondo', type=int, default=30,
                        help='Cuantos registros de entrenamiento usar como fondo SHAP')
    parser.add_argument('--lote', type=int, default=2000,
                        help='Filas por lote para consultar el vecino nucleo en DBSCAN')
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    subcarpeta = ('demostracion_prueba' if args.demo_prueba else 'datos_nuevos')
    salida = (Path(args.salida).expanduser() if args.salida else
              raiz / 'Resultados_Tarea3' / '08_aplicacion_a_nuevas_mediciones' / subcarpeta)
    if not salida.is_absolute():
        salida = raiz / salida
    ejecutar(
        raiz=raiz,
        entrada=Path(args.entrada).expanduser() if args.entrada else None,
        demo=args.demo_prueba,
        salida=salida,
        max_shap=args.max_shap,
        fondo=args.fondo,
        tamano_lote=args.lote,
    )


if __name__ == '__main__':
    main()
