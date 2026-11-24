# -*- coding: utf-8 -*-
"""13 | Isolation Forest multivariable para las diez senales del aerogenerador.

Requisitos: pip install numpy pandas matplotlib scikit-learn joblib

Ejecucion desde carpeta con 08_preparacion_datos.py:
    python 13_isolation_forest.py
    python -u 13_isolation_forest.py --modo final
    python -u 13_isolation_forest.py --modo barrido
    python -u 13_isolation_forest.py --modo final --escalado robusto
    python -u 13_isolation_forest.py --modo final --umbral cuantil --fraccion 0.01
    python 13_isolation_forest.py --autotest

* Ajusta un unico modelo con las 10 senales SIMULTANEAMENTE.
* Recibe todos los registros de entrenamiento del 08 (100492); los arboles
  de Isolation Forest usan submuestras internas por definicion del algoritmo.
  No equivale a entrenar 10 modelos ni a descartar filas de entrada.
* Entrena SOLO con entrenamiento; evalua validacion y prueba sin reentrenar.
* El umbral automatico proviene de scikit-learn (contamination='auto').
  Los umbrales por cuantiles del entrenamiento son alternativas de sensibilidad,
  NO umbrales fisicos ni limites ISO, ni se optimizan con la prueba.
* El SMP ordena registros, NO es tiempo en segundos; no entra al modelo.
* Los colores en PCA representan alertas del MODELO, NO averias confirmadas.
* El analisis de perfiles de senales es descriptivo; para interpretacion
  individual de predicciones corresponde el posterior codigo 14 (SHAP).

Los PNG se generan con Matplotlib, sin abrir ventanas de Windows.
Por defecto guarda en Resultados_Tarea3_Nuevos/05_isolation_forest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

RAIZ = Path(__file__).resolve().parent
PARTES = ('entrenamiento', 'validacion', 'prueba')
COLORES = {'entrenamiento': '#2876A9', 'validacion': '#D58B3F', 'prueba': '#7454A4'}
SMP = 'smp_continuo'
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
SEMILLA = 42
FRACCIONES = [0.005, 0.01, 0.02, 0.05, 0.10]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.16,
})


def guardar_csv(datos, ruta: Path):
    pd.DataFrame(datos).to_csv(ruta, sep=';', index=False, encoding='utf-8-sig',
                                float_format='%.12g')


def guardar_fig(fig, ruta: Path):
    fig.savefig(ruta, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  [GRAFICO]', ruta.name, flush=True)


def leer_preparacion(preparados: Path, escalado: str):
    ruta_meta = preparados / '00_metadatos.json'
    if not ruta_meta.is_file():
        raise FileNotFoundError(f'No existe {ruta_meta}. Ejecuta 08_preparacion_datos.py')
    meta = json.loads(ruta_meta.read_text(encoding='utf-8'))
    nombres = meta.get('variables', [])
    if len(nombres) != 10 or len(set(nombres)) != 10:
        raise ValueError('El codigo 08 debe contener EXACTAMENTE diez senales diferentes.')
    ruta_div = preparados / '00_division_cronologica.csv'
    if not ruta_div.is_file():
        raise FileNotFoundError(f'Falta {ruta_div}. Ejecuta 08_preparacion_datos.py')
    div = pd.read_csv(ruta_div, sep=';', encoding='utf-8-sig', usecols=[SMP, 'conjunto'])
    if div[SMP].duplicated().any() or not div[SMP].is_monotonic_increasing:
        raise ValueError('La division por SMP tiene duplicados o no esta ordenada.')
    datos = {}
    esperadas = [SMP, *[f'z_{n}' for n in nombres]]
    for parte in PARTES:
        ruta = preparados / f'01_{parte}_{escalado}.csv'
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {ruta}. Ejecuta 08_preparacion_datos.py')
        df = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
        if list(df.columns) != esperadas:
            raise ValueError(f'Columnas incorrectas o desordenadas en {ruta.name}')
        smp = df[SMP].to_numpy(dtype=np.int64)
        smp_referencia = div.loc[div['conjunto'] == parte, SMP].to_numpy(dtype=np.int64)
        if not np.array_equal(smp, smp_referencia):
            raise ValueError(f'El orden SMP no coincide con 08 en {parte}')
        X = np.ascontiguousarray(df.iloc[:, 1:].to_numpy(dtype=np.float32))
        if X.shape[1] != 10 or not np.isfinite(X).all():
            raise ValueError(f'Faltan senales o existen valores no finitos en {parte}')
        datos[parte] = {'smp': smp, 'X': X}
    if sum(len(datos[p]['smp']) for p in PARTES) != meta['registros']:
        raise ValueError('El total no coincide con el metadato del codigo 08')
    if (len(datos['entrenamiento']['smp']), len(datos['validacion']['smp']),
            len(datos['prueba']['smp'])) != (100492, 21534, 21535):
        raise ValueError('Esta base debe conservar la division 100492 / 21534 / 21535')
    if not (datos['entrenamiento']['smp'][-1] < datos['validacion']['smp'][0]
            < datos['prueba']['smp'][0]):
        raise ValueError('Se rompio el orden cronologico por SMP')
    return datos, meta


def entrenar(X: np.ndarray, n_estimators: int, max_samples: int,
             max_features: float, trabajadores: int, semilla: int):
    print(f'[ENTRENAMIENTO] {len(X):,} filas x {X.shape[1]} variables, '
          f'{n_estimators} arboles; max_samples={max_samples}; '
          f'max_features={max_features}; semilla={semilla}', flush=True)
    print('  IMPORTANTE: cada arbol usa submuestras internas, '
          'pero se entrega el 100% del entrenamiento al modelo.', flush=True)
    t0 = time.perf_counter()
    modelo = IsolationForest(
        n_estimators=n_estimators, max_samples=max_samples,
        max_features=max_features, contamination='auto',
        random_state=semilla, n_jobs=trabajadores,
    )
    modelo.fit(X)
    duracion = time.perf_counter() - t0
    print(f'  Ajuste completo: {duracion:.2f} segundos; '
          f'submuestras por arbol={modelo.max_samples_}.', flush=True)
    return modelo, duracion


def puntuar(modelo, datos):
    puntajes = {}
    for parte in PARTES:
        t0 = time.perf_counter()
        # score_samples: puntuacion menor = mas atipico.
        # Invertimos el signo: mayor = mas atipico.
        p = -np.asarray(modelo.score_samples(datos[parte]['X']), dtype=np.float64)
        if len(p) != len(datos[parte]['smp']) or not np.isfinite(p).all():
            raise ValueError(f'Puntajes invalidos para {parte}')
        puntajes[parte] = p
        print(f'[PUNTUACION] {parte}: {len(p):,} muestras, '
              f'{time.perf_counter()-t0:.1f} s.', flush=True)
    return puntajes


def umbral_referencia(modelo, puntajes_entreno, metodo: str, fraccion: float):
    if metodo == 'auto':
        # decision_function = score_samples - offset_. Es anomalo si < 0.
        # -score_samples >= -offset_ (empates exactos tratados como alerta).
        return float(-modelo.offset_)
    if metodo == 'cuantil':
        if not 0.0 < fraccion < 0.5:
            raise ValueError('La fraccion del cuantil debe estar entre 0 y 0.5.')
        return float(np.quantile(puntajes_entreno, 1.0-fraccion))
    raise ValueError(f'Metodo de umbral desconocido: {metodo}')


def obtener_episodios(smp, scores, alertas):
    """Secuencias donde hay alertas y SMP realmente consecutivos; sin segundos."""
    activos = np.flatnonzero(alertas)
    columnas = ['episodio', 'smp_inicial', 'smp_final', 'longitud_muestras',
                'puntaje_maximo', 'puntaje_promedio']
    if not len(activos):
        return pd.DataFrame(columns=columnas)
    # Cortar donde dos alertas sucesivas no tienen SMP contiguo.
    cortes = np.r_[0, 1 + np.flatnonzero(np.diff(smp[activos]) != 1), len(activos)]
    resultados = []
    for i, (ini, fin) in enumerate(zip(cortes[:-1], cortes[1:]), 1):
        indices = activos[ini:fin]
        resultados.append((i, int(smp[indices[0]]), int(smp[indices[-1]]),
                           int(len(indices)), float(np.max(scores[indices])),
                           float(np.mean(scores[indices]))))
    return pd.DataFrame(resultados, columns=columnas)


def calcular_resultados(datos, puntajes, umbral, modelo, carpeta):
    resumen, resultados, episodios = [], {}, {}
    for parte in PARTES:
        smp = datos[parte]['smp']
        score = puntajes[parte]
        alertas = score > umbral  # umbral estricto, coincide con decision_function < 0
        tabla = pd.DataFrame({
            SMP: smp, 'puntaje_anomalia': score,
            'score_samples_original': -score,
            'decision_function': -score - modelo.offset_,
            'alerta_isolation_forest': alertas.astype(np.uint8),
        })
        guardar_csv(tabla, carpeta / f'01_resultado_{parte}.csv')
        ep = obtener_episodios(smp, score, alertas)
        guardar_csv(ep, carpeta / f'02_episodios_{parte}.csv')
        resumen.append({
            'conjunto': parte, 'registros': len(score), 'alertas': int(alertas.sum()),
            'porcentaje_alertas': 100.0 * float(alertas.mean()),
            'episodios': len(ep), 'puntaje_min': float(score.min()),
            'puntaje_mediana': float(np.median(score)),
            'puntaje_p95': float(np.quantile(score, .95)),
            'puntaje_p99': float(np.quantile(score, .99)),
            'puntaje_max': float(score.max()),
        })
        resultados[parte] = alertas
        episodios[parte] = ep
    guardar_csv(resumen, carpeta / '00_resumen_isolation_forest.csv')
    return pd.DataFrame(resumen), resultados, episodios


def sensibilidad_umbrales(puntajes, modelo, carpeta):
    filas = []
    umbrales = [('auto', np.nan, -float(modelo.offset_))]
    for frac in FRACCIONES:
        umbrales.append(('cuantil_entrenamiento', frac,
                         float(np.quantile(puntajes['entrenamiento'], 1-frac))))
    for metodo, frac, umbral in umbrales:
        for parte in PARTES:
            a = puntajes[parte] > umbral
            filas.append({
                'metodo': metodo, 'fraccion_referencia': frac,
                'umbral_puntaje_mayor_es_mas_atipico': umbral,
                'conjunto': parte, 'registros': len(a),
                'alertas': int(a.sum()), 'porcentaje_alertas': 100.0*a.mean(),
            })
    tabla = pd.DataFrame(filas)
    guardar_csv(tabla, carpeta / '00_sensibilidad_umbrales.csv')
    return tabla


def tabla_top(datos, puntajes, meta, carpeta, n=100):
    cols = [SMP, 'conjunto', 'puntaje_anomalia', *[f'z_{s}' for s in meta['variables']]]
    partes = []
    for nombre in PARTES:
        p = puntajes[nombre]
        ids = np.argsort(-p, kind='stable')[:n]
        df = pd.DataFrame(datos[nombre]['X'][ids], columns=cols[3:])
        df.insert(0, 'puntaje_anomalia', p[ids])
        df.insert(0, 'conjunto', nombre)
        df.insert(0, SMP, datos[nombre]['smp'][ids])
        partes.append(df)
    todos = pd.concat(partes, ignore_index=True)
    todos = todos.sort_values('puntaje_anomalia', ascending=False, kind='stable').head(n)
    guardar_csv(todos[cols], carpeta / '03_top_100_puntajes.csv')
    return todos


def tabla_perfiles(datos, alertas, meta, carpeta):
    rows = []
    for parte in PARTES:
        X = datos[parte]['X']
        for estado, mascara in [('sin_alerta', ~alertas[parte]),
                               ('con_alerta', alertas[parte])]:
            for j, nombre in enumerate(meta['variables']):
                vals = X[mascara, j]
                rows.append({
                    'conjunto': parte, 'estado': estado, 'variable': nombre,
                    'registros': int(len(vals)),
                    'mediana_escalada': float(np.median(vals)) if len(vals) else np.nan,
                    'mediana_valor_absoluto_escalado':
                        float(np.median(np.abs(vals))) if len(vals) else np.nan,
                    'p95_valor_absoluto_escalado':
                        float(np.quantile(np.abs(vals), .95)) if len(vals) else np.nan,
                })
    df = pd.DataFrame(rows)
    guardar_csv(df, carpeta / '03_perfiles_senales.csv')
    return df


def graficos_distribuciones(puntajes, umbral, resumen, sensibilidad, carpeta):
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    todos = np.concatenate(list(puntajes.values()))
    bins = np.linspace(float(todos.min()), float(todos.max()), 100)
    for parte in PARTES:
        ax.hist(puntajes[parte], bins=bins, density=True, histtype='step',
                linewidth=1.65, label=parte.capitalize(), color=COLORES[parte])
    ax.axvline(umbral, linestyle='--', color='#B53D4C', lw=1.8,
               label=f'Umbral fijado: {umbral:.4f}')
    ax.set(title='Isolation Forest | Distribucion de puntuaciones',
           xlabel='Puntaje de atipicidad (mayor = mas inusual)', ylabel='Densidad')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '01_distribucion_puntajes.png')

    fig, ax = plt.subplots(figsize=(10.3, 5.2))
    for parte in PARTES:
        ordenados = np.sort(puntajes[parte])
        ax.plot(ordenados, np.arange(1, len(ordenados)+1)/len(ordenados),
                label=parte.capitalize(), color=COLORES[parte], lw=1.7)
    ax.axvline(umbral, c='#B53D4C', ls='--', lw=1.5, label='Umbral fijo')
    ax.set(title='Distribucion acumulada de puntajes (CDF)',
           xlabel='Puntaje de atipicidad', ylabel='Fraccion acumulada')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '02_distribucion_acumulada.png')

    fig, ax = plt.subplots(figsize=(8.9, 5))
    ax.boxplot([puntajes[p] for p in PARTES], tick_labels=[p.capitalize() for p in PARTES],
               showfliers=False, patch_artist=True,
               boxprops={'facecolor': '#dce9ef', 'edgecolor': '#466680'},
               medianprops={'color': '#A54B4B', 'linewidth': 2})
    ax.axhline(umbral, c='#B53D4C', ls='--', lw=1.5, label='Umbral')
    ax.set(title='Comparacion de puntuaciones entre conjuntos (sin extremos)',
           ylabel='Puntaje de atipicidad')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '03_boxplot_puntajes.png')

    fig, ax = plt.subplots(figsize=(9, 5.3))
    xpos = np.arange(len(PARTES))
    valores = [float(resumen.loc[resumen.conjunto == p, 'porcentaje_alertas'].iloc[0])
               for p in PARTES]
    barras = ax.bar(xpos, valores, color=[COLORES[p] for p in PARTES], width=.58)
    ax.set_xticks(xpos, [p.capitalize() for p in PARTES])
    ax.set_ylabel('Registros con alerta (%)')
    ax.set_title('Tasa de alertas con el umbral fijo del entrenamiento')
    ax.set_ylim(0, max(max(valores)*1.2, .2))
    for b, val in zip(barras, valores):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9)
    guardar_fig(fig, carpeta / '04_tasas_alerta_conjuntos.png')

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    filtrada = sensibilidad[sensibilidad.metodo == 'cuantil_entrenamiento']
    for p in PARTES:
        t = filtrada[filtrada.conjunto == p]
        ax.plot(t.fraccion_referencia.to_numpy()*100,
                t.porcentaje_alertas, marker='o', linewidth=1.7,
                color=COLORES[p], label=p.capitalize())
    ax.set(title='Sensibilidad al umbral: cuantiles calculados SOLO en entrenamiento',
           xlabel='Fraccion de referencia en entrenamiento (%)',
           ylabel='Alertas por conjunto (%)')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '05_sensibilidad_fraccion_umbrales.png')

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    t = filtrada[filtrada.conjunto == 'entrenamiento']
    ax.plot(t.fraccion_referencia.to_numpy()*100,
            t.umbral_puntaje_mayor_es_mas_atipico, marker='o', color='#2876A9')
    ax.set(title='Umbral de atipicidad vs cuota de referencia',
           xlabel='Fraccion de referencia (%)', ylabel='Umbral de puntaje (solo entrenamiento)')
    guardar_fig(fig, carpeta / '06_umbrales_entrenamiento.png')


def graficos_pca(datos, alertas, carpeta):
    print('[VISUALIZACION] PCA SOLO para proyectar, no para ajustar Isolation Forest.', flush=True)
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(datos['entrenamiento']['X'])
    proy = {p: pca.transform(datos[p]['X']) for p in PARTES}
    # Ejes identicos entre partes, evitar enganar comparacion visual.
    toda = np.concatenate(list(proy.values()), axis=0)
    rangos = []
    zoom = []
    for j in (0, 1):
        lo, hi = float(toda[:, j].min()), float(toda[:, j].max())
        m = max((hi-lo)*.035, .01)
        rangos.append((lo-m, hi+m))
        zl, zh = np.quantile(proy['entrenamiento'][:, j], [.02, .98])
        mz = max(float(zh-zl)*.06, .01)
        zoom.append((float(zl-mz), float(zh+mz)))
    for parte in PARTES:
        P = proy[parte]
        a = alertas[parte]
        for forma, limites in [('completo', rangos), ('zoom', zoom)]:
            fig, ax = plt.subplots(figsize=(10.3, 6.6))
            # Todas las filas; dibujado por capas solo para claridad.
            ax.scatter(P[~a, 0], P[~a, 1], s=3, c='#7891A1', alpha=.23,
                       edgecolors='none', rasterized=True)
            ax.scatter(P[a, 0], P[a, 1], s=7, c='#D65A52', alpha=.68,
                       edgecolors='none', rasterized=True)
            ax.set(xlim=limites[0], ylim=limites[1],
                   xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% varianza)',
                   ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% varianza)',
                   title=f'Isolation Forest | {parte.capitalize()} | PCA {forma} '
                         f'({len(P):,} filas)')
            ax.legend(handles=[
                Line2D([0], [0], marker='o', color='none', markerfacecolor='#7891A1',
                       markersize=7, label='Sin alerta'),
                Line2D([0], [0], marker='o', color='none', markerfacecolor='#D65A52',
                       markersize=7, label='Alerta estadistica (NO falla confirmada)'),
            ], frameon=False, loc='upper right')
            guardar_fig(fig, carpeta / f'07_pca_{parte}_{forma}.png')
    guardar_csv([{'pc': 'PC1', 'varianza_explicada': float(pca.explained_variance_ratio_[0])},
                 {'pc': 'PC2', 'varianza_explicada': float(pca.explained_variance_ratio_[1])}],
                carpeta / '07_pca_varianza_explicada.csv')


def graficos_smp(datos, puntajes, alertas, umbral, carpeta):
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    for p in PARTES:
        smp, s, a = datos[p]['smp'], puntajes[p], alertas[p]
        ax.plot(smp, s, color=COLORES[p], lw=.37, alpha=.49,
                label=f'{p.capitalize()}')
        if a.any():
            ax.scatter(smp[a], s[a], s=2.0, c='#D54750', alpha=.25,
                       edgecolors='none', rasterized=True)
    ax.axhline(umbral, color='#222222', ls='--', lw=1.5, label='Umbral fijo')
    for p in ('validacion', 'prueba'):
        ax.axvline(datos[p]['smp'][0], c='#777777', ls=':', lw=1)
    ax.set(title='Puntajes y alertas por SMP | todos los registros',
           xlabel='SMP continuo (contador, NO segundos)', ylabel='Puntaje de atipicidad')
    ax.legend(frameon=False, ncol=4)
    guardar_fig(fig, carpeta / '08_evolucion_puntajes_smp_completo.png')

    p = 'prueba'
    n = min(3000, len(datos[p]['smp']))
    smp, score, a = datos[p]['smp'][:n], puntajes[p][:n], alertas[p][:n]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(smp, score, c=COLORES[p], lw=.85, alpha=.80, label='Puntaje')
    ax.scatter(smp[a], score[a], c='#C8464B', s=14, label='Alertas', zorder=4)
    ax.axhline(umbral, c='#333333', ls='--', lw=1.5, label='Umbral fijo')
    ax.set(title=f'Detalle: primeras {n} muestras de prueba (zoom por registros)',
           xlabel='SMP continuo (NO segundos)', ylabel='Puntaje de atipicidad')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '09_evolucion_prueba_zoom.png')

    bloques = []
    for p in PARTES:
        smp, a = datos[p]['smp'], alertas[p]
        for i, indices in enumerate(np.array_split(np.arange(len(smp)), 20), 1):
            bloques.append({'conjunto': p, 'bloque': i,
                            'smp_inicio': int(smp[indices[0]]),
                            'smp_fin': int(smp[indices[-1]]),
                            'smp_centro': float(np.median(smp[indices])),
                            'n_registros': int(len(indices)),
                            'n_alertas': int(a[indices].sum()),
                            'porcentaje_alertas': 100*a[indices].mean()})
    guardar_csv(bloques, carpeta / '08_alertas_por_bloques_smp.csv')
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    for p in PARTES:
        b = pd.DataFrame([r for r in bloques if r['conjunto'] == p])
        ax.plot(b['smp_centro'], b['porcentaje_alertas'], marker='o', markersize=2.5,
                lw=1.55, label=p.capitalize(), c=COLORES[p])
    for p in ('validacion', 'prueba'):
        ax.axvline(datos[p]['smp'][0], c='#777777', ls=':', lw=1)
    ax.set(title='Tasa de alertas por bloques cronologicos (20 por conjunto)',
           xlabel='SMP continuo (contador)', ylabel='Alertas del bloque (%)')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '10_tasa_alertas_por_bloques.png')


def graficos_perfiles(datos, alertas, perfiles, top, episodios, carpeta):
    perfiles = perfiles[perfiles.conjunto == 'prueba']
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    pos = np.arange(10)
    for i, (estado, nombre, color) in enumerate([
        ('sin_alerta', 'Sin alerta', '#86A2B8'),
        ('con_alerta', 'Con alerta', '#D65A52'),
    ]):
        t = perfiles[perfiles.estado == estado]
        if len(t) == 10 and t['mediana_valor_absoluto_escalado'].notna().all():
            ax.bar(pos+(i-.5)*.39, t['mediana_valor_absoluto_escalado'].to_numpy(),
                   width=.39, color=color, label=nombre)
    ax.set_xticks(pos, CORTOS)
    ax.set(ylabel='Mediana de |senal escalada| (unidad del escalador)',
           title='Perfil descriptivo por senal | prueba (NO atribucion causal)')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '11_perfil_descriptivo_senales.png')

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    Z = top.iloc[:25, 3:13].to_numpy(float)
    im = ax.imshow(np.clip(Z, -4, 4), aspect='auto', cmap='coolwarm',
                   vmin=-4, vmax=4, interpolation='nearest')
    ax.set_xticks(np.arange(10), CORTOS)
    ax.set_yticks(np.arange(len(Z)), [f'{p[:3]} / SMP {s}' for p, s in
                                 zip(top['conjunto'].iloc[:25], top[SMP].iloc[:25])], fontsize=8)
    ax.set_title('25 puntajes mas altos | senales escaladas (colores truncados en +/- 4)')
    cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
    cb.set_label('Valor escalado, no unidades fisicas')
    guardar_fig(fig, carpeta / '12_mapa_calor_top25.png')

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for p in PARTES:
        eps = episodios[p]
        if len(eps):
            tamanos = eps['longitud_muestras'].to_numpy(dtype=float)
            ax.hist(np.clip(tamanos, 1, 30), bins=np.arange(.5, 31.5, 1),
                    histtype='step', lw=1.6, color=COLORES[p], label=p.capitalize())
    ax.set(title='Longitud de episodios de alertas por SMP consecutivo',
           xlabel='Numero de muestras consecutivas por episodio (>=30 se muestra en 30)',
           ylabel='Numero de episodios')
    ax.legend(frameon=False)
    guardar_fig(fig, carpeta / '13_duracion_episodios_muestras.png')


def modo_final(datos, meta, carpeta, args):
    modelo, ajuste_seg = entrenar(datos['entrenamiento']['X'], args.arboles,
                                  args.max_samples, args.max_features,
                                  args.trabajadores, args.semilla)
    puntajes = puntuar(modelo, datos)
    umbral = umbral_referencia(modelo, puntajes['entrenamiento'], args.umbral, args.fraccion)
    print(f'[UMBRAL] Metodo={args.umbral}; valor={umbral:.8f}; '
          f'(mayor puntaje = mas atipico)', flush=True)
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen, alertas, episodios = calcular_resultados(datos, puntajes, umbral, modelo, carpeta)
    sensibilidad = sensibilidad_umbrales(puntajes, modelo, carpeta)
    top = tabla_top(datos, puntajes, meta, carpeta)
    perfiles = tabla_perfiles(datos, alertas, meta, carpeta)
    joblib.dump(modelo, carpeta / '00_modelo_isolation_forest.joblib', compress=3)
    configuracion = {
        'algoritmo': 'IsolationForest', 'version': 1,
        'variables': meta['variables'], 'n_variables': len(meta['variables']),
        'division': {p: int(len(datos[p]['smp'])) for p in PARTES},
        'entrenado_unicamente_en': 'entrenamiento cronologico',
        'escalado': args.escalado, 'n_estimators': args.arboles,
        'max_samples_pedido': args.max_samples, 'max_samples_real_por_arbol': int(modelo.max_samples_),
        'max_features': args.max_features, 'semilla': args.semilla,
        'contamination_modelo': 'auto', 'umbral_metodo': args.umbral,
        'fraccion_referencia_solo_si_cuantil': args.fraccion if args.umbral == 'cuantil' else None,
        'umbral_score_mayor_mas_atipico': umbral,
        'offset_scikit_learn': float(modelo.offset_),
        'segundos_ajuste': float(ajuste_seg),
        'smp_no_predictor': True, 'pca_solo_visualizacion': True,
        'sin_etiquetas_de_falla': True,
        'advertencia': 'Alerta NO significa falla real ni necesidad de mantenimiento; no hay ground truth.',
        'aclaracion_submuestras': 'Todos los registros se pasan a fit; cada arbol usa max_samples aleatorios internos.',
    }
    (carpeta / '00_configuracion_modelo.json').write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding='utf-8')
    graficos_distribuciones(puntajes, umbral, resumen, sensibilidad, carpeta)
    graficos_pca(datos, alertas, carpeta)
    graficos_smp(datos, puntajes, alertas, umbral, carpeta)
    graficos_perfiles(datos, alertas, perfiles, top, episodios, carpeta)
    (carpeta / '00_LEER_RESULTADOS.txt').write_text(
        '13 - ISOLATION FOREST MULTIVARIABLE\n'
        'Se entreno UN SOLO modelo con las 10 senales juntas.\n'
        'Se usan 100492 registros de entrenamiento, 21534 de validacion y 21535 de prueba.\n'
        'Los arboles de Isolation Forest usan submuestras internas (max_samples).\n'
        'Umbral fijo del entrenamiento o umbral automatico del metodo, segun configuracion.\n'
        'El archivo de sensibilidad muestra que las alertas dependen del umbral.\n'
        'No hay etiquetas de falla; NO se puede calcular exactitud real, recall, precision ni F1.\n'
        'SMP ordena adquisiciones, NO equivale a segundos.\n'
        'PCA solo proyecta los datos para visualizar; NO es entrada del modelo.\n'
        'Perfil de senales es descriptivo, NO causalidad ni explicacion SHAP.\n',
        encoding='utf-8')
    print('\n=== RESUMEN ISOLATION FOREST ===', flush=True)
    print(resumen[['conjunto', 'registros', 'alertas', 'porcentaje_alertas', 'episodios']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)
    print(f'\nResultados: {carpeta}', flush=True)
    print('IMPORTANTE: candidatas a anomalia, NO fallas confirmadas.', flush=True)


def modo_barrido(datos, carpeta, args):
    """Estabilidad/sensibilidad computacional, no seleccion de mejor accuracy sin etiquetas."""
    # Combinaciones elegidas para cambiar por separado dos hiperparametros.
    configs = [(100, 256), (300, 256), (300, 512), (300, 1024), (500, 256)]
    X = datos['entrenamiento']['X']
    filas = []
    base = None
    for i, (arboles, muestras) in enumerate(configs, 1):
        print(f'\n[BARRIDO {i}/{len(configs)}] {arboles} arboles; '
              f'{muestras} muestras/arbol', flush=True)
        modelo, dt = entrenar(X, arboles, muestras, 1.0, args.trabajadores, args.semilla)
        s = -modelo.score_samples(X)
        if base is None:
            base = s.copy()
        # Comparacion relativa de rankings contra la referencia; no accuracy.
        ranks_base = pd.Series(base).rank(method='average').to_numpy(float)
        ranks_act = pd.Series(s).rank(method='average').to_numpy(float)
        rho = float(np.corrcoef(ranks_base, ranks_act)[0, 1])
        q_base = np.quantile(base, .99)
        q = np.quantile(s, .99)
        A, B = base > q_base, s > q
        union = int((A | B).sum())
        jacc = float((A & B).sum()/union) if union else 1.0
        fila = {
            'n_estimators': arboles, 'max_samples_arbol': muestras,
            'n_variables': 10, 'registros_entrenamiento': len(X),
            'duracion_fit_s': dt, 'correlacion_rank_vs_100_256': rho,
            'jaccard_top_1pct_vs_100_256': jacc,
            'alertas_umbral_auto_entrenamiento': int((s > -modelo.offset_).sum()),
            'porcentaje_alertas_umbral_auto': float(100*(s > -modelo.offset_).mean()),
        }
        filas.append(fila)
        carpeta.mkdir(parents=True, exist_ok=True)
        guardar_csv(filas, carpeta / '00_barrido_isolation_forest.csv')
        print(f'  Spearman frente a 100/256: {rho:.4f}; Jaccard top1%: {jacc:.4f}', flush=True)
    tabla = pd.DataFrame(filas)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7))
    etiquetas = [f"{a}/{m}" for a, m in zip(tabla.n_estimators, tabla.max_samples_arbol)]
    axs[0].plot(etiquetas, tabla['correlacion_rank_vs_100_256'], marker='o', color='#2876A9')
    axs[0].set(title='Estabilidad de ranking de puntajes', xlabel='Arboles / muestras por arbol',
               ylabel='Correlacion de rangos vs 100/256', ylim=(0, 1.02))
    axs[1].plot(etiquetas, tabla['jaccard_top_1pct_vs_100_256'], marker='o', color='#D58B3F')
    axs[1].set(title='Concordancia del 1% mas atipico', xlabel='Arboles / muestras por arbol',
               ylabel='Jaccard vs 100/256', ylim=(0, 1.02))
    fig.suptitle('Barrido: consistencia sin etiquetas, NO precision diagnostica')
    guardar_fig(fig, carpeta / '00_barrido_estabilidad.png')
    print('\nBARRIDO TERMINADO. No elijas parametros por numero de fallas: no hay etiquetas.', flush=True)
    print(f'Resultados: {carpeta}', flush=True)


def autotest():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(450, 10)).astype(np.float32)
    model, _ = entrenar(X, 20, 64, 1.0, 1, 42)
    scores = -model.score_samples(X)
    assert np.isfinite(scores).all()
    threshold = umbral_referencia(model, scores, 'auto', .01)
    assert np.array_equal(scores > threshold, model.predict(X) == -1)
    smp = np.array([1, 2, 4, 5, 6, 8, 9])
    a = np.array([True, True, True, False, True, True, True])
    ep = obtener_episodios(smp, np.arange(7, dtype=float), a)
    assert ep['longitud_muestras'].tolist() == [2, 1, 1, 2]
    print('[AUTOTEST] OK: signos/umbral equivalentes a predict; episodios por SMP correctos.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preparados', type=Path,
                        default=RAIZ/'Resultados_Tarea3_Nuevos'/'00_preparacion')
    parser.add_argument('--salida', type=Path, default=None,
                        help='Por defecto 05_isolation_forest; variantes robusto/cuantil en subcarpeta para no sobreescribir estandar-auto')
    parser.add_argument('--modo', choices=['final', 'barrido', 'comprobar'], default='final')
    parser.add_argument('--escalado', choices=['estandar', 'robusto'], default='estandar')
    parser.add_argument('--umbral', choices=['auto', 'cuantil'], default='auto',
                        help='auto: referencia sklearn; cuantil: politica sobre entrenamiento')
    parser.add_argument('--fraccion', type=float, default=.01,
                        help='solo se aplica cuando --umbral cuantil')
    parser.add_argument('--arboles', type=int, default=300)
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--max-features', type=float, default=1.0)
    parser.add_argument('--trabajadores', type=int, default=-1)
    parser.add_argument('--semilla', type=int, default=SEMILLA)
    parser.add_argument('--autotest', action='store_true')
    args = parser.parse_args()
    if args.autotest:
        autotest()
        return
    if args.arboles < 1 or args.max_samples < 2 or not 0 < args.max_features <= 1:
        parser.error('Se requiere arboles>=1, max-samples>=2, 0<max-features<=1')
    if args.umbral == 'cuantil' and not 0 < args.fraccion < .5:
        parser.error('0 < --fraccion < .5')
    if args.salida is None:
        base = RAIZ/'Resultados_Tarea3_Nuevos'/'05_isolation_forest'
        if args.escalado == 'estandar' and args.umbral == 'auto':
            args.salida = base
        else:
            subcarpeta = f'{args.escalado}_{args.umbral}'
            if args.umbral == 'cuantil':
                subcarpeta += f'_{args.fraccion:.4f}'.replace('.', 'p')
            args.salida = base/subcarpeta
    print('[LECTURA] Datos preparados por 08: 10 senales juntas.', flush=True)
    datos, meta = leer_preparacion(args.preparados, args.escalado)
    for p in PARTES:
        print(f'  {p}: {len(datos[p]["smp"]):,} registros', flush=True)
    if args.modo == 'comprobar':
        print('[COMPROBACION OK] Puedes ejecutar --modo final o --modo barrido.', flush=True)
    elif args.modo == 'barrido':
        modo_barrido(datos, args.salida, args)
    else:
        modo_final(datos, meta, args.salida, args)


if __name__ == '__main__':
    main()
