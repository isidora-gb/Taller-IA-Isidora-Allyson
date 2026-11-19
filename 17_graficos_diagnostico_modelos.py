# -*- coding: utf-8 -*-
"""TAREA 3 - Visualizaciones complementarias de K-means, DBSCAN, Isolation Forest y SHAP.

No reentrena algoritmos de deteccion ni reajusta umbrales. Unicamente
ajusta PCA sobre entrenamiento para VISUALIZAR las etiquetas en 2D.

Requiere haber ejecutado los scripts 08, 09, 10, 11 y 12.
Uso desde la carpeta principal del proyecto:
    python 17_graficos_diagnostico_modelos.py
    python 17_graficos_diagnostico_modelos.py --max-puntos 3500

Dependencias: numpy, pandas, matplotlib, scikit-learn, joblib.
(SHAP no se vuelve a calcular: se leen las tablas guardadas por el script 12.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SMP = 'smp_continuo'
PARTES = ('entrenamiento', 'validacion', 'prueba')
ETIQUETAS = {'entrenamiento': 'Entrenamiento', 'validacion': 'Validación', 'prueba': 'Prueba'}
AZUL = '#346B8C'
ROJO = '#B53D57'
MORADO = '#7953A9'
VERDE = '#34947F'
AMARILLO = '#DFA84A'
GRIS = '#ACB7C3'


def leer(ruta: Path, columnas: list[str] | None = None) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f'Archivo necesario no encontrado: {ruta}\n'
                                'Ejecuta antes los códigos 08, 09, 10, 11 y 12.')
    df = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
    if columnas is not None:
        ausentes = sorted(set(columnas) - set(df.columns))
        if ausentes:
            raise ValueError(f'Faltan columnas {ausentes} en {ruta}')
    return df


def guardar(fig: plt.Figure, ruta: Path) -> None:
    fig.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  +', ruta.name)


def estilo() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.grid': False,
        'savefig.facecolor': 'white',
    })


def muestra_ids(tabla: pd.DataFrame, max_puntos: int, semilla: int = 42) -> np.ndarray:
    """Conserva una muestra reproducible de alertas y no alertas; solo para graficar."""
    n = len(tabla)
    if n <= max_puntos:
        return np.arange(n)
    rng = np.random.default_rng(semilla)
    mask = (tabla[['if_alerta', 'km_alerta', 'db_ruido']].to_numpy(int).sum(axis=1) > 0)
    raros, habituales = np.flatnonzero(mask), np.flatnonzero(~mask)
    n_raros = min(len(raros), max(1, int(max_puntos * 0.50)))
    n_habituales = min(len(habituales), max_puntos - n_raros)
    if n_raros + n_habituales < max_puntos:
        n_raros = min(len(raros), max_puntos - n_habituales)
    elegidos = np.r_[rng.choice(raros, size=n_raros, replace=False),
                    rng.choice(habituales, size=n_habituales, replace=False)]
    return np.sort(elegidos)


def fusionar(prep: pd.DataFrame, resultados: Path, parte: str) -> pd.DataFrame:
    a = leer(resultados / '01_isolation_forest' / f'01_resultado_{parte}.csv',
             [SMP, 'puntaje_anomalia', 'alerta_isolation_forest'])
    b = leer(resultados / '02_kmeans' / f'01_resultado_{parte}.csv',
             [SMP, 'cluster', 'distancia_al_centroide', 'alerta_distancia_kmeans'])
    c = leer(resultados / '03_dbscan' / f'01_resultado_{parte}.csv',
             [SMP, 'cluster_dbscan_aproximado', 'distancia_nucleo_mas_cercano', 'alerta_ruido_dbscan'])
    tabla = prep.merge(a, on=SMP, how='inner', validate='one_to_one')
    tabla = tabla.merge(b, on=SMP, how='inner', validate='one_to_one')
    tabla = tabla.merge(c, on=SMP, how='inner', validate='one_to_one')
    if len(tabla) != len(prep):
        raise ValueError(f'Los SMP de resultados no coinciden exactamente en {parte}')
    tabla = tabla.rename(columns={'alerta_isolation_forest': 'if_alerta',
                                  'alerta_distancia_kmeans': 'km_alerta',
                                  'alerta_ruido_dbscan': 'db_ruido'})
    tabla['consenso'] = tabla[['if_alerta', 'km_alerta', 'db_ruido']].astype(int).sum(axis=1)
    return tabla


def pca_scatter(ax, df: pd.DataFrame, columna: str, titulo: str,
                etiquetas: dict | None = None, colores: dict | None = None,
                s: float = 6.0) -> None:
    etiquetas = etiquetas or {}
    colores = colores or {}
    valores = sorted(pd.unique(df[columna]))
    for valor in valores:
        subset = df.loc[df[columna] == valor]
        ax.scatter(subset['PC1'], subset['PC2'], s=s,
                   alpha=0.42 if len(subset) > 600 else 0.67,
                   color=colores.get(valor, None),
                   label=f'{etiquetas.get(valor, str(valor))} (n={len(subset)})',
                   rasterized=True, linewidths=0)
    ax.set(xlabel='PC1 (visualización)', ylabel='PC2 (visualización)', title=titulo)
    ax.legend(loc='best', fontsize=8, frameon=False, markerscale=1.75)


def graficos_kmeans(resultados: Path, salida: Path, datos: pd.DataFrame,
                    modelo: object, pca: PCA) -> None:
    tabla = leer(resultados / '02_kmeans' / '00_seleccion_k.csv',
                 ['K', 'silhouette_muestra_validacion', 'inercia_muestra_entrenamiento'])
    seleccionado = int(modelo.n_clusters)
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    ax.plot(tabla['K'], tabla['silhouette_muestra_validacion'], '-o', color=MORADO, lw=2)
    ax.axvline(seleccionado, color=ROJO, linestyle='--', lw=1.5,
               label=f'K elegido = {seleccionado}')
    ax.set(xlabel='Número de grupos K', ylabel='Coeficiente de silueta',
           title='K-means | Selección de K usando validación')
    ax.grid(axis='y', alpha=.17); ax.legend(frameon=False)
    guardar(fig, salida / '01_kmeans_coeficiente_silueta.png')

    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    ax.plot(tabla['K'], tabla['inercia_muestra_entrenamiento'], '-o', color=AZUL, lw=2)
    ax.axvline(seleccionado, color=ROJO, linestyle='--', lw=1.4,
               label=f'K elegido = {seleccionado}')
    ax.set(xlabel='Número de grupos K', ylabel='Inercia (muestra entrenamiento)',
           title='K-means | Método del codo (referencia)')
    ax.grid(axis='y', alpha=.17); ax.legend(frameon=False)
    guardar(fig, salida / '02_kmeans_metodo_codo.png')

    fig, ax = plt.subplots(figsize=(9.5, 6))
    grupos = sorted(datos['cluster'].unique())
    paleta = plt.get_cmap('tab10')
    colores = {g: paleta(i % 10) for i, g in enumerate(grupos)}
    pca_scatter(ax, datos, 'cluster', 'K-means | Grupos en el conjunto de prueba (PCA 2D)',
                {g: f'Grupo {g}' for g in grupos}, colores)
    centros = pca.transform(modelo.cluster_centers_)
    for i, xy in enumerate(centros):
        ax.scatter(xy[0], xy[1], s=240, marker='X', c=[colores.get(i, '#222222')],
                   edgecolors='black', linewidths=.9, zorder=4)
    guardar(fig, salida / '03_kmeans_grupos_y_centroides_pca.png')

    distrib = leer(resultados / '02_kmeans' / '00_distribucion_clusters.csv',
                   ['conjunto', 'cluster', 'porcentaje'])
    pivote = distrib.pivot(index='conjunto', columns='cluster', values='porcentaje').fillna(0)
    pivote = pivote.reindex(PARTES, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5.1))
    x = np.arange(len(pivote))
    fondo = np.zeros(len(pivote))
    for i, g in enumerate(pivote.columns):
        vals = pivote[g].to_numpy()
        ax.bar(x, vals, bottom=fondo, label=f'Grupo {g}', color=paleta(i % 10))
        fondo += vals
    ax.set_xticks(x, [ETIQUETAS[p] for p in PARTES]); ax.set_ylim(0, 100)
    ax.set(ylabel='Porcentaje de registros (%)',
           title='K-means | Tamaño relativo de los grupos')
    ax.legend(frameon=False)
    guardar(fig, salida / '04_kmeans_porcentaje_por_grupo.png')


def graficos_dbscan(resultados: Path, salida: Path, datos: pd.DataFrame,
                    db_art: dict, pca: PCA, train: pd.DataFrame,
                    cols: list[str]) -> dict:
    modelo = db_art['modelo_dbscan_muestra']
    ids = np.asarray(db_art['indice_muestra_entrenamiento'], dtype=int)
    etiquetas = np.asarray(modelo.labels_, dtype=int)
    if len(ids) != len(etiquetas):
        raise ValueError('Índices y etiquetas DBSCAN no coinciden')
    X = train.iloc[ids][cols].to_numpy(dtype=float)
    plano = pca.transform(X)
    sample = pd.DataFrame({'PC1': plano[:, 0], 'PC2': plano[:, 1], 'cluster': etiquetas})
    clusters = sorted(set(etiquetas) - {-1})
    ruido = int((etiquetas == -1).sum())
    # No tratar el ruido (-1) como un cluster para calcular silhouette.
    no_ruido = etiquetas != -1
    silhouette = None
    motivo = None
    if len(clusters) >= 2 and np.count_nonzero(no_ruido) > len(clusters):
        muestra_limpia = X[no_ruido]
        grupos_limpios = etiquetas[no_ruido]
        n_m = min(2000, len(muestra_limpia))
        try:
            silhouette = float(silhouette_score(muestra_limpia, grupos_limpios,
                                                sample_size=n_m, random_state=42))
        except ValueError as exc:
            motivo = str(exc)
    else:
        motivo = (f'No definido: DBSCAN identificó {len(clusters)} grupo(s) en la '
                  'muestra sin contar el ruido; se necesitan al menos 2 grupos.')

    paleta = plt.get_cmap('tab10')
    colores = {-1: ROJO, **{g: paleta(i % 10) for i, g in enumerate(clusters)}}
    etiquetas_nombres = {-1: 'Ruido (-1)', **{g: f'Cluster {g}' for g in clusters}}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    pca_scatter(ax, sample, 'cluster',
                f'DBSCAN nativo | Muestra de entrenamiento (n={len(sample):,})',
                etiquetas_nombres, colores, 8)
    ax.text(.02, .02,
            ('Silueta sin ruido: '+f'{silhouette:.3f}' if silhouette is not None
             else f'Silueta no definida: {len(clusters)} grupo(s)'),
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#DDDDDD', alpha=.9))
    guardar(fig, salida / '05_dbscan_grupos_originales_pca.png')

    fig, ax = plt.subplots(figsize=(9.5, 6))
    pca_scatter(ax, datos, 'cluster_dbscan_aproximado',
                'DBSCAN | Extensión aproximada sobre prueba (PCA 2D)',
                etiquetas_nombres, colores)
    guardar(fig, salida / '06_dbscan_clusters_y_ruido_prueba_pca.png')

    busqueda = leer(resultados / '03_dbscan' / '00_busqueda_eps.csv',
                    ['eps', 'porcentaje_ruido_muestra', 'seleccionado'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(busqueda['eps'], busqueda['porcentaje_ruido_muestra'], '-o', color=AZUL, lw=2)
    seleccionado = busqueda.loc[busqueda['seleccionado'].astype(str).str.lower().isin(['true', '1'])]
    if not seleccionado.empty:
        ax.scatter(seleccionado['eps'], seleccionado['porcentaje_ruido_muestra'],
                   color=ROJO, marker='X', s=120, label='eps elegido', zorder=5)
    ax.set(xlabel='eps (en 10 variables estandarizadas)',
           ylabel='Porcentaje de ruido en la muestra (%)',
           title='DBSCAN | Sensibilidad al parámetro eps')
    ax.grid(axis='y', alpha=.17); ax.legend(frameon=False)
    guardar(fig, salida / '07_dbscan_sensibilidad_eps.png')
    return {'n_muestra': len(sample), 'clusters_nativos': len(clusters),
            'ruido_en_muestra': ruido, 'silhouette_sin_ruido': silhouette,
            'nota_silhouette': motivo,
            'nota_extensión': 'Prueba: asignación aproximada al núcleo más cercano; no es DBSCAN nativo.'}


def graficos_isolation(resultados: Path, salida: Path,
                       todos: dict[str, pd.DataFrame], datos: pd.DataFrame) -> None:
    config = json.loads((resultados / '01_isolation_forest' /
                         '00_configuracion_modelo.json').read_text(encoding='utf-8'))
    umbral = float(config['umbral_puntuacion'])
    fig, ax = plt.subplots(figsize=(9.7, 5.1))
    paleta = [AZUL, VERDE, MORADO]
    for nombre, color in zip(PARTES, paleta):
        vals = todos[nombre]['puntaje_anomalia'].to_numpy(float)
        ax.hist(vals, bins=75, density=True, color=color, alpha=.35,
                label=ETIQUETAS[nombre])
    ax.axvline(umbral, color='#333333', ls='--', lw=1.7,
               label=f'Umbral entrenamiento = {umbral:.4f}')
    ax.set(xlabel='Puntuación IF (mayor = más inusual)', ylabel='Densidad',
           title='Isolation Forest | Distribución de puntuaciones')
    ax.legend(frameon=False)
    guardar(fig, salida / '08_isolation_forest_distribucion_puntajes.png')

    fig, ax = plt.subplots(figsize=(9.5, 6))
    pca_scatter(ax, datos, 'if_alerta',
                'Isolation Forest | Alertas en prueba (PCA 2D)',
                {0: 'No alerta', 1: 'Alerta'}, {0: AZUL, 1: ROJO})
    guardar(fig, salida / '09_isolation_forest_alertas_pca.png')

    prueba = todos['prueba']
    flags = prueba['if_alerta'].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    ax.plot(prueba[SMP], prueba['puntaje_anomalia'], lw=.55, color=AZUL, alpha=.65)
    ax.scatter(prueba.loc[flags, SMP], prueba.loc[flags, 'puntaje_anomalia'],
               color=ROJO, s=7, alpha=.82, label='Alerta', rasterized=True)
    ax.axhline(umbral, color='#222222', ls='--', lw=1.4, label='Umbral fijo')
    ax.set(xlabel='SMP continuo (NO segundos)', ylabel='Puntuación IF',
           title='Isolation Forest | Detecciones según posición de adquisición')
    ax.legend(frameon=False)
    guardar(fig, salida / '10_isolation_forest_alertas_vs_smp.png')


def graficos_shap(resultados: Path, salida: Path) -> dict:
    shap_dir = resultados / '04_shap'
    importancia = leer(shap_dir / '00_importancias_shap.csv',
                       ['sensor', 'shap_abs_media_total', 'shap_abs_media_alertas',
                        'shap_abs_media_normales'])
    explic = leer(shap_dir / '01_explicaciones_registros_prueba.csv',
                  [SMP, 'puntaje_anomalia', 'alerta_isolation_forest', 'shap_valor_base'])
    cols = [c for c in explic.columns if c.startswith('shap_')
            and c not in ('shap_valor_base', 'shap_suma', 'shap_puntaje_reconstruido')]
    if not cols:
        raise ValueError('El archivo SHAP no contiene columnas shap_<sensor>')
    # Nombres cortos en el CSV de SHAP: Ac X, Gi Y, etc.
    nombres = [c[5:] for c in cols]
    matriz = explic[cols].to_numpy(float)
    nombre_a_col = {n: i for i, n in enumerate(nombres)}
    # Mostrar IMPORTANCIA SOLO de muestra seleccionada, nunca como global validada.
    imp = importancia.sort_values('shap_abs_media_total', ascending=True)
    fig, ax = plt.subplots(figsize=(9.7, 5.5))
    ax.barh(imp['sensor'], imp['shap_abs_media_total'], color=MORADO, alpha=.84)
    ax.set(xlabel='Media de |SHAP| (40 registros seleccionados)',
           title='SHAP | Importancia en muestra de prueba (NO global)')
    ax.grid(axis='x', alpha=.15)
    guardar(fig, salida / '11_shap_importancia_variables_muestra.png')

    a = explic['alerta_isolation_forest'].to_numpy(dtype=int) == 1
    if np.any(a):
        i = np.flatnonzero(a)[np.argmax(explic.loc[a, 'puntaje_anomalia'].to_numpy(float))]
    else:
        i = int(np.argmax(explic['puntaje_anomalia'].to_numpy(float)))
    v = matriz[i]
    orden = np.argsort(np.abs(v))
    fig, ax = plt.subplots(figsize=(9.7, 5.6))
    ax.barh(np.asarray(nombres)[orden], v[orden],
            color=[ROJO if t > 0 else AZUL for t in v[orden]])
    ax.axvline(0, color='#303030', lw=.8)
    ax.set(xlabel='Contribución a la puntuación IF (signo positivo = aumenta)',
           title=f'SHAP | Explicación individual de SMP {int(explic[SMP].iloc[i])}')
    ax.grid(axis='x', alpha=.15)
    guardar(fig, salida / '12_shap_explicacion_alerta_individual.png')

    vmax = max(0.001, float(np.max(np.abs(matriz))))
    fig, ax = plt.subplots(figsize=(11, max(5, .19 * len(explic) + 2)))
    disp = ax.imshow(matriz, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                     norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(np.arange(len(nombres)), nombres, rotation=45, ha='right')
    if len(explic) <= 50:
        ax.set_yticks(np.arange(len(explic)),
                      [f"{int(s)}{' *' if alerta else ''}"
                       for s, alerta in zip(explic[SMP], a)], fontsize=7)
    else:
        ax.set_ylabel('Registros de prueba seleccionados')
    ax.set(title='SHAP | Contribuciones por registro (asterisco = alerta IF)',
           xlabel='Variable física', ylabel='SMP continuo')
    cb = fig.colorbar(disp, ax=ax, shrink=.88)
    cb.set_label('SHAP (negativo ↔ positivo)')
    guardar(fig, salida / '13_shap_mapa_calor_registros.png')

    return {'registros_shap': int(len(explic)),
            'alertas_en_muestra_shap': int(a.sum()),
            'advertencia': 'Interpretaciones locales seleccionadas, no importancia global ni causal.'}


def grafico_comparacion(salida: Path, datos: pd.DataFrame,
                        todos: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    pca_scatter(axs[0, 0], datos, 'if_alerta', 'Isolation Forest',
                {0: 'No alerta', 1: 'Alerta'}, {0: AZUL, 1: ROJO}, 5)
    pca_scatter(axs[0, 1], datos, 'km_alerta', 'K-means (alerta por distancia)',
                {0: 'No alerta', 1: 'Alerta'}, {0: AZUL, 1: ROJO}, 5)
    pca_scatter(axs[1, 0], datos, 'db_ruido', 'DBSCAN (aproximación)',
                {0: 'Grupo', 1: 'Ruido'}, {0: AZUL, 1: ROJO}, 5)
    pca_scatter(axs[1, 1], datos, 'consenso', 'Coincidencias entre modelos',
                {0: '0 métodos', 1: '1 método', 2: '2 métodos', 3: '3 métodos'},
                {0: GRIS, 1: AZUL, 2: AMARILLO, 3: ROJO}, 5)
    fig.suptitle('Resultados sobre los mismos registros de prueba | PCA 2D', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .965))
    guardar(fig, salida / '14_comparacion_cuatro_paneles_pca.png')

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(3)
    ancho = .24
    colores = [AZUL, VERDE, MORADO]
    for j, (campo, titulo) in enumerate([('if_alerta', 'Isolation Forest'),
                                        ('km_alerta', 'K-means'),
                                        ('db_ruido', 'DBSCAN aprox.')]):
        porcentajes = [100 * todos[parte][campo].astype(int).mean() for parte in PARTES]
        ax.bar(x + (j-1)*ancho, porcentajes, width=ancho,
               label=titulo, color=colores[j])
    ax.set_xticks(x, [ETIQUETAS[n] for n in PARTES])
    ax.set(ylabel='Registros señalados (%)',
           title='Comparación de alertas/ruido por conjunto cronológico')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.15)
    guardar(fig, salida / '15_comparacion_porcentajes_alertas.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resultados', type=Path,
                        help='Carpeta Resultados_Tarea3, por defecto junto al script')
    parser.add_argument('--salida', type=Path,
                        help='Carpeta de imágenes, por defecto Resultados_Tarea3/09_graficos_modelos')
    parser.add_argument('--max-puntos', type=int, default=3500,
                        help='Máximo de puntos de prueba para dispersión (default 3500)')
    args = parser.parse_args()
    if args.max_puntos < 100:
        parser.error('--max-puntos debe ser al menos 100')
    raiz = Path(__file__).resolve().parent
    resultados = args.resultados or raiz / 'Resultados_Tarea3'
    salida = args.salida or resultados / '09_graficos_modelos'
    salida.mkdir(parents=True, exist_ok=True)
    estilo()

    preparados = {parte: leer(resultados / '00_preparacion' /
                                f'01_{parte}_escalado.csv', [SMP]) for parte in PARTES}
    cols = [c for c in preparados['entrenamiento'].columns if c.startswith('z_')]
    if len(cols) != 10:
        raise ValueError('Se esperaban 10 variables z_ del código 08')
    todos = {parte: fusionar(preparados[parte], resultados, parte) for parte in PARTES}

    # PCA exclusivamente descriptivo/visual. Se ajusta SOLO con entrenamiento.
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(preparados['entrenamiento'][cols].to_numpy(dtype=float))
    prueba = todos['prueba']
    ids = muestra_ids(prueba, args.max_puntos)
    visibles = prueba.iloc[ids].copy()
    xy = pca.transform(visibles[cols].to_numpy(dtype=float))
    visibles['PC1'], visibles['PC2'] = xy[:, 0], xy[:, 1]
    visibles[[SMP, 'PC1', 'PC2', 'cluster', 'cluster_dbscan_aproximado',
              'if_alerta', 'km_alerta', 'db_ruido', 'consenso']].to_csv(
                  salida / '00_muestra_visualizacion_prueba.csv', sep=';', index=False,
                  encoding='utf-8-sig')

    # joblib puede ejecutar codigo arbitrario: cargar SOLO ficheros propios del proyecto.
    km = joblib.load(resultados / '02_kmeans' / '00_modelo_kmeans.joblib')
    db = joblib.load(resultados / '03_dbscan' / '00_modelo_dbscan_muestra.joblib')
    graficos_kmeans(resultados, salida, visibles, km, pca)
    met_db = graficos_dbscan(resultados, salida, visibles, db, pca,
                            preparados['entrenamiento'], cols)
    graficos_isolation(resultados, salida, todos, visibles)
    met_shap = graficos_shap(resultados, salida)
    grafico_comparacion(salida, visibles, todos)

    meta = {'pca_ajustado_solo_en_entrenamiento': True,
            'pca_varianza_explicada_pc1_pc2_pct':
                round(float(pca.explained_variance_ratio_.sum() * 100), 4),
            'muestras_prueba_total': len(prueba),
            'registros_usados_en_graficos_de_dispersion': len(visibles),
            'dispersion_muestra_estratificada_entre_alertas_y_no_alertas': True,
            'kmeans_k': int(km.n_clusters),
            'dbscan': met_db, 'shap': met_shap,
            'advertencias': [
                'PCA 2D es una proyección; los algoritmos usaron 10 variables.',
                'PCA 2D no necesariamente separa visualmente grupos existentes en 10D.',
                'Silueta mide estructura de agrupación, no detección de fallas.',
                'DBSCAN de prueba es una extensión aproximada de un ajuste en muestra.',
                'SHAP explica la puntuación de Isolation Forest y no establece causalidad.',
                'La muestra de dispersión no representa prevalencias exactas.',
                'Sin etiquetas de falla no se puede medir precisión diagnóstica.'
            ]}
    (salida / '00_metadatos_graficos.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'modelo': 'K-means', 'n_grupos_entrenados': int(km.n_clusters),
                   'silhouette_muestra': np.nan,
                   'nota': 'Curva K=2..7 y silueta de validación en 01_kmeans_coeficiente_silueta.png'},
                  {'modelo': 'DBSCAN nativo muestra',
                   'n_grupos_entrenados': met_db['clusters_nativos'],
                   'silhouette_muestra': met_db['silhouette_sin_ruido'],
                   'nota': met_db['nota_silhouette'] or 'Solo grupos no ruido'}]).to_csv(
                       salida / '00_resumen_silueta.csv', sep=';', index=False,
                       encoding='utf-8-sig')
    print('\nGRÁFICOS DE LOS CUATRO MÉTODOS FINALIZADOS')
    print(f'Carpeta: {salida}')
    print(f'K-means: K={km.n_clusters}')
    print(f"DBSCAN: {met_db['clusters_nativos']} cluster(s) nativo(s) en muestra")
    print('Silueta DBSCAN:', met_db['silhouette_sin_ruido'] if
          met_db['silhouette_sin_ruido'] is not None else met_db['nota_silhouette'])
    print(f'PCA PC1+PC2 = {meta["pca_varianza_explicada_pc1_pc2_pct"]:.2f}% de varianza (visualización)')
    print('Nota: no se reentrenaron detectores, y no se conocen fallas verificadas.')


if __name__ == '__main__':
    main()
