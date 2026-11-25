# -*- coding: utf-8 -*-
"""16 | Ejecucion exploratoria FINAL sobre TODAS las 143.561 mediciones.

NO divide en entrenamiento/validacion/prueba. CADA algoritmo ajusta/analiza
las 143.561 filas y las 10 senales conjuntamente. SMP identifica la muestra;
NO es predictor ni representa segundos.

Esta prueba describe el comportamiento EN LA MISMA BASE utilizada para ajustar
los modelos. No sustituye la evaluacion 70/15/15 de los codigos 08-15, y no
permite medir generalizacion, precision ni fallas fisicas confirmadas.

Uso desde la carpeta con los codigos (ubica/recupera el CSV automaticamente):
    python -u 16_ejecucion_total_143561.py
    python -u 16_ejecucion_total_143561.py --algoritmo kmeans
    python -u 16_ejecucion_total_143561.py --algoritmo dbscan
    python -u 16_ejecucion_total_143561.py --algoritmo hdbscan
    python -u 16_ejecucion_total_143561.py --algoritmo isolation_forest
    python -u 16_ejecucion_total_143561.py --modo comprobar
    python -u 16_ejecucion_total_143561.py --modo graficos
    python -u 16_ejecucion_total_143561.py --entrada "RUTA/datos_multisensor_corregidos.csv"

Requisitos: numpy pandas scipy matplotlib scikit-learn joblib shap; para HDBSCAN:
    python -m pip install hdbscan shap

Resultados en Resultados_Tarea3_Nuevos/08_ejecucion_total_143561.
Modo --modo graficos: usa los CSV ya generados por la version anterior, no
reentrena los cuatro modelos; SOLO reconstruye IF si el archivo de modelo no
existe, y reajusta los K adicionales del codo. Para el arbol HDBSCAN
necesita --arbol-hdbscan, porque las etiquetas CSV no conservan la jerarquia.
SHAP explica una muestra estratificada de la puntuacion del IF entrenado
sobre TODAS las filas; no calcula SHAP individual para 143561 mediciones.
No modifica resultados de los codigos anteriores ni el CSV original.

Los parametros iniciales son EXPLORATORIOS (tomados de los barridos anteriores),
NO se presentan como optimos ni se interpretan los grupos/alertas como averias.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
import threading
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SMP = 'smp_continuo'
VARIABLES = [
    'AcelX1_SisGenEol', 'AcelY1_SisGenEol', 'AcelZ1_SisGenEol',
    'GiroX_SisGenEol', 'GiroY_SisGenEol', 'GiroZ_SisGenEol',
    'MagneX_SisGenEol', 'MagneY_SisGenEol', 'MagneZ_SisGenEol',
    'Ter_SisGenEol',
]
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
MODELOS = ('kmeans', 'dbscan', 'hdbscan', 'isolation_forest')


def avisar(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def latido(nombre: str, cada: int = 15):
    """Informa que el proceso sigue activo; NO simula avance porcentual."""
    parar = threading.Event()
    def seguir():
        transcurrido = 0
        while not parar.wait(cada):
            transcurrido += cada
            avisar(f'  [{nombre}] proceso activo, {transcurrido} segundos sin porcentaje de progreso ...')
    hilo = threading.Thread(target=seguir, daemon=True)
    hilo.start()
    try:
        yield
    finally:
        parar.set()
        hilo.join(timeout=1)


def guardar_csv(df: pd.DataFrame, dest: Path) -> None:
    df.to_csv(dest, sep=';', index=False, encoding='utf-8-sig', float_format='%.12g')


def direccion(valor: str, por_defecto: Path) -> Path:
    if not valor:
        return por_defecto
    p = Path(valor).expanduser()
    return p if p.is_absolute() else BASE / p


def localizar_csv_corregido(entrada: str) -> Path | None:
    """Busca solamente el CSV maestro exacto, no resultados de algoritmos."""
    if entrada:
        p = direccion(entrada, BASE / 'datos_multisensor_corregidos.csv')
        if not p.is_file():
            raise FileNotFoundError(f'La ruta especificada en --entrada no existe: {p}')
        return p.resolve()

    candidatos = [
        BASE / 'datos_multisensor_corregidos.csv',
        BASE / 'Datos_corregidos' / 'datos_multisensor_corregidos.csv',
        BASE / 'Datos_corregidos_desde_excel' / 'datos_multisensor_corregidos.csv',
        BASE / 'Resultados_Tarea3' / 'datos_multisensor_corregidos.csv',
    ]
    # Tambien contempla carpetas personalizadas creadas por el codigo 00.
    candidatos += sorted(BASE.glob('*/datos_multisensor_corregidos.csv'))
    candidatos += sorted(BASE.glob('*/*/datos_multisensor_corregidos.csv'))
    metadatos = BASE / 'Resultados_Tarea3_Nuevos' / '00_preparacion' / '00_metadatos.json'
    if metadatos.is_file():
        try:
            origen = json.loads(metadatos.read_text(encoding='utf-8')).get('origen')
            if origen:
                candidatos.insert(1, Path(origen))
        except (ValueError, OSError):
            pass
    for candidato in candidatos:
        if candidato.is_file():
            return candidato.resolve()
    return None


def recuperar_csv_desde_excel() -> Path | None:
    """Reutiliza el codigo 00 del proyecto si no hay CSV maestro en disco.

    Solo se invoca si existen TANTO el XLSX original COMO el codigo 00.
    No toca los resultados de los modelos y registra la recuperacion en pantalla.
    """
    excel = BASE / 'CF-2023-I-204-1230_DAT.xlsx'
    programa = BASE / '00_validar_y_separar_excel.py'
    if not excel.is_file() or not programa.is_file():
        return None
    salida = BASE / 'Datos_corregidos'
    csv = salida / 'datos_multisensor_corregidos.csv'
    avisar('No aparece el CSV corregido. Encontré el Excel original y el codigo 00.')
    avisar('Reconstruyendo el CSV maestro con la misma auditoria de datos del proyecto ...')
    comando = [sys.executable, '-u', str(programa), '--entrada', str(excel), '--salida', str(salida)]
    subprocess.run(comando, check=True)
    if not csv.is_file():
        raise FileNotFoundError(f'El codigo 00 termino sin generar {csv}')
    return csv.resolve()


def cargar_desde_preparacion_08() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Ultima alternativa: reune 100% de filas del 08 e invierte su escalado.

    Solo se usa si no hay CSV corregido ni es posible regenerarlo desde el
    Excel. Conserva los datos YA IMPUTADOS por 08; esta distincion se registra.
    Luego el 16 ajusta un NUEVO escalado con las 143.561 observaciones.
    """
    carpeta = BASE / 'Resultados_Tarea3_Nuevos' / '00_preparacion'
    rutas = [carpeta / f'01_{parte}_estandar.csv'
             for parte in ('entrenamiento', 'validacion', 'prueba')]
    p_preproc = carpeta / '00_preprocesamiento.joblib'
    ausentes = [str(p) for p in [*rutas, p_preproc] if not p.is_file()]
    if ausentes:
        raise FileNotFoundError(
            'No encuentro el CSV maestro, ni puedo reconstruirlo desde el Excel, '
            'ni estan todos los archivos del codigo 08.\n'
            'Faltantes de preparacion: ' + '; '.join(ausentes) + '\n'
            'Solucion: ejecuta python -u 00_validar_y_separar_excel.py '
            'y luego python -u 16_ejecucion_total_143561.py; '
            'o indica la ruta correcta con --entrada "RUTA\\datos_multisensor_corregidos.csv".'
        )
    avisar('CSV maestro no encontrado: leyendo las tres partes preparadas por 08 ...')
    prep = joblib.load(p_preproc)
    if prep.get('variables') != VARIABLES:
        raise ValueError('Las 10 columnas del preprocesamiento 08 no coinciden con el codigo 16')
    scaler = prep.get('escalador_estandar')
    if scaler is None or not hasattr(scaler, 'inverse_transform'):
        raise ValueError('El codigo 08 no contiene el escalador estandar necesario')
    columns = ['z_' + c for c in VARIABLES]
    tramos = []
    for p in rutas:
        d = pd.read_csv(p, sep=';', encoding='utf-8-sig', low_memory=False)
        if any(c not in d.columns for c in [SMP, *columns]):
            raise ValueError(f'Columnas inesperadas en {p}')
        tramos.append(d[[SMP, *columns]])
    todos = pd.concat(tramos, ignore_index=True)
    if len(todos) != 143561:
        raise ValueError(f'08: esperados 143561 registros; disponibles {len(todos)}')
    smp = todos[SMP].to_numpy(dtype=np.int64)
    if np.unique(smp).size != len(smp) or not (np.diff(smp) > 0).all():
        raise ValueError('08: los SMP no son unicos o estan desordenados')
    z = todos[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError('08: matriz con datos no finitos')
    Xraw = scaler.inverse_transform(z)
    if not np.isfinite(Xraw).all():
        raise ValueError('08: la transformacion inversa no genero datos finitos')
    info = {
        'origen': 'Reconstruccion desde 08_preparacion (tres CSV estandar + transformacion inversa)',
        'archivos_fuente': [str(p) for p in rutas],
        'registros': len(smp), 'senales': VARIABLES, 'nulos_senales': 0,
        'smp_inicial': int(smp[0]), 'smp_final': int(smp[-1]),
        'imputacion_previa_08': True,
        'nota_reconstruccion': 'Las celdas imputadas en 08 no se pueden distinguir de las observadas desde sus CSV escalados. No se han perdido registros.',
        'division': 'NINGUNA en los ajustes del codigo 16; se concatenaron los 3 bloques de 08 solo para recuperar el orden y las 143561 filas',
        'interpretacion': 'Ejercicio exploratorio de ajuste en los mismos datos; no mide generalizacion ni fallas confirmadas',
    }
    avisar('08 recuperado: 143.561 filas reunidas. Se reajustara el preprocesamiento sobre toda la base.')
    return smp, Xraw, pd.DataFrame({SMP: smp}), info


def cargar_todo(ruta: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    if not ruta.is_file():
        raise FileNotFoundError(f'No existe la base: {ruta}')
    # El CSV del proyecto contiene punto y coma, coma decimal, y notacion cientifica.
    df = pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig', low_memory=False)
    faltantes = [c for c in [SMP, *VARIABLES] if c not in df.columns]
    if faltantes:
        raise ValueError(f'Faltan columnas: {faltantes}')
    if len(df) != 143561:
        raise ValueError(f'Esperados 143.561 registros, se encontraron {len(df)}; revisar archivo')
    if df[SMP].isna().any() or df[SMP].duplicated().any() or not df[SMP].is_monotonic_increasing:
        raise ValueError('SMP debe estar completo, ordenado y sin identificadores duplicados')
    smp = df[SMP].to_numpy(dtype=np.int64)
    # Pandas (en ciertas versiones) devuelve una vista NumPy de SOLO LECTURA.
    # np.array(..., copy=True) garantiza un arreglo independiente y modificable.
    Xraw = np.array(
        df[VARIABLES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64),
        dtype=np.float64, copy=True, order='C'
    )
    Xraw[~np.isfinite(Xraw)] = np.nan
    if np.isnan(Xraw).all(axis=0).any() or np.isnan(Xraw).all(axis=1).any():
        raise ValueError('Hay una senal totalmente vacia o una fila sin ninguna senal')
    hash_datos = hashlib.sha256()
    with ruta.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hash_datos.update(chunk)
    info = {
        'origen': str(ruta.resolve()), 'sha256_origen': hash_datos.hexdigest(),
        'registros': len(df), 'senales': VARIABLES, 'nulos_senales': int(np.isnan(Xraw).sum()),
        'smp_inicial': int(smp[0]), 'smp_final': int(smp[-1]),
        'division': 'NINGUNA; ajuste directo en el 100% de los registros',
        'interpretacion': 'Ejercicio exploratorio sobre los datos usados para el ajuste; NO es evaluacion fuera de muestra',
    }
    return smp, Xraw, df, info


def ajustar_preparacion(Xraw: np.ndarray, escalado: str, destino: Path) -> np.ndarray:
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(Xraw)  # TODO el CSV, sin division
    scaler = StandardScaler() if escalado == 'estandar' else RobustScaler()
    X = scaler.fit_transform(X)
    if X.shape != (143561, 10) or not np.isfinite(X).all():
        raise ValueError('Matriz final no tiene 143561 x 10 valores finitos')
    joblib.dump({'imputador': imp, 'escalador': scaler, 'variables': VARIABLES,
                 'entrenado_con_143561': True, 'tipo_escalado': escalado},
                destino / '00_preprocesamiento_143561.joblib', compress=3)
    return np.ascontiguousarray(X, dtype=np.float64)


def dbscan_sobre_todos(X: np.ndarray, eps: float, min_samples: int,
                        max_vecinos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """DBSCAN con TODOS los puntos y sin matriz densa NxN.

    El grafo identifica las componentes conexas de puntos nucleo. Los puntos
    frontera se asignan al nucleo mas cercano; en fronteras ambiguas pueden
    existir varias soluciones igualmente validas de DBSCAN.
    """
    if not (eps > 0 and 2 <= min_samples <= len(X)):
        raise ValueError('eps debe ser > 0 y min_samples >= 2')
    arbol = cKDTree(X)
    avisar('  DBSCAN: calculando distancias k-NN en los 143.561 registros ...')
    with latido('DBSCAN distancias k-NN'):
        d, _ = arbol.query(X, k=min_samples, p=2, workers=min(4, os.cpu_count() or 1))
    dist_k = np.asarray(d[:, -1], dtype=float)
    core_idx = np.flatnonzero(dist_k <= eps + 1e-12)
    lab = np.full(len(X), -1, dtype=np.int32)
    if not len(core_idx):
        return lab, core_idx, dist_k, {'aristas': 0, 'bordes_ambiguos': 'No evaluados'}
    Xcore = X[core_idx]
    arbol_core = cKDTree(Xcore)
    sondas = np.linspace(0, len(core_idx)-1, min(128, len(core_idx)), dtype=int)
    c = arbol_core.query_ball_point(Xcore[sondas], eps, return_length=True)
    c90 = float(np.quantile(c, 0.9))
    estimado = float(np.mean(c) * len(core_idx))
    avisar(f'  DBSCAN: {len(core_idx):,} nucleos; vecindad p90={c90:.0f}; '
          f'conexiones estimadas={estimado:,.0f}')
    if c90 > max_vecinos or estimado > 15_000_000:
        raise RuntimeError('Vecindades demasiado densas: se cancela DBSCAN para no agotar RAM. '
                           'No se generan etiquetas inventadas. Revisa el radio eps.')
    avisar('  DBSCAN: conectando nucleos con pares de vecinos cKDTree (sin matriz densa NxN) ...')
    # query_pairs evita crear una matriz de distancias N x N; devuelve solamente
    # las aristas dentro de eps, una vez por par. Mas eficiente en este conjunto.
    t_conn = time.perf_counter()
    with latido('DBSCAN conexiones nucleares'):
        pares = arbol_core.query_pairs(eps + 1e-12, output_type='ndarray')
    avisar(f'  DBSCAN: {len(pares):,} pares calculados en {time.perf_counter()-t_conn:.1f} s')
    edges = int(2 * len(pares))
    if edges > 15_000_000:
        raise RuntimeError('Exceso de aristas DBSCAN: se cancela para proteger memoria.')
    r = np.r_[pares[:, 0], pares[:, 1]]
    c = np.r_[pares[:, 1], pares[:, 0]]
    g = coo_matrix((np.ones(edges, dtype=np.int8), (r, c)),
                   shape=(len(core_idx), len(core_idx))).tocsr()
    del r, c, pares
    avisar(f'  DBSCAN: grafo disperso conectado ({g.nnz:,} entradas); encontrando componentes ...')
    _, lcore = connected_components(g, directed=False)
    avisar(f'  DBSCAN: componentes listas en {time.perf_counter()-t_conn:.1f} s')
    del g
    avisar('  DBSCAN: asignando todas las observaciones a nucleos o ruido ...')
    dist, nearest = arbol_core.query(X, k=1, workers=min(4, os.cpu_count() or 1))
    adentro = dist <= eps + 1e-12
    lab[adentro] = lcore[nearest[adentro]].astype(np.int32)
    lab[core_idx] = lcore.astype(np.int32)
    return lab, core_idx, dist_k, {'aristas': edges, 'nucleos': int(len(core_idx))}


def ajustar_modelo(nombre: str, X: np.ndarray, args: argparse.Namespace):
    inicio = time.perf_counter()
    if nombre == 'kmeans':
        avisar(f'  K-Means: K={args.k}, ajustando sobre {len(X):,} registros ...')
        modelo = KMeans(n_clusters=args.k, n_init=10, max_iter=300, random_state=42)
        labs = modelo.fit_predict(X).astype(np.int32)
        valores = {'inercia_sse': float(modelo.inertia_), 'iteraciones': int(modelo.n_iter_)}
        extra = {'distancia_centroide': np.min(modelo.transform(X), axis=1)}
    elif nombre == 'dbscan':
        avisar(f'  DBSCAN: eps={args.eps}, min_samples={args.ms_dbscan}')
        labs, nuc, dknn, valores = dbscan_sobre_todos(X, args.eps, args.ms_dbscan, args.max_vecinos)
        extra = {'es_nucleo_dbscan': np.isin(np.arange(len(X)), nuc).astype('int8'),
                 'distancia_k_dbscan': dknn}
    elif nombre == 'hdbscan':
        try:
            import hdbscan
        except (ImportError, OSError) as e:
            raise RuntimeError('Falta biblioteca hdbscan en este Python. '
                               'Instala: python -m pip install hdbscan') from e
        avisar(f'  HDBSCAN: min_cluster_size={args.mcs}, min_samples={args.ms_hdbscan}, '
              f'seleccion={args.seleccion}; ajuste completo en curso ...')
        modelo = hdbscan.HDBSCAN(
            min_cluster_size=args.mcs, min_samples=args.ms_hdbscan,
            cluster_selection_method=args.seleccion, metric='euclidean',
            algorithm='boruvka_kdtree', approx_min_span_tree=True,
            prediction_data=False, core_dist_n_jobs=min(4, os.cpu_count() or 1), leaf_size=60,
        )
        with latido('HDBSCAN ajuste de jerarquia'):
            labs = modelo.fit_predict(X).astype(np.int32)
        extra = {'fuerza_hdbscan': np.asarray(modelo.probabilities_, float)}
        scores = np.asarray(modelo.outlier_scores_, float)
        scores[~np.isfinite(scores)] = np.nan
        extra['glosh_hdbscan'] = scores
        valores = {'persistencia_media': float(np.mean(modelo.cluster_persistence_))
                   if len(modelo.cluster_persistence_) else None}
    elif nombre == 'isolation_forest':
        avisar(f'  Isolation Forest: {args.arboles} arboles, {args.max_samples} muestras '
              f'internas por arbol; todos los {len(X):,} registros se consideran al entrenar ...')
        modelo = IsolationForest(n_estimators=args.arboles, max_samples=args.max_samples,
                                 contamination='auto', random_state=42, n_jobs=-1)
        modelo.fit(X)
        decision = modelo.decision_function(X)
        labs = (decision < 0).astype(np.int8)  # 1 = alerta, 0 = sin alerta
        extra = {'puntuacion_anomalia_if': -modelo.score_samples(X),
                 'decision_function_if': decision}
        valores = {'umbral': 'contamination=auto; decision_function < 0',
                   'advertencia': 'No equivale a probabilidad de falla'}
    else:
        raise ValueError('Modelo desconocido: ' + nombre)
    valores['segundos_ajuste_y_puntuacion'] = round(time.perf_counter() - inicio, 3)
    return labs, extra, valores, modelo if nombre != 'dbscan' else None


def grafico_pca(nombre: str, coord: np.ndarray, labels: np.ndarray,
                var: np.ndarray, destino: Path):
    fig, ax = plt.subplots(figsize=(11.5, 7))
    if nombre == 'isolation_forest':
        for v, color, etiqueta in [(0, '#648eab', 'Sin alerta'), (1, '#cf5b59', 'Alerta')]:
            mask = labels == v
            if mask.any():
                ax.scatter(coord[mask, 0], coord[mask, 1], s=1.4,
                           c=color, alpha=.30, linewidths=0, rasterized=True,
                           label=f'{etiqueta}: {int(mask.sum()):,}')
    else:
        # Graficar los puntos de ruido al fondo para ver clusters minoritarios.
        ruido = labels == -1
        if ruido.any():
            ax.scatter(coord[ruido, 0], coord[ruido, 1], s=1.1, color='#959da8',
                       alpha=.18, linewidths=0, rasterized=True,
                       label=f'Ruido: {int(ruido.sum()):,}')
        clases = np.unique(labels[~ruido])
        colores = plt.get_cmap('tab20')
        for j, clase in enumerate(clases):
            mask = labels == clase
            ax.scatter(coord[mask, 0], coord[mask, 1], s=1.5,
                       color=colores(j % 20), alpha=.40, linewidths=0,
                       rasterized=True, label=f'Grupo {clase}: {int(mask.sum()):,}')
    ax.set_xlabel(f'PC1 ({var[0]*100:.2f}% varianza)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.2f}% varianza)')
    ax.set_title(f'{nombre.replace("_", " ").title()} | 143.561 registros completos (PCA solo para visualizar)')
    ax.grid(alpha=.15)
    h, leg = ax.get_legend_handles_labels()
    if len(leg) > 12:
        h, leg = h[:11], leg[:11] + [f'... y {len(leg)-11} clases mas']
        h = h + [plt.Line2D([], [], color='none')]
    if leg:
        ax.legend(h, leg, loc='best', fontsize=7, markerscale=4)
    fig.tight_layout()
    fig.savefig(destino, dpi=180, bbox_inches='tight')
    plt.close(fig)


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modo', choices=('final', 'comprobar', 'graficos', 'dispersiones'), default='final')
    p.add_argument('--algoritmo', choices=('todos', *MODELOS), default='todos')
    p.add_argument('--entrada', default='')
    p.add_argument('--salida', default='')
    p.add_argument('--escalado', choices=('estandar', 'robusto'), default='estandar')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--eps', type=float, default=1.423777387)
    p.add_argument('--ms-dbscan', type=int, default=20)
    p.add_argument('--max-vecinos', type=int, default=1800)
    p.add_argument('--mcs', type=int, default=100)
    p.add_argument('--ms-hdbscan', type=int, default=5)
    p.add_argument('--seleccion', choices=('eom', 'leaf'), default='leaf')
    p.add_argument('--arboles', type=int, default=300)
    p.add_argument('--max-samples', type=int, default=256)
    p.add_argument('--shap-n', type=int, default=80, help='Filas explicadas con SHAP (no entrena con menos)')
    p.add_argument('--shap-fondo', type=int, default=20, help='Filas de referencia SHAP')
    p.add_argument('--shap-permutaciones', type=int, default=2)
    p.add_argument('--k-max', type=int, default=12, help='K maximo en diagnostico del codo')
    p.add_argument('--sin-barrido-k', action='store_true', help='Omitir K adicionales en los graficos')
    p.add_argument('--arbol-hdbscan', action='store_true', help='En modo graficos, cargar o calcular HDBSCAN para arbol condensado')
    return p.parse_args()


def main():
    args = argumentos()
    salida = direccion(args.salida, BASE / 'Resultados_Tarea3_Nuevos' / '08_ejecucion_total_143561')
    avisar('=== EJECUCION SIN DIVISION: TODOS LOS 143.561 REGISTROS, 10 VARIABLES ===')
    ruta = localizar_csv_corregido(args.entrada)
    if ruta is None:
        ruta = recuperar_csv_desde_excel()
    if ruta is not None:
        avisar(f'Base encontrada: {ruta}')
        smp, Xraw, df, meta = cargar_todo(ruta)
    else:
        smp, Xraw, df, meta = cargar_desde_preparacion_08()
    avisar(f'Base verificada: {len(smp):,} registros / {Xraw.shape[1]} senales; '
          f'faltantes en senales: {meta["nulos_senales"]}')
    if args.modo == 'comprobar':
        avisar('Comprobacion exitosa. NO se entreno ningun modelo.')
        return
    salida.mkdir(parents=True, exist_ok=True)
    if args.modo in ('graficos', 'dispersiones'):
        ruta_prep = salida / '00_preprocesamiento_143561.joblib'
        if ruta_prep.exists():
            prep = joblib.load(ruta_prep)
            if list(prep.get('variables', [])) != VARIABLES:
                raise ValueError('El preprocesamiento guardado no coincide con las 10 senales.')
            X = prep['escalador'].transform(prep['imputador'].transform(Xraw))
            if not np.isfinite(X).all():
                raise ValueError('El preprocesamiento anterior produjo valores no finitos.')
            args.escalado = prep.get('tipo_escalado', args.escalado)
        else:
            avisar('No habia preprocesamiento anterior; reconstruyendo escalado de la base completa ...')
            X = ajustar_preparacion(Xraw, args.escalado, salida)
    else:
        X = ajustar_preparacion(Xraw, args.escalado, salida)
    meta['escalado'] = args.escalado
    meta['parametros_iniciales_exploratorios'] = {
        'kmeans_k': args.k, 'dbscan_eps': args.eps, 'dbscan_min_samples': args.ms_dbscan,
        'hdbscan_mcs': args.mcs, 'hdbscan_min_samples': args.ms_hdbscan,
        'hdbscan_seleccion': args.seleccion,
        'if_arboles': args.arboles, 'if_max_samples_por_arbol': args.max_samples,
    }
    (salida / '00_trazabilidad_143561.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    # PCA se ajusta con toda la base, solo para representar las etiquetas;
    # los algoritmos trabajan con las 10 dimensiones originales escaladas.
    avisar('Calculando PCA de visualizacion (no se usa para entrenar modelos) ...')
    pca = PCA(n_components=2, svd_solver='full')
    coord = pca.fit_transform(X)
    modelos = MODELOS if args.algoritmo == 'todos' else (args.algoritmo,)
    if args.modo == 'dispersiones':
        modo_dispersiones(X, smp, coord, pca.explained_variance_ratio_, salida, args, modelos)
        return
    if args.modo == 'graficos':
        modo_solo_graficos(X, smp, coord, pca.explained_variance_ratio_, salida, args, modelos)
        return
    resumen = []
    fallos = []
    for nombre in modelos:
        avisar(f'\n--- {nombre.upper()} / 143.561 REGISTROS ---')
        try:
            labels, extra, detalles, modelo = ajustar_modelo(nombre, X, args)
            if len(labels) != len(smp) or any(len(v) != len(smp) for v in extra.values()):
                raise RuntimeError('El modelo no genero resultado para los 143.561 registros')
            resultado = pd.DataFrame({SMP: smp})
            if nombre == 'isolation_forest':
                resultado['alerta_isolation_forest'] = labels
                n_grupos = None
                n_raros = int(labels.sum())
            else:
                resultado[f'cluster_{nombre}'] = labels
                n_grupos = int(len(set(labels) - {-1}))
                n_raros = int(np.count_nonzero(labels == -1)) if nombre != 'kmeans' else None
            for col, v in extra.items():
                resultado[col] = v
            guardar_csv(resultado, salida / f'01_resultado_total_{nombre}.csv')
            if nombre == 'hdbscan':
                guardar_modelo_hdbscan(modelo, labels, X, args, salida)
            if nombre == 'isolation_forest':
                joblib.dump(modelo, salida / '00_modelo_total_isolation_forest.joblib', compress=3)
            grafico_pca(nombre, coord, labels, pca.explained_variance_ratio_,
                        salida / f'02_pca_total_{nombre}.png')
            graficos_adicionales(nombre, smp, X, coord, pca.explained_variance_ratio_,
                                 labels, resultado, salida, args, modelo)
            if nombre == 'isolation_forest':
                ejecutar_shap_total(modelo, X, smp, labels, resultado, salida, args)
            fila = {'algoritmo': nombre, 'estado': 'OK', 'registros_utilizados': len(smp),
                    'variables': 10, 'grupos': n_grupos, 'ruido_o_alertas': n_raros,
                    'porcentaje_ruido_o_alertas': 100*n_raros/len(smp)
                    if n_raros is not None else None, 'detalles': json.dumps(detalles, ensure_ascii=False)}
            resumen.append(fila)
            guardar_csv(pd.DataFrame(resumen), salida / '00_resumen_modelos.csv')
            avisar(f'  LISTO: {nombre}; registros={len(smp):,}; grupos={n_grupos}; '
                  f'ruido/alertas={n_raros}; {detalles["segundos_ajuste_y_puntuacion"]:.1f} s')
        except Exception as exc:
            msg = f'{type(exc).__name__}: {exc}'
            fallos.append({'algoritmo': nombre, 'error': msg})
            resumen.append({'algoritmo': nombre, 'estado': 'ERROR',
                            'registros_utilizados': None, 'variables': 10,
                            'grupos': None, 'ruido_o_alertas': None,
                            'porcentaje_ruido_o_alertas': None, 'detalles': msg})
            guardar_csv(pd.DataFrame(resumen), salida / '00_resumen_modelos.csv')
            avisar(f'  ERROR EN {nombre}: {msg}')
            # Continuar con los demas; conservar resultados parciales sin simular exito.
    (salida / '00_errores.json').write_text(json.dumps(fallos, indent=2,
                                       ensure_ascii=False), encoding='utf-8')
    avisar('\n=== RESUMEN DE EJECUCION DIRECTA SOBRE LOS 143.561 REGISTROS ===')
    avisar(pd.DataFrame(resumen).drop(columns=['detalles']).to_string(index=False))
    avisar(f'Archivos guardados en: {salida.resolve()}')
    avisar('NO hay separacion 70/15/15. Estos resultados son DENTRO de muestra; '
          'ruido y alertas NO son averias confirmadas.')
    if fallos:
        avisar(f'ATENCION: {len(fallos)} modelo(s) no completados; revisar 00_errores.json')
        sys.exit(1)



# ---------------------------------------------------------------------------
# FIGURAS ADICIONALES: siempre se calculan con etiquetas REALES del modelo,
# 143.561 registros para el ajuste; cuando hay muestreo es SOLO visual/SHAP.
# ---------------------------------------------------------------------------

def guardar_fig(fig, ruta: Path):
    fig.savefig(ruta, dpi=165, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    avisar(f'    + {ruta.name}')


def armar_grafico(titulo, xlabel='', ylabel='', tam=(11, 5.5)):
    fig, ax = plt.subplots(figsize=tam)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def elegir_para_grafico(labels: np.ndarray, limite=22000, seed=2026):
    """Solo muestrea puntos VISIBLES, sin reducir el ajuste ni los CSV."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n <= limite:
        return np.arange(n)
    clases, counts = np.unique(labels, return_counts=True)
    elegido = []
    # Garantiza grupos minoritarios visibles, sin inventar sus proporciones.
    for c, cnt in zip(clases, counts):
        ids = np.flatnonzero(labels == c)
        cuota = min(cnt, max(100, round(limite * cnt / n)))
        elegido.append(rng.choice(ids, size=cuota, replace=False))
    idx = np.unique(np.concatenate(elegido))
    if len(idx) > limite:
        idx = rng.choice(idx, size=limite, replace=False)
    return np.sort(idx)


def pca_zoom_y_completo(nombre, coord, labels, var, salida):
    idx = elegir_para_grafico(labels, 24000)
    x = coord[idx]
    ys = labels[idx]
    from matplotlib.colors import Normalize
    for zoom in (False, True):
        fig, ax = armar_grafico(
            f'{nombre.replace("_", " ").title()} | PCA {"zona central" if zoom else "completo"} | 143.561 filas ajustadas',
            f'PC1 ({100*var[0]:.2f}% varianza)', f'PC2 ({100*var[1]:.2f}% varianza)', (10, 6.8))
        if nombre == 'isolation_forest':
            orden = [0, 1]
            etiquetas = {0: 'Sin alerta', 1: 'Alerta estadistica'}
        else:
            orden = ([-1] if np.any(ys == -1) else []) + [int(c) for c in np.unique(ys) if c != -1]
            etiquetas = {c: ('Ruido' if c == -1 else f'Grupo {c}') for c in orden}
        cmap = plt.get_cmap('tab20')
        for j, c in enumerate(orden):
            mask = ys == c
            if mask.any():
                color = ('#a0a8b1' if c == -1 else '#ba4057' if nombre == 'isolation_forest' and c == 1
                         else '#527c9a' if nombre == 'isolation_forest' else cmap(j % 20))
                ax.scatter(x[mask, 0], x[mask, 1], s=2 if c != -1 else 1,
                           alpha=.33 if c != -1 else .11, c=[color], rasterized=True,
                           linewidth=0, label=etiquetas[c])
        if zoom:
            ax.set_xlim(*np.quantile(coord[:, 0], [.025, .975]))
            ax.set_ylim(*np.quantile(coord[:, 1], [.025, .975]))
        handles, legend_labels = ax.get_legend_handles_labels()
        if len(handles) <= 18:
            ax.legend(loc='best', fontsize=7, markerscale=4)
        else:
            avisar('      Leyenda PCA abreviada: demasiadas clases; consultar CSV.')
        ax.text(.01, .01, f'Puntos visibles: {len(idx):,}; ajuste: {len(labels):,}',
                transform=ax.transAxes, fontsize=8, va='bottom', alpha=.7)
        guardar_fig(fig, salida / f'03_pca_{nombre}_{"zoom" if zoom else "completo"}.png')


def grafico_tamanos(nombre, labels, salida):
    clases, cuentas = np.unique(labels, return_counts=True)
    orden = np.argsort(cuentas)[::-1]
    clases, cuentas = clases[orden], cuentas[orden]
    fig, ax = armar_grafico(f'{nombre.title()} | Distribucion de observaciones',
                            'Categoria', 'Numero de registros', (12, 5.5))
    visibles = min(len(clases), 24)
    nombres = [('Ruido' if c == -1 else 'Alerta' if nombre == 'isolation_forest' and c == 1
                else 'Sin alerta' if nombre == 'isolation_forest' and c == 0
                else f'G{c}') for c in clases[:visibles]]
    ax.bar(np.arange(visibles), cuentas[:visibles], color='#527c9a')
    ax.set_xticks(np.arange(visibles), nombres, rotation=40 if visibles > 9 else 0)
    for i, n in enumerate(cuentas[:visibles]):
        ax.text(i, n, f'{n:,}', ha='center', va='bottom', fontsize=7)
    if len(clases) > visibles:
        fig.text(.15, .01, f'Se muestran las {visibles} clases mas grandes de {len(clases)}.', fontsize=9)
    guardar_fig(fig, salida / f'04_tamanos_{nombre}.png')


def grafico_evolucion(nombre, smp, labels, salida, n_bloques=120):
    n = len(smp)
    idxs = np.array_split(np.arange(n), n_bloques)
    clases, counts = np.unique(labels, return_counts=True)
    if nombre == 'kmeans':
        elegidas = clases[np.argsort(counts)[::-1][:min(len(clases), 6)]]
    elif nombre == 'isolation_forest':
        elegidas = np.array([1])
    else:
        elegidas = np.array([-1])
    fig, ax = armar_grafico(f'{nombre.replace("_", " ").title()} | Evolucion por contador SMP',
                            'SMP (contador; NO segundos)', 'Porcentaje de registros por bloque')
    mid = np.asarray([np.median(smp[b]) for b in idxs])
    for c in elegidas:
        values = np.asarray([100*np.mean(labels[b] == c) for b in idxs])
        etiqueta = ('Ruido' if c == -1 else 'Alertas' if nombre == 'isolation_forest' else f'Grupo {c}')
        ax.plot(mid, values, lw=1.4, label=etiqueta)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 105)
    guardar_fig(fig, salida / f'05_evolucion_smp_{nombre}.png')
    tabla = {SMP: mid, 'n_registros_bloque': [len(x) for x in idxs]}
    for c in elegidas:
        tabla[f'porcentaje_clase_{c}'] = [100*np.mean(labels[b]==c) for b in idxs]
    guardar_csv(pd.DataFrame(tabla), salida / f'05_evolucion_smp_{nombre}.csv')


def grafico_perfiles(nombre, X, labels, salida):
    clases, cuentas = np.unique(labels, return_counts=True)
    idx = np.argsort(cuentas)[::-1][:min(8, len(clases))]
    cs = clases[idx]
    perfiles = np.stack([np.mean(X[labels == c], axis=0) for c in cs])
    fig, ax = armar_grafico(f'{nombre.replace("_", " ").title()} | Perfil medio de diez senales escaladas',
                            'Senal', 'Media en escala transformada', (12, 5.8))
    for j, c in enumerate(cs):
        ax.plot(CORTOS, perfiles[j], marker='o', ms=3, lw=1.5,
                label='Ruido' if c == -1 else 'Alerta' if nombre=='isolation_forest' and c==1
                else 'Sin alerta' if nombre=='isolation_forest' and c==0 else f'Grupo {c} ({cuentas[idx[j]]:,})')
    ax.axhline(0, color='grey', lw=.8)
    ax.legend(loc='best', fontsize=8)
    fig.text(.12, .01, 'Los perfiles no demuestran causalidad ni fallas de sensores.', fontsize=8)
    guardar_fig(fig, salida / f'06_perfiles_{nombre}.png')
    guardar_csv(pd.DataFrame(perfiles, columns=CORTOS).assign(grupo=cs),
                salida / f'06_perfiles_{nombre}.csv')


def diagnostico_kmeans(X, labels, salida, args, modelo):
    if modelo is not None and hasattr(modelo, 'cluster_centers_'):
        guardar_csv(pd.DataFrame(modelo.cluster_centers_, columns=CORTOS).assign(
                    cluster=np.arange(modelo.n_clusters)), salida / '07_centroides_kmeans.csv')
    if args.sin_barrido_k:
        avisar('    Barrido K omitido por --sin-barrido-k.')
        return
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(1600, len(X)), replace=False)
    max_k = min(max(2, args.k_max), 40)
    ks, inercias, siluetas = [], [], []
    for k in range(1, max_k+1):
        avisar(f'    Barrido K-Means: K={k}/{max_k} (ajuste con 143.561 filas)')
        # Cada K usa TODO X. El muestreo es exclusivamente para silhouette.
        if k == args.k and modelo is not None:
            m = modelo
        else:
            m = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300).fit(X)
        ks.append(k)
        inercias.append(float(m.inertia_))
        lab = m.predict(X[idx])
        sil = float(silhouette_score(X[idx], lab)) if k > 1 and len(np.unique(lab)) > 1 else np.nan
        siluetas.append(sil)
    guardar_csv(pd.DataFrame({'k':ks, 'sse_inercia':inercias,
                  'perdida_pct_respecto_k1':100*np.asarray(inercias)/inercias[0],
                  'silueta_muestra_1600':siluetas}), salida / '07_barrido_kmeans_completo.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ks, inercias, 'o-', ms=4)
    axes[0].set_title('K vs perdida (SSE / inercia)')
    axes[0].set_xlabel('Numero de grupos K')
    axes[0].set_ylabel('SSE (menor es mejor; no basta solo)')
    axes[1].plot(ks[1:], siluetas[1:], 'o-', ms=4)
    axes[1].set_title('K vs silueta sobre 1.600 filas muestreadas')
    axes[1].set_xlabel('Numero de grupos K')
    axes[1].set_ylabel('Silueta promedio')
    for ax in axes: ax.grid(alpha=.18)
    guardar_fig(fig, salida / '07_kmeans_codo_y_silueta.png')


def diagnostico_dbscan(resultado, X, coord, salida, args):
    kd = resultado['distancia_k_dbscan'].to_numpy(float)
    fig, ax = armar_grafico('DBSCAN | curva k-distancia en 143.561 registros',
                            'Observaciones ordenadas por distancia', f'Distancia al vecino {args.ms_dbscan}')
    orden = np.sort(kd)
    ax.plot(np.arange(len(orden))[::max(1, len(orden)//10000)],
            orden[::max(1, len(orden)//10000)], lw=1.5)
    ax.axhline(args.eps, color='#ba4057', lw=1.6, linestyle='--', label=f'eps = {args.eps:.5g}')
    ax.legend()
    guardar_fig(fig, salida / '07_dbscan_curva_k_distancia.png')
    core = resultado['es_nucleo_dbscan'].to_numpy(bool)
    lab = resultado['cluster_dbscan'].to_numpy(int)
    tipos = np.where(core, 0, np.where(lab == -1, 2, 1))
    nombres = ['Nucleo', 'Frontera', 'Ruido']
    fig, ax = armar_grafico('DBSCAN | puntos nucleares, frontera y ruido',
                            'PC1', 'PC2', (11, 6.5))
    idx = elegir_para_grafico(tipos, 25000)
    for t, col in zip((2,1,0), ('#a4a4a4', '#e4a35c', '#367f99')):
        ids = idx[tipos[idx] == t]
        ax.scatter(coord[ids,0], coord[ids,1], c=col, s=2, alpha=.3,
                   linewidth=0, rasterized=True,
                   label=f'{nombres[t]}: {np.count_nonzero(tipos == t):,}')
    ax.legend(markerscale=4)
    guardar_fig(fig, salida / '08_dbscan_nucleo_frontera_ruido.png')


def firma_hdbscan(X, args):
    """Vincula cache con TODOS los datos (10 dimensiones), escalado y parametros."""
    m = hashlib.sha256()
    m.update(np.ascontiguousarray(X).view(np.uint8))
    return {'n_registros': int(len(X)), 'n_senales': int(X.shape[1]),
            'sha256_matriz': m.hexdigest(), 'escalado': args.escalado,
            'min_cluster_size': args.mcs, 'min_samples': args.ms_hdbscan,
            'seleccion': args.seleccion, 'metrica': 'euclidean'}


def guardar_modelo_hdbscan(modelo, labels, X, args, salida):
    """Conserva jerarquia REAL ANTES de cualquier grafico que pudiera fallar."""
    ruta=salida/'00_modelo_total_hdbscan.joblib'
    temporal=ruta.with_name(ruta.name+'.tmp')
    meta=salida/'00_modelo_total_hdbscan_meta.json'
    aviso='  Guardando jerarquia HDBSCAN para no tener que recalcularla si falla un grafico ...'
    avisar(aviso)
    joblib.dump(modelo, temporal, compress=3)
    temporal.replace(ruta)
    info=firma_hdbscan(X,args)
    info['sha256_etiquetas']=hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int32).view(np.uint8)).hexdigest()
    meta.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    avisar(f'    + {ruta.name} (reutilizable en siguientes ejecuciones)')


def cargar_modelo_hdbscan(X, labels, args, salida):
    ruta=salida/'00_modelo_total_hdbscan.joblib'
    meta=salida/'00_modelo_total_hdbscan_meta.json'
    if not ruta.is_file() or not meta.is_file():
        return None
    info=json.loads(meta.read_text(encoding='utf-8'))
    actual=firma_hdbscan(X,args)
    coincide=all(info.get(k)==v for k,v in actual.items())
    hash_labels=hashlib.sha256(np.ascontiguousarray(labels,dtype=np.int32).view(np.uint8)).hexdigest()
    if not coincide or info.get('sha256_etiquetas')!=hash_labels:
        avisar('  Cache de HDBSCAN con otros parametros/datos/etiquetas. No se reutiliza.')
        return None
    avisar('  Reutilizando jerarquia HDBSCAN guardada: no se vuelve a entrenar.')
    return joblib.load(ruta)


def dibujar_jerarquia_hdbscan_resumida(arbol, salida):
    """Plan B: muestra ramas entre clusters reales, sin parches Ellipse."""
    df=arbol.to_pandas()
    guardar_csv(df, salida/'08_hdbscan_arbol_condensado_datos.csv')
    nodos=df.loc[df['child_size']>1,['parent','child','lambda_val','child_size']].copy()
    fig,ax=armar_grafico('HDBSCAN | jerarquia condensada: ramas entre grupos',
                        'Ramas ordenadas (escala visual)', 'Lambda de separacion', (13,6.8))
    if nodos.empty:
        ax.text(.5,.5,'El arbol no presenta subdivisiones entre grupos',
                transform=ax.transAxes,ha='center',va='center')
    else:
        # Los puntos individuales se omiten SOLO DEL DIBUJO; se exportan al CSV.
        # El orden de hojas se deriva de la estructura, no de posiciones PCA.
        hijos={}
        for pa,ch,lam,tam in nodos.itertuples(index=False, name=None):
            hijos.setdefault(int(pa),[]).append((int(ch),float(lam),int(tam)))
        for pa in hijos:
            hijos[pa].sort(key=lambda t:(-t[2],t[0]))
        raiz=min(hijos)
        x={}; y={raiz:0.0}; contador=0
        pila=[(raiz,False)]; visitados=set()
        while pila:
            nodo,final=pila.pop()
            if final:
                sub=hijos.get(nodo,[])
                if sub:
                    x[nodo]=float(np.mean([x[c] for c,_,_ in sub if c in x]))
                elif nodo not in x:
                    x[nodo]=contador; contador+=1
                continue
            if nodo in visitados:
                continue
            visitados.add(nodo)
            pila.append((nodo,True))
            for c,lam,tam in reversed(hijos.get(nodo,[])):
                y[c]=lam
                pila.append((c,False))
        from matplotlib.collections import LineCollection
        segmentos=[]
        for pa,ch,lam,tam in nodos.itertuples(index=False,name=None):
            pa=int(pa); ch=int(ch)
            if pa in x and ch in x and np.isfinite(lam):
                segmentos.append([(x[pa],y.get(pa,0.0)),(x[pa],float(lam))])
                segmentos.append([(x[pa],float(lam)),(x[ch],float(lam))])
        if segmentos:
            ax.add_collection(LineCollection(segmentos,colors='#4b7897',linewidths=.9,alpha=.8))
            ax.autoscale_view()
        ax.text(.01,.98,f'{len(nodos):,} ramas entre clusters; {len(df):,} filas en CSV',
                transform=ax.transAxes,va='top',fontsize=9,
                bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    ax.text(.01,-.14,'Visualizacion simplificada de la jerarquia real; los puntos individuales estan en el CSV.',
            transform=ax.transAxes,fontsize=9)
    guardar_fig(fig,salida/'08_hdbscan_arbol_condensado.png')


def diagnostico_hdbscan(resultado, modelo, salida):
    p = resultado['fuerza_hdbscan'].to_numpy(float)
    g = resultado['glosh_hdbscan'].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(p[np.isfinite(p)], bins=40, color='#4a8299')
    axes[0].set(title='Fuerza de pertenencia', xlabel='Probabilidad HDBSCAN (no falla)', ylabel='Cantidad')
    axes[1].hist(g[np.isfinite(g)], bins=50, color='#b56770')
    axes[1].set(title='Puntaje GLOSH', xlabel='Mayor = mas atipico localmente', ylabel='Cantidad')
    for a in axes: a.grid(alpha=.15)
    guardar_fig(fig, salida / '07_hdbscan_pertenencia_y_glosh.png')
    if modelo is not None:
        if hasattr(modelo, 'condensed_tree_'):
            # BUG de compatibilidad: con select_clusters=True, algunas versiones
            # de HDBSCAN entregan un alto numpy(1,) a matplotlib.patches.Ellipse.
            # La jerarquia no cambia si evitamos esas elipses decorativas.
            try:
                fig, ax = plt.subplots(figsize=(13, 6))
                modelo.condensed_tree_.plot(axis=ax, select_clusters=False, colorbar=False)
                ax.set_title('HDBSCAN | arbol condensado de 143.561 registros')
                guardar_fig(fig, salida / '08_hdbscan_arbol_condensado.png')
                # CSV completo para inspeccion y trazabilidad.
                guardar_csv(modelo.condensed_tree_.to_pandas(),
                            salida / '08_hdbscan_arbol_condensado_datos.csv')
            except (ValueError, TypeError, OverflowError, MemoryError) as e:
                plt.close('all')
                avisar(f'  Grafico nativo incompatible ({type(e).__name__}). '
                      'Dibujando jerarquia real con un metodo alternativo ...')
                dibujar_jerarquia_hdbscan_resumida(modelo.condensed_tree_, salida)
        pers = np.asarray(getattr(modelo, 'cluster_persistence_', []), dtype=float)
        if len(pers):
            fig, ax = armar_grafico('HDBSCAN | persistencia de grupos', 'Grupo', 'Persistencia')
            ax.bar(np.arange(len(pers)), pers, color='#6583aa')
            guardar_fig(fig, salida / '09_hdbscan_persistencia.png')
    else:
        avisar('    Arbol condensado no disponible: version anterior no guardo modelo HDBSCAN. ' 
              'Para generarlo: --modo graficos --arbol-hdbscan')


def diagnostico_if(resultado, smp, salida):
    score = resultado['puntuacion_anomalia_if'].to_numpy(float)
    decision = resultado['decision_function_if'].to_numpy(float)
    alerta = resultado['alerta_isolation_forest'].to_numpy(int).astype(bool)
    fig, ax = armar_grafico('Isolation Forest | distribucion de puntuacion anomala',
                            'Puntuacion (mas alta = mas atipica)', 'Numero de registros')
    ax.hist(score[~alerta], bins=70, alpha=.72, label='Sin alerta', color='#527c9a')
    ax.hist(score[alerta], bins=70, alpha=.72, label='Alerta', color='#b44558')
    ax.axvline(.5, color='black', ls='--', label='Umbral contamination=auto (0.5)')
    ax.legend()
    guardar_fig(fig, salida / '07_if_histograma_puntuaciones.png')
    fig, ax = armar_grafico('Isolation Forest | puntuacion a lo largo de SMP',
                            'SMP (contador de adquisicion)', 'Puntuacion de anomalia')
    salto = max(1, len(smp)//12000)
    ax.scatter(smp[::salto], score[::salto], s=2, alpha=.28, rasterized=True)
    ax.axhline(.5, color='#b44558', ls='--', label='Umbral automatico 0.5')
    ax.legend()
    guardar_fig(fig, salida / '08_if_evolucion_puntuacion.png')
    # El porcentaje de alertas por bloque ya esta en 05_evolucion_smp_isolation_forest.
    extremo = np.argsort(score)[-50:][::-1]
    guardar_csv(pd.DataFrame({SMP:smp[extremo], 'puntuacion_anomalia':score[extremo],
                   'alerta':alerta[extremo]}), salida / '09_if_50_mayores_puntajes.csv')


def graficos_adicionales(nombre, smp, X, coord, var, labels, resultado, salida, args, modelo=None):
    avisar(f'  [GRAFICOS] {nombre}: PCA + distribuciones + perfiles + evolucion ...')
    pca_zoom_y_completo(nombre, coord, labels, var, salida)
    grafico_tamanos(nombre, labels, salida)
    grafico_evolucion(nombre, smp, labels, salida)
    grafico_perfiles(nombre, X, labels, salida)
    if nombre == 'kmeans':
        diagnostico_kmeans(X, labels, salida, args, modelo)
    elif nombre == 'dbscan':
        diagnostico_dbscan(resultado, X, coord, salida, args)
    elif nombre == 'hdbscan':
        diagnostico_hdbscan(resultado, modelo, salida)
    else:
        diagnostico_if(resultado, smp, salida)


def ejecutar_shap_total(modelo, X, smp, labels, resultado, salida, args):
    """Explica el nuevo IF ENTRENADO SOBRE 143.561 filas; SHAP usa muestra.

    No se atribuyen explicaciones a HDBSCAN/DBSCAN; SHAP aqui explica
    exclusivamente puntuacion = -IsolationForest.score_samples(X).
    """
    try:
        import shap
    except ImportError as e:
        raise RuntimeError('Para generar SHAP instala en este Python: python -m pip install shap') from e
    if args.shap_n < 2 or args.shap_fondo < 2 or args.shap_permutaciones < 1:
        raise ValueError('--shap-n y --shap-fondo >= 2, --shap-permutaciones >= 1')
    avisar(f'  [SHAP] Explicando puntuacion de IF completo con una muestra de '
          f'{args.shap_n} mediciones y fondo de {args.shap_fondo}; modelo fijo.')
    rng = np.random.default_rng(117)
    alerta = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    n_alerta = min(len(alerta), args.shap_n // 2)
    n_normal = min(len(normal), args.shap_n - n_alerta)
    if n_alerta+n_normal < args.shap_n:
        n_alerta = min(len(alerta), args.shap_n-n_normal)
    idx = np.sort(np.concatenate([rng.choice(alerta, n_alerta, replace=False) if n_alerta else np.array([],dtype=int),
                                   rng.choice(normal, n_normal, replace=False) if n_normal else np.array([],dtype=int)]))
    fondo = X[rng.choice(len(X), min(args.shap_fondo, len(X)), replace=False)]
    def funcion(Z):
        return -modelo.score_samples(np.asarray(Z, dtype=np.float32))
    masker = shap.maskers.Independent(fondo, max_samples=len(fondo))
    exp = shap.Explainer(funcion, masker, algorithm='permutation', feature_names=CORTOS, seed=42)
    chunks = []
    max_evals = (2*X.shape[1]+1)*args.shap_permutaciones
    for a in range(0, len(idx), 10):
        b = idx[a:a+10]
        avisar(f'    SHAP filas explicadas {a+len(b)}/{len(idx)} ...')
        e = exp(X[b].astype(np.float64), max_evals=max_evals, batch_size=128, silent=True)
        chunks.append((np.asarray(e.values, dtype=float), np.asarray(e.base_values, dtype=float).reshape(-1)))
    sv = np.concatenate([c[0] for c in chunks])
    base_vals = np.concatenate([c[1] for c in chunks])
    score = funcion(X[idx])
    reconstruccion = base_vals + sv.sum(axis=1)
    max_error = float(np.max(np.abs(reconstruccion-score)))
    if sv.shape != (len(idx), len(VARIABLES)) or max_error > 2e-5:
        raise RuntimeError(f'No se reconstruyen correctamente las puntuaciones SHAP: {max_error}')
    tabla = pd.DataFrame({SMP:smp[idx], 'alerta_isolation_forest':labels[idx],
                          'score_real':score, 'score_reconstruido':reconstruccion,
                          'valor_base_shap':base_vals})
    for j, c in enumerate(CORTOS):
        tabla['SHAP_'+c.replace(' ','_')] = sv[:,j]
    guardar_csv(tabla, salida / '10_shap_contribuciones_muestra.csv')
    prevalencia = float(np.mean(labels == 1))
    # El muestreo dentro de cada clase es aleatorio; ponderamos su prevalencia.
    pesos = np.where(labels[idx] == 1,
                     prevalencia / max(n_alerta,1),
                     (1-prevalencia) / max(n_normal,1))
    pesos = pesos / pesos.sum()
    importancia = np.sum(np.abs(sv)*pesos[:,None], axis=0)
    orden = np.argsort(importancia)[::-1]
    guardar_csv(pd.DataFrame({'variable':np.array(CORTOS)[orden],
                              'importancia_media_absoluta_ponderada':importancia[orden],
                              'importancia_relativa_pct':100*importancia[orden]/importancia.sum()}),
                salida / '10_shap_importancia.csv')
    # 1. Importancia ponderada: las diez variables.
    fig, ax = armar_grafico('SHAP | importancia de las 10 senales en Isolation Forest total',
                            'Media ponderada de |SHAP| (puntuacion IF)', 'Senal', (11, 6))
    ax.barh(np.array(CORTOS)[orden][::-1], importancia[orden][::-1], color='#527c9a')
    guardar_fig(fig, salida / '10_shap_importancia_10_senales.png')
    # 2. Beeswarm: posicion SHAP, color valor escalado.
    fig, ax = armar_grafico('SHAP | distribucion de contribuciones (muestra estratificada)',
                            'SHAP sobre puntuacion de anomalia', 'Senal', (11, 6.8))
    rng_jitter = np.random.default_rng(9)
    for rank,j in enumerate(orden[::-1]):
        vals = X[idx,j]
        lo,hi = np.quantile(vals,[.05,.95])
        col = np.clip((vals-lo)/max(hi-lo,1e-10),0,1)
        ax.scatter(sv[:,j], rank+rng_jitter.uniform(-.25,.25,size=len(idx)),
                   s=15,c=col,cmap='coolwarm',vmin=0,vmax=1,alpha=.8, linewidths=0)
    ax.set_yticks(range(10), np.array(CORTOS)[orden[::-1]])
    ax.axvline(0,color='black',lw=.8)
    guardar_fig(fig, salida / '11_shap_beeswarm.png')
    # 3. Heatmap SHAP: todas las explicaciones de la muestra.
    ordenar_filas = np.argsort(score)
    fig, ax = plt.subplots(figsize=(12, 6))
    lim = max(float(np.quantile(np.abs(sv),.98)),1e-8)
    im = ax.imshow(sv[ordenar_filas].T,cmap='coolwarm',aspect='auto',vmin=-lim,vmax=lim)
    ax.set_yticks(range(10),CORTOS)
    ax.set_xlabel('Muestras ordenadas por puntuacion de anomalia')
    ax.set_title('SHAP | contribucion por senal y observacion')
    fig.colorbar(im,ax=ax,label='Contribucion SHAP')
    guardar_fig(fig, salida / '12_shap_mapa_calor.png')
    # 4. Ejemplo local: alerta mas atipica en la muestra.
    caso = int(np.argmax(score))
    order = np.argsort(np.abs(sv[caso]))
    fig,ax = armar_grafico(f'SHAP | explicacion individual SMP={smp[idx[caso]]}',
                           'Contribucion a puntuacion de anomalia', 'Senal', (10, 6))
    ax.barh(np.array(CORTOS)[order],sv[caso,order],
            color=np.where(sv[caso,order]>=0,'#b44558','#527c9a'))
    ax.axvline(0,color='black',lw=.9)
    fig.text(.12,.02,f'Score = base ({base_vals[caso]:.5f}) + suma SHAP = {score[caso]:.5f}; '
             f'muestra, no diagnostico fisico.',fontsize=8)
    guardar_fig(fig, salida / '13_shap_explicacion_individual.png')
    # 5. Dependencia de variable con mayor importancia.
    v=int(orden[0])
    fig,ax=armar_grafico(f'SHAP | dependencia {CORTOS[v]}',
                         f'Valor escalado: {CORTOS[v]}', 'Contribucion SHAP')
    ax.scatter(X[idx,v],sv[:,v],s=25,alpha=.55,c=np.where(labels[idx]==1,'#b44558','#527c9a'))
    ax.axhline(0,color='grey',lw=.8)
    guardar_fig(fig,salida/'14_shap_dependencia_variable_principal.png')
    # 6. Comparacion SHAP alerta vs sin alerta.
    fig,ax=armar_grafico('SHAP | importancia media por tipo de observacion',
                          'Media de |SHAP| en muestra aleatoria de cada clase', 'Senal', (11,6))
    altura=np.arange(10)
    anchura=.38
    if n_normal:
        media_n=np.mean(np.abs(sv[labels[idx]==0]),axis=0)
        ax.barh(altura-anchura/2,media_n,anchura,color='#527c9a',label='Sin alerta')
    if n_alerta:
        media_a=np.mean(np.abs(sv[labels[idx]==1]),axis=0)
        ax.barh(altura+anchura/2,media_a,anchura,color='#b44558',label='Alerta')
    ax.set_yticks(altura,CORTOS)
    ax.legend()
    guardar_fig(fig,salida/'15_shap_por_condicion.png')
    (salida/'10_shap_metodologia.json').write_text(json.dumps({
       'modelo':'IsolationForest entrenado con todos los 143561 registros',
       'registros_entrenamiento':len(X), 'registros_explicados_shap':len(idx),
       'fondo':len(fondo), 'alertas_explicadas':int(n_alerta),
       'normales_explicadas':int(n_normal), 'prevalencia_alertas_total':prevalencia,
       'puntuacion_explicada':'-modelo.score_samples(X) (mayor = mas atipica)',
       'metodo':'PermutationExplainer', 'permutaciones':args.shap_permutaciones,
       'maximo_error_aditividad':max_error,
       'limitacion':'SHAP es una muestra de explicaciones, no se calcula individualmente para todas las filas; no demuestra causa fisica.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    avisar(f'  SHAP completado: {len(idx)} observaciones explicadas; error aditividad {max_error:.3g}')


def modo_solo_graficos(X, smp, coord, var, salida, args, modelos):
    avisar('=== MODO GRAFICOS: SE REUTILIZAN ETIQUETAS YA CALCULADAS ===')
    avisar('No se reajustan K-Means, DBSCAN ni HDBSCAN por defecto.')
    errores = []
    realizados = []
    for nombre in modelos:
        path = salida / f'01_resultado_total_{nombre}.csv'
        if not path.exists():
            avisar(f'  FALTA {path.name}: ejecuta --modo final --algoritmo {nombre}')
            errores.append(nombre)
            continue
        df=pd.read_csv(path,sep=';',encoding='utf-8-sig')
        if len(df)!=len(smp) or not np.array_equal(df[SMP].to_numpy(np.int64),smp):
            raise ValueError(f'Los SMP en {path} no corresponden a las 143.561 filas actuales.')
        columna='alerta_isolation_forest' if nombre=='isolation_forest' else 'cluster_'+nombre
        if columna not in df:
            raise ValueError(f'Falta {columna} en {path}')
        labels=df[columna].to_numpy(np.int32)
        modelo=None
        if nombre=='kmeans' and not args.sin_barrido_k:
            # Para la curva se reajustan modelos de distintos K, NO cambia las etiquetas originales.
            avisar('  La curva del codo evaluara K adicionales con todas las filas.')
        if nombre=='hdbscan' and args.arbol_hdbscan:
            modelo=cargar_modelo_hdbscan(X,labels,args,salida)
            if modelo is None:
                avisar('  Sin cache compatible: se ajustara HDBSCAN UNA VEZ para recuperar el arbol ...')
                new_labels, ex, info, modelo=ajustar_modelo(nombre,X,args)
                if not np.array_equal(labels,new_labels):
                    raise ValueError('HDBSCAN reentrenado no coincide con las etiquetas guardadas. '
                                     'Revisa parametros/escalado; NO se dibujara un arbol incongruente.')
                guardar_modelo_hdbscan(modelo,labels,X,args,salida)
        if nombre=='isolation_forest':
            path_modelo=salida/'00_modelo_total_isolation_forest.joblib'
            if path_modelo.is_file():
                modelo=joblib.load(path_modelo)
            else:
                avisar('  Modelo de la version anterior no fue guardado: reconstruyendo IF con '
                      'los mismos parametros para poder calcular SHAP REAL, sin simular resultados ...')
                new_labels, ex, info, modelo=ajustar_modelo(nombre,X,args)
                joblib.dump(modelo,path_modelo,compress=3)
            comprobar=np.unique(np.linspace(0,len(X)-1,75,dtype=int))
            if np.max(np.abs(-modelo.score_samples(X[comprobar])-
                                 df['puntuacion_anomalia_if'].to_numpy(float)[comprobar]))>2e-7:
                raise ValueError('El Isolation Forest cargado no coincide con las puntuaciones anteriores. '
                                 'Es necesario ejecutar --modo final --algoritmo isolation_forest.')
            if not np.array_equal((modelo.decision_function(X[comprobar]) < 0).astype(int),labels[comprobar]):
                raise ValueError('Etiquetas de IF guardadas no coinciden con el modelo reconstruido.')
        avisar(f'  [REUTILIZANDO] {nombre}: {len(df):,} filas verificadas por SMP.')
        graficos_adicionales(nombre,smp,X,coord,var,labels,df,salida,args,modelo)
        if nombre=='isolation_forest':
            ejecutar_shap_total(modelo,X,smp,labels,df,salida,args)
        realizados.append(nombre)
    avisar('Graficos completados para: '+', '.join(realizados))
    avisar(f'Carpeta: {salida}')
    if errores:
        raise FileNotFoundError('Modelos sin resultados previos: '+', '.join(errores))



def _seleccion_estratificada(ident, etiquetas, max_puntos=33000):
    """Muestra solo para graficar. No altera datos/modelos ni proporciones reportadas."""
    rng=np.random.default_rng(2026)
    u, n=np.unique(etiquetas, return_counts=True)
    tomadas=[]
    for v, nn in zip(u,n):
        idx=np.flatnonzero(etiquetas==v)
        if v==(-1 if ident=='hdbscan' else 0):
            cuota=min(nn, 19000)
        else:
            cuota=min(nn, max(120, int(max_puntos*nn/len(etiquetas))))
        tomadas.append(rng.choice(idx, size=cuota, replace=False))
    return np.sort(np.concatenate(tomadas))


def _dispersion_coloreada(nombre, xy, labels, nombres_ejes, salida, prefijo, zoom=False, solo_interes=False):
    """Dispersión sobre datos reales: IF 0/1; HDBSCAN -1 o grupos."""
    if nombre not in ('hdbscan','isolation_forest'):
        return
    idx=_seleccion_estratificada(nombre, labels)
    x=xy[idx]
    labs=labels[idx]
    fig,ax=plt.subplots(figsize=(10.5,7.5))
    if nombre=='isolation_forest':
        total_alert=int((labels==1).sum()); total_normal=int((labels==0).sum())
        if not solo_interes:
            m=labs==0
            ax.scatter(x[m,0],x[m,1],s=5.5,alpha=.24,c='#448bcb',
                       linewidths=0,rasterized=True,label=f'Sin alerta ({total_normal:,})',zorder=1)
        m=labs==1
        ax.scatter(x[m,0],x[m,1],s=8.5,alpha=.78,c='#eb8a23',
                   linewidths=0,rasterized=True,label=f'Alerta estadistica ({total_alert:,})',zorder=3)
        tipo='SOLO ALERTAS (se ocultan normales)' if solo_interes else 'NORMALES vs ALERTAS'
    else:
        grupos=np.unique(labels[labels>=0])
        ruido=int(np.count_nonzero(labels==-1))
        if not solo_interes:
            m=labs==-1
            ax.scatter(x[m,0],x[m,1],s=3,alpha=.12,c='#7c8797',
                       linewidths=0,rasterized=True,label=f'Ruido ({ruido:,})',zorder=1)
        colores=plt.get_cmap('tab20')
        for i,g in enumerate(grupos):
            m=labs==g
            if m.any():
                ax.scatter(x[m,0],x[m,1],s=12,alpha=.92,
                           c=[colores(i % 20)],edgecolors='none',rasterized=True,
                           label=f'Grupo {g} ({np.count_nonzero(labels==g):,})',zorder=2+i/100)
        if not len(grupos):
            ax.text(.5,.52,'HDBSCAN no selecciono grupos.\n100 % ruido.',
                    transform=ax.transAxes,ha='center',va='center',fontsize=15,
                    bbox={'facecolor':'white','alpha':.9,'edgecolor':'#cccccc'})
        tipo=('SOLO GRUPOS (se omite ruido)' if solo_interes else 'GRUPOS y RUIDO')
    if zoom:
        interesantes=labels>=0 if nombre=='hdbscan' else labels>=0
        pts=xy[interesantes]
        if len(pts):
            xmin,xmax=np.quantile(pts[:,0],[.02,.98]); ymin,ymax=np.quantile(pts[:,1],[.02,.98])
            if xmax>xmin: ax.set_xlim(xmin-.04*(xmax-xmin),xmax+.04*(xmax-xmin))
            if ymax>ymin: ax.set_ylim(ymin-.04*(ymax-ymin),ymax+.04*(ymax-ymin))
    ax.set_title(f'{"Isolation Forest" if nombre=="isolation_forest" else "HDBSCAN"} | {tipo}\n143.561 registros evaluados, 10 senales usadas en el modelo',
                 fontsize=12,fontweight='bold')
    ax.set_xlabel(nombres_ejes[0]);ax.set_ylabel(nombres_ejes[1]);ax.grid(alpha=.11)
    ax.spines[['top','right']].set_visible(False)
    handles, labs_ley=ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:19],labs_ley[:19],loc='upper right',fontsize=8,markerscale=2,
                  framealpha=.93)
    ax.text(.015,.012,f'Muestra visual: {len(idx):,} puntos de {len(labels):,}; conteos de leyenda = total real.\n'
                        'Coordenadas 2D: no representan las 10 dimensiones completas.',
             transform=ax.transAxes,fontsize=8,va='bottom',bbox={'facecolor':'white','alpha':.83,'edgecolor':'none'})
    fig.tight_layout()
    ruta=salida/f'{prefijo}_{nombre}.png'
    guardar_fig(fig,ruta)
    avisar('  + '+ruta.name)


def modo_dispersiones(X,smp,coord,var,salida,args,modelos):
    """Nueva opción rápida: gráficos de puntos SIN reajustar algoritmos ni SHAP."""
    seleccion=['isolation_forest','hdbscan'] if args.algoritmo=='todos' else [m for m in modelos if m in ('isolation_forest','hdbscan')]
    if not seleccion:
        raise ValueError('El modo dispersiones admite --algoritmo isolation_forest, hdbscan o todos.')
    for nombre in seleccion:
        path=salida/f'01_resultado_total_{nombre}.csv'
        if not path.is_file():
            avisar(f'  SIN DATOS para {nombre}: falta {path.name}. Ejecuta el modelo final antes.')
            continue
        df=pd.read_csv(path,sep=';',encoding='utf-8-sig')
        if len(df)!=len(smp) or not np.array_equal(df[SMP].to_numpy(np.int64),smp):
            raise ValueError(f'El archivo {path.name} no coincide con los 143.561 SMP del original')
        col='alerta_isolation_forest' if nombre=='isolation_forest' else 'cluster_hdbscan'
        if col not in df:
            raise ValueError(f'Falta columna {col} en {path.name}')
        labels=df[col].to_numpy(int)
        avisar(f'Generando dispersiones de {nombre}, sin volver a entrenar, con {len(labels):,} etiquetas...')
        pcanombres=[f'PC1 ({100*var[0]:.1f}% de varianza)',f'PC2 ({100*var[1]:.1f}% de varianza)']
        _dispersion_coloreada(nombre,coord,labels,pcanombres,salida,'16_dispersion_pca_contraste',zoom=False)
        _dispersion_coloreada(nombre,coord,labels,pcanombres,salida,'17_dispersion_pca_central',zoom=True)
        _dispersion_coloreada(nombre,coord,labels,pcanombres,salida,'18_dispersion_pca_solo_destacados',zoom=False,solo_interes=True)
        # Proyeccion observacional en DOS señales para facilitar interpretacion.
        # No se usa para entrenar. Ejes predefinidos; no son escogidos para fingir separacion.
        _dispersion_coloreada(nombre,X[:,[0,5]],labels,
            [f'{CORTOS[0]} (estandarizado)',f'{CORTOS[5]} (estandarizado)'],
            salida,'19_dispersion_acelerometro_x_giroscopio_z')
    avisar('Listo: nuevas dispersiones. No se entreno ni reajusto ningun algoritmo.')
    avisar(f'Carpeta: {salida}')

if __name__ == '__main__':
    main()
