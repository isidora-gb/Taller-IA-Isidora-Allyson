# -*- coding: utf-8 -*-
"""17 | COMPARACION EXPLORATORIA INTEGRAL (143.561 mediciones, 10 senales).

Lee los CSV ya calculados por 16_ejecucion_total_143561.py.
NO reentrena, NO ejecuta PCA, NO modifica los resultados originales,
NO mezcla la evaluacion cronologica 70/15/15 con la ejecucion total.

Desde la carpeta del proyecto (PowerShell):
  python -u 17_comparacion_final_143561.py --modo comprobar
  python -u 17_comparacion_final_143561.py

Opcional si los resultados estan en otra carpeta:
  python -u 17_comparacion_final_143561.py --entrada "RUTA_A_08_ejecucion_total_143561"

Requisitos: pip install numpy pandas matplotlib

Lee cuatro archivos obligatorios:
  01_resultado_total_kmeans.csv
  01_resultado_total_dbscan.csv
  01_resultado_total_hdbscan.csv
  01_resultado_total_isolation_forest.csv
SHAP es complementario (10_shap_importancia.csv, 10_shap_metodologia.json).

SALIDA: Resultados_Tarea3_Nuevos/09_comparacion_total_143561

IMPORTANTE: ruido DBSCAN/HDBSCAN y alertas de Isolation Forest representan
criterios diferentes; coincidencia NO es rendimiento diagnostico, ni averia.
Esta comparacion es sobre los mismos datos usados para ajustar los modelos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
INPUT_DEFAULT = ROOT / 'Resultados_Tarea3_Nuevos' / '08_ejecucion_total_143561'
OUTPUT_DEFAULT = ROOT / 'Resultados_Tarea3_Nuevos' / '09_comparacion_total_143561'
EXPECTED = 143_561
FILES = {
    'kmeans': '01_resultado_total_kmeans.csv',
    'dbscan': '01_resultado_total_dbscan.csv',
    'hdbscan': '01_resultado_total_hdbscan.csv',
    'isolation_forest': '01_resultado_total_isolation_forest.csv',
}
REQUIRED = {
    'kmeans': ['smp_continuo', 'cluster_kmeans'],
    'dbscan': ['smp_continuo', 'cluster_dbscan'],
    'hdbscan': ['smp_continuo', 'cluster_hdbscan'],
    'isolation_forest': ['smp_continuo', 'alerta_isolation_forest', 'puntuacion_anomalia_if'],
}
DISPLAY = {'kmeans': 'K-Means', 'dbscan': 'DBSCAN',
           'hdbscan': 'HDBSCAN', 'isolation_forest': 'Isolation Forest'}
COLOR = {'dbscan': '#337DB3', 'hdbscan': '#CC8452',
         'isolation_forest': '#328C77', 'kmeans': '#775DA3'}
MARKERS = ['ruido_dbscan', 'ruido_hdbscan', 'alerta_isolation_forest']
SHORT = {'ruido_dbscan': 'DBSCAN: ruido', 'ruido_hdbscan': 'HDBSCAN: ruido',
         'alerta_isolation_forest': 'IF: alerta'}


def msg(s):
    print(str(s), flush=True)


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig', float_format='%.12g')


def fmt(n):
    return f'{int(n):,}'.replace(',', '.')


def pct(x):
    return f'{100*float(x):.2f}%'.replace('.', ',')


def clean_axes(ax, grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(axis='y', alpha=.19, color='#78838B', zorder=0)
    ax.set_axisbelow(True)


def figsave(fig, out, name):
    fig.savefig(out / name, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    msg('  + ' + name)


def as_int_labels(series, name, allowed=None):
    v = pd.to_numeric(series, errors='coerce')
    if v.isna().any() or not np.isfinite(v.to_numpy(dtype='float64')).all():
        raise ValueError(f'La columna {name} tiene etiquetas vacias o no numericas.')
    if not np.all(np.equal(v, np.floor(v))):
        raise ValueError(f'La columna {name} debe contener enteros.')
    arr = v.astype('int32')
    if allowed is not None and not arr.isin(allowed).all():
        raise ValueError(f'{name} tiene valores fuera de {sorted(allowed)}.')
    if allowed is None and (arr < -1).any():
        raise ValueError(f'{name} contiene un grupo inferior a -1.')
    return arr


def read_input(folder: Path, expected: int, allow_partial: bool):
    if not folder.is_dir():
        raise FileNotFoundError(f'No existe carpeta de resultados: {folder}\n'
                                'Ejecuta primero el codigo 16 o usa --entrada "RUTA".')
    tables = {}
    meta_warnings = []
    missing = []
    for model, filename in FILES.items():
        p = folder / filename
        if not p.is_file():
            missing.append(filename)
            continue
        d = pd.read_csv(p, sep=';', encoding='utf-8-sig', low_memory=False)
        absent = [c for c in REQUIRED[model] if c not in d.columns]
        if absent:
            raise ValueError(f'{p.name} no tiene las columnas necesarias: {absent}\n'
                             f'Columnas encontradas: {list(d.columns)}')
        s = pd.to_numeric(d['smp_continuo'], errors='coerce')
        if s.isna().any() or not np.all(np.isfinite(s)) or not np.all(s == np.floor(s)):
            raise ValueError(f'{p.name}: SMP debe ser un contador entero valido.')
        d['smp_continuo'] = s.astype('int64')
        if d['smp_continuo'].duplicated().any():
            example = d.loc[d['smp_continuo'].duplicated(), 'smp_continuo'].iloc[0]
            raise ValueError(f'{p.name} tiene SMP duplicado, por ejemplo: {example}.')
        if len(d) != expected:
            raise ValueError(f'{p.name} contiene {fmt(len(d))} filas; se esperaban '
                             f'{fmt(expected)}. No se debe comparar un subconjunto como si fuera la base completa.')
        col = 'alerta_isolation_forest' if model == 'isolation_forest' else 'cluster_' + model
        d[col] = as_int_labels(d[col], col, {0, 1} if model == 'isolation_forest' else None)
        if model == 'kmeans' and (d[col] == -1).any():
            raise ValueError('K-Means no debe tener etiqueta de ruido -1.')
        if model == 'isolation_forest':
            d['puntuacion_anomalia_if'] = pd.to_numeric(d['puntuacion_anomalia_if'], errors='coerce')
            if not np.isfinite(d['puntuacion_anomalia_if'].to_numpy()).all():
                raise ValueError('IF: las puntuaciones de anomalia deben ser finitas.')
        if model == 'hdbscan':
            for c in ('fuerza_hdbscan', 'glosh_hdbscan'):
                if c in d:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
        if model == 'dbscan' and 'es_nucleo_dbscan' in d:
            d['es_nucleo_dbscan'] = pd.to_numeric(d['es_nucleo_dbscan'], errors='coerce')
        tables[model] = d.sort_values('smp_continuo').reset_index(drop=True)
        msg(f'  [OK] {DISPLAY[model]}: {fmt(len(d))} registros, SMP unico.')
    if missing and not allow_partial:
        raise FileNotFoundError('Faltan archivos del codigo 16:\n  - ' + '\n  - '.join(missing)
                                + '\nEjecuta los modelos faltantes en el codigo 16 en modo final.\n'
                                  'No hace falta repetir los que ya existen.')
    if 'kmeans' not in tables:
        raise FileNotFoundError('Falta 01_resultado_total_kmeans.csv, necesario para alinear el análisis.')
    if len(tables) < 2:
        raise ValueError('Hacen falta al menos dos modelos para comparar.')
    original_smp = tables['kmeans']['smp_continuo'].to_numpy()
    for model, table in tables.items():
        if not np.array_equal(table['smp_continuo'].to_numpy(), original_smp):
            only_left = np.setdiff1d(original_smp, table['smp_continuo'].to_numpy())
            only_right = np.setdiff1d(table['smp_continuo'].to_numpy(), original_smp)
            raise ValueError(f'Los SMP de {DISPLAY[model]} no coinciden con los de K-Means. '
                             f'Faltan={len(only_left)}, extra={len(only_right)}. '
                             'No unir resultados por posicion de fila.')
    merged = pd.DataFrame({'smp_continuo': original_smp})
    all_columns = set(merged)
    for model, t in tables.items():
        extra = t.drop(columns=['smp_continuo'])
        overlap = set(extra.columns) & all_columns
        if overlap:
            raise ValueError(f'Columnas ambiguas en {model}: {overlap}')
        merged = merged.join(extra)
        all_columns.update(extra.columns)
    if 'dbscan' in tables:
        merged['ruido_dbscan'] = (merged['cluster_dbscan'] == -1).astype('int8')
    if 'hdbscan' in tables:
        merged['ruido_hdbscan'] = (merged['cluster_hdbscan'] == -1).astype('int8')
    meta_path = folder / '00_trazabilidad_143561.json'
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if int(meta.get('registros', -1)) != expected:
                meta_warnings.append('El JSON de trazabilidad no coincide con el numero de filas.')
            if len(meta.get('senales', [])) != 10:
                meta_warnings.append('El JSON de trazabilidad NO acredita las 10 variables previstas.')
            if meta.get('division', '').lower().find('ninguna') == -1:
                meta_warnings.append('El JSON de trazabilidad NO confirma ejecucion sin division.')
            # El codigo 16 sobrescribe metadatos globales al ejecutar modelos individualmente.
            meta_warnings.append('Los parametros del JSON de trazabilidad son globales y pudieron '
                                 'ser sobrescritos: no demuestran los parametros exactos de cada CSV.')
        except (OSError, ValueError, TypeError) as exc:
            meta_warnings.append(f'No se pudo interpretar trazabilidad: {exc}')
    else:
        meta_warnings.append('No se encontro 00_trazabilidad_143561.json.')
    return merged, tables, meta, missing, meta_warnings


def load_shap(folder: Path, overall_alert_rate: float | None):
    p = folder / '10_shap_importancia.csv'
    mp = folder / '10_shap_metodologia.json'
    warnings_ = []
    if not p.is_file():
        return None, None, ['Falta 10_shap_importancia.csv: no se inventara importancia SHAP.']
    d = pd.read_csv(p, sep=';', encoding='utf-8-sig')
    cols = ['variable', 'importancia_relativa_pct']
    if not all(c in d.columns for c in cols):
        return None, None, [f'SHAP sin columnas {cols}. No se graficara.']
    d['importancia_relativa_pct'] = pd.to_numeric(d['importancia_relativa_pct'], errors='coerce')
    if d['importancia_relativa_pct'].isna().any() or (d['importancia_relativa_pct'] < 0).any():
        return None, None, ['La tabla SHAP tiene importancias invalidas.']
    if d['variable'].duplicated().any() or len(d) != 10:
        return None, None, ['SHAP no contiene diez variables distintas.']
    if not np.isclose(d['importancia_relativa_pct'].sum(), 100, atol=.1):
        return None, None, ['SHAP no suma aproximadamente 100%; revisar resultados.']
    info = {}
    if mp.is_file():
        try:
            info = json.loads(mp.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            warnings_.append('No se pudo leer la metodologia SHAP.')
    else:
        warnings_.append('Sin 10_shap_metodologia.json: no se conoce tamano de muestra SHAP.')
    if overall_alert_rate is not None and 'prevalencia_alertas_total' in info:
        if abs(float(info['prevalencia_alertas_total']) - overall_alert_rate) > .001:
            warnings_.append('ATENCION: la tasa de alertas del SHAP no coincide con la del CSV '
                             'de Isolation Forest. SHAP podria corresponder a otro ajuste.')
            return None, info, warnings_
    if info.get('registros_entrenamiento') is not None and int(info['registros_entrenamiento']) != EXPECTED:
        warnings_.append('SHAP corresponde a otro volumen de entrenamiento; revisar antes de interpretar.')
        return None, info, warnings_
    return d.sort_values('importancia_relativa_pct', ascending=False), info, warnings_


def summarize(merged: pd.DataFrame, tables):
    n = len(merged)
    rows = []
    for model in FILES:
        if model not in tables:
            continue
        if model == 'isolation_forest':
            flags = merged['alerta_isolation_forest'].to_numpy(dtype=bool)
            num_groups = np.nan
            positive = int(flags.sum())
            concept = 'Alerta IF (umbral automatico)'
            grouped = np.nan
        else:
            label = merged['cluster_' + model].to_numpy()
            num_groups = len(np.unique(label[label >= 0]))
            positive = int(np.sum(label == -1)) if model != 'kmeans' else np.nan
            grouped = int(np.sum(label >= 0))
            concept = ('Ruido por densidad' if model in ('dbscan','hdbscan')
                       else 'K-Means solo crea grupos')
        rows.append({'algoritmo': DISPLAY[model], 'registros': n,
                     'grupos': num_groups, 'asignados_a_grupos': grouped,
                     'n_ruido_o_alerta': positive,
                     'pct_ruido_o_alerta': (100 * positive / n if not pd.isna(positive) else np.nan),
                     'criterio': concept})
    return pd.DataFrame(rows)


def concordance(merged):
    cols = [c for c in MARKERS if c in merged]
    n = len(merged)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            u = merged[a].to_numpy(dtype=bool)
            v = merged[b].to_numpy(dtype=bool)
            both = int(np.sum(u & v))
            count_a, count_b = int(u.sum()), int(v.sum())
            union = int(np.sum(u | v))
            neither = n - union
            denom = math.sqrt(count_a * (n-count_a) * count_b * (n-count_b))
            phi = ((both * neither) - ((count_a-both)*(count_b-both))) / denom if denom else np.nan
            rows.append({
                'criterio_A': SHORT[a], 'criterio_B': SHORT[b],
                'n_total': n, 'n_A': count_a, 'n_B': count_b,
                'ambos': both, 'solo_A': count_a-both, 'solo_B': count_b-both,
                'ninguno': neither, 'union': union,
                'jaccard': both/union if union else np.nan,
                'pct_de_A_que_tambien_B': 100*both/count_a if count_a else np.nan,
                'pct_de_B_que_tambien_A': 100*both/count_b if count_b else np.nan,
                'phi': phi,
            })
    return pd.DataFrame(rows)


def combinations(merged):
    cols = [c for c in MARKERS if c in merged]
    labels = [SHORT[c] for c in cols]
    if len(cols) < 2:
        return pd.DataFrame(), None
    bit = np.zeros(len(merged), dtype='int32')
    for i, c in enumerate(cols):
        bit |= merged[c].to_numpy(dtype='int32') << i
    value_counts = np.bincount(bit, minlength=1 << len(cols))
    result = []
    for v in range(1 << len(cols)):
        active = [labels[i] for i in range(len(cols)) if v & (1 << i)]
        name = ' + '.join(active) if active else 'Ninguno'
        result.append({'combinacion': name, 'codigo': v,
                       'n_registros': int(value_counts[v]),
                       'pct_total': float(100*value_counts[v]/len(merged))})
    result = pd.DataFrame(result).sort_values('n_registros', ascending=False).reset_index(drop=True)
    return result, bit


def temporal_blocks(merged, n_blocks: int):
    # Orden SMP es un contador de muestras; no conocemos segundos ni frecuencia.
    n = len(merged)
    ids = np.floor(np.arange(n, dtype=float)*min(n_blocks,n)/n).astype(int)
    columns = [c for c in MARKERS if c in merged]
    temp = merged[['smp_continuo', *columns]].copy()
    temp.insert(0, 'bloque', ids+1)
    agg = temp.groupby('bloque', sort=True).agg(
        smp_desde=('smp_continuo','min'), smp_hasta=('smp_continuo','max'),
        registros=('smp_continuo','size')).reset_index()
    for c in columns:
        agg['pct_' + c] = 100*temp.groupby('bloque')[c].mean().to_numpy()
    if len(columns) > 1:
        agg['pct_coincidencia_todos'] = 100*temp.groupby('bloque')[columns].apply(
            lambda d: d.to_numpy(dtype=bool).all(axis=1).mean()).to_numpy()
    return agg


def per_kmeans(merged):
    cols = [c for c in MARKERS if c in merged]
    agg = merged.groupby('cluster_kmeans').agg(registros=('smp_continuo','size')).reset_index()
    for c in cols:
        agg['n_' + c] = merged.groupby('cluster_kmeans')[c].sum().to_numpy(dtype=int)
        agg['pct_' + c] = 100*merged.groupby('cluster_kmeans')[c].mean().to_numpy()
    return agg



def supplemental_diagnostics(folder: Path, merged: pd.DataFrame):
    """Recupera mediciones complementarias si existe fuente verificable.

    No atribuye un valor a un modelo diferente del que efectivamente produjo
    el CSV. Los resultados del barrido K son exploratorios y se rotulan asi.
    """
    rows = []
    warns = []
    km_path = folder / '07_barrido_kmeans_completo.csv'
    if km_path.is_file():
        try:
            tab = pd.read_csv(km_path, sep=';', encoding='utf-8-sig')
            k_actual = int(merged['cluster_kmeans'].nunique())
            selected = tab[pd.to_numeric(tab['k'], errors='coerce') == k_actual]
            if len(selected) == 1:
                r = selected.iloc[0]
                for c, txt in [('sse_inercia','SSE barrido K-Means'),
                               ('silueta_muestra_1600','Silueta estimada en muestra (barrido)'),
                               ('perdida_pct_respecto_k1','Inercia K/K=1 (%)')]:
                    if c in tab.columns:
                        v = pd.to_numeric(pd.Series([r[c]]), errors='coerce').iloc[0]
                        if pd.notna(v) and np.isfinite(v):
                            rows.append({'modelo':'K-Means','indicador':txt, 'valor':float(v),
                                         'fuente':km_path.name,
                                         'nota':'Exploratorio; no audita que sea el mismo ajuste guardado'})
                warns.append('Los indicadores SSE y silueta proceden del barrido K y no '
                             'demuestran que sea exactamente el mismo ajuste que las etiquetas CSV.')
        except (KeyError, ValueError, OSError) as ex:
            warns.append(f'No se pudo incorporar barrido K-Means: {ex}')
    cache_path = folder / '00_modelo_total_hdbscan_meta.json'
    if cache_path.is_file() and 'cluster_hdbscan' in merged:
        try:
            hmeta = json.loads(cache_path.read_text(encoding='utf-8'))
            labels = np.ascontiguousarray(merged['cluster_hdbscan'].to_numpy(dtype=np.int32))
            digest = hashlib.sha256(labels.view(np.uint8)).hexdigest()
            if hmeta.get('sha256_etiquetas') == digest and int(hmeta.get('n_registros',-1)) == len(merged):
                for name in ('min_cluster_size','min_samples'):
                    v = hmeta.get(name)
                    if v is not None:
                        rows.append({'modelo':'HDBSCAN','indicador':name, 'valor':int(v),
                                     'fuente':cache_path.name,
                                     'nota':'Verificado por hash contra las etiquetas leidas'})
                selection = hmeta.get('seleccion', '?')
                warns.append(f'Parametros HDBSCAN verificados con firma de etiquetas: '
                             f'min_cluster_size={hmeta.get("min_cluster_size")}, '
                             f'min_samples={hmeta.get("min_samples")}, '
                             f'seleccion={selection}, escalado={hmeta.get("escalado")}.')
            else:
                warns.append('La firma del cache HDBSCAN NO coincide con las etiquetas del CSV. '
                             'No se asumen sus parametros.')
        except (OSError, ValueError, TypeError) as ex:
            warns.append(f'No se pudo validar la firma HDBSCAN: {ex}')
    return pd.DataFrame(rows,columns=['modelo','indicador','valor','fuente','nota']), warns


def q_if(merged):
    if 'puntuacion_anomalia_if' not in merged:
        return pd.DataFrame()
    score = merged['puntuacion_anomalia_if'].to_numpy(dtype=float)
    vals = np.quantile(score, [0,.01,.05,.25,.5,.75,.95,.99,1])
    return pd.DataFrame({'cuantil': ['min', 'p01', 'p05', 'p25', 'mediana', 'p75', 'p95', 'p99', 'max'],
                         'puntuacion_if': vals})


def fig_summary(summary, out):
    data = summary.dropna(subset=['pct_ruido_o_alerta']).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(11,5.5))
    col = [COLOR[k] for k in ('dbscan','hdbscan','isolation_forest') if DISPLAY[k] in set(data['algoritmo'])]
    names = data['algoritmo'].tolist()
    col = [COLOR[next(k for k,v in DISPLAY.items() if v == a)] for a in names]
    x = np.arange(len(data))
    b = ax.bar(x, data['pct_ruido_o_alerta'], color=col, width=.55, zorder=2)
    for rect, val in zip(b, data['pct_ruido_o_alerta']):
        ax.text(rect.get_x()+rect.get_width()/2, val+1.2, f'{val:.2f}%', ha='center', fontsize=11)
    ax.set_xticks(x, [f'{n}\n'+('ALERTA' if n=='Isolation Forest' else 'RUIDO') for n in names])
    ax.set_ylim(0, max(107, data['pct_ruido_o_alerta'].max()*1.12))
    ax.set_ylabel('Porcentaje de las 143.561 mediciones (%)')
    ax.set_title('Criterios de interes reportados por cada modelo', weight='bold', fontsize=14)
    ax.text(.5, -.22, 'Ruido de densidad y alertas de IF NO son la misma clase de evento. K-Means no define ruido.',
            ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#565D62')
    clean_axes(ax)
    figsave(fig, out, '01_porcentaje_ruido_y_alertas.png')


def fig_overlap(conc, merged, out):
    """Matriz Jaccard con indice y numero REAL de coincidencias por casilla."""
    cols = [c for c in MARKERS if c in merged]
    labels = [SHORT[c] for c in cols]
    n = len(cols)
    if n < 2:
        return
    matrix = np.eye(n, dtype=float)
    counts = np.zeros((n, n), dtype='int64')
    flags = [merged[c].to_numpy(dtype=bool) for c in cols]
    for i in range(n):
        counts[i, i] = np.count_nonzero(flags[i])
        for j in range(i + 1, n):
            both = int(np.count_nonzero(flags[i] & flags[j]))
            union = int(np.count_nonzero(flags[i] | flags[j]))
            matrix[i, j] = matrix[j, i] = (both / union if union else np.nan)
            counts[i, j] = counts[j, i] = both
    fig, ax = plt.subplots(figsize=(10.6, 8.3))
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='Blues')
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if np.isfinite(val) and val >= .55 else '#14283A'
            main = '—' if not np.isfinite(val) else f'{val:.3f}'
            ax.text(j, i - .14, main, color=color, fontsize=15,
                    ha='center', va='center', weight='bold')
            extra = ('Marcados: ' if i == j else 'Coinciden: ') + fmt(counts[i, j])
            ax.text(j, i + .19, extra, color=color, fontsize=10,
                    ha='center', va='center')
    ax.set_xticks(range(n), labels, rotation=23, ha='right')
    ax.set_yticks(range(n), labels)
    ax.set_title('Matriz de coincidencia: Jaccard y cantidad de registros',
                 weight='bold', fontsize=14, pad=15)
    fig.text(.50, .02,
             'Numero superior: indice Jaccard (0 a 1). Inferior: registros marcados por ambos criterios. '
             'Diagonal: total marcado por ese criterio.',
             ha='center', fontsize=9, color='#475569')
    cbar = fig.colorbar(im, ax=ax, shrink=.78)
    cbar.set_label('Indice Jaccard (0 a 1)')
    fig.subplots_adjust(bottom=.19)
    figsave(fig, out, '02_matriz_jaccard.png')


def fig_combos(combos, merged, out):
    if combos.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, .63*len(combos)+1.5)))
    colors = ['#8293A0' if s=='Ninguno' else '#5082A5' for s in combos['combinacion']]
    bars = ax.barh(combos['combinacion'][::-1], combos['n_registros'][::-1], color=colors[::-1], height=.72)
    for p, n in zip(bars, combos['n_registros'][::-1]):
        ax.text(p.get_width()+len(merged)*.007, p.get_y()+p.get_height()/2,
                f'{fmt(n)} ({100*n/len(merged):.2f}%)', va='center', fontsize=9)
    ax.set_xlim(0, max(1, combos['n_registros'].max()*1.29))
    ax.set_xlabel('Numero de mediciones')
    ax.set_title('Coincidencias exclusivas entre los criterios de interes', weight='bold', fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    clean_axes(ax)
    figsave(fig, out, '03_combinaciones_coincidencias.png')


def fig_temporal(temp, out):
    cols = [c for c in MARKERS if 'pct_'+c in temp]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(13,5.5))
    for c in cols:
        key = 'dbscan' if 'dbscan' in c and 'hdbscan' not in c else ('hdbscan' if 'hdbscan' in c else 'isolation_forest')
        ax.plot(temp['bloque'], temp['pct_'+c], label=SHORT[c], linewidth=1.65, color=COLOR[key])
    ax.set_ylim(-3, 103)
    ax.set_xlabel('Bloques consecutivos de registros ordenados por SMP (NO segundos)')
    ax.set_ylabel('Porcentaje de registros en cada bloque')
    ax.set_title('Evolucion de ruido y alertas a lo largo del contador SMP', fontsize=13, weight='bold')
    ax.legend(loc='best', frameon=False, fontsize=10)
    clean_axes(ax)
    figsave(fig, out, '04_evolucion_por_bloques_smp.png')


def fig_cluster_rates(per, out):
    cols = [c for c in MARKERS if 'pct_'+c in per]
    if not cols:
        return
    data = per.sort_values('cluster_kmeans')
    fig, ax = plt.subplots(figsize=(11,5.7))
    ind = np.arange(len(data))
    width = .75 / len(cols)
    for j, c in enumerate(cols):
        key = 'dbscan' if c=='ruido_dbscan' else 'hdbscan' if c=='ruido_hdbscan' else 'isolation_forest'
        ax.bar(ind + (j-(len(cols)-1)/2)*width,
               data['pct_'+c], width=width*.94, label=SHORT[c], color=COLOR[key])
    ax.set_xticks(ind, [f'Grupo {k}\nn={fmt(n)}' for k,n in zip(data['cluster_kmeans'],data['registros'])])
    ax.set_ylim(0, 108)
    ax.set_ylabel('Porcentaje dentro del grupo K-Means (%)')
    ax.set_title('Ruido y alertas dentro de los grupos definidos por K-Means', weight='bold')
    ax.legend(loc='upper right', frameon=False)
    clean_axes(ax)
    figsave(fig, out, '05_criterios_dentro_de_kmeans.png')


def fig_if_hist(merged, out):
    if 'puntuacion_anomalia_if' not in merged:
        return
    score = merged['puntuacion_anomalia_if'].to_numpy(dtype=float)
    flag = merged['alerta_isolation_forest'].to_numpy(dtype=bool)
    bins = np.linspace(np.quantile(score,.001), np.quantile(score,.999), 65)
    fig, ax = plt.subplots(figsize=(11,5.6))
    ax.hist(score[~flag], bins=bins, color='#4A869D', alpha=.68, label='Sin alerta')
    ax.hist(score[flag], bins=bins, color='#D47754', alpha=.79, label='Con alerta')
    if 'decision_function_if' in merged:
        dec = pd.to_numeric(merged['decision_function_if'], errors='coerce')
        threshold = np.median((score + dec).to_numpy(dtype=float))
        if np.allclose((score+dec).to_numpy(dtype=float), threshold, rtol=0, atol=1e-5):
            ax.axvline(threshold, ls='--', color='#323943', lw=1.8, label=f'Umbral IF = {threshold:.3f}')
    ax.set_ylabel('Numero de mediciones')
    ax.set_xlabel('Puntuacion de anomalia de IF (mayor = mas atipica)')
    ax.set_title('Distribucion real de puntuaciones | Isolation Forest', weight='bold')
    ax.legend(frameon=False)
    clean_axes(ax)
    figsave(fig, out, '06_distribucion_puntuaciones_if.png')


def fig_shap(shap, info, out):
    if shap is None:
        return
    fig, ax = plt.subplots(figsize=(11,6))
    d = shap.sort_values('importancia_relativa_pct')
    bar = ax.barh(d['variable'], d['importancia_relativa_pct'], color='#537C97')
    for r, x in zip(bar,d['importancia_relativa_pct']):
        ax.text(x+.15, r.get_y()+r.get_height()/2, f'{x:.2f}%', va='center', fontsize=9)
    ax.set_xlim(0, max(15, float(d['importancia_relativa_pct'].max())*1.18))
    ax.set_xlabel('Participacion relativa de importancia SHAP (%)')
    n = info.get('registros_explicados_shap', 'no informado') if info else 'no informado'
    ax.set_title(f'SHAP | contribuciones a la puntuacion de IF (muestra n={n})', weight='bold')
    clean_axes(ax)
    figsave(fig, out, '07_shap_importancia_10_senales.png')


def fig_confusion_pairs(merged, out):
    """Tablas cruzadas con conteo y porcentaje legible en TODAS las casillas."""
    cols = [c for c in MARKERS if c in merged]
    if len(cols) < 2:
        return
    from itertools import combinations as pairs
    pairlist = list(pairs(cols, 2))
    n_total = len(merged)
    fig, axes = plt.subplots(1, len(pairlist),
                            figsize=(6.2 * len(pairlist), 5.9), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairlist):
        A = merged[a].to_numpy(dtype=bool)
        B = merged[b].to_numpy(dtype=bool)
        tab = np.array([[int(np.sum(~A & ~B)), int(np.sum(~A & B))],
                        [int(np.sum(A & ~B)), int(np.sum(A & B))]])
        scale = np.log1p(tab)
        im = ax.imshow(scale, cmap='Blues', vmin=0, vmax=np.log1p(n_total))
        for i in range(2):
            for j in range(2):
                color = 'white' if scale[i, j] > np.log1p(n_total) * .67 else '#162A39'
                ax.text(j, i - .12, fmt(tab[i, j]), ha='center', va='center',
                        fontsize=15, weight='bold', color=color)
                ax.text(j, i + .18, f'{100 * tab[i, j] / n_total:.2f}%',
                        ha='center', va='center', fontsize=10, color=color)
        ax.set_xticks([0, 1], ['No', 'Sí'])
        ax.set_yticks([0, 1], ['No', 'Sí'])
        ax.set_xlabel(SHORT[b], fontsize=11)
        ax.set_ylabel(SHORT[a], fontsize=11)
        ax.set_title('Conteo y % del total', weight='medium', fontsize=12)
    fig.suptitle('Matrices cruzadas: cantidad de registros en cada casilla',
                 fontsize=15, weight='bold', y=.98)
    fig.text(.5, .04,
             'Fila = criterio del eje vertical; columna = criterio del eje horizontal. '
             'El porcentaje se calcula sobre las 143.561 mediciones.',
             ha='center', fontsize=10, color='#475569')
    fig.subplots_adjust(wspace=.40, top=.84, bottom=.20)
    figsave(fig, out, '08_tablas_cruzadas.png')


def fig_pair_smp_scatter(merged, out, max_draw=26000):
    # Visualization only, no train and no modified labels.
    cols = [c for c in MARKERS if c in merged]
    if not cols:
        return
    n=len(merged)
    draw=np.linspace(0,n-1,num=min(n,max_draw),dtype=int)
    fig, ax=plt.subplots(figsize=(13,5.8))
    for j,c in enumerate(cols):
        flag=merged[c].to_numpy(dtype=bool)[draw]
        smp=merged['smp_continuo'].to_numpy()[draw]
        # Draw only positive events to reveal where their frequency concentrates.
        key = 'dbscan' if c=='ruido_dbscan' else 'hdbscan' if c=='ruido_hdbscan' else 'isolation_forest'
        ax.scatter(smp[flag], np.full(int(flag.sum()),j), s=3, alpha=.35,
                   color=COLOR[key], rasterized=True)
    ax.set_yticks(range(len(cols)),[SHORT[c] for c in cols])
    ax.set_ylim(-.6,len(cols)-.4)
    ax.set_xlabel('Contador SMP (no es una unidad de tiempo)')
    ax.set_title('Ubicacion de registros marcados | muestra para visualizacion',weight='bold')
    ax.grid(axis='x',alpha=.18)
    clean_axes(ax,grid=False)
    figsave(fig,out,'09_marcaciones_vs_smp.png')


def write_text(summary, conc, combos, per, shap, info, meta, warns, out, partial, diagnostics):
    n=int(summary['registros'].iloc[0])
    txt=['# Comparacion exploratoria de algoritmos | base total', '',
         f'- Mediciones alineadas por SMP: **{fmt(n)}**.',
         '- Senales usadas por los ajustes del codigo 16: 10 simultaneamente (segun su diseno).',
         '- **No se reentrena ningun modelo** en el codigo 17; se comparan sus salidas guardadas.',
         '- Se trata de resultados **dentro de muestra**, no de prueba independiente.',
         '- SMP ordena adquisiciones, NO indica segundos ni intervalos regulares.',
         '- Ruido por densidad, grupos K-Means y alertas IF son conceptos distintos.',
         '- No hay fallas confirmadas: NO se calcula precision, recall ni exactitud diagnostica.', '']
    if partial:
        txt += ['**ADVERTENCIA: EJECUCION PARCIAL. No usar como comparacion de cuatro algoritmos.**','']
    txt += ['## Resumen por modelo', '',
            '| Modelo | Grupos | Marcados ruido / alertas | % | Criterio |',
            '|:---|---:|---:|---:|:---|']
    for r in summary.itertuples(index=False):
        gr='—' if pd.isna(r.grupos) else fmt(r.grupos)
        marked='—' if pd.isna(r.n_ruido_o_alerta) else fmt(r.n_ruido_o_alerta)
        markpct='—' if pd.isna(r.pct_ruido_o_alerta) else f'{r.pct_ruido_o_alerta:.2f}%'
        txt.append(f'| {r.algoritmo} | {gr} | {marked} | {markpct} | {r.criterio} |')
    txt+=['','## Concordancia entre criterios de interes', '',
          'Indice Jaccard = cantidad marcada por ambos / cantidad marcada por al menos uno.',
          'No es una metrica de acierto frente a fallas reales.','',
          '| Criterio A | Criterio B | Ambas marcas | Jaccard | % de A que tambien es B |',
          '|:---|:---|---:|---:|---:|']
    for r in conc.itertuples(index=False):
        txt.append(f'| {r.criterio_A} | {r.criterio_B} | {fmt(r.ambos)} | '
                   f'{r.jaccard:.4f} | {r.pct_de_A_que_tambien_B:.2f}% |')
    txt+=['', '## SHAP: interpretacion de Isolation Forest', '']
    if shap is not None:
        top=shap.iloc[:3]
        txt.append('Senales de mayor importancia estimada en la muestra SHAP: '
                   + ', '.join(f'{r.variable} ({r.importancia_relativa_pct:.2f}%)'
                               for r in top.itertuples(index=False)) + '.')
        sample = info.get('registros_explicados_shap',None) if info else None
        txt.append(f'SHAP explica **{sample if sample is not None else "una muestra"}** mediciones; '
                   'NO se calcularon contribuciones individuales para las 143.561.')
        txt.append('Una contribucion SHAP explica una prediccion del modelo; '
                   'no demuestra causalidad ni falla de una pieza.')
    else:
        txt.append('No hay tabla SHAP valida para esta ejecucion; se omite este resultado.')
    txt += ['', '## Lectura tecnica sugerida', '',
            '1. La cantidad de grupos NO determina la calidad del agrupamiento.',
            '2. El porcentaje de ruido de HDBSCAN NO implica ese porcentaje de averias.',
            '3. Que IF y DBSCAN marquen simultaneamente una medicion respalda concordancia '
            'estadistica, pero NO valida que exista una falla.',
            '4. Sin etiquetas de fallas, registros de mantenimiento, datos nominales del '
            'equipo y frecuencia temporal, no se puede cuantificar desempeno de deteccion ni '
            'definir frecuencia de mantencion.',
            '5. La seleccion de parametros en la base completa es exploratoria; '
            'conservar por separado la evaluacion cronologica 70/15/15.']
    if warns:
        txt+=['','## Advertencias de trazabilidad','']
        txt+=['- '+w for w in warns]
    txt+=['','## Archivos generados','',
          '- `00_auditoria_insumos.csv`: archivos, columnas y muestras.',
          '- `01_resultados_alineados_143561.csv`: etiquetas y puntuaciones unidas por SMP.',
          '- `02_resumen_comparativo.csv`, `03_concordancia.csv`, `04_combinaciones.csv`,',
          '  `05_evolucion_bloques_smp.csv`, `06_por_cluster_kmeans.csv`,',
          '  `07_cuantiles_puntuacion_if.csv`: tablas de apoyo.',
          '- Figuras 01 a 09: resumen, solapamiento, evolucion, perfiles y SHAP.', '']
    (out/'00_lectura_para_informe.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')


def args_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modo', choices=['final','comprobar'], default='final')
    p.add_argument('--entrada', default=str(INPUT_DEFAULT), help='Carpeta que contiene los CSV del codigo 16.')
    p.add_argument('--salida', default=str(OUTPUT_DEFAULT), help='Carpeta NUEVA para resultados del codigo 17.')
    p.add_argument('--registros-esperados',type=int,default=EXPECTED)
    p.add_argument('--bloques',type=int,default=120,help='Bloques consecutivos para evolucion SMP.')
    p.add_argument('--permitir-parcial',action='store_true', help='Diagnostico: NO exige los 4 archivos, pero rotula resultado parcial.')
    return p.parse_args()


def main():
    args=args_parser()
    folder=Path(args.entrada).expanduser().resolve()
    out=Path(args.salida).expanduser().resolve()
    if args.registros_esperados<=0 or args.bloques<2:
        raise ValueError('registros-esperados > 0, bloques >= 2')
    msg('=== 17 | COMPARACION FINAL SIN REENTRENAR MODELOS ===')
    msg('Carpeta de lectura: '+str(folder))
    merged,tables,meta,missing,meta_warnings=read_input(folder,args.registros_esperados,args.permitir_parcial)
    msg(f'[OK] {fmt(len(merged))} SMP coincidentes entre {len(tables)} modelos.')
    summary=summarize(merged,tables)
    msg(summary.to_string(index=False, na_rep='--',float_format=lambda x:f'{x:.3f}'))
    shap,shapinfo,shap_warnings=load_shap(folder, float(merged['alerta_isolation_forest'].mean())
                                      if 'alerta_isolation_forest' in merged else None)
    diagnostic, diagnostic_warns = supplemental_diagnostics(folder,merged)
    warns=meta_warnings+shap_warnings+diagnostic_warns
    if args.modo=='comprobar':
        msg('SHAP: '+('encontrado y validado' if shap is not None else 'no disponible/valido'))
        for w in warns: msg('[AVISO] '+w)
        msg('COMPROBACION CORRECTA. No se crearon archivos ni se reentreno ningun modelo.')
        return
    if out==folder:
        raise ValueError('Salida debe ser diferente a entrada para NO sobrescribir resultados del codigo 16.')
    out.mkdir(parents=True,exist_ok=True)
    audit=[]
    for k,t in tables.items():
        name=FILES[k]
        audit.append({'algoritmo':DISPLAY[k], 'archivo':name, 'filas':len(t),
                      'smp_min':int(t['smp_continuo'].min()),'smp_max':int(t['smp_continuo'].max()),
                      'smp_unicos':t['smp_continuo'].nunique(),
                      'columnas': '|'.join(t.columns)})
    save_csv(pd.DataFrame(audit),out/'00_auditoria_insumos.csv')
    save_csv(merged,out/'01_resultados_alineados_143561.csv')
    save_csv(summary,out/'02_resumen_comparativo.csv')
    conc=concordance(merged)
    save_csv(conc,out/'03_concordancia.csv')
    combos,_=combinations(merged)
    save_csv(combos,out/'04_combinaciones.csv')
    temp=temporal_blocks(merged,args.bloques)
    save_csv(temp,out/'05_evolucion_bloques_smp.csv')
    per=per_kmeans(merged)
    save_csv(per,out/'06_por_cluster_kmeans.csv')
    q=q_if(merged)
    save_csv(q,out/'07_cuantiles_puntuacion_if.csv')
    if shap is not None:
        save_csv(shap,out/'08_shap_importancia_consolidada.csv')
    if not diagnostic.empty:
        save_csv(diagnostic,out/'09_diagnosticos_adicionales.csv')
    jsonout={'registros_comparados':len(merged),'algoritmos':list(tables),
             'total_esperado':args.registros_esperados,
             'division':'ninguna; comparacion de resultados en muestra',
             'no_reentrena':True,'shap_es_muestra':True,
             'shap_n':shapinfo.get('registros_explicados_shap') if shapinfo else None,
             'falta_algoritmo':missing,'advertencias':warns,
             'carpeta_origen':str(folder),
        'diagnosticos_adicionales':len(diagnostic)}
    (out/'00_metodologia_comparativa.json').write_text(json.dumps(jsonout,indent=2,ensure_ascii=False),encoding='utf-8')
    msg('Generando graficos ...')
    fig_summary(summary,out)
    fig_overlap(conc,merged,out)
    fig_combos(combos,merged,out)
    fig_temporal(temp,out)
    fig_cluster_rates(per,out)
    fig_if_hist(merged,out)
    fig_shap(shap,shapinfo,out)
    fig_confusion_pairs(merged,out)
    fig_pair_smp_scatter(merged,out)
    write_text(summary,conc,combos,per,shap,shapinfo,meta,warns,out,bool(missing),diagnostic)
    msg('  + 00_lectura_para_informe.md')
    if missing: msg('ADVERTENCIA: EJECUCION PARCIAL, faltan: '+', '.join(missing))
    for w in warns: msg('[AVISO] '+w)
    msg('=== LISTO: se generaron tablas y graficos SIN REENTRENAR ===')
    msg('Carpeta: '+str(out))


if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError,ValueError,KeyError) as exc:
        print('\nERROR DE VALIDACION: '+str(exc), file=sys.stderr, flush=True)
        sys.exit(2)
