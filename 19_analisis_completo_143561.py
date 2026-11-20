
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 42
SMP = 'smp_continuo'
PARTES = ('entrenamiento', 'validacion', 'prueba')
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
AZUL = '#2872A3'
NARANJA = '#D49142'
MORADO = '#8C5AA9'
ROJO = '#C84859'
GRIS = '#8C98A6'
VERDE = '#3B907D'
PALETA = ['#2872A3', '#D49142', '#8C5AA9', '#3B907D', '#C84859', '#7276B3']


def csv_out(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, sep=';', index=False, encoding='utf-8-sig', float_format='%.10g')


def guardar(fig, ruta: Path) -> None:
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close(fig)


def preparar_estilo() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.0,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def existe(ruta: Path) -> Path:
    if not ruta.is_file():
        raise FileNotFoundError(f'No encuentro: {ruta}. Ejecuta antes 08, 09, 10 y 11.')
    return ruta


def leer_json(ruta: Path) -> dict:
    return json.loads(existe(ruta).read_text(encoding='utf-8'))


def leer_entrada(ruta: Path) -> pd.DataFrame:
    with existe(ruta).open('r', encoding='utf-8-sig') as f:
        encabezado = f.readline()
    sep = ';' if ';' in encabezado else ','
    return pd.read_csv(ruta, sep=sep, encoding='utf-8-sig', dtype='string')


def numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype('string').str.strip().str.replace(',', '.', regex=False),
                         errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def obtener_fuente(raiz: Path, argumento: str | None, preparados: Path) -> Path:
    candidatos = []
    if argumento:
        p = Path(argumento).expanduser()
        candidatos.append(p if p.is_absolute() else raiz / p)
    else:
        meta = preparados / '00_metadatos.json'
        if meta.is_file():
            original = leer_json(meta).get('archivo_fuente')
            if original:
                candidatos.append(Path(original))
        candidatos.extend([
            raiz / 'Datos_corregidos' / 'datos_multisensor_corregidos.csv',
            raiz / 'Datos_corregidos_desde_excel' / 'datos_multisensor_corregidos.csv',
            raiz / 'datos_multisensor_corregidos.csv',
        ])
    for c in candidatos:
        if c.is_file():
            return c
    raise FileNotFoundError('No se encontró datos_multisensor_corregidos.csv; usa --entrada ruta/datos_multisensor_corregidos.csv')


def cargar(raiz: Path, base: Path, fuente_arg: str | None):
    prepdir = base / '00_preparacion'
    ifdir = base / '01_isolation_forest'
    kmdir = base / '02_kmeans'
    dbdir = base / '03_dbscan'
    obj_prep = joblib.load(existe(prepdir / '00_preprocesamiento.joblib'))
    modelo_if = joblib.load(existe(ifdir / '00_modelo_isolation_forest.joblib'))
    modelo_km = joblib.load(existe(kmdir / '00_modelo_kmeans.joblib'))
    db = joblib.load(existe(dbdir / '00_modelo_dbscan_muestra.joblib'))
    conf_if = leer_json(ifdir / '00_configuracion_modelo.json')
    conf_km = leer_json(kmdir / '00_configuracion_kmeans.json')
    fuente = obtener_fuente(raiz, fuente_arg, prepdir)
    raw = leer_entrada(fuente)
    sensores = list(obj_prep['variables'])
    if len(sensores) != 10 or list(db['variables']) != ['z_' + x for x in sensores]:
        raise ValueError('El orden de los sensores de los artefactos no coincide (08 vs 11).')
    faltan = [c for c in [SMP, *sensores] if c not in raw.columns]
    if faltan:
        raise ValueError(f'Faltan columnas en la base corregida: {faltan}')
    raw[SMP] = numero(raw[SMP])
    if raw[SMP].isna().any() or (raw[SMP] % 1 != 0).any():
        raise ValueError('SMP continuo incompleto/no entero. Revisa la base corregida.')
    raw[SMP] = raw[SMP].astype('int64')
    if raw[SMP].duplicated().any():
        raise ValueError('SMP continuo contiene duplicados.')
    raw = raw.sort_values(SMP, kind='mergesort').reset_index(drop=True)
    for c in sensores:
        raw[c] = numero(raw[c])

    # Usamos EXACTAMENTE las entradas escaladas previas para reproducir los
    # resultados publicados de los códigos 09, 10 y 11 (sin reprocesar umbrales).
    trozos = []
    for nombre in PARTES:
        archivo = existe(prepdir / f'01_{nombre}_escalado.csv')
        t = pd.read_csv(archivo, sep=';', encoding='utf-8-sig')
        if list(t.columns) != [SMP, *('z_' + v for v in sensores)]:
            raise ValueError(f'Columnas escaladas incompatibles en {archivo}')
        t['conjunto'] = nombre
        trozos.append(t)
    escalados = pd.concat(trozos, ignore_index=True).sort_values(SMP, kind='mergesort').reset_index(drop=True)
    if not np.array_equal(raw[SMP].to_numpy(), escalados[SMP].to_numpy()):
        raise ValueError('La base corregida no coincide con los SMP del código 08. No mezcles versiones de CSV.')
    if not np.isfinite(escalados[['z_' + v for v in sensores]].to_numpy(float)).all():
        raise ValueError('Entradas escaladas con NaN/infinito.')
    if (escalados['conjunto'].value_counts().reindex(PARTES, fill_value=0) == 0).any():
        raise ValueError('Falta algún conjunto de entrenamiento, validación o prueba.')
    # Trazabilidad: comparar el reescalado de la fuente con las salidas 08.
    muestra = np.linspace(0, len(raw)-1, min(len(raw), 300), dtype=int)
    transformado = obj_prep['escalador'].transform(
        obj_prep['imputador'].transform(raw.iloc[muestra][sensores]))
    guardado = escalados.loc[muestra, ['z_' + v for v in sensores]].to_numpy(float)
    diferencia = float(np.max(np.abs(transformado - guardado)))
    if diferencia > 1e-5:
        raise ValueError('Los valores físicos del CSV no coinciden con los escalados del 08. Diferencia: ' + str(diferencia))
    return {
        'raw': raw, 'esc': escalados, 'sensores': sensores, 'if': modelo_if,
        'km': modelo_km, 'db': db,
        'umbral_if': float(conf_if['umbral_puntuacion']),
        'umbral_km': float(conf_km['umbral_distancia']),
        'fuente': str(fuente.resolve()), 'base': base,
        'error_verificacion_escalado': diferencia,
    }


def ajuste_final_con_todos(obj: dict, salida: Path):
    """OPCIONAL: reentrena con todos los datos para un modelo final de uso.

    Al usar los 143.561 para ajustar, NO quedan muestras independientes de
    evaluación. Los resultados 'prueba' posteriores son DESCRIPTIVOS y no una
    nueva prueba externa. Los modelos previos NO se sobrescriben.
    """
    raw, sensores = obj['raw'], obj['sensores']
    n=len(raw)
    print('AJUSTE FINAL: usando TODOS los registros. Sin prueba independiente.',flush=True)
    imp=SimpleImputer(strategy='median')
    scaler=StandardScaler()
    xa=scaler.fit_transform(imp.fit_transform(raw[sensores]))
    for j,c in enumerate(sensores):
        obj['esc']['z_'+c]=xa[:,j]
    obj['if']=IsolationForest(n_estimators=300,max_samples='auto',
                              contamination='auto',random_state=SEED,n_jobs=-1).fit(xa)
    scores=-obj['if'].score_samples(xa)
    fraccion_if=0.01
    obj['umbral_if']=float(np.quantile(scores,1-fraccion_if))
    k=int(obj['km'].n_clusters) # K elegido previamente con la validación.
    obj['km']=KMeans(n_clusters=k,n_init=10,random_state=SEED,max_iter=300).fit(xa)
    dist=np.min(obj['km'].transform(xa),axis=1)
    obj['umbral_km']=float(np.quantile(dist,.99))
    db_original=obj['db']['modelo_dbscan_muestra']
    n_sample=min(n, int(len(db_original.labels_)))
    rng=np.random.default_rng(SEED)
    ids=np.sort(rng.choice(n,size=n_sample,replace=False))
    xm=xa[ids]
    min_samples=int(db_original.min_samples)
    vecinos=NearestNeighbors(n_neighbors=min_samples,n_jobs=1).fit(xm)
    distancias,_=vecinos.kneighbors(xm)
    kdist=distancias[:,-1]
    candidatos=sorted({float(np.quantile(kdist,q)) for q in (.65,.75,.85,.90,.95,.98) if np.quantile(kdist,q)>0})
    if not candidatos:
        raise ValueError('No se pudo seleccionar eps para el ajuste final de DBSCAN.')
    posibles=[]
    for eps in candidatos:
        dbm=DBSCAN(eps=eps,min_samples=min_samples,n_jobs=1).fit(xm)
        n_clusters=len(set(dbm.labels_) - {-1})
        penal=100 if n_clusters==0 or len(dbm.core_sample_indices_)==0 else 0
        err=penal+abs(float(np.mean(dbm.labels_==-1))-.05)
        posibles.append((err,eps,dbm))
    _,eps,dbm=min(posibles,key=lambda x:(x[0],x[1]))
    if not len(dbm.core_sample_indices_):
        raise ValueError('DBSCAN sin núcleos después de ajustar todos los datos.')
    nn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(dbm.components_)
    obj['db']={'modelo_dbscan_muestra':dbm,'vecinos_core':nn,
               'etiquetas_nucleos':dbm.labels_[dbm.core_sample_indices_],
               'eps':float(eps),'min_samples':min_samples,
               'variables':['z_'+c for c in sensores],
               'indice_muestra_entrenamiento':ids}
    obj['ajuste_final_todos']=True
    dir_m=salida/'00_modelos_reajustados_todos'
    dir_m.mkdir(parents=True,exist_ok=True)
    joblib.dump({'imputador':imp,'escalador':scaler,'variables':sensores,'columna_smp':SMP},
                dir_m/'preprocesador_final.joblib')
    joblib.dump(obj['if'],dir_m/'isolation_forest_final.joblib')
    joblib.dump(obj['km'],dir_m/'kmeans_final.joblib')
    joblib.dump(obj['db'],dir_m/'dbscan_muestra_final.joblib')
    (dir_m/'umbrales_finales.json').write_text(json.dumps({
        'umbral_if':obj['umbral_if'], 'umbral_kmeans':obj['umbral_km'],
        'eps_dbscan':float(eps),'K':k,
        'muestra_dbscan':n_sample,
        'advertencia':'Reajustado con los 143.561 registros: NO hay prueba externa independiente.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    return xa


def puntuar_modelos(obj: dict, tamano_lote: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    raw, esc, cols = obj['raw'], obj['esc'], ['z_' + c for c in obj['sensores']]
    X = esc[cols].to_numpy(float)
    n = len(X)
    score = np.empty(n, dtype=float)
    dist = np.empty(n, dtype=float)
    grupos_km = np.empty(n, dtype=np.int32)
    grupos_db = np.empty(n, dtype=np.int32)
    dist_nucleo = np.empty(n, dtype=float)
    # No ajustar los modelos; aplicación sobre TODOS los registros existentes.
    vecinos = obj['db']['vecinos_core']
    etiquetas_core = np.asarray(obj['db']['etiquetas_nucleos'], dtype=int)
    eps = float(obj['db']['eps'])
    for inicio in range(0, n, tamano_lote):
        fin = min(inicio+tamano_lote, n)
        x = X[inicio:fin]
        score[inicio:fin] = -obj['if'].score_samples(x)
        dkm = obj['km'].transform(x)
        grupos_km[inicio:fin] = np.argmin(dkm, axis=1)
        dist[inicio:fin] = np.min(dkm, axis=1)
        dd, idx = vecinos.kneighbors(x, n_neighbors=1)
        dd = dd[:, 0]
        etiqueta = etiquetas_core[idx[:, 0]].astype(np.int32, copy=True)
        etiqueta[dd > eps] = -1
        dist_nucleo[inicio:fin] = dd
        grupos_db[inicio:fin] = etiqueta
        if n > 50000 and (inicio == 0 or (fin // tamano_lote) % 5 == 0):
            print(f'  Aplicación de modelos: {fin:,} / {n:,} registros', flush=True)
    base = raw[[SMP, *obj['sensores']]].copy()
    base['conjunto'] = esc['conjunto'].to_numpy()
    base['puntaje_if'] = score
    base['alerta_if'] = (score >= obj['umbral_if']).astype('int8')
    base['cluster_kmeans'] = grupos_km
    base['distancia_centroide_kmeans'] = dist
    base['alerta_kmeans'] = (dist >= obj['umbral_km']).astype('int8')
    base['cluster_dbscan_aproximado'] = grupos_db
    base['distancia_nucleo_dbscan'] = dist_nucleo
    base['ruido_dbscan_aproximado'] = (grupos_db == -1).astype('int8')
    columnas_alarma = ['alerta_if', 'alerta_kmeans', 'ruido_dbscan_aproximado']
    base['votos_modelos'] = base[columnas_alarma].sum(axis=1).astype('int8')
    base['consenso_al_menos_dos'] = (base['votos_modelos'] >= 2).astype('int8')
    base['consenso_tres'] = (base['votos_modelos'] == 3).astype('int8')
    base['clasificacion_revision_propuesta'] = np.select(
        [base['votos_modelos'] == 3, base['votos_modelos'] == 2, base['votos_modelos'] == 1],
        ['revisar_prioritariamente', 'revisar', 'seguir'], default='sin_alerta')
    meta = {
        'umbral_if': obj['umbral_if'], 'umbral_distancia_kmeans': obj['umbral_km'],
        'eps_dbscan': eps, 'K_kmeans': int(obj['km'].n_clusters),
        'dbscan_clusters_muestra': int(len(set(obj['db']['modelo_dbscan_muestra'].labels_) - {-1})),
        'dbscan_registros_muestra': int(len(obj['db']['modelo_dbscan_muestra'].labels_)),
        'dbscan_ruido_muestra': int(np.sum(obj['db']['modelo_dbscan_muestra'].labels_ == -1)),
        'dbscan_nucleos_muestra': int(len(obj['db']['modelo_dbscan_muestra'].core_sample_indices_)),
        'filas': n,
    }
    return base, X, meta


def construir_episodios(df: pd.DataFrame, nombre_columna: str) -> pd.DataFrame:
    smp = df[SMP].to_numpy(np.int64)
    alerta = df[nombre_columna].to_numpy(bool)
    if not alerta.any():
        return pd.DataFrame(columns=['episodio','smp_inicial','smp_final','longitud_muestras','conjunto_inicial','conjunto_final'])
    inicios = np.flatnonzero(alerta & np.r_[True, (~alerta[:-1]) | (np.diff(smp) != 1)])
    finales = np.flatnonzero(alerta & np.r_[(~alerta[1:]) | (np.diff(smp) != 1), True])
    return pd.DataFrame({
        'episodio': np.arange(1, len(inicios)+1),
        'smp_inicial': smp[inicios], 'smp_final': smp[finales],
        'longitud_muestras': finales - inicios + 1,
        'conjunto_inicial': df['conjunto'].to_numpy()[inicios],
        'conjunto_final': df['conjunto'].to_numpy()[finales],
    })


def jaccard(a, b):
    union = int(np.count_nonzero(a | b))
    inter = int(np.count_nonzero(a & b))
    return inter, 100*inter/union if union else float('nan')


def informes_tabulares(df: pd.DataFrame, obj: dict, salida: Path, tam_bloque: int):
    alertas = [('Isolation Forest', 'alerta_if'), ('K-means', 'alerta_kmeans'),
               ('DBSCAN aproximado', 'ruido_dbscan_aproximado'),
               ('Consenso 2+', 'consenso_al_menos_dos'), ('Consenso 3', 'consenso_tres')]
    filas = []
    for conjunto in [*PARTES, 'todos']:
        t = df if conjunto == 'todos' else df.loc[df['conjunto'] == conjunto]
        for nombre, c in alertas:
            episodios = construir_episodios(t, c)
            filas.append({'conjunto': conjunto, 'modelo': nombre, 'registros': len(t),
                          'alertas': int(t[c].sum()), 'porcentaje': 100*float(t[c].mean()),
                          'alertas_por_1000': 1000*float(t[c].mean()),
                          'episodios': len(episodios)})
            if conjunto == 'todos':
                archivo = {'alerta_if':'05_episodios_isolation_forest.csv',
                           'alerta_kmeans':'06_episodios_kmeans.csv',
                           'ruido_dbscan_aproximado':'07_episodios_dbscan.csv',
                           'consenso_al_menos_dos':'08_episodios_consenso.csv',
                           'consenso_tres':'09_episodios_consenso_tres.csv'}[c]
                csv_out(episodios, salida / archivo)
    resumen = pd.DataFrame(filas)
    csv_out(resumen, salida / '00_resumen_por_conjunto_y_total.csv')
    grupos = df.groupby(['conjunto','cluster_kmeans'], sort=True).size().reset_index(name='registros')
    totalgrupos = df.groupby('cluster_kmeans').size().reset_index(name='registros')
    totalgrupos.insert(0, 'conjunto', 'todos')
    csv_out(pd.concat([grupos, totalgrupos], ignore_index=True), salida / '02_tamanos_grupos_kmeans.csv')
    perfil = df.groupby('cluster_kmeans', sort=True)[obj['sensores']].agg(['mean','std','median']).reset_index()
    perfil.columns = ['cluster_kmeans' if c[0]=='cluster_kmeans' else f'{c[0]}__{c[1]}' for c in perfil.columns]
    csv_out(perfil, salida / '03_perfil_fisico_clusters_kmeans.csv')
    dbdis = df.groupby(['conjunto','cluster_dbscan_aproximado']).size().reset_index(name='registros')
    dbtodos = df.groupby('cluster_dbscan_aproximado').size().reset_index(name='registros')
    dbtodos.insert(0, 'conjunto', 'todos')
    csv_out(pd.concat([dbdis, dbtodos], ignore_index=True), salida / '04_distribucion_dbscan_aproximado.csv')
    # Comparación de niveles de variables entre ruido y no ruido.
    perfil_db = df.assign(es_ruido_dbscan=np.where(df['ruido_dbscan_aproximado'].eq(1), 'ruido', 'no_ruido'))
    perfil_db = perfil_db.groupby('es_ruido_dbscan')[obj['sensores']].agg(['mean','std','median']).reset_index()
    perfil_db.columns = ['es_ruido_dbscan' if c[0]=='es_ruido_dbscan' else f'{c[0]}__{c[1]}' for c in perfil_db.columns]
    csv_out(perfil_db, salida / '04b_perfil_fisico_dbscan.csv')

    pares = [('IF','K-means','alerta_if','alerta_kmeans'),
             ('IF','DBSCAN','alerta_if','ruido_dbscan_aproximado'),
             ('K-means','DBSCAN','alerta_kmeans','ruido_dbscan_aproximado')]
    acuerdos = []
    for conjunto in [*PARTES, 'todos']:
        t = df if conjunto == 'todos' else df.loc[df['conjunto'] == conjunto]
        for a,b,ca,cb in pares:
            ab, jac = jaccard(t[ca].to_numpy(bool), t[cb].to_numpy(bool))
            acuerdos.append({'conjunto':conjunto,'modelo_A':a,'modelo_B':b,
                             'coincidencias':ab,'jaccard_pct':jac})
    csv_out(pd.DataFrame(acuerdos), salida / '10_jaccard_conjuntos_y_total.csv')
    tabla_combinaciones = df.groupby(['alerta_if','alerta_kmeans','ruido_dbscan_aproximado']).size().reset_index(name='registros')
    csv_out(tabla_combinaciones, salida / '11_combinaciones_alertas_total.csv')

    # Los BLOQUES cubren TODOS los SMP, no solamente prueba.
    b = df[[SMP, 'conjunto', 'alerta_if', 'alerta_kmeans',
            'ruido_dbscan_aproximado', 'consenso_al_menos_dos']].copy()
    b['bloque_registros'] = (np.arange(len(b)) // tam_bloque) + 1
    bloques = b.groupby('bloque_registros', sort=True).agg(
        smp_inicial=(SMP,'min'), smp_final=(SMP,'max'), registros=(SMP,'size'),
        IF=('alerta_if','sum'), Kmeans=('alerta_kmeans','sum'),
        DBSCAN=('ruido_dbscan_aproximado','sum'), consenso=('consenso_al_menos_dos','sum'),
    ).reset_index()
    for c in ('IF','Kmeans','DBSCAN','consenso'):
        bloques[f'{c}_pct'] = 100 * bloques[c] / bloques['registros']
    csv_out(bloques, salida / '12_alertas_por_bloques_smp.csv')
    return resumen, grupos, bloques


def pca_completo(df: pd.DataFrame, X: np.ndarray, salida: Path, ajuste_final_todos=False):
    mask = (np.ones(len(df),dtype=bool) if ajuste_final_todos
            else df['conjunto'].to_numpy() == 'entrenamiento')
    pca = PCA(n_components=X.shape[1], svd_solver='full')
    pca.fit(X[mask])
    coords = pca.transform(X)
    nombres = [f'PC{i}' for i in range(1, X.shape[1]+1)]
    for i, c in enumerate(nombres):
        df[c] = coords[:, i]
    valores = pd.DataFrame({
        'componente': nombres, 'varianza_individual_pct': 100*pca.explained_variance_ratio_,
        'varianza_acumulada_pct':100*np.cumsum(pca.explained_variance_ratio_),
        'autovalor':pca.explained_variance_,
    })
    csv_out(valores, salida / '13_pca_varianza_10_componentes.csv')
    cargas = pd.DataFrame(pca.components_.T, columns=nombres)
    cargas.insert(0, 'sensor', CORTOS)
    csv_out(cargas, salida / '14_pca_coeficientes_componentes.csv')
    stds = np.std(X[mask], axis=0, ddof=1)
    corr = pca.components_.T * np.sqrt(pca.explained_variance_)[None,:] / stds[:,None]
    corr = np.clip(corr, -1, 1)
    cc = pd.DataFrame(corr, columns=nombres)
    cc.insert(0, 'sensor', CORTOS)
    csv_out(cc, salida / '15_pca_correlaciones_flechas.csv')
    return pca, coords, corr, valores


def diagnostico_silueta(obj: dict, df: pd.DataFrame, X: np.ndarray,
                       salida: Path, tam_silueta: int):
    seleccion = obj['base'] / '02_kmeans' / '00_seleccion_k.csv'
    tabla = pd.read_csv(existe(seleccion), sep=';', encoding='utf-8-sig')
    csv_out(tabla, salida / '16_silueta_elige_k_original.csv')
    elegido = int(obj['km'].n_clusters)
    fila = tabla.loc[tabla['K'] == elegido]
    sil_criterio = float(fila['silhouette_muestra_validacion'].iloc[0]) if len(fila) else float('nan')
    # Calcular silueta de un SUBCONJUNTO aleatorio sin matrices NxN gigantes.
    n = len(X)
    rng = np.random.default_rng(SEED)
    elegido_muestra = np.sort(rng.choice(n, size=min(tam_silueta, n), replace=False))
    etiquetas = df['cluster_kmeans'].to_numpy()[elegido_muestra]
    if 2 <= len(np.unique(etiquetas)) <= len(etiquetas)-1:
        valores = silhouette_samples(X[elegido_muestra], etiquetas)
        diag = float(np.mean(valores))
        tabla_puntos = pd.DataFrame({SMP:df[SMP].to_numpy()[elegido_muestra],
                                     'cluster_kmeans':etiquetas,'silueta':valores})
        csv_out(tabla_puntos, salida / '17_silueta_por_registro_muestra.csv')
    else:
        valores = np.array([])
        diag = float('nan')
        csv_out(pd.DataFrame(columns=[SMP,'cluster_kmeans','silueta']), salida/'17_silueta_por_registro_muestra.csv')
    mdb = obj['db']['modelo_dbscan_muestra']
    etiquetas_db = np.asarray(mdb.labels_, dtype=int)
    sin_ruido = etiquetas_db >= 0
    if 2 <= np.unique(etiquetas_db[sin_ruido]).size <= sin_ruido.sum()-1:
        # Para DBSCAN de muestra necesitamos los X originales de la muestra de entrenamiento:
        indices_muestra = np.asarray(obj['db']['indice_muestra_entrenamiento'], dtype=int)
        x_base = X if obj.get('ajuste_final_todos') else X[df['conjunto'].to_numpy() == 'entrenamiento']
        xm = x_base[indices_muestra]
        sil_db = float(silhouette_score(xm[sin_ruido], etiquetas_db[sin_ruido]))
        nota_db = 'Solo puntos no ruido en la muestra nativa DBSCAN.'
    else:
        sil_db = None
        nota_db = 'No definida: DBSCAN nativo tiene solo 1 grupo no ruido (o no hay suficientes grupos).'
    res = pd.DataFrame([{
        'K_seleccionado': elegido,
        'silueta_validacion_usada_para_elegir_K': sil_criterio,
        'silueta_diagnostica_muestra_de_todos': diag,
        'n_registros_diagnostico_todos': int(len(elegido_muestra)),
        'silueta_dbscan_muestra_sin_ruido': sil_db,
        'nota_dbscan': nota_db,
    }])
    csv_out(res, salida / '18_resumen_silueta.csv')
    return tabla, elegido_muestra, valores, res


def grafico_pca_circulo(corr: np.ndarray, pair: tuple[int,int], salida: Path) -> None:
    i, j = pair
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.add_patch(plt.Circle((0,0), 1, fill=False, edgecolor=GRIS, linewidth=1.2))
    ax.axhline(0, lw=0.8, c='#9CA7B4')
    ax.axvline(0, lw=0.8, c='#9CA7B4')
    for k, nombre in enumerate(CORTOS):
        x, y = corr[k,i], corr[k,j]
        ax.annotate('', xy=(x,y), xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=1.7, color=PALETA[k%len(PALETA)]))
        ax.text(x*1.09, y*1.09, nombre, size=8, color=PALETA[k%len(PALETA)],
                ha='center', va='center')
    ax.set(xlim=(-1.17,1.17), ylim=(-1.17,1.17), xlabel=f'PC{i+1}', ylabel=f'PC{j+1}',
           title=f'Círculo de correlaciones | PC{i+1}–PC{j+1}')
    ax.set_aspect('equal', adjustable='box')
    guardar(fig, salida / f'24_circulo_angulos_PC{i+1}_PC{j+1}.png')


def grafico_dispersion(df: pd.DataFrame, obj: dict, pca: PCA,
                      pair: tuple[int, int], tipo: str, salida: Path) -> None:
    i,j = pair
    x, y = df[f'PC{i+1}'].to_numpy(), df[f'PC{j+1}'].to_numpy()
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    if tipo == 'kmeans':
        labels = df['cluster_kmeans'].to_numpy(int)
        for k in np.unique(labels):
            m = labels == k
            ax.scatter(x[m], y[m], s=2.5, alpha=.25, color=PALETA[int(k)%len(PALETA)],
                       label=f'Cluster {k} (n={m.sum():,})', rasterized=True)
        centros_pca = pca.transform(obj['km'].cluster_centers_)
        ax.scatter(centros_pca[:,i], centros_pca[:,j], marker='X', s=85, c='black',
                   linewidths=.7, edgecolors='white', zorder=4, label='Centroides')
        titulo = 'K-means: clusters y centroides'
    elif tipo == 'dbscan':
        labels = df['cluster_dbscan_aproximado'].to_numpy(int)
        for k in np.unique(labels):
            m = labels == k
            color = ROJO if k == -1 else PALETA[int(k)%len(PALETA)]
            nombre = 'Ruido aprox.' if k == -1 else f'Cluster {k} aprox.'
            ax.scatter(x[m], y[m], s=2.3, alpha=.27, color=color,
                       label=f'{nombre} (n={m.sum():,})', rasterized=True)
        titulo = 'DBSCAN: clasificación aproximada por núcleos'
    elif tipo == 'if':
        flags = df['alerta_if'].to_numpy(bool)
        ax.scatter(x[~flags], y[~flags], s=2, c=GRIS, alpha=.10,
                   rasterized=True, label=f'Sin alerta ({(~flags).sum():,})')
        ax.scatter(x[flags], y[flags], s=4, c=ROJO, alpha=.65,
                   rasterized=True, label=f'Alertas IF ({flags.sum():,})')
        titulo = 'Isolation Forest: alertas'
    else:
        votos = df['votos_modelos'].to_numpy(int)
        for k in range(4):
            m = votos == k
            ax.scatter(x[m], y[m], s=2.5, color=[GRIS,AZUL,NARANJA,ROJO][k],
                       alpha=.15 if k==0 else .55, rasterized=True,
                       label=f'{k} modelos (n={m.sum():,})')
        titulo = 'Consenso entre modelos'
    ax.set(xlabel=f'PC{i+1} (PCA ajustado '+('con todo el histórico' if obj.get('ajuste_final_todos') else 'solo con entrenamiento')+')', ylabel=f'PC{j+1}',
           title=f'{titulo} | PC{i+1}–PC{j+1} | {len(df):,} registros')
    if tipo == 'kmeans':
        # Leyenda con tamaños fijos, para evitar que la X de centroides salga gigante.
        from matplotlib.lines import Line2D
        items = [
            Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                   markerfacecolor=PALETA[int(k) % len(PALETA)],
                   markeredgecolor='none',
                   label=f'Cluster {k} (n={int((labels == k).sum()):,})')
            for k in np.unique(labels)
        ]
        items.append(Line2D([0], [0], marker='X', linestyle='None',
                            markersize=8, markerfacecolor='black',
                            markeredgecolor='white', markeredgewidth=0.6,
                            label='Centroides'))
        ax.legend(handles=items, frameon=False, fontsize=8, loc='best')
    else:
        ax.legend(frameon=False, markerscale=3, fontsize=8, loc='best')
    ax.grid(alpha=.10)
    guardar(fig, salida / f'25_proyeccion_{tipo}_PC{i+1}_PC{j+1}.png')


def dibujar_generales(df, obj, pca, corr, var, sel, indices_sil, valores_sil, resumen,
                       bloques, salida: Path):
    png = salida / 'Graficos'
    png.mkdir(parents=True, exist_ok=True)
    preparar_estilo()
    # K-means: elección K, inercia, silueta individual y tamaño real grupos.
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    ax.plot(sel['K'], sel['silhouette_muestra_validacion'], 'o-', color=MORADO, lw=2)
    ax.axvline(obj['km'].n_clusters, color=ROJO, ls='--', label=f'K seleccionado = {obj["km"].n_clusters}')
    ax.set(xlabel='Número de clusters K', ylabel='Coeficiente de silueta (muestra validación)',
           title='K-means | Selección de K mediante silueta')
    ax.legend(frameon=False); guardar(fig, png/'01_kmeans_seleccion_K_silueta.png')
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    ax.plot(sel['K'], sel['inercia_muestra_entrenamiento'], 'o-', color=AZUL, lw=2)
    ax.set(xlabel='K', ylabel='Inercia (muestra de entrenamiento)',
           title='K-means | Método del codo (misma muestra que la búsqueda K)')
    guardar(fig, png/'02_kmeans_codo.png')
    fig, ax = plt.subplots(figsize=(7.7, 4.3))
    t = df.groupby('cluster_kmeans').size()
    ax.bar([f'Grupo {i}' for i in t.index], t.values, color=PALETA[:len(t)])
    for k, n in enumerate(t.values): ax.text(k,n+0.01*len(df),f'{n:,}',ha='center')
    ax.set(ylabel='Registros', title='K-means | Tamaño de clusters en TODOS los datos')
    guardar(fig, png/'03_kmeans_tamano_grupos_total.png')
    if len(valores_sil):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        tab = pd.DataFrame({'grupo':df['cluster_kmeans'].to_numpy()[indices_sil], 'valor':valores_sil})
        grupos = sorted(tab['grupo'].unique())
        ax.boxplot([tab.loc[tab.grupo==g,'valor'] for g in grupos], tick_labels=[f'Grupo {g}' for g in grupos])
        ax.axhline(0, color=ROJO, ls='--', alpha=.7)
        ax.set(ylabel='Silueta', title='K-means | Silueta por grupo (submuestra de TODOS los datos)')
        guardar(fig, png/'04_kmeans_silueta_por_grupo.png')
    # DBSCAN muestra: eps y k-distancias.
    ruta_busqueda = obj['base'] / '03_dbscan' / '00_busqueda_eps.csv'
    if ruta_busqueda.is_file():
        busca = pd.read_csv(ruta_busqueda, sep=';', encoding='utf-8-sig')
        fig, ax = plt.subplots(figsize=(7.7, 4.5))
        ax.plot(busca['eps'], busca['porcentaje_ruido_muestra'], 'o-', color=ROJO)
        ax.axvline(float(obj['db']['eps']), ls='--', color='#333333', label=f'eps = {obj["db"]["eps"]:.3f}')
        ax.set(xlabel='eps candidato',ylabel='Ruido en muestra nativa (%)',title='DBSCAN | Sensibilidad de eps')
        ax.legend(frameon=False)
        guardar(fig, png/'05_dbscan_sensibilidad_eps.png')
    muestra_inds = np.asarray(obj['db']['indice_muestra_entrenamiento'], dtype=int)
    n_train = int((df['conjunto'] == 'entrenamiento').sum())
    X_base = obj['_X'] if obj.get('ajuste_final_todos') else obj['_X'][:n_train]
    modelo_db = obj['db']['modelo_dbscan_muestra']
    xm = X_base[muestra_inds]
    vecinos = NearestNeighbors(n_neighbors=int(modelo_db.min_samples)).fit(xm)
    distancias_k, _ = vecinos.kneighbors(xm)
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ax.plot(np.sort(distancias_k[:,-1]), color=AZUL, lw=1.4)
    ax.axhline(float(modelo_db.eps),ls='--',color=ROJO, label=f'eps = {modelo_db.eps:.3f}')
    ax.set(xlabel='Muestras ordenadas',ylabel=f'Distancia al {modelo_db.min_samples}º vecino',
           title='DBSCAN | Curva k-distance en muestra nativa')
    ax.legend(frameon=False); guardar(fig, png/'06_dbscan_distancias_k.png')
    fig,ax = plt.subplots(figsize=(7.7,4.4))
    tt = df.groupby('cluster_dbscan_aproximado').size()
    ax.bar([('Ruido' if k==-1 else f'Cluster {k}') for k in tt.index], tt.values,
           color=[ROJO if k==-1 else AZUL for k in tt.index])
    ax.set(ylabel='Registros', title='DBSCAN | Distribución total (extensión aproximada)')
    guardar(fig, png/'07_dbscan_grupos_ruido_total.png')
    # Isolation Forest: puntuación y umbral en TODOS.
    fig, ax = plt.subplots(figsize=(9,5))
    for c,col in zip(PARTES,[AZUL,NARANJA,MORADO]):
        v = df.loc[df['conjunto']==c,'puntaje_if'].to_numpy()
        ax.hist(v,bins=70,density=True,alpha=.30,color=col,label=c.capitalize())
    ax.axvline(obj['umbral_if'], color=ROJO,ls='--',label=f'Umbral IF = {obj["umbral_if"]:.4f}')
    ax.set(xlabel='-score_samples (más alto = más anómalo)',ylabel='Densidad',
           title='Isolation Forest | Distribución de puntuaciones en los 143.561 registros')
    ax.legend(frameon=False); guardar(fig,png/'08_if_distribucion_puntuaciones_todos.png')
    fig,ax=plt.subplots(figsize=(11.0,4.7))
    ax.plot(df[SMP],df['puntaje_if'],lw=.25,c=AZUL,alpha=.5,label='Puntuación IF')
    alertas = df['alerta_if'].eq(1)
    ax.scatter(df.loc[alertas,SMP],df.loc[alertas,'puntaje_if'],s=2.5,c=ROJO,
               alpha=.6,rasterized=True,label='Alertas')
    ax.axhline(obj['umbral_if'],color='#222222',ls='--',label='Umbral fijo')
    ax.set(xlabel='SMP continuo (no segundos)',ylabel='Puntuación de anomalía',
           title='Isolation Forest | Todas las mediciones en orden SMP')
    ax.legend(frameon=False); guardar(fig,png/'09_if_puntuacion_vs_smp_total.png')
    # Alertas por partición y todo; comparativo.
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    v=resumen[resumen['modelo'].isin(['Isolation Forest','K-means','DBSCAN aproximado'])].copy()
    piv=v.pivot(index='conjunto', columns='modelo', values='porcentaje').reindex([*PARTES,'todos'])
    xs=np.arange(len(piv)); anch=.23
    for k,(name,c) in enumerate(zip(['Isolation Forest','K-means','DBSCAN aproximado'],[AZUL,NARANJA,ROJO])):
        ax.bar(xs+(k-1)*anch,piv[name].to_numpy(),width=anch,label=name,color=c)
    ax.set_xticks(xs,piv.index)
    ax.set(ylabel='Registros señalados (%)',title='Porcentaje de alertas | Entrenamiento, validación, prueba y TOTAL')
    ax.legend(frameon=False); guardar(fig,png/'10_comparacion_alertas_total_y_particiones.png')
    fig, ax = plt.subplots(figsize=(10.5,4.7))
    for c, color in [('IF_pct',AZUL),('Kmeans_pct',NARANJA),('DBSCAN_pct',ROJO),('consenso_pct',VERDE)]:
        ax.plot(bloques[SMP if False else 'smp_inicial'],bloques[c],label=c.replace('_pct',''),lw=1.25,c=color)
    ax.set(xlabel='SMP continuo (inicio de bloque)',ylabel='% de registros señalados',
           title='Alertas por bloques de adquisición | Todas las mediciones')
    ax.legend(frameon=False,ncol=4); guardar(fig,png/'11_alertas_por_bloques_todos.png')
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    dist = df['votos_modelos'].value_counts().reindex(range(4),fill_value=0)
    ax.bar([str(i) for i in range(4)],dist.values,color=[GRIS,AZUL,NARANJA,ROJO])
    ax.set(xlabel='Número de modelos que emiten alerta',ylabel='Registros',
           title='Consenso entre los tres algoritmos | Datos completos')
    guardar(fig,png/'12_consenso_todos.png')
    # PCA varianza individual/acumulada y ángulos.
    fig, ax=plt.subplots(figsize=(8.5,4.7))
    ax.bar(var['componente'],var['varianza_individual_pct'],color=AZUL,alpha=.7,label='Individual')
    ax2=ax.twinx()
    ax2.plot(var['componente'],var['varianza_acumulada_pct'],'o-',c=ROJO,label='Acumulada')
    ax2.axhline(90,ls='--',lw=.8,c='#777777')
    ax.set(ylabel='Varianza individual (%)',xlabel='Componente',
           title=('PCA ajustado con TODO el histórico' if obj.get('ajuste_final_todos') else 'PCA ajustado solo en entrenamiento'))
    ax2.set(ylabel='Varianza acumulada (%)',ylim=(0,105))
    guardar(fig,png/'13_pca_varianza.png')
    fig, ax=plt.subplots(figsize=(9,5.4))
    im=ax.imshow(corr,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(10),[f'PC{i}' for i in range(1,11)])
    ax.set_yticks(range(10),CORTOS)
    ax.set(title='Correlación de variables con componentes PCA')
    fig.colorbar(im,ax=ax,label='Correlación')
    guardar(fig,png/'14_pca_cargas_correlaciones.png')
    for par in ((0,1),(2,3),(0,4)):
        grafico_pca_circulo(corr,par,png)
        for tipo in ('kmeans','dbscan','if','consenso'):
            grafico_dispersion(df,obj,pca,par,tipo,png)


def elegir_shap(df: pd.DataFrame, cantidad: int, todos: bool) -> np.ndarray:
    if todos:
        return np.arange(len(df), dtype=int)
    if cantidad < 0:
        raise ValueError('--shap-cantidad no puede ser negativo')
    if cantidad == 0:
        return np.array([],dtype=int)
    rng = np.random.default_rng(SEED)
    # En cada conjunto: incluir alertas y normales en proporción equilibrada.
    cada_parte = max(1, cantidad // (2*len(PARTES)))
    ids = []
    for nombre in PARTES:
        m_part = np.flatnonzero(df['conjunto'].to_numpy() == nombre)
        mask_a = df['alerta_if'].to_numpy(bool)[m_part]
        aa = m_part[mask_a]
        nn = m_part[~mask_a]
        for pool in (aa,nn):
            if len(pool):
                ids.extend(rng.choice(pool, size=min(cada_parte,len(pool)), replace=False).tolist())
    faltan = max(0,cantidad-len(ids))
    if faltan:
        otros = np.setdiff1d(np.arange(len(df)),np.asarray(ids,dtype=int),assume_unique=True)
        ids.extend(rng.choice(otros,size=min(faltan,len(otros)),replace=False).tolist())
    return np.sort(np.asarray(ids,dtype=int))


def shap_informe_puntuacion(df: pd.DataFrame, X: np.ndarray, obj: dict, salida: Path,
                 cantidad: int, fondo_n: int, shap_todos: bool, tamano: int):
    try:
        import shap
    except ImportError as err:
        raise RuntimeError('Falta instalar SHAP: python -m pip install shap') from err
    ids = elegir_shap(df, cantidad, shap_todos)
    if len(ids)==0:
        print('SHAP omitido por --shap-cantidad 0')
        return {'registros_shap':0}
    if len(ids)>10000:
        print('AVISO: explicar todos los registros mediante Permutation SHAP puede llevar MUCHÍSIMO tiempo.')
    mask_train = np.ones(len(df),dtype=bool) if obj.get('ajuste_final_todos') else df['conjunto'].to_numpy() == 'entrenamiento'
    x_train = X[mask_train]
    rng = np.random.default_rng(SEED)
    bg=x_train[rng.choice(len(x_train),size=min(fondo_n,len(x_train)),replace=False)]
    def puntuar(matriz):
        return -obj['if'].score_samples(np.asarray(matriz,float))
    ex = shap.Explainer(puntuar, shap.maskers.Independent(bg,max_samples=len(bg)),
                        algorithm='permutation', feature_names=CORTOS, seed=SEED)
    archivo=salida/'19_shap_contribuciones.csv'
    importancia_suma=np.zeros(10,dtype=float)
    alertas_abs=np.zeros(10,dtype=float)
    normales_abs=np.zeros(10,dtype=float)
    cont_a=cont_n=0
    max_error=0.0
    top_explicacion=None
    heat_val=[];heat_smp=[]
    nombres=['shap_'+c.replace(' ','_') for c in CORTOS]
    for ini in range(0,len(ids),tamano):
        sub=ids[ini:ini+tamano]
        exp=ex(X[sub],max_evals=2*X.shape[1]+1,batch_size=64,silent=True)
        val=np.asarray(exp.values,float)
        base=np.broadcast_to(np.asarray(exp.base_values,float).ravel(),len(sub))
        scores=df['puntaje_if'].to_numpy()[sub]
        err=np.abs(base+val.sum(axis=1)-scores)
        max_error=max(max_error,float(err.max()))
        filas=df.iloc[sub][[SMP,'conjunto','puntaje_if','alerta_if']].copy().reset_index(drop=True)
        filas['base_shap']=base
        filas['puntaje_reconstruido']=base+val.sum(axis=1)
        filas['error_reconstruccion']=err
        for j,nom in enumerate(nombres):filas[nom]=val[:,j]
        filas.to_csv(archivo,sep=';',index=False,encoding='utf-8-sig',mode='w' if ini==0 else 'a',
                     header=(ini==0),float_format='%.10g')
        importancia_suma+=np.abs(val).sum(axis=0)
        a=df['alerta_if'].to_numpy(bool)[sub]
        if a.any():
            alertas_abs+=np.abs(val[a]).sum(axis=0)
            cont_a+=int(a.sum())
        if (~a).any():
            normales_abs+=np.abs(val[~a]).sum(axis=0)
            cont_n+=int((~a).sum())
        if top_explicacion is None or np.any(scores>top_explicacion[0]):
            pos=int(np.argmax(scores))
            if top_explicacion is None or float(scores[pos]) > top_explicacion[0]:
                top_explicacion=(float(scores[pos]),int(df[SMP].iloc[sub[pos]]),val[pos].copy())
        if len(heat_val)<60:
            espacio=60-len(heat_val)
            heat_val.extend(val[:espacio].tolist())
            heat_smp.extend(df[SMP].iloc[sub[:espacio]].tolist())
        if len(ids)>200 and (ini == 0 or (ini//tamano) % 10 == 0):
            print(f'  SHAP: {min(ini+tamano,len(ids)):,} / {len(ids):,} registros', flush=True)
    imp=pd.DataFrame({'sensor':CORTOS,'shap_abs_media':importancia_suma/len(ids),
                      'shap_abs_media_alertas':alertas_abs/max(cont_a,1),
                      'shap_abs_media_normales':normales_abs/max(cont_n,1)})
    imp=imp.sort_values('shap_abs_media',ascending=False)
    csv_out(imp,salida/'20_shap_importancias.csv')
    png=salida/'Graficos'
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(imp['sensor'].iloc[::-1],imp['shap_abs_media'].iloc[::-1],color=MORADO)
    ax.set(xlabel='Media |SHAP| (puntuación IF)',title=f'SHAP | {len(ids):,} registros explicados')
    guardar(fig,png/'30_shap_importancia.png')
    if top_explicacion is not None:
        puntaje,smp,v=top_explicacion
        orden=np.argsort(np.abs(v))
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(np.asarray(CORTOS)[orden],v[orden],color=[ROJO if v[k]>0 else AZUL for k in orden])
        ax.axvline(0,c='#444444',lw=.8)
        ax.set(xlabel='Contribución SHAP al puntaje IF (+ aumenta, - disminuye)',
               title=f'SHAP | Ejemplo SMP {smp} (score {puntaje:.4f})')
        guardar(fig,png/'31_shap_explicacion_individual.png')
    if heat_val:
        fig,ax=plt.subplots(figsize=(11,6))
        im=ax.imshow(np.asarray(heat_val),cmap='coolwarm',aspect='auto')
        ax.set_xticks(range(10),CORTOS,rotation=45,ha='right')
        ax.set(ylabel='Registros explicados (hasta 60)',title='SHAP | Mapa de calor de contribuciones locales')
        fig.colorbar(im,ax=ax,label='Valor SHAP (signo)')
        guardar(fig,png/'32_shap_mapa_calor.png')
    return {'registros_shap':len(ids),'alertas_shap':cont_a,'normales_shap':cont_n,
            'fondo_shap':len(bg),'error_maximo_reconstruccion_shap':max_error,
            'shap_todos':shap_todos}




def longitud_normalizacion_if(max_samples: int) -> float:
    """Mismo c(n) que scikit-learn para pasar longitud media a anomaly score."""
    n = int(max_samples)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    # np.euler_gamma es la constante de Euler-Mascheroni.
    return 2.0 * (np.log(n-1.0) + float(np.euler_gamma)) - 2.0*(n-1.0)/n


def shap_informe_arboles(df: pd.DataFrame, X: np.ndarray, obj: dict, salida: Path,
                         cantidad: int, shap_todos: bool, tamano: int):
    """TreeSHAP fiel a la longitud de rutas del IF. Puede incluir TODAS las filas.

    MUY IMPORTANTE: el valor explicado es la LONGITUD MEDIA de aislamiento;
    no es directamente -score_samples. El score se reconstruye aplicando la
    transformación score = 2**(-longitud/c(n)). SHAP no conmuta con una
    transformación no lineal: las contribuciones de longitud NO son SHAP del score.
    """
    try:
        import shap
    except ImportError as err:
        raise RuntimeError('Instala shap: python -m pip install shap') from err
    ids = elegir_shap(df, cantidad, shap_todos)
    if not len(ids):
        return {'registros_shap': 0}
    if len(ids) > 10000:
        print(f'AVISO: TreeSHAP evaluará {len(ids):,} filas; puede tardar bastante.', flush=True)
    explainer=shap.TreeExplainer(obj['if'])
    valor_base=float(np.asarray(explainer.expected_value,float).reshape(-1)[0])
    c=longitud_normalizacion_if(int(obj['if'].max_samples_))
    nombrecols=['shap_longitud_'+v.replace(' ','_') for v in CORTOS]
    ruta=salida/'19_shap_contribuciones_LONGITUD.csv'
    tot_abs=np.zeros(X.shape[1],dtype=float)
    a_abs=np.zeros(X.shape[1],dtype=float)
    n_abs=np.zeros(X.shape[1],dtype=float)
    cont_a=cont_n=0
    max_error=0.
    top=None
    heat=[]
    for ini in range(0,len(ids),tamano):
        sub=ids[ini:ini+tamano]
        valores=np.asarray(explainer.shap_values(X[sub]),dtype=float)
        if valores.ndim != 2 or valores.shape[1]!=10:
            raise ValueError('TreeSHAP ha generado una salida de dimensiones inesperadas.')
        longitud=valor_base+valores.sum(axis=1)
        score_reco=np.exp2(-longitud/c)
        score=df['puntaje_if'].to_numpy(float)[sub]
        dif=np.abs(score-score_reco)
        max_error=max(max_error,float(np.max(dif)))
        filas=df.iloc[sub][[SMP,'conjunto','puntaje_if','alerta_if']].copy().reset_index(drop=True)
        filas['longitud_base_esperada']=valor_base
        filas['longitud_arboles_reconstruida']=longitud
        filas['score_if_reconstruido_desde_longitud']=score_reco
        filas['error_reconstruccion_score']=dif
        for j,name in enumerate(nombrecols):
            filas[name]=valores[:,j]
        filas.to_csv(ruta,sep=';',index=False,encoding='utf-8-sig',
                     mode='w' if ini==0 else 'a',header=ini==0,float_format='%.10g')
        tot_abs+=np.abs(valores).sum(axis=0)
        flags=df['alerta_if'].to_numpy(bool)[sub]
        if flags.any():a_abs+=np.abs(valores[flags]).sum(axis=0);cont_a+=int(flags.sum())
        if (~flags).any():n_abs+=np.abs(valores[~flags]).sum(axis=0);cont_n+=int((~flags).sum())
        if top is None or float(np.max(score)) > top[0]:
            k=int(np.argmax(score))
            top=(float(score[k]), int(df[SMP].iloc[sub[k]]), valores[k].copy())
        if len(heat)<60:
            heat.extend(valores[:60-len(heat)].tolist())
        if len(ids)>300 and (ini==0 or (ini//tamano)%10==0):
            print(f'  TreeSHAP: {min(ini+tamano,len(ids)):,} / {len(ids):,}',flush=True)
    imp=pd.DataFrame({'sensor':CORTOS,
                      'shap_abs_media_longitud':tot_abs/len(ids),
                      'shap_abs_media_alertas':a_abs/max(1,cont_a),
                      'shap_abs_media_normales':n_abs/max(1,cont_n)}).sort_values('shap_abs_media_longitud',ascending=False)
    csv_out(imp,salida/'20_shap_importancias_LONGITUD.csv')
    png=salida/'Graficos'
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(imp.sensor.iloc[::-1],imp.shap_abs_media_longitud.iloc[::-1],color=MORADO)
    ax.set(xlabel='Promedio |TreeSHAP| (unidades de longitud de aislamiento)',
           title=f'TreeSHAP | {len(ids):,} registros explicados')
    guardar(fig,png/'30_treeshap_importancia_longitud.png')
    if top is not None:
        score,smp,val=top;orden=np.argsort(np.abs(val))
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(np.asarray(CORTOS)[orden],val[orden],
                color=[AZUL if val[i]>0 else ROJO for i in orden])
        ax.axvline(0,c='#444444',lw=.8)
        ax.set(xlabel='SHAP longitud (+ alarga camino, - acorta camino y aumenta anomalía)',
               title=f'TreeSHAP | SMP {smp} | IF score {score:.4f}')
        guardar(fig,png/'31_treeshap_explica_registro.png')
    if len(heat):
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(np.asarray(heat),aspect='auto',cmap='coolwarm')
        ax.set_xticks(range(10),CORTOS,rotation=45,ha='right')
        ax.set(ylabel='Registros explicados (hasta 60)',title='TreeSHAP | Contribuciones a longitud de aislamiento')
        fig.colorbar(im,ax=ax,label='SHAP de longitud')
        guardar(fig,png/'32_treeshap_mapa_calor.png')
    return {'modo_shap':'rapido_arboles_longitud', 'registros_shap':len(ids),
            'alertas_shap':cont_a,'normales_shap':cont_n,'longitud_base':valor_base,
            'normalizacion_score':c,'error_maximo_reconstruccion_score':max_error,
            'advertencia':'Las contribuciones SHAP corresponden a longitud de aislamiento; no son SHAP de la puntuación -score_samples.'}



def referencia_estadistica_tarea2(df: pd.DataFrame, sensores: list[str], salida: Path):
    """Reglas exploratorias sobre todo el período, como en la Tarea 2.

    NO son etiquetas reales, NO se utilizan para entrenar ni calibrar modelos.
    Sus parámetros emplean el histórico completo solo como DESCRIPCIÓN.
    """
    resumen=[]
    alerta_robusta=np.zeros(len(df),dtype=bool)
    critico_robusto=np.zeros(len(df),dtype=bool)
    alerta_6sigma=np.zeros(len(df),dtype=bool)
    for nombre,col in zip(CORTOS,sensores):
        x=df[col].to_numpy(float)
        med=float(np.nanmedian(x))
        mad=float(np.nanmedian(np.abs(x-med)))
        mu=float(np.nanmean(x))
        sigma=float(np.nanstd(x,ddof=1))
        if mad>0:
            zm=np.abs(.6745*(x-med)/mad)
            a=(zm>3.5)&(zm<=6.)
            c=(zm>6.)
        else:
            # MAD 0 => z modificado indefinido. Evitar etiquetas falsas.
            a=np.zeros(len(x),dtype=bool)
            c=np.zeros(len(x),dtype=bool)
        seis=(np.abs(x-mu)>6*sigma) if sigma>0 else np.zeros(len(x),dtype=bool)
        alerta_robusta|=a
        critico_robusto|=c
        alerta_6sigma|=seis
        resumen.append({'sensor':nombre,'columna':col,'media':mu,'desv_estandar':sigma,
                        'mediana':med,'MAD':mad,'alertas_Z_modificado':int(a.sum()),
                        'extremos_Z_modificado':int(c.sum()),
                        'fuera_6sigma':int(seis.sum()),
                        'nota':'Valores orientativos; no etiquetas de falla.' if mad>0 else 'MAD=0: no se definió Z modificado.'})
    df['alerta_z_modificado_global']=alerta_robusta.astype('int8')
    df['extremo_z_modificado_global']=critico_robusto.astype('int8')
    df['fuera_6sigma_global']=alerta_6sigma.astype('int8')
    df['estado_robusto_global']=np.where(critico_robusto,'extremo',
                                              np.where(alerta_robusta,'alerta','normal'))
    csv_out(pd.DataFrame(resumen),salida/'21_referencia_estadistica_Zmod_6sigma_Tarea2.csv')
    tabla=pd.DataFrame({'estado_robusto_global':['normal','alerta','extremo'],
                        'filas':[(df['estado_robusto_global']==c).sum() for c in ['normal','alerta','extremo']]})
    csv_out(tabla,salida/'22_estados_robustos_Tarea2_global.csv')
    return tabla


def anexar_validacion_5_cortes(obj: dict, salida: Path):
    """Incorpora RESULTADOS previos de los cinco cortes, sin repetir entrenamientos."""
    fuente=obj['base']/'06_validacion_5_cortes'/'01_resultados_por_corte.csv'
    if not fuente.is_file():
        return False
    t=pd.read_csv(fuente,sep=';',encoding='utf-8-sig')
    requeridas={'corte','modelo','porcentaje_alertas_validacion'}
    if not requeridas.issubset(t.columns):
        return False
    csv_out(t,salida/'23_validacion_5_cortes_PREVIA.csv')
    fig,ax=plt.subplots(figsize=(9.4,4.8))
    for modelo, grupo in t.groupby('modelo'):
        ax.plot(grupo['corte'],grupo['porcentaje_alertas_validacion'],
                'o-',lw=1.7,label=str(modelo))
    ax.set(xlabel='Corte cronológico dentro del entrenamiento original',
           ylabel='Alertas (%)',title='Validación original de cinco cortes (no recalculada)')
    ax.legend(frameon=False)
    guardar(fig,salida/'Graficos'/'33_validacion_5_cortes_previa.png')
    return True


def graficos_sensores_completos(df, sensores, salida):
    """Distribuciones y cambios por SMP de los 10 canales, usando todos los datos."""
    fig, axs=plt.subplots(2,5,figsize=(18,7))
    for i,ax in enumerate(axs.flat):
        v=df[sensores[i]].to_numpy(float)
        ax.hist(v[np.isfinite(v)],bins=80,color=AZUL,alpha=.83)
        ax.set(title=CORTOS[i],ylabel='Registros' if i%5==0 else None)
    fig.suptitle('Distribuciones de los 10 sensores | TODOS los registros',y=1.02)
    guardar(fig,salida/'Graficos'/'34_histogramas_10_sensores_todos.png')
    # Mediana de cada canal cada 1.000 filas: la forma tiene en cuenta la
    # adquisición, no afirma tiempo en segundos.
    g=np.arange(len(df))//1000
    med=df[sensores].groupby(g).median()
    med_smp=df[SMP].groupby(g).median()
    fig,axs=plt.subplots(5,2,figsize=(13,13))
    for i,ax in enumerate(axs.flat):
        ax.plot(med_smp,med[sensores[i]],c=PALETA[i%len(PALETA)],lw=1.4)
        ax.set(title=CORTOS[i],xlabel='SMP continuo (posición, no segundos)')
    fig.suptitle('Evolución de medianas por 1.000 registros | TODOS los sensores',y=1.01)
    guardar(fig,salida/'Graficos'/'35_evolucion_mediana_sensores_todos.png')

def ejecutar(args):
    raiz=Path(__file__).resolve().parent
    base=Path(args.resultados).expanduser() if args.resultados else raiz/'Resultados_Tarea3'
    if not base.is_absolute():base=raiz/base
    salida=Path(args.salida).expanduser() if args.salida else base/'11_analisis_completo_143561'
    if not salida.is_absolute():salida=raiz/salida
    salida.mkdir(parents=True,exist_ok=True)
    if args.lote < 1 or args.silueta_muestra < 50 or args.bloque < 50 or args.shap_lote < 1 or args.shap_fondo < 1:
        raise ValueError('Revisa --lote, --silueta-muestra, --bloque, --shap-lote y --shap-fondo.')
    print('\nCargando TODOS los registros y los modelos ya entrenados...',flush=True)
    obj=cargar(raiz,base,args.entrada)
    n=len(obj['raw'])
    if args.esperados>0 and n!=args.esperados:
        raise ValueError(f'El CSV contiene {n:,} registros, no {args.esperados:,}. Usa --esperados 0 para otra base.')
    print(f'Se procesarán {n:,} filas. No se elimina ningún registro.')
    if args.ajuste_final_todos:
        ajuste_final_con_todos(obj,salida)
    df,X,meta=puntuar_modelos(obj,args.lote)
    obj['_X']=X
    print('Calculando grupos, episodios, consensos y estadísticas...',flush=True)
    resumen,grupos,bloques=informes_tabulares(df,obj,salida,args.bloque)
    referencia=referencia_estadistica_tarea2(df,obj['sensores'],salida)
    print('Calculando PCA sobre las 10 variables (ajuste ' + ('con TODO' if args.ajuste_final_todos else 'solo en entrenamiento') + ', proyección de TODAS)...',flush=True)
    pca,coords,corr,var=pca_completo(df,X,salida,ajuste_final_todos=args.ajuste_final_todos)
    print('Calculando silueta K-means con muestra controlada y recuperando K original...',flush=True)
    sel,ids_sil,val_sil,tabla_sil=diagnostico_silueta(obj,df,X,salida,args.silueta_muestra)
    print('Generando gráficos de TODOS los registros...',flush=True)
    dibujar_generales(df,obj,pca,corr,var,sel,ids_sil,val_sil,resumen,bloques,salida)
    graficos_sensores_completos(df,obj['sensores'],salida)
    anexo_cortes=anexar_validacion_5_cortes(obj,salida)
    # Se guarda EL CSV de todas las filas, con variables físicas y PC1..PC10.
    csv_out(df,salida/f'01_resultados_TODOS_{n}_registros.csv')
    # SHAP requiere cálculo aparte: predicciones IF ya se hicieron sobre TODOS.
    met_shap={'registros_shap':0,'nota':'SHAP desactivado (--sin-shap).'}
    if not args.sin_shap and (args.shap_todos or args.shap_cantidad>0):
        print('Calculando SHAP; esta parte es más costosa que predecir...',flush=True)
        if args.shap_modo == 'rapido':
            met_shap=shap_informe_arboles(df,X,obj,salida,args.shap_cantidad,
                                         args.shap_todos,max(1,args.shap_lote))
        else:
            met_shap=shap_informe_puntuacion(df,X,obj,salida,args.shap_cantidad,args.shap_fondo,
                                            args.shap_todos,max(1,args.shap_lote))
    metadata={
        'fuente':obj['fuente'], 'registros_analizados':n,
        'smp_inicial':int(df[SMP].min()), 'smp_final':int(df[SMP].max()),
        'grupos_kmeans':int(obj['km'].n_clusters),
        'grupos_dbscan_nativo_muestra':meta['dbscan_clusters_muestra'],
        'muestra_dbscan':meta['dbscan_registros_muestra'],
        'registros_ruido_dbscan_muestra_nativa':meta['dbscan_ruido_muestra'],
        'umbral_if':meta['umbral_if'], 'umbral_kmeans':meta['umbral_distancia_kmeans'],
        'eps_dbscan':meta['eps_dbscan'],
        'silhouette_kmeans_validacion_elegir_K':float(tabla_sil['silueta_validacion_usada_para_elegir_K'].iloc[0]),
        'silhouette_kmeans_submuestra_todos':float(tabla_sil['silueta_diagnostica_muestra_de_todos'].iloc[0]),
        'ajuste_final_todos':bool(args.ajuste_final_todos),
        'PCA_ajustado_solo_entrenamiento':not args.ajuste_final_todos,
        'PCA_componentes_1_2_varianza_pct':float(var['varianza_individual_pct'].iloc[:2].sum()),
        'PCA_componentes_1_5_varianza_pct':float(var['varianza_individual_pct'].iloc[:5].sum()),
        'shap':met_shap,
        'anexo_validacion_5_cortes_original':anexo_cortes,
        'clasificacion_robusta_estadistica':referencia.to_dict(orient='records'),
        'errores_maximos_validacion_escalado_fuente':obj['error_verificacion_escalado'],
        'metodologia':[
            ('Ajuste final con 143.561 filas: NO hay conjunto de prueba independiente.' if args.ajuste_final_todos else
             'Aplicación a las 143.561 filas; modelos entrenados con 70%, sin usar prueba para ajustarlos.'),
            'K-means forma grupos usando 10 variables escaladas; alertas por distancia a centroide con umbral fijo.',
            'DBSCAN nativo fue ajustado en muestra; fuera de muestra etiqueta por núcleo más cercano y eps; es aproximación.',
            'Coeficiente de silueta es muestral para evitar costos cuadráticos sobre 143 mil puntos.',
            'TreeSHAP por defecto explica longitud de aislamiento; otro modo --shap-modo puntuacion reproduce la explicación del código 12.',
            'SHAP es costoso y por defecto explica muestra; --shap-todos explica todos con alto costo.',
            'PCA se ajusta con entrenamiento y proyecta todos; cada círculo muestra correlaciones en ese plano.',
            'Episodios requieren SMP consecutivo; SMP no tiene unidad de segundos.',
            'Las reglas Z modificado y 6sigma se computan sobre el histórico completo SOLO con fin descriptivo.',
            'No existen etiquetas reales de falla: no calcular precisión, recall, F1 ni matriz de confusión.',
            'La tasa de alertas en TODO incluye entrenamiento: no debe confundirse con evaluación externa del conjunto de prueba.',
        ]}
    (salida/'00_metodologia_y_limitaciones.json').write_text(
        json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('\nANÁLISIS INTEGRAL TERMINADO',flush=True)
    print('Total de registros: ',len(df))
    print('K-means: grupos =',obj['km'].n_clusters,'| silueta para elegir K =',f"{metadata['silhouette_kmeans_validacion_elegir_K']:.4f}")
    print('DBSCAN: grupos en muestra =',meta['dbscan_clusters_muestra'],'| asignación TOTAL = APROXIMADA')
    print('Resultados para TODOS:')
    print(resumen[resumen.conjunto=='todos'].to_string(index=False))
    print('SHAP:',met_shap)
    print('Carpeta:',salida)
    print('IMPORTANTE: ' + ('se reentrenaron sobre todos; sin prueba independiente.' if args.ajuste_final_todos
            else 'no se reentrenaron modelos; prueba externa conservada.') + ' No se conocen fallas confirmadas.')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--entrada',help='CSV fuente corregido. Por defecto se busca en Datos_corregidos/.')
    ap.add_argument('--resultados',help='Carpeta Resultados_Tarea3 del proyecto.')
    ap.add_argument('--salida',help='Carpeta de resultados de este programa.')
    ap.add_argument('--esperados',type=int,default=143561,help='Número de registros esperado (0 desactiva comprobación).')
    ap.add_argument('--lote',type=int,default=10000,help='Tamaño de lotes para aplicar modelos.')
    ap.add_argument('--bloque',type=int,default=1000,help='Bloques de adquisición para evolución de alertas.')
    ap.add_argument('--silueta-muestra',type=int,default=2000,help='Muestra para calcular silueta diagnóstica de todos los datos.')
    ap.add_argument('--ajuste-final-todos',action='store_true',
                    help='OPCIONAL: reentrenar tres detectores con los 143.561 para uso final; sin test independiente.')
    ap.add_argument('--sin-shap',action='store_true',help='Desactivar SHAP para una ejecución más rápida.')
    ap.add_argument('--shap-cantidad',type=int,default=120,help='Registros a explicar con SHAP (sobre todo el período).')
    ap.add_argument('--shap-modo',choices=['rapido','puntuacion'],default='rapido',
                    help='rapido = TreeSHAP de longitud; puntuacion = Permutation SHAP del puntaje exacto, mucho más lento.')
    ap.add_argument('--shap-fondo',type=int,default=30,help='Registros de entrenamiento usados como fondo SHAP.')
    ap.add_argument('--shap-lote',type=int,default=300,help='Registros SHAP por lote (en modo puntuacion sugerido 10).')
    ap.add_argument('--shap-todos',action='store_true',help='Explicar los 143.561 con SHAP (MUY costoso).')
    ejecutar(ap.parse_args())


if __name__=='__main__':
    main()
