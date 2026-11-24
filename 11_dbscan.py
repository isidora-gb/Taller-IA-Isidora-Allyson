
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
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components

PARTES = ('entrenamiento', 'validacion', 'prueba')
SMP = 'smp_continuo'
METRICAS = {'euclidiana': 2, 'manhattan': 1, 'chebyshev': np.inf}
CUANTILES = (0.10, 0.25, 0.35)


def guardar(df, ruta):
    pd.DataFrame(df).to_csv(ruta, sep=';', index=False, encoding='utf-8-sig', float_format='%.10g')


def leer_datos(carpeta, escalado):
    partes = {}
    for nombre in PARTES:
        ruta = carpeta / f'01_{nombre}_{escalado}.csv'
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {ruta}. Primero ejecuta 08_preparacion_datos.py')
        partes[nombre] = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
    variables = [c for c in partes['entrenamiento'].columns if c.startswith('z_')]
    if len(variables) != 10:
        raise ValueError('Se requieren las diez señales estandarizadas del codigo 08')
    for parte, df in partes.items():
        if df.empty or list(df.columns) != [SMP, *variables]:
            raise ValueError(f'Estructura inesperada en {parte}')
        if df[SMP].isna().any() or df[SMP].duplicated().any() or not df[SMP].is_monotonic_increasing:
            raise ValueError(f'Contador SMP invalido en {parte}')
        if not np.isfinite(df[variables].to_numpy(float)).all():
            raise ValueError(f'Datos no finitos en {parte}')
    if not (partes['entrenamiento'][SMP].iloc[-1] < partes['validacion'][SMP].iloc[0] <
            partes['validacion'][SMP].iloc[-1] < partes['prueba'][SMP].iloc[0]):
        raise ValueError('La division no es cronologica')
    # Verificar datos y columnas del 08, prohibir analisis por familia.
    metadatos = carpeta / '00_metadatos.json'
    if not metadatos.is_file():
        raise FileNotFoundError(f'Falta {metadatos}; ejecuta primero 08_preparacion_datos.py')
    info = json.loads(metadatos.read_text(encoding='utf-8'))
    if info.get('registros') != sum(len(t) for t in partes.values()):
        raise ValueError('Particiones no corresponden a los metadatos del codigo 08')
    if variables != ['z_' + x for x in info.get('variables', [])]:
        raise ValueError('Los nombres/orden de variables difieren del codigo 08')
    return partes, variables, info


def calcular_distancia_k(X, arbol, k, p):
    # cKDTree.query incluye al propio punto (distancia 0), como DBSCAN.
    dist, _ = arbol.query(X, k=k, p=p, workers=2)
    return np.asarray(dist[:, -1], dtype=float)


def dbscan_completo(X, eps, min_samples, p, arbol=None, dist_k=None, max_vecinos=1800):
    """DBSCAN sin matriz completa de distancias y con TODO X.

    Core: al menos min_samples puntos (incluyendose) a distancia <= eps.
    Clusters: componentes conexas de la relacion entre puntos core.
    Border: asignacion al punto core mas cercano dentro de eps.

    Si el vecindario resulta densisimo, no se aproxima un resultado: se rechaza
    explicitamente esa configuracion para proteger la memoria/equipo.
    """
    if eps <= 0 or min_samples < 2 or min_samples > len(X):
        raise ValueError('Parametros invalidos')
    arbol = arbol if arbol is not None else cKDTree(X)
    if dist_k is None:
        dist_k = calcular_distancia_k(X, arbol, min_samples, p)
    core_idx = np.flatnonzero(dist_k <= eps + 1e-12)
    n = len(X)
    etiquetas = np.full(n, -1, dtype=np.int32)
    if not len(core_idx):
        return etiquetas, core_idx, np.empty(0, np.int32), np.full(n, np.inf)
    cores = X[core_idx]
    arbol_core = cKDTree(cores)
    # Estimacion de densidad SOLO para control computacional; no selecciona modelos.
    sondas = np.linspace(0, len(cores)-1, min(96, len(cores)), dtype=int)
    counts = arbol_core.query_ball_point(cores[sondas], eps, p=p, return_length=True, workers=1)
    q90 = float(np.quantile(counts, 0.9))
    if q90 > max_vecinos:
        raise RuntimeError(f'Vecindarios demasiado densos para este barrido (p90={q90:.0f} > {max_vecinos}); '
                           'se omite la configuracion, NO se inventan etiquetas')
    # Componentes conexas exactas entre nucleos, calculadas en SciPy (codigo C).
    # Se estima memoria para evitar construir accidentalmente grafos enormes.
    # NO se afirma un resultado de agrupamiento cuando esta configuracion se omite.
    aristas_estimadas = float(len(cores) * np.mean(counts))
    tope_aristas = 12_000_000
    if aristas_estimadas > tope_aristas:
        raise RuntimeError(f'Grafo demasiado denso: ~{aristas_estimadas:.0f} conexiones estimadas '
                           f'(limite {tope_aristas}); configuracion NO evaluada')
    grafo = arbol_core.sparse_distance_matrix(arbol_core, eps, p=p,
                                               output_type='coo_matrix').tocsr()
    _, labels_core = connected_components(grafo, directed=False)
    labels_core = labels_core.astype(np.int32, copy=False)
    del grafo
    # Clasificar TODOS los registros, incluidos los border.
    # El punto core mas cercano da una asignacion DBSCAN legitima para border;
    # esta extension NO se usa para inferir clusters nuevos fuera del entrenamiento.
    d, cual = arbol_core.query(X, k=1, p=p, workers=2)
    dentro = d <= eps + 1e-12
    etiquetas[dentro] = labels_core[cual[dentro]]
    etiquetas[core_idx] = labels_core
    return etiquetas, core_idx, labels_core, np.asarray(d, dtype=float)


def resumen_labels(labels, n_core):
    labels = np.asarray(labels)
    vals, cant = np.unique(labels[labels >= 0], return_counts=True)
    grupos = int(len(vals))
    ruido = int(np.count_nonzero(labels == -1))
    return {'clusters': grupos, 'ruido_registros': ruido, 'ruido_pct': 100*ruido/len(labels),
            'nucleos': int(n_core), 'mayor_grupo_pct': 100*max(cant)/len(labels) if grupos else np.nan,
            'grupo_mas_pequeno': int(min(cant)) if grupos else 0}



def graficar_curva_k(dist, metrica, min_samples, salida, cuantiles, eps_referencia=None):
    """Traza las 100.492 distancias ordenadas, sin recortar muestras."""
    orden = np.sort(np.asarray(dist, dtype=float))
    fig, ax = plt.subplots(figsize=(10.7, 5.1))
    ax.plot(np.arange(1, len(orden) + 1), orden, color='#347ca3', lw=1.1,
            rasterized=True, label='Distancia al vecino k-ésimo (todos los registros)')
    for q in cuantiles:
        y = float(np.quantile(orden, q))
        ax.axhline(y, lw=1.0, ls='--', alpha=.8, label=f'Cuantil {100*q:.0f}%: eps={y:.3f}')
    if eps_referencia is not None:
        ax.axhline(eps_referencia, lw=1.6, color='#b44455', label='eps seleccionado')
    ax.set(xlabel='Registros de entrenamiento ordenados por distancia',
           ylabel=f'Distancia k-NN ({metrica}, k={min_samples})',
           title=f'DBSCAN | Curva k-distancia: {metrica}; min_samples={min_samples}')
    ax.grid(axis='y', alpha=.16)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(salida / f'01_curva_k_distancias_{metrica}_ms{min_samples}.png',
                dpi=220, bbox_inches='tight')
    plt.close(fig)


def graficar_categorias_pca(pca, X, labels, cores, salida, nombre):
    """Nucleos, frontera y ruido de entrenamiento en PCA para inspeccionar DBSCAN."""
    Y = pca.transform(X)
    tipo = np.full(len(X), 'Frontera', dtype=object)
    tipo[np.asarray(labels) == -1] = 'Ruido'
    tipo[np.asarray(cores, dtype=int)] = 'Nucleo'
    fig, ax = plt.subplots(figsize=(9.3, 6.1))
    for cat, col, size, alpha in [('Nucleo','#327b9c',1.0,.27),
                                  ('Frontera','#d69a35',1.8,.46),
                                  ('Ruido','#b84d5f',2.0,.43)]:
        mask=tipo==cat
        if mask.any():
            ax.scatter(Y[mask,0],Y[mask,1],s=size,c=col,alpha=alpha,
                       linewidths=0,rasterized=True,
                       label=f'{cat}: {int(mask.sum()):,}')
    v=100*pca.explained_variance_ratio_
    ax.set(xlabel=f'PC1 ({v[0]:.1f}% de varianza)',
           ylabel=f'PC2 ({v[1]:.1f}% de varianza)',
           title='DBSCAN | Tipos de observaciones del entrenamiento')
    ax.legend(frameon=False,fontsize=9)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(salida/'05_nucleos_frontera_ruido_pca.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def graficar_tamanos_y_evolucion(etiquetas_por_parte, partes, salida):
    fig, ax=plt.subplots(figsize=(10.5,4.8))
    for parte, vals in etiquetas_por_parte.items():
        labs, counts=np.unique(vals,return_counts=True)
        if len(labs) > 30:
            orden=np.argsort(counts)[-30:]; labs, counts=labs[orden],counts[orden]
        ax.scatter([str(int(v)) for v in labs],100*counts/len(vals),s=26,
                   label=parte.capitalize(),alpha=.8)
    ax.set(xlabel='Etiqueta del grupo (-1 = ruido)',ylabel='Registros (%)',
           title='DBSCAN | Tamaño relativo de grupos por conjunto')
    ax.grid(axis='y',alpha=.17);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(salida/'06_tamanos_grupos.png',dpi=220,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.7))
    for parte, labels in etiquetas_por_parte.items():
        smp=partes[parte][SMP].to_numpy(dtype=int)
        # Por bloques: todas las observaciones entran en el promedio de cada bloque.
        n_bloques=min(80,max(2,len(smp)//300))
        ids=np.array_split(np.arange(len(smp)),n_bloques)
        xs=[int(np.median(smp[g])) for g in ids]
        ys=[100*np.mean(np.asarray(labels)[g]==-1) for g in ids]
        ax.plot(xs,ys,lw=1.4,label=parte.capitalize())
    ax.set(xlabel='SMP continuo (contador, no segundos)',
           ylabel='Ruido (%) en bloque de observaciones',
           title='DBSCAN | Evolucion del porcentaje de ruido por SMP')
    ax.grid(alpha=.16);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(salida/'07_evolucion_ruido_smp.png',dpi=220,bbox_inches='tight');plt.close(fig)


def pca_dos_dimensiones(partes, variables):
    """PCA de visualizacion, solo aprendido del entrenamiento COMPLETO."""
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(partes['entrenamiento'][variables].to_numpy(float))
    return pca


def dibujar_clusters_pca(pca, X, etiquetas, nombre, destino, zoom=False, nota=''):
    """Todos los puntos de cada particion, proyectados pero NO entrenados con PCA."""
    X2=pca.transform(X)
    etiquetas=np.asarray(etiquetas,dtype=np.int32)
    if len(X2)!=len(etiquetas):raise ValueError('Longitud de etiquetas distinta de datos')
    fig,ax=plt.subplots(figsize=(10.7,6.6))
    labs, counts=np.unique(etiquetas,return_counts=True)
    pal=plt.get_cmap('tab20')
    for i,g in enumerate(sorted([int(v) for v in labs if v!=-1])):
        mask=etiquetas==g
        ax.scatter(X2[mask,0],X2[mask,1],s=1.5,c=[pal(i%20)],alpha=.37,
                   linewidths=0,rasterized=True,label=f'Grupo {g}: {mask.sum():,}')
    mask=etiquetas==-1
    if mask.any():
        ax.scatter(X2[mask,0],X2[mask,1],s=2,c='#777b81',alpha=.47,
                   linewidths=0,rasterized=True,label=f'Ruido (-1): {mask.sum():,}')
    v=100*pca.explained_variance_ratio_
    if zoom:
        lo,hi=np.quantile(X2,[.005,.995],axis=0)
        delta=np.maximum(hi-lo,1e-5)
        ax.set_xlim(lo[0]-.08*delta[0],hi[0]+.08*delta[0]);ax.set_ylim(lo[1]-.08*delta[1],hi[1]+.08*delta[1])
    ax.set(xlabel=f'PC1 ({v[0]:.1f}% de varianza)',
           ylabel=f'PC2 ({v[1]:.1f}% de varianza)',
           title=f'DBSCAN | Grupos y ruido en {nombre} ({"zoom central" if zoom else "vista completa"})')
    ax.grid(alpha=.14)
    if len(labs)<=12:ax.legend(fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(1.01,1))
    else:
        ax.legend(fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    fig.text(.03,.005,f'{len(X):,} registros representados; clusters definidos en 10D, no en 2D. '
             f'Ruido no implica falla. {nota}',fontsize=8,color='#555')
    fig.savefig(destino,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def dibujar_pca_tres_partes(partes, variables, etiquetas_por_parte, salida, nota=''):
    pca=pca_dos_dimensiones(partes,variables)
    for parte in PARTES:
        X=partes[parte][variables].to_numpy(float)
        for zoom in (False,True):
            dibujar_clusters_pca(pca,X,etiquetas_por_parte[parte],parte,
                salida/f'03_grupos_{parte}_{"zoom" if zoom else "completo"}.png',zoom,
                nota=('Asignacion aproximada a nucleos fijos. '+nota if parte!='entrenamiento' else nota))
    guardar(pd.DataFrame({'componente':['PC1','PC2'],
                          'varianza_explicada_pct':100*pca.explained_variance_ratio_}),
            salida/'03_varianza_pca.csv')
    return pca


def diagnostico(partes, variables, salida, nombres_metricas, min_samples_lista, cuantiles, tope):
    X = partes['entrenamiento'][variables].to_numpy(float)
    arbol = cKDTree(X)
    filas, dist_tablas = [], []
    visualizables = {}  # Una configuracion ilustrativa por metrica y min_samples.
    for metrica in nombres_metricas:
        p = METRICAS[metrica]
        for minimo in min_samples_lista:
            if minimo > len(X):
                raise ValueError('min_samples no puede exceder al entrenamiento')
            print(f'Distancias completas: {metrica}, min_samples={minimo}, N={len(X):,}', flush=True)
            kdist = calcular_distancia_k(X, arbol, minimo, p)
            graficar_curva_k(kdist,metrica,minimo,salida,cuantiles)
            for q in cuantiles:
                eps = float(np.quantile(kdist, q))
                if eps <= 0:
                    continue
                fila = {'metrica':metrica, 'min_samples':minimo,
                        'cuantil_distancia_k':q, 'eps':eps,
                        'n_entrenamiento':len(X), 'ajuste_sobre_total':True,
                        'estado':'OK', 'observacion':''}
                try:
                    y, idx, _, _ = dbscan_completo(X, eps, minimo, p, arbol, kdist, tope)
                    fila.update(resumen_labels(y, len(idx)))
                    fila['frontera_registros']=int(len(X)-len(idx)-fila['ruido_registros'])
                    # Guardar SOLO el candidato con cuantil mas cercano a 0.55.
                    # No se usa para elegir modelo final ni contamina prueba.
                    llave = (metrica, minimo)
                    if (llave not in visualizables or
                            abs(q-.55) < visualizables[llave][0]):
                        visualizables[llave] = (abs(q-.55), eps, q, y.copy())
                    print(f'  eps={eps:.5f} | grupos={fila["clusters"]} | ruido={fila["ruido_pct"]:.2f}% ', flush=True)
                except RuntimeError as e:
                    fila['estado']='OMITIDO_POR_COSTO'; fila['observacion']=str(e)
                    fila.update({c: np.nan for c in ('clusters','ruido_registros','ruido_pct',
                                                          'nucleos','mayor_grupo_pct','grupo_mas_pequeno')})
                    print(f'  eps={eps:.5f} | OMITIDO: {e}', flush=True)
                filas.append(fila)
            # Distribucion completa de las distancias (sin muestrear registros).
            dist_tablas.append({'metrica':metrica,'min_samples':minimo,
                                'p05':float(np.quantile(kdist,.05)),
                                'p25':float(np.quantile(kdist,.25)),
                                'p50':float(np.quantile(kdist,.50)),
                                'p75':float(np.quantile(kdist,.75)),
                                'p90':float(np.quantile(kdist,.90)),
                                'p95':float(np.quantile(kdist,.95)),
                                'n_registros':len(X)})
    tabla = pd.DataFrame(filas)
    guardar(tabla, salida / '00_busqueda_eps.csv')
    guardar(dist_tablas, salida / '00_distribucion_distancias_k.csv')
    graficar_barrido(tabla, salida)
    if visualizables:
        pca = pca_dos_dimensiones(partes, variables)
        cuadros = []
        for (metrica, minimo), (_, eps, cuantil, etiquetas) in visualizables.items():
            nombre_fig=f'03_barrido_grupos_{metrica}_ms{minimo}_completo.png'
            X=partes['entrenamiento'][variables].to_numpy(float)
            for z in (False,True):
                dibujar_clusters_pca(pca,X,etiquetas,'entrenamiento (barrido)',
                    salida/f'03_barrido_grupos_{metrica}_ms{minimo}_{"zoom" if z else "completo"}.png',z,
                    f'Metrica={metrica}, min_samples={minimo}, eps={eps:.4g}; solo ilustrativo')
            cuadros.append({'metrica':metrica,'min_samples':minimo,
                    'cuantil':cuantil,'eps':eps,'archivo':nombre_fig,
                    'estado':'ILUSTRATIVO, NO SELECCIONADO'})
        guardar(cuadros,salida/'03_dbscan_pca_barrido_descripcion.csv')
    (salida / '00_lectura_resultados.txt').write_text(
        'BARRIDO SOBRE TODO EL ENTRENAMIENTO (100 % de sus filas).\n'
        'Un grupo, muchos grupos o mucho ruido no son fallas de codigo: evaluar utilidad y estabilidad.\n'
        'No existe un eps optimo sin un criterio del problema.\n'
        'No se ha elegido aun una configuracion definitiva.\n'
        'Validacion y prueba quedan reservadas para la configuracion que se elija.\n'
        'OMITIDO_POR_COSTO significa que NO se calculo esa configuracion.\n',encoding='utf-8')
    print(f'\nBarrido guardado: {salida / "00_busqueda_eps.csv"}', flush=True)


def graficar_barrido(tabla, salida):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    ok = tabla[tabla.estado == 'OK']
    if ok.empty:
        return
    for atributo, ylabel, titulo, fname in [
        ('ruido_pct','Registros clasificados como ruido (%)','DBSCAN | Sensibilidad del ruido a eps','01_sensibilidad_eps_ruido.png'),
        ('clusters','Cantidad de grupos','DBSCAN | Grupos al variar eps','02_sensibilidad_eps_clusters.png'),
        ('nucleos','Cantidad de observaciones nucleares','DBSCAN | Núcleos al variar eps','02_sensibilidad_eps_nucleos.png')]:
        fig, ax = plt.subplots(figsize=(9.6,5.1))
        for (m, k), t in ok.groupby(['metrica','min_samples']):
            t = t.sort_values('eps')
            ax.plot(t.eps, t[atributo], marker='o', label=f'{m} | min_samples={k}')
        ax.set(xlabel='Radio eps (unidades de la metrica; no comparable directamente entre metricas)',
               ylabel=ylabel, title=titulo)
        ax.grid(alpha=.2); ax.legend(fontsize=8,frameon=False)
        fig.tight_layout();fig.savefig(salida/fname,dpi=240,bbox_inches='tight');plt.close(fig)


def prediccion_core(X, cores, labels_core, eps, p):
    if len(cores) == 0:
        return np.full(len(X), -1, dtype=np.int32), np.full(len(X), np.inf)
    core_tree = cKDTree(cores)
    dist, ids = core_tree.query(X, k=1, p=p, workers=2)
    etiquetas = np.where(dist <= eps + 1e-12, labels_core[ids], -1).astype(np.int32)
    return etiquetas, dist


def episodios(df):
    x = df[SMP].to_numpy(np.int64)
    b = df['alerta_ruido_dbscan'].to_numpy(bool)
    ini = np.flatnonzero(b & np.r_[True, ~b[:-1] | (np.diff(x)!=1)])
    fin = np.flatnonzero(b & np.r_[~b[1:] | (np.diff(x)!=1), True])
    return pd.DataFrame({'episodio':np.arange(1,len(ini)+1),
                         'smp_inicial':x[ini], 'smp_final':x[fin],
                         'longitud_muestras':fin-ini+1})


def final(partes, variables, salida, metrica, min_samples, eps, tope):
    Xtr = partes['entrenamiento'][variables].to_numpy(float)
    print(f'Ajustando DBSCAN final con TODAS las {len(Xtr):,} filas de entrenamiento',flush=True)
    p = METRICAS[metrica]
    arbol = cKDTree(Xtr)
    dist_k = calcular_distancia_k(Xtr, arbol, min_samples, p)
    labtr, indices_core, labels_core, dtr = dbscan_completo(Xtr,eps,min_samples,p,arbol,dist_k,tope)
    cores = Xtr[indices_core]
    resumen, dist = [], []
    etiquetas_por_parte = {}
    for nombre in PARTES:
        datos = partes[nombre]
        X = datos[variables].to_numpy(float)
        if nombre == 'entrenamiento':
            labs, distances = labtr, dtr
            metodo = 'DBSCAN sobre el entrenamiento completo'
        else:
            labs, distances = prediccion_core(X, cores, labels_core, eps, p)
            metodo = 'Asignacion a nucleo fijo mas proximo (fuera de muestra)'
        res = pd.DataFrame({SMP:datos[SMP].to_numpy(np.int64),
                            'cluster_dbscan':labs,
                            'distancia_nucleo_mas_cercano':distances,
                            'alerta_ruido_dbscan':(labs == -1).astype(np.int8)})
        guardar(res, salida/f'01_resultado_{nombre}.csv')
        etiquetas_por_parte[nombre] = labs
        guardar(episodios(res), salida/f'02_episodios_{nombre}.csv')
        info = resumen_labels(labs,len(indices_core) if nombre == 'entrenamiento' else 0)
        info.update({'conjunto':nombre,'registros':len(datos),'metrica':metrica,
                     'min_samples':min_samples,'eps':eps,'metodo':metodo})
        resumen.append(info)
        for label,n in zip(*np.unique(labs,return_counts=True)):
            dist.append({'conjunto':nombre,'cluster':int(label), 'registros':int(n),
                         'porcentaje':100*int(n)/len(labs)})
    guardar(resumen,salida/'00_resumen_dbscan.csv')
    pca=dibujar_pca_tres_partes(partes, variables, etiquetas_por_parte, salida,
                              f'Metrica={metrica}; min_samples={min_samples}; eps={eps:.6g}.')
    graficar_categorias_pca(pca,Xtr,labtr,indices_core,salida,'entrenamiento')
    graficar_tamanos_y_evolucion(etiquetas_por_parte,partes,salida)
    graficar_curva_k(dist_k,metrica,min_samples,salida,[.10,.25,.35],eps_referencia=eps)
    guardar(dist,salida/'00_distribucion_clusters_dbscan.csv')
    joblib.dump({'cores':cores,'labels_core':labels_core,'eps':eps,'metrica':metrica,
                 'min_samples':min_samples,'variables':variables},
                salida/'00_modelo_dbscan_completo.joblib',compress=3)
    conf = {'algoritmo':'DBSCAN','parametros':{'eps':eps,'min_samples':min_samples,'metrica':metrica},
            'registros_entrenamiento':len(Xtr),'registros_validacion':len(partes['validacion']),
            'registros_prueba':len(partes['prueba']),
            'uso_datos':'Entrenamiento completo; aplicacion sin reentrenar a validacion y prueba',
            'limitaciones':['El ruido no es una falla confirmada',
                            'La asignacion fuera de muestra al nucleo fijo NO equivale a ejecutar DBSCAN sobre el nuevo lote',
                            'Los puntos frontera proximos a varios grupos pueden recibir otra etiqueta valida',
                            'El objetivo no dispone de etiquetas para precision/recall',
                            'El uso del conjunto de prueba para ajustar eps produciria fuga de informacion']}
    (salida/'00_configuracion_dbscan.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(resumen).to_string(index=False),flush=True)
    print('Modelo y resultados finales en:',salida,flush=True)


def autotest():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    X, _ = make_blobs(n_samples=240,centers=3,random_state=42,cluster_std=.25)
    for p, nombre in [(2,'euclidiana'),(1,'manhattan'),(np.inf,'chebyshev')]:
        eps=0.35 if p!=1 else 0.48
        y, core, _, _ = dbscan_completo(X,eps,5,p,max_vecinos=999999)
        otro = DBSCAN(eps=eps,min_samples=5,metric={'euclidiana':'euclidean',
                       'manhattan':'manhattan','chebyshev':'chebyshev'}[nombre]).fit_predict(X)
        assert np.array_equal(y == -1, otro == -1), 'No coinciden las clasificaciones de ruido'
        assert adjusted_rand_score(y,otro) > .999, 'No coinciden agrupaciones'
    print('AUTOTEST correcto: DBSCAN equivalente a sklearn en datos de control')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modo',choices=['barrido','final'],default='barrido')
    parser.add_argument('--escalado',choices=['estandar','robusto'],default='estandar',
                        help='Usar siempre las diez señales juntas')
    parser.add_argument('--preparados',type=Path)
    parser.add_argument('--salida',type=Path)
    parser.add_argument('--metricas',default='euclidiana',
                        help='Metricas en modo barrido (separadas por coma)')
    parser.add_argument('--min-samples',default='20',
                        help='Enteros separados por coma (barrido) o uno solo (final)')
    parser.add_argument('--cuantiles',default='0.10,0.25,0.35',
                        help='Cuantiles de la distancia k-NN del entrenamiento (barrido)')
    parser.add_argument('--metrica',choices=METRICAS,default='euclidiana',help='Solo modo final')
    parser.add_argument('--eps',type=float,help='Obligatorio en modo final: elegido del diagnostico')
    parser.add_argument('--tope-vecinos',type=int,default=600,
                        help='Proteccion si 90%% de vecinos son demasiados; informa omision')
    parser.add_argument('--autotest',action='store_true')
    a=parser.parse_args()
    if a.autotest:
        autotest(); return
    raiz=Path(__file__).resolve().parent
    prep=a.preparados or raiz/'Resultados_Tarea3_Nuevos'/'00_preparacion'
    out=a.salida or raiz/'Resultados_Tarea3_Nuevos'/'03_dbscan'
    minimos=[int(v.strip()) for v in a.min_samples.split(',')]
    if not minimos or any(v<2 for v in minimos):
        raise ValueError('min_samples debe ser al menos 2')
    partes, variables, meta=leer_datos(prep,a.escalado)
    out.mkdir(parents=True,exist_ok=True)
    (out/'00_contexto_datos.json').write_text(json.dumps({
         'origen_08':str(prep.resolve()),'sha256_origen':meta.get('sha256_origen'),
         'escalado':a.escalado,'variables':meta.get('variables'),
         'filas_por_conjunto':{p:len(partes[p]) for p in PARTES},
         'total_registros':sum(len(partes[p]) for p in PARTES),
         'sin_modelos_por_sensor':True},ensure_ascii=False,indent=2),encoding='utf-8')
    if a.modo=='barrido':
        metricas=[s.strip() for s in a.metricas.split(',')]
        if any(m not in METRICAS for m in metricas):
            raise ValueError('Metrica desconocida. Usa euclidiana, manhattan o chebyshev')
        cuantiles=[float(q) for q in a.cuantiles.split(',')]
        if any(not 0 < q < 1 for q in cuantiles):
            raise ValueError('Cuantiles deben estar entre 0 y 1')
        diagnostico(partes,variables,out,metricas,minimos,cuantiles,a.tope_vecinos)
    else:
        if a.eps is None or a.eps<=0:
            parser.error('El modo final requiere --eps POSITIVO luego de revisar el barrido')
        if len(minimos)!=1:
            parser.error('El modo final requiere UN solo --min-samples')
        final(partes,variables,out,a.metrica,minimos[0],a.eps,a.tope_vecinos)

if __name__=='__main__':
    main()
