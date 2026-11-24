
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import hdbscan
    from hdbscan.prediction import approximate_predict
except (ImportError, OSError) as exc:
    raise SystemExit(
        "No puedo importar la biblioteca 'hdbscan'. Instalala en el entorno ACTIVO de VS Code: "
        "python -m pip install hdbscan\nDetalle: " + str(exc)
    ) from exc

RAIZ = Path(__file__).resolve().parent
PARTES = ('entrenamiento', 'validacion', 'prueba')
SMP = 'smp_continuo'
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
SEMILLA = 42


def exportar(tabla, ruta: Path):
    pd.DataFrame(tabla).to_csv(ruta, sep=';', encoding='utf-8-sig', index=False, float_format='%.10g')


def figura(fig, ruta: Path):
    fig.savefig(ruta, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  +', ruta.name, flush=True)


def leer_08(preparados: Path, escalado: str):
    ruta_meta = preparados / '00_metadatos.json'
    if not ruta_meta.is_file():
        raise FileNotFoundError(f'No se encuentra {ruta_meta}. Ejecuta primero 08_preparacion_datos.py')
    meta = json.loads(ruta_meta.read_text(encoding='utf-8'))
    nombres = meta.get('variables', [])
    if len(nombres) != 10 or len(set(nombres)) != 10:
        raise ValueError('El 08 debe definir 10 variables de sensores distintas.')
    division = pd.read_csv(preparados / '00_division_cronologica.csv', sep=';', encoding='utf-8-sig',
                           usecols=[SMP, 'conjunto'])
    if division[SMP].duplicated().any() or not division[SMP].is_monotonic_increasing:
        raise ValueError('SMP duplicados o sin orden en la division del 08.')
    datos = {}
    for parte in PARTES:
        ruta = preparados / f'01_{parte}_{escalado}.csv'
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {ruta}; ejecuta 08_preparacion_datos.py')
        d = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
        esperadas = [SMP] + [f'z_{c}' for c in nombres]
        if list(d.columns) != esperadas:
            raise ValueError(f'Columnas distintas a las 10 señales esperadas en {ruta.name}')
        s = d[SMP].to_numpy(np.int64)
        referencia = division.loc[division.conjunto == parte, SMP].to_numpy(np.int64)
        if not np.array_equal(s, referencia):
            raise ValueError(f'No coincide la division cronologica del 08 con {ruta.name}')
        X = d.iloc[:, 1:].to_numpy(dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError(f'Hay datos no finitos en {ruta.name}')
        datos[parte] = (s, X)
    total = sum(len(datos[p][0]) for p in PARTES)
    if total != meta.get('registros') or total != 143561:
        raise ValueError(f'Base inesperada: {total} registros; se requieren los 143561 acordados')
    if not (datos['entrenamiento'][0][-1] < datos['validacion'][0][0] < datos['prueba'][0][0]):
        raise ValueError('Los conjuntos no estan en orden cronologico')
    return datos, meta


def entrenar(X, min_cluster_size: int, min_samples: int, metodo: str):
    if min_samples < 1 or min_cluster_size < 2:
        raise ValueError('min_samples >= 1 y min_cluster_size >= 2')
    # hdbscan-contrib: min_samples no incluye al punto mismo; no confundir
    # numericamente con la convencion del HDBSCAN de scikit-learn.
    modelo = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric='euclidean', cluster_selection_method=metodo,
        prediction_data=True, approx_min_span_tree=True,
        core_dist_n_jobs=1, algorithm='best',
    )
    modelo.fit(X)  # TODAS las filas de entrenamiento; ninguna muestra omitida
    return modelo


def evaluar_partes(datos, modelo):
    salida = {}
    for p in PARTES:
        smp, X = datos[p]
        if p == 'entrenamiento':
            etiquetas = np.asarray(modelo.labels_, dtype=np.int32)
            fuerza = np.asarray(modelo.probabilities_, dtype=float)
            out = np.asarray(modelo.outlier_scores_, dtype=float)
            out[~np.isfinite(out)] = np.nan
        else:
            # No se pueden asignar observaciones a grupos inexistentes.
            # Etiquetarlas como ruido es una conclusion negativa, NO crear grupos.
            if len(set(modelo.labels_) - {-1}) == 0:
                etiquetas = np.full(len(X), -1, dtype=np.int32)
                fuerza = np.zeros(len(X), dtype=float)
            else:
                # Asignacion aproximada a clusters preexistentes, SIN reentrenar.
                etiquetas, fuerza = approximate_predict(modelo, X)
                etiquetas = np.asarray(etiquetas, dtype=np.int32)
                fuerza = np.asarray(fuerza, dtype=float)
            out = np.full(len(X), np.nan)
        if len(etiquetas) != len(smp) or len(fuerza) != len(smp):
            raise RuntimeError(f'Longitud incorrecta de predicciones en {p}')
        salida[p] = pd.DataFrame({SMP: smp, 'cluster_hdbscan': etiquetas,
                                   'ruido_hdbscan': (etiquetas == -1).astype(np.int8),
                                   'fuerza_pertenencia': fuerza,
                                   'puntaje_glosh_solo_entrenamiento': out})
    return salida


def calcular_resumen(tablas, modelo):
    r = []
    for p in PARTES:
        t = tablas[p]
        l = t['cluster_hdbscan'].to_numpy()
        r.append({'conjunto': p, 'registros': len(l), 'grupos_presentes': len(set(l) - {-1}),
                  'ruido': int(np.count_nonzero(l == -1)), 'ruido_pct': 100 * np.mean(l == -1),
                  'fuerza_media_sin_ruido': float(t.loc[t.cluster_hdbscan != -1,
                                                       'fuerza_pertenencia'].mean()) if (l != -1).any() else np.nan,
                  'metodo_asignacion': 'fit_hdbscan' if p == 'entrenamiento' else 'approximate_predict'})
    pers = np.asarray(modelo.cluster_persistence_)
    persistencia = pd.DataFrame({'cluster_hdbscan': np.arange(len(pers)), 'persistencia': pers})
    grupos = []
    for p, t in tablas.items():
        vc = t['cluster_hdbscan'].value_counts().sort_index()
        for k, n in vc.items():
            grupos.append({'conjunto': p, 'cluster_hdbscan': int(k), 'registros': int(n),
                           'porcentaje': 100 * n / len(t)})
    return pd.DataFrame(r), persistencia, pd.DataFrame(grupos)


def paleta_etiquetas(modelo):
    claves = sorted(int(c) for c in np.unique(modelo.labels_) if c != -1)
    colores = plt.get_cmap('turbo')
    return {c: colores((i+0.5)/max(1, len(claves))) for i,c in enumerate(claves)}


def limites_pca(P):
    # Limite COMPLETO comun calculado sobre las tres partes; no se eliminan puntos.
    todo = np.concatenate([P[p] for p in PARTES])
    a = todo.min(axis=0); b = todo.max(axis=0)
    margen = np.maximum((b-a)*0.06, 0.05)
    completo = [(float(a[j]-margen[j]),float(b[j]+margen[j])) for j in range(2)]
    # Zoom compartido definido con TRAIN; recorta SOLO la vista, nunca el ajuste.
    x = P['entrenamiento']
    a = np.quantile(x,[.01,.99],axis=0)
    margen = np.maximum((a[1]-a[0])*.11,.05)
    zoom = [(float(a[0,j]-margen[j]),float(a[1,j]+margen[j])) for j in range(2)]
    return completo, zoom


def scatter_pca(x, etiquetas, nombre, limites, tipo, destino, colores, varianza):
    fig, ax = plt.subplots(figsize=(11.8,6.3))
    etiquetas = np.asarray(etiquetas)
    ruido = etiquetas == -1
    if ruido.any():
        ax.scatter(x[ruido,0],x[ruido,1],s=1.65,alpha=.28,color='#9299a2',
                   linewidths=0,rasterized=True,zorder=1)
    ids = sorted(k for k in np.unique(etiquetas) if k != -1)
    for k in ids:
        m = etiquetas == k
        ax.scatter(x[m,0], x[m,1], s=1.5, alpha=.45,
                   c=[colores.get(k,'#327db2')], linewidths=0,rasterized=True,zorder=2)
    ax.set_xlim(*limites[0]); ax.set_ylim(*limites[1])
    ax.set_xlabel(f'PC1 ({varianza[0]*100:.1f}% de variabilidad)')
    ax.set_ylabel(f'PC2 ({varianza[1]*100:.1f}% de variabilidad)')
    ax.set_title(f'HDBSCAN | {nombre} | 10 señales juntas | {tipo}')
    ax.grid(alpha=.12)
    tot = len(etiquetas)
    top = sorted(ids,key=lambda k: -np.count_nonzero(etiquetas == k))[:10]
    manejadores=[]
    if ruido.any():
        manejadores.append(Line2D([],[],linestyle='',marker='o',markersize=6,
                                  color='#9299a2',label=f'Ruido: {ruido.sum():,} ({ruido.mean()*100:.1f}%)'))
    for k in top:
        n=int(np.count_nonzero(etiquetas == k))
        manejadores.append(Line2D([],[],linestyle='',marker='o',markersize=6,
                                  color=colores.get(k,'#327db2'),label=f'Grupo {k}: {n:,} ({100*n/tot:.1f}%)'))
    if len(ids)>len(top):
        manejadores.append(Line2D([],[],linestyle='',color='none',label=f'+ {len(ids)-len(top)} grupos'))
    if manejadores:
        ax.legend(handles=manejadores,loc='upper left',bbox_to_anchor=(1.005,1),
                  frameon=False,fontsize=8,title='Etiquetas (no son fallas)')
    visibles=((x[:,0]>=limites[0][0])&(x[:,0]<=limites[0][1])&
              (x[:,1]>=limites[1][0])&(x[:,1]<=limites[1][1])).sum()
    fig.text(.03,.005, f'{tot:,} registros utilizados. {visibles:,} visibles en {tipo}.'
             ' El PCA es SOLO visualizacion; agrupamiento en 10 dimensiones. '
             'Las etiquetas -1 no prueban fallas.', fontsize=8,color='#555555')
    figura(fig,destino)


def grafico_arbol(modelo, salida):
    fig, ax=plt.subplots(figsize=(13,6.5))
    try:
        modelo.condensed_tree_.plot(axis=ax, select_clusters=True,
                                    label_clusters=False, colorbar=True)
        ax.set_title('HDBSCAN | Arbol condensado de densidades (entrenamiento)')
        figura(fig,salida/'02_arbol_condensado.png')
    except Exception as exc:
        plt.close(fig)
        (salida/'02_arbol_condensado_error.txt').write_text(
            f'El arbol condensado no pudo representarse; no se invento figura.\n{type(exc).__name__}: {exc}',encoding='utf-8')
        print('  AVISO: no se pudo dibujar arbol condensado:',repr(exc),flush=True)


def grafico_persistencia(pers, salida):
    fig,ax=plt.subplots(figsize=(10.5,4.6))
    if pers.empty:
        ax.text(.5,.5,'No se detectaron grupos persistentes',ha='center',va='center',transform=ax.transAxes)
    else:
        sel=pers.nlargest(35,'persistencia').sort_values('persistencia')
        ax.barh(sel.cluster_hdbscan.astype(str),sel.persistencia,color='#3e83a6')
        ax.set_xlabel('Persistencia relativa (0-1)')
        ax.set_xlim(0,max(1.0,float(sel.persistencia.max())*1.1))
        if len(pers)>len(sel):
            ax.text(.99,.01,f'Se muestran los {len(sel)} de {len(pers)} grupos mas persistentes',
                    transform=ax.transAxes,ha='right',fontsize=8)
    ax.set_ylabel('Grupo')
    ax.set_title('HDBSCAN | Estabilidad de los grupos (entrenamiento)')
    figura(fig,salida/'03_persistencia_grupos.png')


def grafico_fuerzas(tablas, salida):
    fig, ax=plt.subplots(figsize=(9.5,4.9))
    for p,c in zip(PARTES,['#327ca5','#d18b42','#8758a8']):
        v=tablas[p].loc[tablas[p].cluster_hdbscan!=-1,'fuerza_pertenencia'].to_numpy()
        if len(v):
            ax.hist(v,bins=np.linspace(0,1,41),density=True,histtype='step',linewidth=1.65,
                    label=f'{p.capitalize()} (n={len(v):,})',color=c)
    ax.set(xlabel='Fuerza de pertenencia a un grupo (0-1)',ylabel='Densidad',
           title='HDBSCAN | Fuerza de pertenencia de observaciones NO ruido')
    if any((tablas[p].cluster_hdbscan != -1).any() for p in PARTES):
        ax.legend(frameon=False)
    else:
        ax.text(.5,.5,'Sin grupos: todas las observaciones son ruido',
                transform=ax.transAxes,ha='center',va='center')
    ax.grid(axis='y',alpha=.15)
    figura(fig,salida/'04_fuerza_pertenencia.png')


def grafico_tamanos(grupos, salida):
    fig,ax=plt.subplots(figsize=(12,5.5))
    d=grupos.pivot(index='cluster_hdbscan',columns='conjunto',values='registros').fillna(0)
    d=d.reindex(columns=list(PARTES),fill_value=0)
    sel=d.sum(axis=1).sort_values(ascending=False).head(24).index
    for p,c,o in zip(PARTES,['#327ca5','#d18b42','#8758a8'],[-.25,0,.25]):
        ax.bar(np.arange(len(sel))+o,d.loc[sel,p].to_numpy(),width=.24,label=p.capitalize(),color=c)
    ax.set_xticks(np.arange(len(sel)),[('Ruido' if i==-1 else str(int(i))) for i in sel],rotation=65)
    ax.set(xlabel='Etiqueta de grupo (los 24 mas frecuentes)',ylabel='Registros',
           title='HDBSCAN | Distribucion de etiquetas entre conjuntos')
    ax.legend(frameon=False)
    figura(fig,salida/'05_tamanos_grupos.png')


def grafico_evolucion(tablas, colores, salida):
    fig,axs=plt.subplots(3,1,figsize=(13,8.4),sharex=True,constrained_layout=True)
    for ax,p in zip(axs,PARTES):
        t=tablas[p]
        x=t[SMP].to_numpy()
        labels=t.cluster_hdbscan.to_numpy()
        # TODOS los registros representados (nube de puntos rasterizada).
        ruido=labels==-1
        ax.scatter(x[ruido],labels[ruido],s=.9,color='#8e969d',alpha=.28,rasterized=True)
        for k in np.unique(labels[~ruido]):
            sel=labels==k
            ax.scatter(x[sel],labels[sel],s=.9,color=colores.get(k,'#327db2'),
                       alpha=.50,rasterized=True)
        ax.set_ylabel(p.capitalize()+'\nEtiqueta')
        ax.grid(axis='y',alpha=.1)
    axs[-1].set_xlabel('SMP continuo (contador; no segundos)')
    fig.suptitle('HDBSCAN | Evolucion de etiquetas por SMP (los 10 sensores juntos)')
    figura(fig,salida/'06_evolucion_grupos_smp.png')


def grafico_perfiles(datos, tablas, salida):
    X=datos['entrenamiento'][1]
    labs=tablas['entrenamiento'].cluster_hdbscan.to_numpy()
    ids=sorted((int(k) for k in set(labs)-{-1}),key=lambda k: -(labs==k).sum())[:16]
    fig,ax=plt.subplots(figsize=(10.7,max(3.5,0.32*len(ids)+2.1)))
    if not ids:
        ax.text(.5,.5,'Sin grupos; no hay perfiles que comparar',ha='center',va='center',transform=ax.transAxes)
        ax.axis('off')
    else:
        centro=np.vstack([np.median(X[labs==k],axis=0) for k in ids])
        lim=max(.01,float(np.nanquantile(np.abs(centro),.99)))
        im=ax.imshow(np.clip(centro,-lim,lim),aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
        ax.set_xticks(range(10),CORTOS,rotation=30)
        ax.set_yticks(range(len(ids)),[f'Grupo {k} (n={(labs==k).sum():,})' for k in ids])
        fig.colorbar(im,ax=ax,label='Mediana de señal escalada (en el conjunto de entrenamiento)')
    ax.set_title('HDBSCAN | Perfil descriptivo de grupos (todas las senales, no modelos separados)')
    figura(fig,salida/'07_perfil_grupos.png')


def grafico_glosh(modelo, salida):
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    x=np.asarray(modelo.outlier_scores_,float)
    x=x[np.isfinite(x)]
    if len(x):
        ax.hist(x,bins=65,color='#527fa2',edgecolor='white',linewidth=.25)
        ax.set(xlabel='Puntaje GLOSH (solo datos de entrenamiento)',ylabel='Registros')
    else:
        ax.text(.5,.5,'No hay puntuaciones GLOSH finitas',ha='center',va='center',transform=ax.transAxes)
    ax.set_title('HDBSCAN | Puntajes de atipicidad del entrenamiento')
    figura(fig,salida/'08_puntajes_glosh.png')


def guardar_graficos(datos, tablas, modelo, pers, grupos, salida):
    Pca=PCA(n_components=2,random_state=SEMILLA)
    Pca.fit(datos['entrenamiento'][1])
    proyecciones={p:Pca.transform(datos[p][1]) for p in PARTES}
    completo, zoom=limites_pca(proyecciones)
    colores=paleta_etiquetas(modelo)
    for p in PARTES:
        l=tablas[p].cluster_hdbscan.to_numpy()
        for tipo, limites in [('completo',completo),('zoom',zoom)]:
            scatter_pca(proyecciones[p],l,p,limites,tipo,
                        salida/f'01_grupos_{p}_{tipo}.png',colores,
                        Pca.explained_variance_ratio_)
    exportar(pd.DataFrame({'componente':['PC1','PC2'],
                           'fraccion_varianza':Pca.explained_variance_ratio_}),
             salida/'01_pca_varianza_explicada.csv')
    grafico_arbol(modelo,salida)
    grafico_persistencia(pers,salida)
    grafico_fuerzas(tablas,salida)
    grafico_tamanos(grupos,salida)
    grafico_evolucion(tablas,colores,salida)
    grafico_perfiles(datos,tablas,salida)
    grafico_glosh(modelo,salida)


def silueta_solo_descriptiva(X, etiquetas, max_n=1500):
    # Nunca usada para declarar fallas ni para forzar agrupacion. Ruido excluido.
    ok=np.asarray(etiquetas)!=-1
    a=X[ok]; l=np.asarray(etiquetas)[ok]
    if len(np.unique(l))<2 or len(a)<4:
        return np.nan,0
    rng=np.random.default_rng(SEMILLA)
    # Muestra estratificada que preserve algunos elementos de grupos pequenos.
    inds=[]
    grupos=np.unique(l)
    if len(grupos)>=max_n:
        return np.nan,0
    # Dos ejemplos por grupo cuando existen, luego completar de forma aleatoria.
    for k in grupos:
        t=np.flatnonzero(l==k)
        inds.extend(rng.choice(t,size=min(len(t),2),replace=False).tolist())
    inds=np.unique(inds)
    n_extra=min(max_n-len(inds),len(a)-len(inds))
    if n_extra>0:
        resto=np.setdiff1d(np.arange(len(a)),inds,assume_unique=True)
        inds=np.r_[inds,rng.choice(resto,size=n_extra,replace=False)]
    lb=l[inds]
    if len(np.unique(lb))<2 or len(np.unique(lb))>=len(lb):
        return np.nan,len(inds)
    try:
        return float(silhouette_score(a[inds],lb)), len(inds)
    except ValueError:
        return np.nan,len(inds)


def resumen_modelo(modelo,X_train,ms,mcs,metodo,escalado,tiempo):
    labels=np.asarray(modelo.labels_)
    n_clusters=len(set(labels)-{-1})
    return {'escalado':escalado,'min_cluster_size':mcs,'min_samples':ms,
            'cluster_selection_method':metodo,'registros_entrenamiento':len(X_train),
            'grupos':n_clusters,'ruido_pct':100*np.mean(labels==-1),
            'ruido_n':int(np.count_nonzero(labels==-1)),
            'persistencia_media_no_ponderada':float(np.mean(modelo.cluster_persistence_)) if n_clusters else np.nan,
            'tiempo_ajuste_segundos':round(tiempo,3)}


def barrido(datos, escalado, salida, configs, metodo):
    X=datos['entrenamiento'][1]
    resultados=[]
    for i,(mcs,ms) in enumerate(configs,1):
        print(f'[{i}/{len(configs)}] Las 10 senales | mcs={mcs} | ms={ms} | filas={len(X):,}',flush=True)
        t0=time.perf_counter()
        try:
            modelo=entrenar(X,mcs,ms,metodo)
            fila=resumen_modelo(modelo,X,ms,mcs,metodo,escalado,time.perf_counter()-t0)
            fila['estado']='OK'; fila['error']=''
        except (MemoryError, ValueError, RuntimeError) as exc:
            fila={'escalado':escalado,'min_cluster_size':mcs,'min_samples':ms,
                  'cluster_selection_method':metodo,'registros_entrenamiento':len(X),
                  'grupos':np.nan,'ruido_pct':np.nan,'ruido_n':np.nan,
                  'persistencia_media_no_ponderada':np.nan,
                  'tiempo_ajuste_segundos':round(time.perf_counter()-t0,3),
                  'estado':'ERROR','error':str(exc)}
        resultados.append(fila)
        print('  ', fila['estado'], '| grupos:',fila.get('grupos'),'| ruido %:',fila.get('ruido_pct'),flush=True)
        exportar(pd.DataFrame(resultados),salida/'00_barrido_hdbscan.csv')
    d=pd.DataFrame(resultados)
    fig,axs=plt.subplots(2,1,figsize=(10.5,7),sharex=True,constrained_layout=True)
    x=np.arange(len(d))
    axs[0].plot(x,d['grupos'],marker='o',color='#3c80a8')
    axs[1].plot(x,d['ruido_pct'],marker='o',color='#bd644c')
    axs[0].set_ylabel('Cantidad de grupos')
    axs[1].set_ylabel('Ruido (%)')
    axs[1].set_xlabel('Configuraciones (todas las 10 senales)')
    axs[1].set_xticks(x,[f'{r.min_cluster_size}/{r.min_samples}' for r in d.itertuples()],rotation=35)
    axs[0].set_title('HDBSCAN | Sensibilidad a min_cluster_size / min_samples')
    for ax in axs:ax.grid(axis='y',alpha=.16)
    figura(fig,salida/'00_barrido_parametros.png')
    print('No se elige automaticamente una configuracion: revisar tabla y grafico.',flush=True)


def final(datos,meta,escalado,salida,mcs,ms,metodo):
    X=datos['entrenamiento'][1]
    print(f'Ajustando HDBSCAN sobre las 10 senales y TODO entrenamiento ({len(X):,})',flush=True)
    t=time.perf_counter()
    modelo=entrenar(X,mcs,ms,metodo)
    duracion=time.perf_counter()-t
    tablas=evaluar_partes(datos,modelo)
    resumen,persistencia,grupos=calcular_resumen(tablas,modelo)
    for p in PARTES:
        exportar(tablas[p],salida/f'01_resultado_{p}.csv')
    exportar(resumen,salida/'00_resumen_hdbscan.csv')
    exportar(grupos,salida/'00_tamanos_grupos.csv')
    exportar(persistencia,salida/'00_persistencia_grupos.csv')
    sil,n_silueta=silueta_solo_descriptiva(datos['validacion'][1],tablas['validacion'].cluster_hdbscan.to_numpy())
    configuracion={'metodo':'HDBSCAN contrib, 10 variables juntas','escalado':escalado,
                   'min_cluster_size':mcs,'min_samples':ms,'metrica':'euclidiana',
                   'cluster_selection_method':metodo,'filas':{p:len(datos[p][0]) for p in PARTES},
                   'origen_sha256':meta.get('sha256_origen'), 'tiempo_ajuste_segundos':duracion,
                   'prediccion_fuera_muestra':'hdbscan.approximate_predict, NO reentrena jerarquia',
                   'silueta_validacion_no_ruido_muestra':sil,
                   'n_muestra_silueta':n_silueta,
                   'advertencia':'Agrupaciones y ruido NO son fallas verificadas; sin etiquetas no hay precision diagnostica'}
    (salida/'00_configuracion_hdbscan.json').write_text(json.dumps(configuracion,indent=2,ensure_ascii=False),encoding='utf-8')
    joblib.dump(modelo,salida/'00_modelo_hdbscan.joblib')
    guardar_graficos(datos,tablas,modelo,persistencia,grupos,salida)
    print('\nHDBSCAN COMPLETADO',flush=True)
    print(resumen.to_string(index=False),flush=True)
    print(f'Silueta descriptiva en muestra de validacion, sin ruido: {sil} (n={n_silueta})',flush=True)
    print('Resultado:',salida.resolve(),flush=True)
    print('ADVERTENCIA: ruido y grupos no equivalen a averias confirmadas.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modo',choices=['barrido','final'],default='final')
    parser.add_argument('--escalado',choices=['estandar','robusto'],default='estandar')
    parser.add_argument('--min-cluster-size',type=int,default=200)
    parser.add_argument('--min-samples',type=int,default=10)
    parser.add_argument('--seleccion',choices=['eom','leaf'],default='eom')
    parser.add_argument('--configs',default='100:5,200:10,500:20',
                        help='Solo barrido: parejas min_cluster_size:min_samples separadas por coma')
    parser.add_argument('--preparados',type=Path,default=None)
    parser.add_argument('--salida',type=Path,default=None)
    args=parser.parse_args()
    prep=args.preparados or RAIZ/'Resultados_Tarea3_Nuevos'/'00_preparacion'
    salida=args.salida or RAIZ/'Resultados_Tarea3_Nuevos'/'04_hdbscan'
    datos,meta=leer_08(prep,args.escalado)
    salida.mkdir(parents=True,exist_ok=True)
    if args.modo=='barrido':
        try:
            configs=[tuple(map(int,p.split(':'))) for p in args.configs.split(',')]
            if not configs or any(len(c)!=2 or c[0]<2 or c[1]<1 for c in configs):
                raise ValueError()
        except ValueError:
            parser.error('--configs debe tener forma 100:5,200:10,500:20')
        barrido(datos,args.escalado,salida,configs,args.seleccion)
    else:
        final(datos,meta,args.escalado,salida,args.min_cluster_size,args.min_samples,args.seleccion)


if __name__=='__main__':
    main()
