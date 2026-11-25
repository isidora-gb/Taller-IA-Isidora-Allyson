
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent
SMP = 'smp_continuo'
PARTES = ('entrenamiento', 'validacion', 'prueba')
SEED = 42
ABREV = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
COLS = ['id', 'escalado', 'min_cluster_size', 'min_samples', 'seleccion', 'motor',
        'n_entrenamiento', 'grupos', 'ruido_n', 'ruido_pct', 'cobertura_pct',
        'val_ruido_pct', 'val_cobertura_pct', 'val_grupos_presentes',
        'diferencia_cobertura_pp', 'persistencia_media', 'persistencia_ponderada',
        'fuerza_media_agrupados', 'dbcv_aprox_mst', 'silueta_solo_agrupados_muestra',
        'n_silueta_muestra', 'segundos', 'estado', 'error']


def guardarcsv(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(p, sep=';', index=False, encoding='utf-8-sig', float_format='%.10g')


def guardarjson(d, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def finito_o_nulo(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def guardarfig(fig, p):
    fig.savefig(p, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  +', p.name, flush=True)


def formatear_error(e):
    return f'{type(e).__name__}: {str(e)[:500]}'


def leer(preparacion, escalas):
    meta_p = preparacion/'00_metadatos.json'
    if not meta_p.exists():
        raise FileNotFoundError(f'Falta {meta_p}. Primero ejecuta python 08_preparacion_datos.py')
    meta = json.loads(meta_p.read_text(encoding='utf-8'))
    if meta.get('registros') != 143561 or len(meta.get('variables', [])) != 10:
        raise ValueError('Se requieren exactamente 143561 registros y diez señales en los metadatos 08.')
    division = pd.read_csv(preparacion/'00_division_cronologica.csv', sep=';', encoding='utf-8-sig')
    if not division[SMP].is_unique or not division[SMP].is_monotonic_increasing:
        raise ValueError('La division SMP contiene identificadores duplicados o desordenados.')
    datos = {}
    for esc in escalas:
        d = {}
        for parte in PARTES:
            p = preparacion/f'01_{parte}_{esc}.csv'
            if not p.exists():
                raise FileNotFoundError(f'No existe {p}; vuelve a ejecutar el 08.')
            tabla = pd.read_csv(p, sep=';', encoding='utf-8-sig')
            esperadas = [SMP] + [f'z_{x}' for x in meta['variables']]
            if list(tabla.columns) != esperadas:
                raise ValueError(f'Columnas inesperadas en {p.name}.')
            smp = tabla[SMP].to_numpy(dtype=np.int64)
            ref = division.loc[division.conjunto.eq(parte), SMP].to_numpy(dtype=np.int64)
            if not np.array_equal(smp, ref):
                raise ValueError(f'Division SMP no coincide en {p.name}.')
            X = np.ascontiguousarray(tabla.iloc[:, 1:].to_numpy(dtype=np.float64))
            if not np.isfinite(X).all():
                raise ValueError(f'Valores no finitos en {p.name}.')
            d[parte] = (smp, X)
        if [len(d[p][0]) for p in PARTES] != [100492, 21534, 21535]:
            raise ValueError('Tamaños de conjuntos no coinciden con el proyecto.')
        datos[esc] = d
    return meta, datos


def importar_hdbscan():
    try:
        import hdbscan
        from hdbscan.prediction import approximate_predict
        return hdbscan, approximate_predict
    except (ImportError, OSError) as e:
        raise RuntimeError(f'Falta la biblioteca hdbscan en {sys.executable}. Instala: python -m pip install hdbscan. Detalle: {e}') from e


def resumen_ram():
    try:
        import psutil
        return f' RAM Python {psutil.Process().memory_info().rss/2**30:.2f} GiB'
    except Exception:
        return ''


def ajuste(X, cfg, salida, trabajadores, motor, prediccion):
    lib, _ = importar_hdbscan()
    if cfg['min_samples'] >= len(X) or cfg['min_cluster_size'] > len(X):
        raise ValueError('Los parametros superan el numero de registros.')
    cache = salida/'00_cache_calculos'/cfg['escalado']
    cache.mkdir(parents=True, exist_ok=True)
    modelo = lib.HDBSCAN(
        min_cluster_size=cfg['min_cluster_size'], min_samples=cfg['min_samples'],
        cluster_selection_method=cfg['seleccion'], metric='euclidean',
        algorithm=motor, core_dist_n_jobs=trabajadores, leaf_size=60,
        approx_min_span_tree=True, gen_min_span_tree=True,
        prediction_data=prediccion,
        memory=Memory(location=str(cache), verbose=0),
    )
    fin = threading.Event()
    def latido():
        i = 0
        while not fin.wait(20):
            i += 1
            print(f'    [HDBSCAN] proceso activo; {i*20} s transcurridos.{resumen_ram()}', flush=True)
    hilo = threading.Thread(target=latido, daemon=True)
    hilo.start()
    try:
        modelo.fit(X)
    finally:
        fin.set(); hilo.join(timeout=2)
    return modelo


def predecir(modelo, X):
    _, approximate_predict = importar_hdbscan()
    if len(np.unique(np.asarray(modelo.labels_)[np.asarray(modelo.labels_) >= 0])) == 0:
        return np.full(len(X), -1, dtype=np.int32), np.zeros(len(X), dtype=float)
    if getattr(modelo, 'prediction_data_', None) is None:
        modelo.generate_prediction_data()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        l, prob = approximate_predict(modelo, X)
        if w:
            print('    Nota de prediccion aproximada:', str(w[0].message)[:190], flush=True)
    return np.asarray(l, dtype=np.int32), np.asarray(prob, dtype=float)


def silueta_diagnostica(X, labels):
    ok = np.flatnonzero(labels >= 0)
    if len(ok) < 4 or len(np.unique(labels[ok])) < 2:
        return np.nan, 0
    rng = np.random.default_rng(SEED)
    grupos = np.unique(labels[ok])
    elegido = []
    for k in grupos:
        ids = ok[labels[ok] == k]
        elegido.extend(rng.choice(ids, min(2, len(ids)), replace=False))
    elegido = np.asarray(sorted(set(elegido)), dtype=int)
    restantes = np.setdiff1d(ok, elegido, assume_unique=True)
    cantidad = min(1200-len(elegido), len(restantes))
    if cantidad > 0:
        elegido = np.r_[elegido, rng.choice(restantes, cantidad, replace=False)]
    if len(np.unique(labels[elegido])) < 2:
        return np.nan, len(elegido)
    return float(silhouette_score(X[elegido], labels[elegido])), len(elegido)


def identificador(esc, mcs, ms, sel):
    return f'{esc}_mcs{mcs}_ms{ms}_{sel}'


def configs(args):
    if args.perfil == 'inicial':
        mcs = [30, 100]
        ms = [2, 10]
    else:
        mcs = [30, 100, 300]
        ms = [2, 5, 15]
    if args.mcs:
        mcs = [int(x.strip()) for x in args.mcs.split(',')]
    if args.ms:
        ms = [int(x.strip()) for x in args.ms.split(',')]
    if not mcs or not ms or min(mcs) < 2 or min(ms) < 1:
        raise ValueError('mcs>=2 y ms>=1; listas no vacias.')
    metodos = args.selecciones.split(',')
    escalas = args.escalados.split(',')
    if not set(metodos) <= {'eom','leaf'} or not set(escalas) <= {'estandar','robusto'}:
        raise ValueError('Las selecciones solo pueden ser eom/leaf, escalados estandar/robusto.')
    return [dict(id=identificador(esc, c, m, met), escalado=esc, min_cluster_size=c,
                 min_samples=m, seleccion=met)
            for esc in escalas for m in ms for c in mcs for met in metodos]


def reanudar_tabla(p):
    if p.is_file():
        df = pd.read_csv(p, sep=';', encoding='utf-8-sig')
        if not {'id', 'estado'}.issubset(df.columns):
            raise ValueError('Archivo de barrido incompatible; utiliza una carpeta de salida nueva.')
        return {str(r['id']): r for r in df.to_dict('records')}
    return {}


def barrido(args, datos, salida, propuestas):
    tabla_p = salida/'00_barrido_optimizacion.csv'
    anterior = reanudar_tabla(tabla_p)
    # Reutilizar barridos previos evita repetir ajustes costosos, pero hay
    # que distinguir configuraciones cuyo motor cambie.
    ok_previo = [c for c in propuestas if c['id'] in anterior and anterior[c['id']]['estado'] == 'OK']
    if ok_previo:
        print(f'Reanudacion: {len(ok_previo)} configuraciones OK ya guardadas; no se repiten.', flush=True)
    nuevos = 0
    for i, cfg in enumerate(propuestas, 1):
        cid = cfg['id']
        if cid in anterior and anterior[cid]['estado'] == 'OK':
            continue
        if args.max_configs and nuevos >= args.max_configs:
            print('Detencion solicitada por --max-configs. El barrido queda PARCIAL.', flush=True)
            break
        nuevos += 1
        esc = cfg['escalado']
        smp, Xtr = datos[esc]['entrenamiento']
        Xval = datos[esc]['validacion'][1]
        print(f'[{i}/{len(propuestas)}] {cid} | n={len(Xtr):,}, 10 senales', flush=True)
        inicio = time.perf_counter()
        fila = dict(id=cid, escalado=esc, min_cluster_size=cfg['min_cluster_size'],
                    min_samples=cfg['min_samples'], seleccion=cfg['seleccion'],
                    motor=args.motor, n_entrenamiento=len(Xtr), estado='ERROR', error='')
        try:
            modelo = ajuste(Xtr, cfg, salida, args.trabajadores, args.motor, False)
            labels = np.asarray(modelo.labels_, dtype=np.int32)
            k = len(set(labels)-{-1})
            ruido = int(np.count_nonzero(labels == -1))
            cobertura = float(100*(1-ruido/len(labels)))
            probs = np.asarray(modelo.probabilities_, dtype=float)
            per = np.asarray(modelo.cluster_persistence_, dtype=float)
            cuenta = pd.Series(labels[labels>=0]).value_counts().sort_index().to_numpy()
            pers_pond = np.average(per, weights=cuenta) if len(per) and len(cuenta)==len(per) else np.nan
            try:
                dbcv_aprox = float(modelo.relative_validity_)
            except (AttributeError, ValueError, RuntimeError, TypeError, ZeroDivisionError) as e:
                print('    DBCV aprox. no disponible:', str(e)[:140], flush=True)
                dbcv_aprox = np.nan
            # Si no hay grupos no tiene sentido proyectar validacion.
            if k:
                pred_val, fuerza = predecir(modelo, Xval)
            else:
                pred_val = np.full(len(Xval), -1, dtype=np.int32)
            val_cobertura = 100 * np.mean(pred_val >= 0)
            sil, n_sil = silueta_diagnostica(Xtr, labels)
            fila.update(grupos=k, ruido_n=ruido, ruido_pct=100-cobertura,
                        cobertura_pct=cobertura, val_ruido_pct=100-val_cobertura,
                        val_cobertura_pct=val_cobertura,
                        val_grupos_presentes=len(set(pred_val)-{-1}),
                        diferencia_cobertura_pp=abs(val_cobertura-cobertura),
                        persistencia_media=float(np.mean(per)) if len(per) else np.nan,
                        persistencia_ponderada=float(pers_pond),
                        fuerza_media_agrupados=float(np.mean(probs[labels>=0])) if (labels>=0).any() else np.nan,
                        dbcv_aprox_mst=dbcv_aprox,
                        silueta_solo_agrupados_muestra=sil, n_silueta_muestra=n_sil,
                        estado='OK', error='')
            # Guardar etiquetas para examinar sensibilidad sin nuevos ajustes.
            np.savez_compressed(salida/'00_etiquetas_barrido'/f'{cid}.npz',
                                smp=smp, labels=labels)
            print(f'    grupos={k}; cobertura train={cobertura:.2f}%; val={val_cobertura:.2f}%; '
                  f'persistencia={fila["persistencia_media"]:.4f}; DBCV-aprox={dbcv_aprox:.4g}', flush=True)
            del modelo
        except (Exception, KeyboardInterrupt) as e:
            fila['error'] = formatear_error(e)
            fila['estado'] = 'ERROR'
            print('    ERROR:', fila['error'], flush=True)
            if isinstance(e, KeyboardInterrupt):
                fila['segundos'] = round(time.perf_counter()-inicio, 2)
                anterior[cid] = fila
                guardarcsv(pd.DataFrame(anterior.values()).reindex(columns=COLS), tabla_p)
                raise
        fila['segundos'] = round(time.perf_counter()-inicio, 2)
        anterior[cid] = fila
        guardarcsv(pd.DataFrame(anterior.values()).reindex(columns=COLS), tabla_p)
        print('    + fila guardada; el barrido es reanudable',flush=True)
    print('Barrido guardado:', tabla_p,flush=True)
    return pd.DataFrame(anterior.values()).reindex(columns=COLS)


def normalizar(x):
    x = pd.Series(x, dtype=float).replace([np.inf,-np.inf],np.nan)
    if x.notna().sum() < 2 or x.max() == x.min():
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return ((x-x.min())/(x.max()-x.min())).fillna(0)


def ranking(args, salida, propuestas):
    p = salida/'00_barrido_optimizacion.csv'
    if not p.is_file():
        raise FileNotFoundError(f'Falta {p}. Ejecuta --modo barrido primero.')
    df = pd.read_csv(p, sep=';', encoding='utf-8-sig')
    ids = {c['id'] for c in propuestas}
    df = df[df['id'].isin(ids)].copy()
    completos = df[df.estado.eq('OK')].copy()
    if completos.empty:
        raise RuntimeError('No existe ninguna configuracion HDBSCAN exitosa para seleccionar.')
    pendientes = sorted(ids-set(completos.id))
    if pendientes:
        print(f'AVISO: barrido incompleto, faltan {len(pendientes)} de {len(propuestas)} configuraciones.',flush=True)
    # Ningun indice se transforma en "precision de falla".
    # Puntuacion EXPLORATORIA, solo para ordenar entre opciones no degeneradas.
    validos = completos[completos.grupos.ge(2)].copy()
    if validos.empty:
        exportar = {'estado':'sin_grupos_comparables','faltantes':pendientes}
        guardarjson(exportar, salida/'00_seleccion_exploratoria.json')
        print('No existen configuraciones con dos o mas grupos; no se recomienda elegir una.',flush=True)
        graficar_barrido(completos, salida)
        return None
    validos['fidelidad_cobertura'] = 1 - np.clip(validos.diferencia_cobertura_pp/100,0,1)
    # DBCV aprox. solo si disponible en todas las alternativas consideradas;
    # no se comparan escalas con unos valores inexistentes y otros validos.
    dbcv_completo = validos.dbcv_aprox_mst.notna().all()
    factores = {'cobertura_pct':.30, 'persistencia_ponderada':.30,
                'val_cobertura_pct':.20, 'fidelidad_cobertura':.20}
    estabilidad_p = salida/'03_estabilidad_por_configuracion.csv'
    if estabilidad_p.exists():
        estable = pd.read_csv(estabilidad_p,sep=';',encoding='utf-8-sig')
        validos = validos.merge(estable[['id','jaccard_mediana_vecinos']], how='left', on='id')
    else:
        validos['jaccard_mediana_vecinos'] = np.nan
    if dbcv_completo:
        factores={'cobertura_pct':.22,'persistencia_ponderada':.22,
                  'val_cobertura_pct':.18,'fidelidad_cobertura':.13,
                  'dbcv_aprox_mst':.25}
    if validos.jaccard_mediana_vecinos.notna().all():
        factores={k:w*.9 for k,w in factores.items()}
        factores['jaccard_mediana_vecinos']=.10
    validos['score_exploratorio'] = sum(w*normalizar(validos[k]).to_numpy() for k,w in factores.items())
    validos['pasa_puerta_orientativa'] = (
        validos.cobertura_pct.ge(args.cobertura_min) &
        validos.val_cobertura_pct.ge(args.cobertura_min) &
        validos.persistencia_ponderada.ge(args.persistencia_min)
    )
    if dbcv_completo:
        validos['pasa_puerta_orientativa'] &= validos.dbcv_aprox_mst.gt(args.dbcv_min)
    validos = validos.sort_values('score_exploratorio', ascending=False).reset_index(drop=True)
    guardarcsv(validos, salida/'01_ranking_exploratorio.csv')
    graficar_barrido(completos, salida, validos)
    viables=validos[validos.pasa_puerta_orientativa]
    elegido=viables.iloc[0] if len(viables) else validos.iloc[0]
    estado='candidato_seleccionado_bajo_criterios_orientativos' if len(viables) else 'NO_HAY_CANDIDATO_QUE_PASE_FILTROS'
    registro={
        'estado':estado,
        'id_candidato_exploratorio':str(elegido['id']),
        'cumple_puertas_orientativas':bool(len(viables)),
        'numero_configuraciones_probadas':int(len(completos)),
        'numero_configuraciones_propuestas':int(len(propuestas)),
        'barrido_completo':not pendientes,
        'configuraciones_pendientes':pendientes,
        'umbrales_orientativos_no_normativos':{'cobertura_train_y_validacion_pct':args.cobertura_min,
                                               'persistencia_ponderada':args.persistencia_min,
                                               'dbcv_aprox_mst_si_disponible':args.dbcv_min},
        'score_exploratorio_pesos':factores,
        'dbcv_aprox_mst_completo':bool(dbcv_completo),
        'nota':'No es un optimo global, ni clasificacion de averias. La prueba no participo en la seleccion.'
    }
    guardarjson(registro, salida/'00_seleccion_exploratoria.json')
    texto = ['# Seleccion exploratoria HDBSCAN', '',
             f'- Configuraciones evaluadas: {len(completos)} / {len(propuestas)}.',
             f'- Candidato comparativo: `{elegido["id"]}`.',
             f'- Estado: **{estado}**.',
             f'- Grupos: {int(elegido["grupos"])}.',
             f'- Cobertura entrenamiento: {elegido["cobertura_pct"]:.2f} %.',
             f'- Cobertura validacion aproximada: {elegido["val_cobertura_pct"]:.2f} %.',
             f'- Persistencia ponderada: {elegido["persistencia_ponderada"]:.4f}.',
             f'- DBCV aproximado por MST: {elegido["dbcv_aprox_mst"]}.',
             '', 'Los filtros de representatividad son criterios de analisis, no normas de mantenimiento.',
             'Una cobertura baja no prueba falla del equipo; indica dificultad para formar grupos.',
             'La prueba cronologica se conserva fuera de la busqueda de parametros.',
             'Si el barrido es parcial, repetir hasta completar antes de defender la eleccion.',
             'Estabilidad entre configuraciones: Jaccard de puntos agrupados, sin usar coincidencia masiva de ruido.',
             'Si DBCV aproximado es NO DISPONIBLE, la seleccion es mas debil y se debe justificar esa limitacion.']
    (salida/'00_lectura_optimizacion.md').write_text('\n'.join(texto),encoding='utf-8')
    print('Candidato EXPLORATORIO:',elegido['id'],'|',estado,flush=True)
    print('Ranking:',salida/'01_ranking_exploratorio.csv',flush=True)
    return str(elegido['id'])


def graficar_barrido(completos, salida, ranking_df=None):
    t=completos.copy().sort_values(['escalado','seleccion','min_samples','min_cluster_size']).reset_index(drop=True)
    nombres=t.id.astype(str).str.replace('estandar_', 'E:', regex=False).str.replace('robusto_', 'R:',regex=False)
    x=np.arange(len(t))
    fig,ax=plt.subplots(3,1,figsize=(max(11,len(t)*.34),10),sharex=True,constrained_layout=True)
    for j,(a,b,tit) in enumerate([('cobertura_pct','val_cobertura_pct','Cobertura (%)'),
                                  ('persistencia_ponderada','persistencia_media','Persistencia de grupos'),
                                  ('dbcv_aprox_mst',None,'DBCV aproximado por MST')]):
        ax[j].plot(x,t[a],'.-',label=a)
        if b: ax[j].plot(x,t[b],'.--',label=b)
        ax[j].set_ylabel(tit);ax[j].grid(axis='y',alpha=.15);ax[j].legend(fontsize=8,frameon=False)
    ax[-1].set_xticks(x,nombres,rotation=75,fontsize=7)
    fig.suptitle('HDBSCAN: barrido sobre 100.492 mediciones (diez senales juntas)')
    guardarfig(fig, salida/'01_resultados_barrido.png')
    # Matrices de sensibilidad con NUMEROS EN LAS CASILLAS: una por escalado/metodo.
    pares=sorted(set(zip(completos.escalado,completos.seleccion)))
    if pares:
        fig,axs=plt.subplots(len(pares),2,figsize=(11,max(3.5,len(pares)*2.75)),squeeze=False,constrained_layout=True)
        for i,(esc,met) in enumerate(pares):
            sub=completos[(completos.escalado==esc)&(completos.seleccion==met)]
            for j,(campo,titulo,fmt) in enumerate([('cobertura_pct','Cobertura agrupada (%)','.1f'),
                                                    ('dbcv_aprox_mst','DBCV aprox. (MST)','.3f')]):
                pivot=sub.pivot_table(index='min_samples',columns='min_cluster_size',values=campo)
                ax=axs[i,j]
                if pivot.empty:
                    ax.text(.5,.5,'Sin metrica disponible',ha='center',va='center',transform=ax.transAxes)
                    ax.set_axis_off();continue
                valores=pivot.to_numpy(dtype=float)
                im=ax.imshow(np.ma.masked_invalid(valores),cmap='viridis',aspect='auto')
                ax.set_xticks(range(len(pivot.columns)),pivot.columns)
                ax.set_yticks(range(len(pivot.index)),pivot.index)
                ax.set_xlabel('min_cluster_size');ax.set_ylabel('min_samples')
                ax.set_title(f'{esc} / {met} | {titulo}',fontsize=10)
                for iy in range(valores.shape[0]):
                    for ix in range(valores.shape[1]):
                        v=valores[iy,ix]
                        ax.text(ix,iy,format(v,fmt) if np.isfinite(v) else 'N/D',
                                ha='center',va='center',fontsize=9,color='white',weight='bold')
                fig.colorbar(im,ax=ax,shrink=.75)
        guardarfig(fig,salida/'03_matrices_sensibilidad_numeradas.png')
    if ranking_df is not None and not ranking_df.empty:
        t=ranking_df.head(20).iloc[::-1]
        fig,ax=plt.subplots(figsize=(10,max(4,len(t)*.34+1)))
        ax.barh(t['id'],t['score_exploratorio'],color=['#2879a1' if k else '#ba8a55' for k in t['pasa_puerta_orientativa']])
        for i,r in enumerate(t.itertuples()):
            ax.text(r.score_exploratorio+.012,i,f'{r.cobertura_pct:.1f}% en grupos',va='center',fontsize=8)
        ax.set_xlim(0,max(1.25,t.score_exploratorio.max()+.22))
        ax.set_xlabel('Puntuacion relativa (NO precision de averias)')
        ax.set_title('Ranking exploratorio de configuraciones HDBSCAN')
        ax.grid(axis='x',alpha=.15)
        guardarfig(fig,salida/'02_ranking_exploratorio.png')


def elegir_config(args, propuestas, salida):
    if args.id:
        coincidencias=[c for c in propuestas if c['id']==args.id]
        if not coincidencias:
            raise ValueError('ID no incluido en la grilla actual. Usa el mismo perfil/escalas/listas del barrido.')
        return coincidencias[0]
    seleccion=salida/'00_seleccion_exploratoria.json'
    if not seleccion.exists():
        raise FileNotFoundError('Falta 00_seleccion_exploratoria.json; ejecuta --modo seleccionar o especifica --id.')
    d=json.loads(seleccion.read_text(encoding='utf-8'))
    if not d.get('barrido_completo'):
        raise ValueError('El barrido esta incompleto; no se puede seleccionar implicitamente.')
    if not d.get('cumple_puertas_orientativas') and not args.aceptar_exploratorio:
        raise ValueError('Ninguna configuracion paso los filtros. Usa --id explicitamente o --aceptar-exploratorio si decides estudiarla igual.')
    return elegir_config(argparse.Namespace(id=d['id_candidato_exploratorio']), propuestas, salida)


def comparar_estabilidad(salida):
    p=salida/'00_barrido_optimizacion.csv'
    if not p.exists():
        return
    t=pd.read_csv(p,sep=';',encoding='utf-8-sig')
    t=t[t.estado.eq('OK')].copy()
    if len(t)<2: return
    from sklearn.metrics import adjusted_rand_score
    resultados=[]
    records=t.to_dict('records')
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['escalado']!=b['escalado']: continue
            # Comparar solo opciones cercanas: mismo metodo y una dimension de parametro cambia.
            diferencias=sum(a[k]!=b[k] for k in ['min_cluster_size','min_samples','seleccion'])
            if diferencias!=1: continue
            fa=salida/'00_etiquetas_barrido'/f'{a["id"]}.npz'
            fb=salida/'00_etiquetas_barrido'/f'{b["id"]}.npz'
            if not (fa.exists() and fb.exists()): continue
            with np.load(fa) as za, np.load(fb) as zb:
                if not np.array_equal(za['smp'],zb['smp']):
                    raise ValueError('SMP no coincide entre configuraciones del barrido.')
                la,lb=za['labels'],zb['labels']
            ina, inb=la>=0,lb>=0
            union=int(np.count_nonzero(ina|inb))
            inter=int(np.count_nonzero(ina&inb))
            ari=float(adjusted_rand_score(la[ina|inb],lb[ina|inb])) if union>2 else np.nan
            resultados.append({'id_A':a['id'],'id_B':b['id'],
                               'jaccard_puntos_agrupados':inter/union if union else np.nan,
                               'ari_solo_union_agrupados':ari,
                               'union_agrupados':union,
                               'comentario':'ARI solo union de agrupados; excluye ruido concordante para evitar inflacion'})
    if resultados:
        tabla=pd.DataFrame(resultados)
        guardarcsv(tabla,salida/'03_sensibilidad_entre_configuraciones.csv')
        elementos=[]
        for r in resultados:
            for nombre in (r['id_A'],r['id_B']):
                elementos.append({'id':nombre,'jaccard_vecino':r['jaccard_puntos_agrupados']})
        mediana=pd.DataFrame(elementos).groupby('id',as_index=False)['jaccard_vecino'].median()
        mediana=mediana.rename(columns={'jaccard_vecino':'jaccard_mediana_vecinos'})
        guardarcsv(mediana,salida/'03_estabilidad_por_configuracion.csv')
        print('  + 03_sensibilidad_entre_configuraciones.csv',flush=True)


def grafico_pca(X, labels, salida, nombre, var=None):
    pca=PCA(n_components=2,random_state=SEED)
    C=pca.fit_transform(X)
    # Solo el dibujo usa muestra reproducible; el ajuste conserva todos los registros.
    rng=np.random.default_rng(SEED)
    ruido=np.flatnonzero(labels==-1)
    agrupados=np.flatnonzero(labels>=0)
    ruido_p= rng.choice(ruido,min(8000,len(ruido)),replace=False) if len(ruido) else np.array([],int)
    agrup_p= rng.choice(agrupados,min(5000,len(agrupados)),replace=False) if len(agrupados) else np.array([],int)
    fig,ax=plt.subplots(figsize=(10,6))
    if len(ruido_p):ax.scatter(C[ruido_p,0],C[ruido_p,1],s=2,alpha=.16,c='#999999',rasterized=True,label='Ruido')
    colores=plt.get_cmap('tab20')
    ks=sorted(set(labels)-{-1})
    for i,k in enumerate(ks[:20]):
        ix=agrup_p[labels[agrup_p]==k]
        if len(ix):ax.scatter(C[ix,0],C[ix,1],s=5,alpha=.7,c=[colores(i/20)],rasterized=True,label=f'Grupo {k}')
    ax.set(xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f} %)',
           ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f} %)',
           title=f'HDBSCAN | {nombre} | PCA descriptivo (no entrenamiento)')
    ax.legend(fontsize=8,loc='best',frameon=False,ncol=2)
    ax.grid(alpha=.11)
    guardarfig(fig,salida/f'05_pca_{nombre}.png')


def finalizar(args, datos, salida, propuestas, total=False):
    cfg = elegir_config(args, propuestas, salida)
    esc=cfg['escalado']
    ds=datos[esc]
    Xtr=ds['entrenamiento'][1]
    smp_tr=ds['entrenamiento'][0]
    destino=salida/('04_ajuste_total' if total else '04_ajuste_cronologico')
    destino.mkdir(parents=True,exist_ok=True)
    if total:
        X=np.ascontiguousarray(np.vstack([ds[p][1] for p in PARTES]))
        smp=np.concatenate([ds[p][0] for p in PARTES])
        print('AJUSTE TOTAL: 143561 registros, las diez senales. NO es prueba independiente.',flush=True)
    else:
        X,smp=Xtr,smp_tr
        print('AJUSTE SELECCIONADO: 100492 entrenamiento; validacion y prueba son predicciones aproximadas.',flush=True)
    t0=time.perf_counter()
    modelo=ajuste(X,cfg,salida,args.trabajadores,args.motor,True)
    dur=time.perf_counter()-t0
    # Guardar el modelo ANTES de hacer figuras u otros calculos.
    joblib.dump(modelo,destino/'00_modelo_hdbscan_optimizado.joblib',compress=3)
    labels=np.asarray(modelo.labels_,dtype=np.int32)
    v=np.asarray(modelo.probabilities_,dtype=float)
    data={'smp_continuo':smp, 'cluster_hdbscan':labels, 'ruido_hdbscan':(labels==-1).astype('int8'),
          'fuerza_pertenencia':v, 'glosh':np.asarray(modelo.outlier_scores_,dtype=float)}
    guardarcsv(pd.DataFrame(data),destino/'01_resultado_entrenado.csv')
    res={'id':cfg['id'], 'escalado':esc,'parametros':cfg,'filas_ajuste':len(X),'tiempo_segundos':round(dur,2),
         'grupos':len(set(labels)-{-1}),'ruido_pct':100*float(np.mean(labels==-1)),
         'persistencia_grupos':np.asarray(modelo.cluster_persistence_).tolist(),
         'datos': 'todos 143561' if total else 'solo 100492 entrenamiento',
         'advertencia': 'ruido no equivale a falla; proyeccion 2D PCA es solo descriptiva'}
    if not total:
        for p in ['validacion','prueba']:
            l,f=predecir(modelo,ds[p][1])
            guardarcsv(pd.DataFrame({SMP:ds[p][0], 'cluster_hdbscan':l,
                                    'ruido_hdbscan':(l==-1).astype('int8'),
                                    'fuerza_pertenencia':f}), destino/f'02_resultado_{p}.csv')
            res[f'{p}_ruido_pct_aproximado']=100*float(np.mean(l==-1))
    guardarjson(res,destino/'00_resumen_ajuste.json')
    guardarcsv(pd.DataFrame({'cluster':np.arange(len(modelo.cluster_persistence_)),
                             'persistencia':modelo.cluster_persistence_}),destino/'02_persistencia.csv')
    grafico_pca(X,labels,destino,'total' if total else 'entrenamiento')
    # Exportar arbol REAL para auditabilidad aun si un backend no lo puede pintar.
    try:
        guardarcsv(modelo.condensed_tree_.to_pandas(),destino/'03_arbol_condensado_datos.csv')
        fig,ax=plt.subplots(figsize=(12,6))
        modelo.condensed_tree_.plot(axis=ax,select_clusters=False)
        ax.set_title('HDBSCAN | Arbol condensado (sin elipses de seleccion)')
        guardarfig(fig,destino/'03_arbol_condensado.png')
    except (ValueError,TypeError,AttributeError) as e:
        print('No se pudo dibujar el arbol en esta version de Matplotlib:',e,flush=True)
        plt.close('all')
    print('Ajuste completo y guardado:',destino,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modo',choices=['comprobar','barrido','seleccionar','final','total'],default='comprobar')
    parser.add_argument('--perfil',choices=['inicial','ampliado'],default='ampliado')
    parser.add_argument('--mcs',default=None,help='Ejemplo: 30,100,300')
    parser.add_argument('--ms',default=None,help='Ejemplo: 2,5,15')
    parser.add_argument('--selecciones',default='eom,leaf')
    parser.add_argument('--escalados',default='estandar,robusto')
    parser.add_argument('--preparacion',type=Path,default=BASE/'Resultados_Tarea3_Nuevos'/'00_preparacion')
    parser.add_argument('--salida',type=Path,default=BASE/'Resultados_Tarea3_Nuevos'/'10_optimizacion_hdbscan')
    parser.add_argument('--motor',default='boruvka_kdtree')
    parser.add_argument('--trabajadores',type=int,default=max(1,min(4,(os.cpu_count() or 2)-1)))
    parser.add_argument('--max-configs',type=int,default=0,help='Solo limita cantidad de AJUSTES, no filas entrenadas')
    parser.add_argument('--cobertura-min',type=float,default=10.0,help='Puerta orientativa; NO norma, porcentaje')
    parser.add_argument('--persistencia-min',type=float,default=0.05,help='Puerta orientativa, no norma')
    parser.add_argument('--dbcv-min',type=float,default=0.0,help='Puerta orientativa si DBCV aproximado disponible')
    parser.add_argument('--id',type=str,default=None,help='Identificador exacto desde 01_ranking_exploratorio.csv')
    parser.add_argument('--aceptar-exploratorio',action='store_true')
    args=parser.parse_args(argv)
    propuestas=configs(args)
    args.salida.mkdir(parents=True,exist_ok=True)
    print('=== 18 | OPTIMIZACION HDBSCAN MULTIVARIABLE ===',flush=True)
    print(f'Propuestas: {len(propuestas)}. Cada ajuste: 100492 registros / 10 senales.',flush=True)
    escalas=sorted({c['escalado'] for c in propuestas})
    meta, datos=leer(args.preparacion, escalas)
    print('Datos comprobados: 100492 train + 21534 val + 21535 test = 143561.',flush=True)
    rastreo_p=args.salida/'00_trazabilidad_entrada.json'
    registro={'sha256_origen_08':meta.get('sha256_origen'),
              'variable_names':meta['variables'],
              'registros':143561,
              'division':'70/15/15 cronologica',
              'motor':args.motor}
    if rastreo_p.exists():
        antiguo=json.loads(rastreo_p.read_text(encoding='utf-8'))
        if any(antiguo.get(k)!=registro[k] for k in ('sha256_origen_08','variable_names','registros','division','motor')):
            raise ValueError('La fuente o motor cambió. Usa otra carpeta --salida, no mezcles los resultados.')
    else:
        guardarjson(registro,rastreo_p)
    if args.modo=='comprobar':
        print('COMPROBACION CORRECTA. No se entrena. Para empezar: --modo barrido --perfil inicial',flush=True)
        return
    if args.modo=='barrido':
        importar_hdbscan()
        (args.salida/'00_etiquetas_barrido').mkdir(exist_ok=True)
        barrido(args,datos,args.salida,propuestas)
        comparar_estabilidad(args.salida)
        print('Luego: --modo seleccionar --perfil '+args.perfil,flush=True)
    elif args.modo=='seleccionar':
        comparar_estabilidad(args.salida)
        ranking(args,args.salida,propuestas)
    elif args.modo in ('final','total'):
        importar_hdbscan()
        finalizar(args,datos,args.salida,propuestas,total=args.modo=='total')


if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f'\nERROR CONTROLADO: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
