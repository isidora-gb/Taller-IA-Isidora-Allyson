
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
SMP = 'smp_continuo'
PARTES = ('entrenamiento', 'validacion', 'prueba')
SEEDS = (42, 123)
CUANTILES = (0.01, 0.02, 0.05)
ROWS = ['id', 'escalado', 'n_estimators', 'max_samples', 'semilla', 'registros_entrenamiento',
        'variables', 'segundos_ajuste', 'segundos_puntuar', 'auto_alertas_entrenamiento_pct',
        'auto_alertas_validacion_pct', 'q01_val_pct', 'q02_val_pct', 'q05_val_pct',
        'umbral_auto', 'umbral_q01', 'umbral_q02', 'umbral_q05', 'estado', 'error']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.grid':True,'grid.alpha':.18})


def csv_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(obj).to_csv(path, sep=';', index=False, encoding='utf-8-sig',
                             float_format='%.12g')


def json_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')


def fig_save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  [GRAFICO]',path.name,flush=True)


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for fragment in iter(lambda: f.read(8*1024*1024),b''):
            h.update(fragment)
    return h.hexdigest()


def files_and_meta(prep, escalado):
    meta_file = prep/'00_metadatos.json'
    if not meta_file.is_file():
        raise FileNotFoundError(f'No existe {meta_file}. Ejecuta el codigo 08 o pasa --preparados RUTA.')
    meta=json.loads(meta_file.read_text(encoding='utf-8'))
    cols=meta.get('variables', [])
    if len(cols)!=10 or len(set(cols))!=10 or meta.get('registros')!=143561:
        raise ValueError('Metadatos incompatibles: se esperan 143561 registros y 10 senales distintas.')
    div=prep/'00_division_cronologica.csv'
    if not div.is_file():
        raise FileNotFoundError(f'Falta {div}')
    paths={p:prep/f'01_{p}_{escalado}.csv' for p in PARTES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Falta {path} ({name}). Ejecuta 08.')
    return meta, div, paths


def read_sets(prep, escalado, include_test=False):
    meta, div_path, paths=files_and_meta(prep, escalado)
    division=pd.read_csv(div_path,sep=';',encoding='utf-8-sig',usecols=[SMP,'conjunto'])
    if not division[SMP].is_unique or not division[SMP].is_monotonic_increasing:
        raise ValueError('SMP de division tiene duplicados o esta desordenado.')
    datos={}
    for name in PARTES if include_test else PARTES[:2]:
        df=pd.read_csv(paths[name],sep=';',encoding='utf-8-sig')
        exp=[SMP]+[f'z_{c}' for c in meta['variables']]
        if list(df.columns)!=exp:
            raise ValueError(f'Columnas inesperadas en {paths[name].name}')
        smp=df[SMP].to_numpy(dtype=np.int64,copy=True)
        esperado=division.loc[division['conjunto'].eq(name),SMP].to_numpy(dtype=np.int64)
        if not np.array_equal(smp,esperado):
            raise ValueError(f'Orden SMP no corresponde a la division 08 ({name}).')
        X=np.ascontiguousarray(df.iloc[:,1:].to_numpy(dtype=np.float32,copy=True))
        if X.shape[1]!=10 or not np.isfinite(X).all():
            raise ValueError(f'Valores no finitos / no son 10 variables: {name}')
        datos[name]={'smp':smp,'X':X}
    sizes=(len(datos['entrenamiento']['smp']),len(datos['validacion']['smp']))
    if sizes != (100492,21534):
        raise ValueError(f'Tamanos no esperados: {sizes}, debieran ser 100492 y 21534.')
    if include_test and len(datos['prueba']['smp'])!=21535:
        raise ValueError('Prueba debe tener exactamente 21535 registros.')
    if datos['entrenamiento']['smp'][-1]>=datos['validacion']['smp'][0]:
        raise ValueError('Se rompio orden cronologico de las particiones.')
    if include_test and datos['validacion']['smp'][-1]>=datos['prueba']['smp'][0]:
        raise ValueError('Se rompio el orden cronologico respecto a prueba.')
    return datos,meta,paths


def manifest(prep, escalado, salida, paths, crear):
    expected={'version':1,'escala':escalado,'modelo':'IsolationForest',
              'n_variables':10,'particiones_train_val':{
                  p:hash_file(paths[p]) for p in ('entrenamiento','validacion')}}
    target=salida/'00_manifiesto_datos.json'
    if target.is_file():
        previo=json.loads(target.read_text(encoding='utf-8'))
        if previo!=expected:
            raise ValueError('Los archivos de entrada cambiaron desde el barrido previo. '
                             'No es seguro mezclar resultados. Usa otra --salida o respalda la carpeta.')
    elif crear:
        json_save(expected,target)
    return expected


def configs(profile, escalado):
    if profile=='inicial':
        pairs=[(100,256),(300,256),(300,512),(300,1024),(500,256),(500,512)]
    else:
        pairs=[(n,m) for n in (100,200,300,500) for m in (256,512,1024,2048)]
    return [dict(id=f'{escalado}_a{n}_m{m}',escalado=escalado,
                 n_estimators=n,max_samples=m) for n,m in pairs]


def topmask(s, fraction):
    # Ids de las puntuaciones mas altas; no depende del valor de umbral.
    n=max(1,int(math.ceil(len(s)*fraction)))
    ids=np.argpartition(s,-n)[-n:]
    mask=np.zeros(len(s),dtype=bool)
    mask[ids]=True
    return mask


def jaccard(a,b):
    both=np.count_nonzero(a & b)
    union=np.count_nonzero(a | b)
    return float(both/union) if union else float('nan')


def run_fit(X_train,X_val,config,seed,n_jobs):
    t=time.perf_counter()
    model=IsolationForest(n_estimators=config['n_estimators'],
                          max_samples=config['max_samples'],max_features=1.0,
                          contamination='auto',random_state=seed,n_jobs=n_jobs)
    model.fit(X_train)
    fit_sec=time.perf_counter()-t
    t=time.perf_counter()
    # -score_samples = puntaje creciente con atipicidad.
    train=-np.asarray(model.score_samples(X_train),dtype=np.float64)
    val=-np.asarray(model.score_samples(X_val),dtype=np.float64)
    score_sec=time.perf_counter()-t
    if not np.isfinite(train).all() or not np.isfinite(val).all():
        raise RuntimeError('Puntajes no finitos.')
    return model,train,val,fit_sec,score_sec


def read_barrido(path):
    if not path.is_file():
        return pd.DataFrame(columns=ROWS)
    t=pd.read_csv(path,sep=';',encoding='utf-8-sig')
    if len(t) and not set(ROWS).issubset(set(t.columns)):
        raise ValueError('CSV del barrido tiene esquema incompatible; respalda antes de continuar.')
    return t


def row_from_scores(cfg,seed,model,train,val,fit_sec,score_sec):
    qs={q:float(np.quantile(train,1-q)) for q in CUANTILES}
    thr=float(-model.offset_)
    return {'id':cfg['id'],'escalado':cfg['escalado'],
            'n_estimators':cfg['n_estimators'],'max_samples':cfg['max_samples'],
            'semilla':seed,'registros_entrenamiento':len(train),'variables':10,
            'segundos_ajuste':fit_sec,'segundos_puntuar':score_sec,
            'auto_alertas_entrenamiento_pct':100*float(np.mean(train>thr)),
            'auto_alertas_validacion_pct':100*float(np.mean(val>thr)),
            'q01_val_pct':100*float(np.mean(val>qs[.01])),
            'q02_val_pct':100*float(np.mean(val>qs[.02])),
            'q05_val_pct':100*float(np.mean(val>qs[.05])),
            'umbral_auto':thr,'umbral_q01':qs[.01],
            'umbral_q02':qs[.02],'umbral_q05':qs[.05],
            'estado':'OK','error':''}


def modo_barrido(datos,prep,escalado,salida,args,paths):
    salida.mkdir(parents=True,exist_ok=True)
    manifest(prep,escalado,salida,paths,True)
    target=salida/'00_barrido_optimizacion.csv'
    done=read_barrido(target)
    agenda=configs(args.perfil,escalado)
    cache=salida/'00_cache_puntajes'
    cache.mkdir(exist_ok=True)
    total=len(agenda)*len(SEEDS)
    for i,cfg in enumerate(agenda,1):
        for seed in SEEDS:
            previous=done.loc[(done['id'].eq(cfg['id'])) &
                              (pd.to_numeric(done['semilla'],errors='coerce')==seed) &
                              (done['estado'].eq('OK'))]
            score_file=cache/f'{cfg["id"]}_seed{seed}.npz'
            if len(previous) and score_file.is_file():
                print(f'[REUTILIZANDO {i}/{len(agenda)}] {cfg["id"]} semilla {seed}',flush=True)
                continue
            print(f'\n[ENTRENAMIENTO {i}/{len(agenda)}] {cfg["id"]} semilla={seed}; '
                  f'{len(datos["entrenamiento"]["X"]):,} filas x 10 senales',flush=True)
            try:
                mod,tr,va,ft,st=run_fit(datos['entrenamiento']['X'],
                                       datos['validacion']['X'],cfg,seed,args.trabajadores)
                row=row_from_scores(cfg,seed,mod,tr,va,ft,st)
                # Primero cache; luego fila CSV. Evita clasificar como completado sin puntajes.
                np.savez_compressed(score_file,train=tr,val=va)
                print(f'  Termino: {ft+st:.1f}s; alertas auto ent={row["auto_alertas_entrenamiento_pct"]:.2f}%; '
                      f'val={row["auto_alertas_validacion_pct"]:.2f}%',flush=True)
            except Exception as e:
                row={k:np.nan for k in ROWS}
                row.update({'id':cfg['id'],'escalado':escalado,
                            'n_estimators':cfg['n_estimators'],'max_samples':cfg['max_samples'],
                            'semilla':seed,'registros_entrenamiento':100492,
                            'variables':10,'estado':'ERROR', 'error':f'{type(e).__name__}: {e}'[:500]})
                print('  [ERROR]',row['error'],flush=True)
            done=done.loc[~((done['id'].eq(cfg['id'])) &
                            (pd.to_numeric(done['semilla'],errors='coerce')==seed))].copy()
            done=pd.concat([done,pd.DataFrame([row])],ignore_index=True)
            done=done[ROWS].sort_values(['escalado','n_estimators','max_samples','semilla'])
            csv_save(done,target)
    print(f'\nBarrido guardado en: {target}',flush=True)
    ranking_and_figures(done,cache,salida,args.perfil,escalado)


def ranked_data(barrido,cache,perfil,escalado):
    rows=[]
    requested=configs(perfil,escalado)
    for cfg in requested:
        subset=barrido[(barrido.id==cfg['id']) & (barrido.estado=='OK')]
        if set(pd.to_numeric(subset.semilla).astype(int))!=set(SEEDS):
            continue
        sub=subset.set_index('semilla').loc[list(SEEDS)]
        files=[cache/f'{cfg["id"]}_seed{seed}.npz' for seed in SEEDS]
        if not all(f.is_file() for f in files):
            continue
        with np.load(files[0]) as a, np.load(files[1]) as b:
            tr0=a['train']; v0=a['val']; tr1=b['train']; v1=b['val']
        if len(tr0)!=100492 or len(v0)!=21534 or len(tr1)!=100492 or len(v1)!=21534:
            raise ValueError('Cache incompatible con tamanos de particiones.')
        rho=float(spearmanr(v0,v1).statistic)
        j1=jaccard(topmask(v0,.01),topmask(v1,.01))
        j2=jaccard(topmask(v0,.02),topmask(v1,.02))
        a0=v0>float(sub.loc[SEEDS[0],'umbral_auto'])
        a1=v1>float(sub.loc[SEEDS[1],'umbral_auto'])
        jauto=jaccard(a0,a1)
        tim=float((sub.segundos_ajuste+sub.segundos_puntuar).median())
        mean_rate=float(sub.auto_alertas_validacion_pct.mean())
        row={'id':cfg['id'],'escalado':escalado,
             'n_estimators':cfg['n_estimators'],'max_samples':cfg['max_samples'],
             'semillas_verificadas':2,'spearman_val_entre_semillas':rho,
             'jaccard_top1pct_val':j1,'jaccard_top2pct_val':j2,
             'jaccard_alertas_auto_val':jauto,
             'auto_alertas_validacion_pct_promedio':mean_rate,
             'q02_val_pct_promedio':float(sub.q02_val_pct.mean()),
             'segundos_mediana_fit_y_puntuacion':tim}
        # Criterio de seleccion: reproducibilidad del ranking y top-2% en validacion.
        # Se favorece costo SOLO como desempate suave, no se usa la tasa de alertas.
        row['puntaje_estabilidad_exploratorio']=(.45*rho+.35*j2+.20*j1 -
             .02*math.log2(max(tim, .1)))
        row['criterio_orientativo_cumplido']=bool(rho>=.98 and j2>=.70 and j1>=.60)
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_and_figures(barrido,cache,salida,perfil,escalado):
    t=ranked_data(barrido,cache,perfil,escalado)
    if t.empty:
        print('[SELECCION] Faltan pares de semillas completados. No hay ranking todavia.',flush=True)
        return None
    t=t.sort_values('puntaje_estabilidad_exploratorio',ascending=False,kind='stable').reset_index(drop=True)
    t.insert(0,'posicion_exploratoria',np.arange(1,len(t)+1))
    csv_save(t,salida/'01_ranking_estabilidad.csv')
    elegibles=t[t.criterio_orientativo_cumplido]
    completo=(len(t)==len(configs(perfil,escalado)))
    candidato=(elegibles.iloc[0] if len(elegibles) and completo else None)
    selection={'criterio':'Reproducibilidad de puntuaciones/ranking y top-1/2% en validacion, NO precision diagnostica',
               'comparaciones':int(len(t)), 'configuraciones_perfil':len(configs(perfil,escalado)),
               'barrido_perfil_completo':bool(completo),
               'escala':escalado,'candidato_orientativo':None if candidato is None else str(candidato['id']),
               'id_lider_ranking_aunque_no_cumpla_filtros':str(t.iloc[0]['id']),
               'filtros_exploratorios':{'spearman_val_min':.98,'jaccard_top1_min':.60,'jaccard_top2_min':.70},
               'prueba_utilizada_para_seleccionar':False,
               'advertencia':'No existe etiqueta de falla; ninguna configuracion demuestra acierto diagnostico.'}
    json_save(selection,salida/'00_seleccion_exploratoria.json')
    charts_ranking(t,salida,escalado)
    lines=['# Optimizacion de Isolation Forest | lectura para informe','',
           f'- Configuraciones con ambas semillas completadas: **{len(t)}**.',
           '- Se emplean todas las filas de entrenamiento (100.492) y las diez senales juntas.',
           '- Validacion tiene 21.534 filas; prueba queda reservada hasta modo final.',
           '- Los modelos difieren en arboles y submuestras internas por arbol; se fija max_features=1.0.',
           '- Se usa contamination=auto solo como referencia; cuotas de 1/2/5% se calculan en entrenamiento.',
           f'- Barrido de perfil completo: {completo}. Un perfil incompleto no autoriza elegir candidato.',
           '- El ranking es exploratorio, en ausencia de etiquetas de falla. No indica precision o recall.',
           '',f'- Lider del ranking: `{t.iloc[0]["id"]}` (no equivale a mejor detector de fallas).',
           f'- Candidato que supera filtros orientativos: `{selection["candidato_orientativo"]}`.'
           if candidato is not None else '- No se selecciona candidato: perfil incompleto o ninguno supera filtros orientativos.',
           '', '## Criterios y cautelas',
           '- Spearman compara el orden de puntuaciones de DOS semillas en validacion.',
           '- Jaccard top-1% y top-2% comparan los mismos registros mas inusuales sin aplicar umbral de falla.',
           '- Jaccard de alertas auto es sensible a umbrales diferentes entre ajustes.',
           '- La tasa de alertas NO es tasa de averias.',
           '- El filtro de estabilidad es una convencion de esta investigacion, NO una norma ISO.',
           '- No usar el conjunto de prueba para elegir parametros o umbrales.',
           '- SMP no representa tiempo ni duracion.',
           '']
    (salida/'00_lectura_optimizacion.md').write_text('\n'.join(lines),encoding='utf-8')
    print('[RANKING] primer lugar:',t.iloc[0]['id'],
          '| candidato orientativo:',selection['candidato_orientativo'],flush=True)
    return t


def charts_ranking(t,salida,escalado):
    nvals=sorted(t.n_estimators.unique())
    mvals=sorted(t.max_samples.unique())
    heat=np.full((len(nvals),len(mvals)),np.nan)
    for _,r in t.iterrows():
        heat[nvals.index(r.n_estimators),mvals.index(r.max_samples)]=r.jaccard_top2pct_val
    fig,ax=plt.subplots(figsize=(max(8.4,len(mvals)*1.75+2),max(5.4,len(nvals)*.9+2)))
    im=ax.imshow(heat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(np.arange(len(mvals)),mvals)
    ax.set_yticks(np.arange(len(nvals)),nvals)
    ax.set(xlabel='max_samples (muestras internas/arbol)',ylabel='Numero de arboles',
           title='Jaccard top-2% entre semillas | validacion (mayor = mas reproducible)')
    for i in range(len(nvals)):
        for j in range(len(mvals)):
            if np.isfinite(heat[i,j]):
                ax.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',
                        color='white' if heat[i,j]>.58 else '#152b40',weight='bold')
    fig.colorbar(im,ax=ax,label='Jaccard (0 a 1)')
    fig_save(fig,salida/'01_matriz_estabilidad_numerada.png')

    fig,ax=plt.subplots(figsize=(10.5,5.3))
    for n,group in t.groupby('n_estimators'):
        group=group.sort_values('max_samples')
        ax.plot(group.max_samples,group.jaccard_top2pct_val,marker='o',linewidth=1.9,
                label=f'{n} arboles')
    ax.axhline(.70,color='#b35d52',linestyle='--',alpha=.85,label='Umbral exploratorio 0.70')
    ax.set(xlabel='Muestras internas por arbol',ylabel='Jaccard top-2% en validacion',
           title='Sensibilidad de las alertas mas extremas a los parametros',ylim=(0,1.03))
    ax.legend(ncol=2,frameon=False)
    fig_save(fig,salida/'02_jaccard_vs_muestras.png')

    fig,ax=plt.subplots(figsize=(9.5,5.5))
    sc=ax.scatter(t.segundos_mediana_fit_y_puntuacion,t.jaccard_top2pct_val,
                  c=t.n_estimators,cmap='viridis',s=65+np.array(t.max_samples)/25,
                  alpha=.88,edgecolors='white',linewidths=.7)
    for _,r in t.iterrows():
        ax.annotate(f'{int(r.n_estimators)}/{int(r.max_samples)}',
                    (r.segundos_mediana_fit_y_puntuacion,r.jaccard_top2pct_val),
                    xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Segundos de ajuste y puntuacion (mediana entre semillas)',
           ylabel='Jaccard top-2% validacion',
           title='Compromiso entre reproducibilidad y costo computacional')
    fig.colorbar(sc,ax=ax,label='Numero de arboles')
    fig_save(fig,salida/'03_estabilidad_vs_costo.png')

    fig,ax=plt.subplots(figsize=(10.4,5.2))
    for n,g in t.groupby('n_estimators'):
        g=g.sort_values('max_samples')
        ax.plot(g.max_samples,g.auto_alertas_validacion_pct_promedio,
                marker='o',label=f'{n} arboles')
    ax.set(xlabel='Muestras internas por arbol',ylabel='Alertas en validacion (%)',
           title='Tasa de alertas con umbral automatico (NO mide exactitud)')
    ax.legend(ncol=2,frameon=False)
    fig_save(fig,salida/'04_sensibilidad_alertas_auto.png')

    fig,ax=plt.subplots(figsize=(10.6,max(4.5,len(t)*.4+1.7)))
    ordered=t.sort_values('puntaje_estabilidad_exploratorio').tail(16)
    ax.barh(ordered.id,ordered.puntaje_estabilidad_exploratorio,color='#497e9b')
    ax.set(xlabel='Indice heuristico de estabilidad y costo (NO precision)',
           title='Ranking exploratorio de configuraciones ensayadas')
    fig_save(fig,salida/'05_ranking_exploratorio.png')


def pick_config(salida,id_requested):
    path=salida/'01_ranking_estabilidad.csv'
    if not path.is_file():
        raise FileNotFoundError('Primero ejecuta --modo barrido y --modo seleccionar.')
    t=pd.read_csv(path,sep=';',encoding='utf-8-sig')
    if id_requested:
        s=t[t.id.eq(id_requested)]
        if s.empty:
            raise ValueError(f'ID {id_requested} no existe en ranking. Revisa 01_ranking_estabilidad.csv')
        r=s.iloc[0]
        print('[AVISO] ID fijado explicitamente por usuario; revisa su estabilidad.',flush=True)
    else:
        selected=salida/'00_seleccion_exploratoria.json'
        if not selected.is_file():
            raise FileNotFoundError('Ejecuta --modo seleccionar primero.')
        sel=json.loads(selected.read_text(encoding='utf-8'))
        if sel.get('candidato_orientativo') is None:
            raise RuntimeError('Ninguna configuracion pasa los filtros. No se selecciona automaticamente. '
                               'Revisa las metricas y, si justificas un experimento, indica --id.')
        r=t[t.id.eq(sel['candidato_orientativo'])].iloc[0]
    return {'id':str(r['id']),'escalado':str(r['escalado']),
            'n_estimators':int(r['n_estimators']),'max_samples':int(r['max_samples'])}


def final_fit(datos,meta,salida,cfg,args,full=False):
    sub=salida/('03_ajuste_total' if full else '02_ajuste_cronologico')
    sub.mkdir(parents=True,exist_ok=True)
    Xtr=datos['entrenamiento']['X']
    if full:
        Xtr=np.concatenate([datos[p]['X'] for p in PARTES],axis=0)
    if len(Xtr)!=(143561 if full else 100492):
        raise ValueError('Cantidad de registros incorrecta para modo elegido.')
    print(f'[FINAL] {cfg["id"]}; ajustando con {len(Xtr):,} filas completas de este conjunto.',flush=True)
    t=time.perf_counter()
    mod=IsolationForest(n_estimators=cfg['n_estimators'],max_samples=cfg['max_samples'],
                        max_features=1.0,contamination='auto',random_state=42,
                        n_jobs=args.trabajadores)
    mod.fit(Xtr)
    print(f'  Ajuste terminado: {time.perf_counter()-t:.2f}s',flush=True)
    joblib.dump(mod,sub/'00_modelo_optimizado.joblib',compress=3)
    thr_auto=-float(mod.offset_)
    baseline=-mod.score_samples(datos['entrenamiento']['X'])
    qs={q:float(np.quantile(baseline,1-q)) for q in CUANTILES}
    summary=[]
    results={}
    for part in PARTES:
        d=datos[part]
        score=-np.asarray(mod.score_samples(d['X']),dtype=float)
        a=score>thr_auto
        df=pd.DataFrame({SMP:d['smp'],'puntaje_anomalia':score,
                         'alerta_auto':a.astype('uint8')})
        for q,v in qs.items():
            df[f'alerta_cuantil_{int(q*100):02d}']=np.asarray(score>v,dtype=np.uint8)
        csv_save(df,sub/f'01_resultado_{part}.csv')
        results[part]={'smp':d['smp'],'scores':score,'auto':a}
        for method,threshold in [('auto',thr_auto),
                                 *[(f'cuantil_{int(q*100):02d}',qs[q]) for q in CUANTILES]]:
            n=int(np.count_nonzero(score>threshold))
            summary.append({'conjunto':part,'criterio':method,'umbral':threshold,
                            'registros':len(score),'marcaciones':n,
                            'porcentaje':100*n/len(score)})
    csv_save(summary,sub/'00_resumen_umbrales.csv')
    json_save({'configuracion':cfg,'modo':'total' if full else 'cronologico',
               'n_ajuste':len(Xtr),'escalado_fijado_por_08':cfg['escalado'],
               'umbral_auto':thr_auto,
               'umbrales_cuantiles_entrenamiento_08':{str(q):v for q,v in qs.items()},
               'n_senales':10,'seleccion_sin_etiquetas':True,
               'nota':'En modo total son resultados dentro de muestra, NO test independiente'},
              sub/'00_configuracion_optimizada.json')
    graph_final(results,thr_auto,sub,datos,summary)
    print(f'  [LISTO] resultados y graficos en {sub}',flush=True)


def graph_final(results,threshold,out,datos,summary):
    colors={'entrenamiento':'#2379a6','validacion':'#d18c42','prueba':'#7961a9'}
    scores=np.concatenate([results[p]['scores'] for p in PARTES])
    bins=np.linspace(float(np.min(scores)),float(np.max(scores)),100)
    fig,ax=plt.subplots(figsize=(10.4,5.5))
    for p in PARTES:
        ax.hist(results[p]['scores'],bins=bins,density=True,histtype='step',
                linewidth=1.7,color=colors[p],label=p.capitalize())
    ax.axvline(threshold,color='#b33c46',linestyle='--',lw=1.4,label='Auto fijo')
    ax.set(xlabel='Puntuacion de atipicidad (mayor = mas inusual)',ylabel='Densidad',
           title='Distribucion de puntajes | mismas escalas y umbral')
    ax.legend(frameon=False)
    fig_save(fig,out/'01_distribucion_puntajes.png')

    df=pd.DataFrame(summary)
    fig,ax=plt.subplots(figsize=(10.5,5.2))
    methods=['auto','cuantil_01','cuantil_02','cuantil_05']
    x=np.arange(len(methods))
    for i,p in enumerate(PARTES):
        vals=[float(df[(df.conjunto==p)&(df.criterio==m)]['porcentaje'].iloc[0]) for m in methods]
        ax.plot(x,vals,marker='o',color=colors[p],label=p.capitalize())
        for xx,v in zip(x,vals):
            ax.annotate(f'{v:.1f}',(xx,v),xytext=(0,5),textcoords='offset points',
                        ha='center',fontsize=8)
    ax.set_xticks(x,['Auto','Top 1% ent.','Top 2% ent.','Top 5% ent.'])
    ax.set(xlabel='Criterio de alerta',ylabel='Alertas (%)',
           title='Sensibilidad al umbral (cuantiles SOLO de entrenamiento)')
    ax.legend(frameon=False)
    fig_save(fig,out/'02_sensibilidad_umbrales.png')

    fig,ax=plt.subplots(figsize=(12.7,5.3))
    for p in PARTES:
        r=results[p]
        ax.plot(r['smp'],r['scores'],c=colors[p],lw=.36,alpha=.5,label=p.capitalize())
        inds=np.flatnonzero(r['auto'])
        if len(inds):
            take=inds if len(inds)<=3500 else inds[np.linspace(0,len(inds)-1,3500,dtype=int)]
            ax.scatter(r['smp'][take],r['scores'][take],c='#be4450',s=3,
                       alpha=.55,rasterized=True)
    ax.axhline(threshold,color='#292929',ls='--',lw=1.2,label='Auto')
    ax.set(xlabel='SMP continuo (contador, NO segundos)',ylabel='Puntuacion de atipicidad',
           title='Evolucion por SMP y alertas automaticas')
    ax.legend(frameon=False,ncol=4)
    fig_save(fig,out/'03_evolucion_smp.png')

    pca=PCA(n_components=2,random_state=42)
    pca.fit(datos['entrenamiento']['X'])
    Z=pca.transform(datos['prueba']['X'])
    a=results['prueba']['auto']
    rng=np.random.default_rng(42)
    ids=np.arange(len(Z))
    if len(ids)>17000:
        flagged=np.flatnonzero(a)
        normal=np.flatnonzero(~a)
        chosen=np.r_[flagged if len(flagged)<=8500 else rng.choice(flagged,8500,replace=False),
                     rng.choice(normal,min(8500,len(normal)),replace=False)]
    else:
        chosen=ids
    fig,ax=plt.subplots(figsize=(10,6.5))
    for flag,color,label in [(False,'#6a8eaa','Sin alerta'),(True,'#d66348','Alerta')]:
        m=a[chosen]==flag
        ax.scatter(Z[chosen[m],0],Z[chosen[m],1],c=color,s=5 if not flag else 9,
                   alpha=.38 if not flag else .66,edgecolors='none',label=label,rasterized=True)
    ax.set(xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var.)',
           ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var.)',
           title='Prueba | PCA SOLO visualizacion (10 variables para el modelo)')
    ax.legend(frameon=False)
    fig_save(fig,out/'04_pca_prueba_alertas.png')


def main():
    pa=argparse.ArgumentParser(description='19 | Optimizar Isolation Forest sin etiquetas de averias')
    pa.add_argument('--modo',choices=['comprobar','barrido','seleccionar','final','total'],
                    default='comprobar')
    pa.add_argument('--perfil',choices=['inicial','ampliado'],default='inicial')
    pa.add_argument('--escalado',choices=['estandar','robusto'],default='estandar')
    pa.add_argument('--trabajadores',type=int,default=4)
    pa.add_argument('--preparados',type=Path,
                    default=ROOT/'Resultados_Tarea3_Nuevos'/'00_preparacion')
    pa.add_argument('--salida',type=Path,
                    default=ROOT/'Resultados_Tarea3_Nuevos'/'11_optimizacion_isolation_forest')
    pa.add_argument('--id',default=None,
                    help='ID explicitamente seleccionado del 01_ranking_estabilidad.csv')
    args=pa.parse_args()
    print('=== OPTIMIZACION ISOLATION FOREST | 10 SENALES JUNTAS ===',flush=True)
    print('Particion 100492/21534/21535; SMP NO equivale a segundos.',flush=True)
    if args.trabajadores==0:
        pa.error('--trabajadores no puede ser cero')
    if args.modo=='seleccionar':
        manifest_file=args.salida/'00_manifiesto_datos.json'
        if not manifest_file.exists():
            raise FileNotFoundError('No hay barrido todavia. Primero ejecuta --modo barrido.')
        m=json.loads(manifest_file.read_text(encoding='utf-8'))
        if args.escalado!=m['escala']:
            raise ValueError('El escalado no coincide con el del barrido. Usa la misma --escalado y --salida.')
        # Comprobar fuente en modo seleccionar evita usar resultados desactualizados.
        _,_,paths=files_and_meta(args.preparados,args.escalado)
        manifest(args.preparados,args.escalado,args.salida,paths,False)
        ranking_and_figures(read_barrido(args.salida/'00_barrido_optimizacion.csv'),
                            args.salida/'00_cache_puntajes',args.salida,args.perfil,args.escalado)
        return
    include_test=args.modo in ('final','total')
    datos,meta,paths=read_sets(args.preparados,args.escalado,include_test)
    print(f'Datos verificados: entrenamiento={len(datos["entrenamiento"]["smp"]):,}; '
          f'validacion={len(datos["validacion"]["smp"]):,}; '
          f'prueba={len(datos.get("prueba",{}).get("smp",[])) if include_test else "RESERVADA"}; '
          f'variables=10; escalado={args.escalado}.',flush=True)
    if args.modo=='comprobar':
        print('[OK] No se ha entrenado nada. Puedes ejecutar el barrido.',flush=True)
        return
    args.salida.mkdir(parents=True,exist_ok=True)
    manifest(args.preparados,args.escalado,args.salida,paths,True)
    if args.modo=='barrido':
        modo_barrido(datos,args.preparados,args.escalado,args.salida,args,paths)
    else:
        cfg=pick_config(args.salida,args.id)
        if cfg['escalado']!=args.escalado:
            raise ValueError('El escalado del modelo elegido debe coincidir con --escalado.')
        final_fit(datos,meta,args.salida,cfg,args,full=args.modo=='total')


if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError,ValueError,RuntimeError) as err:
        print(f'\nERROR: {err}',flush=True)
        raise SystemExit(1)
