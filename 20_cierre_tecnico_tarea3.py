
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMP = 'smp_continuo'
PARTES = ('entrenamiento', 'validacion', 'prueba')
PARES = (
    ('Isolation Forest', 'alerta_if', '01_isolation_forest', 'alerta_isolation_forest'),
    ('K-means', 'alerta_kmeans', '02_kmeans', 'alerta_distancia_kmeans'),
    ('DBSCAN aproximado', 'ruido_dbscan_aproximado', '03_dbscan', 'alerta_ruido_dbscan'),
)
COL = {'Isolation Forest':'#316D9C','K-means':'#BC7648','DBSCAN (aprox.)':'#62917B'}
NOMBRES = {
    'AcelX1_SisGenEol':'Ac X', 'AcelY1_SisGenEol':'Ac Y', 'AcelZ1_SisGenEol':'Ac Z',
    'GiroX_SisGenEol':'Gi X', 'GiroY_SisGenEol':'Gi Y', 'GiroZ_SisGenEol':'Gi Z',
    'MagneX_SisGenEol':'Ma X', 'MagneY_SisGenEol':'Ma Y', 'MagneZ_SisGenEol':'Ma Z',
    'Ter_SisGenEol':'Temp',
}


def read(path, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Falta {path}. Ejecuta el codigo que genera este resultado.')
        return None
    return pd.read_csv(path, sep=';', encoding='utf-8-sig', low_memory=False)


def save(df, path):
    pd.DataFrame(df).to_csv(path, index=False, sep=';', encoding='utf-8-sig', float_format='%.10g')


def check_columns(df, cols, origen):
    miss = set(cols)-set(df.columns)
    if miss:
        raise ValueError(f'{origen}: faltan columnas {sorted(miss)}')


def add(logs, tema, status, detail):
    logs.append({'comprobacion':tema,'estado':status,'detalle':str(detail)})


def figsave(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def find_source(root, results):
    meta = results/'00_preparacion'/'00_metadatos.json'
    candidates = [root/'Datos_corregidos'/'datos_multisensor_corregidos.csv',
                  root/'Datos_corregidos_desde_excel'/'datos_multisensor_corregidos.csv',
                  root/'datos_multisensor_corregidos.csv']
    if meta.is_file():
        item = json.loads(meta.read_text(encoding='utf-8')).get('archivo_fuente')
        if item:
            candidates = [Path(item), *candidates]
    for p in candidates:
        if p.is_file():
            return p
    return None


def read_source(path):
    with path.open('r', encoding='utf-8-sig') as f:
        head = f.readline()
    sep = ';' if ';' in head else ','
    raw = pd.read_csv(path, sep=sep, encoding='utf-8-sig', dtype='string')
    check_columns(raw,[SMP,*NOMBRES],str(path))
    for c in [SMP,*NOMBRES]:
        raw[c] = pd.to_numeric(raw[c].str.strip().str.replace(',','.',regex=False), errors='coerce')
    return raw.sort_values(SMP,kind='mergesort').reset_index(drop=True)


def perfil_grupos(df, sensores, result_dir, logs):
    tr = df.loc[df['conjunto']=='entrenamiento', sensores]
    mu = tr.mean(axis=0)
    ds = tr.std(axis=0, ddof=0).replace(0, np.nan)
    g = df.groupby('cluster_kmeans',sort=True)
    medios = g[sensores].mean()
    count = g.size()
    table = []
    for c in medios.index:
        for v in sensores:
            table.append({'cluster_kmeans':int(c),'sensor':NOMBRES[v],
                          'registros_cluster':int(count.loc[c]),
                          'media_fisica_cluster':float(medios.loc[c,v]),
                          'media_en_desviaciones_entrenamiento':float((medios.loc[c,v]-mu[v])/ds[v])})
    p = pd.DataFrame(table)
    save(p,result_dir/'03_perfil_clusters_estandarizado.csv')
    dif = pd.DataFrame({'sensor':[NOMBRES[v] for v in sensores]})
    if len(medios)==2:
        ids = list(medios.index)
        dif['cluster_a'] = int(ids[0]); dif['cluster_b'] = int(ids[1])
        dif['diferencia_medias_en_DSen_train'] = [float((medios.loc[ids[1],v]-medios.loc[ids[0],v])/ds[v]) for v in sensores]
        dif['diferencia_absoluta'] = np.abs(dif['diferencia_medias_en_DSen_train'])
        dif = dif.sort_values('diferencia_absoluta',ascending=False)
    save(dif,result_dir/'04_variables_que_distinguen_clusters.csv')
    if len(medios)>=2:
        pivot = p.pivot(index='sensor',columns='cluster_kmeans',values='media_en_desviaciones_entrenamiento')
        pivot = pivot.reindex([NOMBRES[v] for v in sensores])
        a = pivot.to_numpy()
        fig, ax = plt.subplots(figsize=(8,5.3))
        lim = max(.1, float(np.nanmax(np.abs(a))))
        im=ax.imshow(a,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
        ax.set_xticks(range(len(pivot.columns)),[f'Cluster {i}' for i in pivot.columns]);
        ax.set_yticks(range(len(pivot.index)),pivot.index)
        ax.set_title('Perfil de clusters K-means\nMedia respecto del entrenamiento, en DS')
        fig.colorbar(im,ax=ax,label='(Media cluster - media train) / DS train')
        figsave(fig,result_dir/'01_perfil_clusters_estandarizado.png')
    add(logs,'Perfiles K-means','OK',f'Clusters: {len(count)}; tamaños: '+repr(dict(count)))
    return p, dif


def comparacion_modelos(df, root_results, salida, logs):
    res=[]
    for name,col,folder,othercol in PARES:
        for part in PARTES:
            orig=read(root_results/folder/f'01_resultado_{part}.csv',required=False)
            bloque=df.loc[df.conjunto==part,[SMP,col]].copy()
            if orig is None:
                add(logs,f'{name} {part}','NO_VERIFICADO','Falta salida original del modelo')
                continue
            check_columns(orig,[SMP,othercol],f'Original {name}')
            if orig[SMP].duplicated().any() or bloque[SMP].duplicated().any():
                add(logs,f'{name} {part}','ERROR','SMP duplicado'); continue
            combined=bloque.merge(orig[[SMP,othercol]],how='outer',on=SMP,indicator=True,validate='one_to_one')
            coincide=int(((combined['_merge']=='both') &
                (pd.to_numeric(combined[col],errors='coerce')==pd.to_numeric(combined[othercol],errors='coerce'))).sum())
            n=len(bloque)
            ok=(len(combined)==n and coincide==n)
            status='OK' if ok else 'ERROR'
            add(logs,f'Coherencia {name} / {part}',status,
                f'Coincidencias de bandera: {coincide}/{n}; filas merge: {len(combined)}')
            res.append({'modelo':name,'conjunto':part,'registros':n,'coinciden':coincide,'estado':status})
    save(res,salida/'02_consistencia_salidas_modelos.csv')


def silueta_archivos(results,salida,logs):
    ks=read(results/'02_kmeans'/'00_seleccion_k.csv')
    check_columns(ks,['K','silhouette_muestra_validacion'],'seleccion K')
    ks['K']=pd.to_numeric(ks['K'],errors='coerce')
    ks['silhouette_muestra_validacion']=pd.to_numeric(ks['silhouette_muestra_validacion'],errors='coerce')
    sil=read(results/'11_analisis_completo_143561'/'18_resumen_silueta.csv',required=False)
    if sil is None:
        add(logs,'Silueta diagnostica codigo 19','NO_VERIFICADO','Ejecutar codigo 19 en modo normal')
    else:
        save(sil,salida/'06_silueta_diagnostica_codigo19.csv')
    mejor=ks.dropna(subset=['silhouette_muestra_validacion']).sort_values(['silhouette_muestra_validacion','K'],ascending=[False,True])
    if len(mejor)==0:
        add(logs,'Silueta K','ERROR','No existen valores válidos'); return None
    best=mejor.iloc[0]
    add(logs,'Selección K por silueta','OK',f"K={int(best.K)}, silhouette validación={best.silhouette_muestra_validacion:.6f}")
    save(ks,salida/'05_k_seleccion_silueta_original.csv')
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    ax.plot(ks.K,ks.silhouette_muestra_validacion,marker='o',color='#7B548E')
    ax.scatter([best.K],[best.silhouette_muestra_validacion],s=100,color='#C67949',zorder=3)
    ax.annotate(f'K={int(best.K)}; silueta={best.silhouette_muestra_validacion:.3f}',
                (best.K,best.silhouette_muestra_validacion),xytext=(8,15),textcoords='offset points')
    ax.set(xlabel='Número de grupos (K)',ylabel='Silueta en muestra de validación',title='Selección original de K')
    ax.grid(axis='y',alpha=.2)
    figsave(fig,salida/'02_seleccion_k_silueta_anotada.png')
    return {'K':int(best.K),'silueta':float(best.silhouette_muestra_validacion),
            'silueta_19':sil.to_dict(orient='records') if sil is not None else None}


def cortes_y_corte4(raw, results,salida,logs, sensors):
    folder=results/'06_validacion_5_cortes'
    summary=read(folder/'00_resumen_cortes.csv',required=False)
    resultados=read(folder/'01_resultados_por_corte.csv',required=False)
    if summary is None or resultados is None:
        add(logs,'Validación cinco cortes','NO_VERIFICADO','Falta codigo 14: 00_resumen_cortes o 01_resultados_por_corte')
        return None
    check_columns(summary,['corte','train_smp_inicio','train_smp_fin','validacion_smp_inicio','validacion_smp_fin'], 'Cortes')
    check_columns(resultados,['corte','modelo','porcentaje_alertas_validacion'], 'Resultados cortes')
    keys=summary.sort_values('corte')['corte'].tolist()
    if len(keys)!=5 or len(set(keys))!=5:
        add(logs,'Cinco cortes','ERROR',f'Se esperaban cinco cortes, recibidos: {keys}')
    else:
        add(logs,'Cinco cortes','OK',f'Cortes presentes: {keys}')
    records=[]
    long_serie=[]
    for _,row in summary.iterrows():
        idx=int(row.corte)
        tr=raw[(raw[SMP]>=row.train_smp_inicio)&(raw[SMP]<=row.train_smp_fin)]
        va=raw[(raw[SMP]>=row.validacion_smp_inicio)&(raw[SMP]<=row.validacion_smp_fin)]
        if not len(tr) or not len(va):
            add(logs,f'Corte {idx}','ERROR','No se encontraron intervalos de SMP en el CSV corregido');continue
        if int(tr[SMP].max())>=int(va[SMP].min()):
            add(logs,f'Corte {idx}','ERROR','Solapamiento temporal entrenamiento/validación');continue
        for v in sensors:
            mu_t=float(tr[v].mean()); mu_v=float(va[v].mean());
            std=float(tr[v].std(ddof=0)); q25,q75=tr[v].quantile([.25,.75]);
            iqr=float(q75-q25); median_t=float(tr[v].median()); median_v=float(va[v].median());
            sdshift=float((mu_v-mu_t)/std) if std>1e-12 else float('nan')
            robust_shift=float((median_v-median_t)/iqr) if iqr>1e-12 else float('nan')
            records.append({'corte':idx,'sensor':NOMBRES[v],
                            'n_entrenamiento':len(tr),'n_validacion':len(va),
                            'media_entrenamiento':mu_t,'media_validacion':mu_v,
                            'delta_media_DSen_train':sdshift,
                            'abs_delta_media_DSen_train':abs(sdshift),
                            'delta_mediana_IQRentrenamiento':robust_shift,
                            'desv_estandar_entrenamiento':std,
                            'desv_estandar_validacion':float(va[v].std(ddof=0)),
                            'mediana_entrenamiento':median_t,'mediana_validacion':median_v})
        long_serie.append({'corte':idx,'train':tr,'val':va})
    drifts=pd.DataFrame(records)
    save(drifts,salida/'07_cambios_variables_por_corte.csv')
    save(resultados,salida/'08_alertas_5_cortes_original.csv')
    fig,ax=plt.subplots(figsize=(9,5.2))
    for n in ['Isolation Forest','K-means','DBSCAN (aprox.)']:
        rr=resultados[resultados.modelo==n].sort_values('corte')
        if len(rr): ax.plot(rr.corte,rr.porcentaje_alertas_validacion,label=n,marker='o',linewidth=1.8,color=COL[n])
    ax.set(xlabel='Corte cronológico',ylabel='Alertas en validación (%)',title='Estabilidad temporal de los tres detectores')
    ax.legend(frameon=False)
    ax.set_xticks(sorted(resultados.corte.unique()));ax.grid(axis='y',alpha=.18)
    figsave(fig,salida/'03_alertas_cinco_cortes.png')
    if not len(drifts): return None
    pivot=drifts.pivot(index='sensor',columns='corte',values='delta_media_DSen_train').reindex([NOMBRES[v] for v in sensors])
    vals=pivot.to_numpy(float)
    lim=max(.05,float(np.nanmax(np.abs(vals))))
    fig,ax=plt.subplots(figsize=(8,5.5))
    im=ax.imshow(vals,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
    ax.set_yticks(range(len(pivot.index)),pivot.index)
    ax.set_xticks(range(len(pivot.columns)),[f'Corte {int(k)}' for k in pivot.columns])
    fig.colorbar(im,ax=ax,label='(Media validación - media train) / DS train')
    ax.set_title('Cambios estadísticos por corte\nLos modelos se reajustan en cada corte')
    figsave(fig,salida/'04_mapa_cambios_variables_cinco_cortes.png')
    c4=drifts[drifts.corte==4].sort_values('abs_delta_media_DSen_train',ascending=False)
    save(c4,salida/'09_diagnostico_corte_4_ordenado.csv')
    if len(c4):
        fig,ax=plt.subplots(figsize=(8.8,4.9))
        ax.barh(c4.sensor.iloc[::-1],c4.delta_media_DSen_train.iloc[::-1],color='#5C84AB')
        ax.axvline(0,color='gray',lw=1)
        ax.set(xlabel='Cambio de media respecto al entrenamiento (DS del entrenamiento)',
               title='Corte 4: variables con mayor cambio de media')
        ax.grid(axis='x',alpha=.18)
        figsave(fig,salida/'05_corte4_cambios_medias.png')
        q=next((item for item in long_serie if item['corte']==4),None)
        if q:
            top=c4.head(3)['sensor'].tolist()
            for i,name in enumerate(top,1):
                col=next(c for c,v in NOMBRES.items() if v==name)
                tr=q['train'][col].dropna().to_numpy();va=q['val'][col].dropna().to_numpy()
                # Submuestras acotadas sólo para la figura; cuantiles se calculan con todos.
                t=tr[np.linspace(0,len(tr)-1,min(7000,len(tr)),dtype=int)]
                v=va[np.linspace(0,len(va)-1,min(7000,len(va)),dtype=int)]
                fig,ax=plt.subplots(figsize=(8.4,4.5))
                ax.hist(t,bins=35,density=True,alpha=.5,label='Subentrenamiento anterior')
                ax.hist(v,bins=35,density=True,alpha=.5,label='Bloque validación corte 4')
                ax.set(title=f'Corte 4: distribución de {name}',xlabel=f'{name} (escala registrada)',ylabel='Densidad')
                ax.legend(frameon=False)
                figsave(fig,salida/f'06_corte4_distribucion_{i}_{name.replace(" ","")}.png')
            add(logs,'Corte 4, mayor desplazamiento','OK',', '.join(f"{r.sensor}: {r.delta_media_DSen_train:+.3f} DS" for r in c4.head(3).itertuples()))
    return {'top_corte4':c4.head(5).to_dict(orient='records'),'cortes':5}


def auditar_imputaciones(raiz,raw,salida,logs):
    # No convertir 0 imputaciones nuevas en afirmacion sobre celdas historicas.
    candidates=[]
    for p in [raiz/'registro_imputaciones.csv', raiz/'Datos_corregidos'/'registro_imputaciones.csv',
              raiz/'Datos_corregidos_desde_excel'/'registro_imputaciones.csv',
              raiz/'Resultados_corregidos'/'registro_imputaciones.csv']:
        if p.is_file(): candidates.append(p)
    for sub in [raiz/'Datos_corregidos',raiz/'Datos_corregidos_desde_excel']:
        if sub.is_dir(): candidates += list(sub.glob('*/registro_imputaciones.csv'))
    candidates=list(dict.fromkeys(candidates))
    audit=[]
    for p in candidates:
        try:
            temp=pd.read_csv(p,encoding='utf-8-sig',sep=None,engine='python')
            audit.append({'archivo':str(p),'registros_auditoria':len(temp),
                          'columnas':' | '.join(map(str,temp.columns)),'nota':'Documento de procedencia, no imputaciones nuevas'})
        except Exception as ex:
            audit.append({'archivo':str(p),'registros_auditoria':None,'columnas':'', 'nota':f'No legible: {ex}'})
    save(audit if audit else [{'archivo':'No localizado','registros_auditoria':np.nan,
                              'columnas':'','nota':'No se puede certificar la procedencia histórica sin registro_imputaciones.csv'}],
         salida/'10_auditoria_documentos_imputacion.csv')
    if candidates:
        add(logs,'Procedencia imputaciones','OK',f'Se localizaron {len(candidates)} registros de auditoría. Revisar contenido.')
    else:
        add(logs,'Procedencia imputaciones','NO_VERIFICADO','No se encontró registro_imputaciones.csv; CSV corregido ya contiene datos reemplazados')
    p=raw.tail(3)
    checks=[]
    if len(p)>=3:
        for col in ['MagneZ_SisGenEol','Ter_SisGenEol']:
            if col in p.columns:
                ult=float(p[col].iloc[-1]); ant=float(p[col].iloc[-3:-1].mean())
                checks.append({'sensor':NOMBRES[col],'ultimo_valor_corregido':ult,
                               'promedio_dos_anteriores':ant,'diferencia':ult-ant,
                               'coincide_con_promedio_dos_anteriores':bool(np.isclose(ult,ant,rtol=0,atol=1e-8))})
    save(checks,salida/'11_comprobacion_valores_terminales.csv')
    if checks:
        add(logs,'Ultima fila corregida','OK' if all(x['coincide_con_promedio_dos_anteriores'] for x in checks) else 'ADVERTENCIA',
            'Comparación con dos registros anteriores; no verifica dato real ni reemplaza evidencia de imputación')


def sap_suficiente(results,salida,logs):
    # Separa SHAP puntuación (código 12) de TreeSHAP longitud (código 19).
    resumen=[]
    info=[('SHAP de puntuación (código 12)',results/'04_shap'/'00_importancias_shap.csv'),
          ('TreeSHAP de longitud (código 19)',results/'11_analisis_completo_143561'/'20_shap_importancias_LONGITUD.csv')]
    for metodo,path in info:
        d=read(path,required=False)
        if d is None:
            add(logs,metodo,'NO_VERIFICADO',f'Falta {path.name}');continue
        col=next((c for c in d.columns if 'abs_media' in c or 'shap_abs' in c),None)
        if col and 'sensor' in d.columns:
            for r in d.sort_values(col,ascending=False).head(10).itertuples(index=False):
                row=dict(zip(d.columns,r))
                resumen.append({'metodo':metodo,'sensor':row['sensor'], 'importancia_abs_media':row[col],
                                'tipo_salida_explicada':'puntaje' if 'puntuación' in metodo else 'longitud'})
            add(logs,metodo,'OK',f'Importancias existentes; no comparar valores entre métodos SHAP')
        else:
            add(logs,metodo,'ADVERTENCIA',f'Columnas no reconocidas: {list(d.columns)}')
    save(resumen,salida/'12_shap_importancias_separadas.csv')



def auditar_preprocesamiento(raw,results,salida,logs):
    """Verifica que las estadisticas de imputacion/escalado vengan solo de train."""
    try:
        import joblib
        prep=joblib.load(results/'00_preparacion'/'00_preprocesamiento.joblib')
        variables=list(prep['variables'])
        train=raw.iloc[:int(len(raw)*.70)][variables]
        # SimpleImputer de mediana: calcular su salida, y la media/escala del StandardScaler.
        imputer=prep['imputador']; scaler=prep['escalador']
        xt=imputer.transform(train)
        median=np.nanmedian(train.to_numpy(float),axis=0)
        mean=np.mean(xt,axis=0)
        stdev=np.std(xt,axis=0,ddof=0)
        stdev=np.where(stdev==0,1.0,stdev)
        a=np.asarray(imputer.statistics_,dtype=float)
        b=np.asarray(scaler.mean_,dtype=float)
        c=np.asarray(scaler.scale_,dtype=float)
        dif=np.array([np.max(abs(a-median)),np.max(abs(b-mean)),np.max(abs(c-stdev))])
        save(pd.DataFrame({'variable':variables,'mediana_train_calculada':median,
             'imputador_mediana_guardada':a,'media_train_calculada':mean,
             'escalador_media_guardada':b,'desv_train_calculada':stdev,
             'escalador_escala_guardada':c}),salida/'13_auditoria_preprocesamiento_entrenamiento.csv')
        add(logs,'Sin filtración en preparación 08',
            'OK' if (dif.max()<1e-7) else 'ERROR',
            f'Diferencias absolutas máximas [imputación, media, escala]: {dif.tolist()}')
    except Exception as exc:
        add(logs,'Sin filtración en preparación 08','ERROR',f'No se pudo verificar: {exc}')


def auditar_episodios(df,results,salida,logs):
    """Corrobora los episodios globales por SMP sin juntar huecos."""
    spec=[('Isolation Forest','alerta_if','05_episodios_isolation_forest.csv'),
          ('K-means','alerta_kmeans','06_episodios_kmeans.csv'),
          ('DBSCAN aproximado','ruido_dbscan_aproximado','07_episodios_dbscan.csv'),
          ('Consenso 2+', 'consenso_al_menos_dos','08_episodios_consenso.csv')]
    if not {'consenso_al_menos_dos',SMP}.issubset(df.columns):
        add(logs,'Episodios','ERROR','Faltan columnas para auditar consensos')
        return
    smp=df[SMP].to_numpy(dtype=np.int64)
    records=[]
    for nombre,col,filename in spec:
        flags=pd.to_numeric(df[col],errors='coerce').to_numpy()==1
        empieza=flags & np.r_[True,(~flags[:-1])|(np.diff(smp)!=1)]
        computed=int(np.sum(empieza))
        old=read(results/'11_analisis_completo_143561'/filename,required=False)
        existing=len(old) if old is not None else -1
        length=int(old['longitud_muestras'].sum()) if old is not None and 'longitud_muestras' in old.columns else -1
        ok=(computed==existing and length==int(flags.sum()))
        records.append({'modelo':nombre,'alertas_por_fila':int(flags.sum()),'episodios_recalculados':computed,
                        'episodios_guardados':existing,'suma_longitud_episodios':length,'estado':'OK' if ok else 'ERROR'})
        add(logs,'Episodios '+nombre,'OK' if ok else 'ERROR',
            f'Calculados={computed}, guardados={existing}, alertas={flags.sum()}, suma longitudes={length}')
    save(records,salida/'14_auditoria_episodios_SMP.csv')


def create_report(salida,logs,n,ks,cut4,counts,source):
    issues=[x for x in logs if x['estado']=='ERROR']
    warn=[x for x in logs if x['estado'] in ('ADVERTENCIA','NO_VERIFICADO')]
    lines=[
        '# Cierre técnico de Python — Tarea 3', '',
        f'**Fuente analizada:** `{source}`',
        f'**Filas en resultados del código 19:** {n:,}',
        '**Modalidad:** auditoría descriptiva. No se reentrenan modelos, no se modifican umbrales.', '',
        '## Qué se verificó',
        'Se contrastaron las banderas de los tres modelos entre los códigos 09–11 y el 19 por SMP y por partición; se analizaron tamaños y perfiles de grupos, silueta, desplazamientos entre cortes e imputaciones históricas.', '',
        '## K-means y DBSCAN',
    ]
    if ks:
        lines.append(f"- K elegido: **{ks['K']}**; silueta original en muestra de validación: **{ks['silueta']:.6f}** (no es exactitud de fallas).")
        if ks['silueta_19']:
            x=ks['silueta_19'][0]
            lines.append(f"- Silueta diagnóstica sobre muestra de todas las particiones: **{x.get('silueta_diagnostica_muestra_de_todos')}**; no se usa para escoger K.")
            lines.append(f"- DBSCAN: {x.get('nota_dbscan')}")
    lines.append('- Los perfiles estandarizados de los clusters son descriptivos; K-means no asigna el significado «sano/falla» a los grupos.')
    lines.extend(['','## Cinco cortes y revisión del corte 4'])
    if cut4:
        lines.append('Se calcularon diferencias de medias respecto del propio entrenamiento de cada corte, expresadas en desviaciones estándar del entrenamiento. No equivalen a límites físicos o a diagnósticos.')
        for row in cut4['top_corte4'][:5]:
            lines.append(f"- {row['sensor']}: {row['delta_media_DSen_train']:+.4f} DS (corte 4).")
    else:
        lines.append('No fue posible revisar el corte 4: faltan salidas del código 14.')
    lines.extend(['','## Procedencia y límites',
                  '- El CSV corregido ya puede contener dos imputaciones de la Tarea 2: el contador de nuevas imputaciones del prototipo no certifica las históricas.',
                  '- No existe una matriz de confusión válida sin etiquetas verificadas de averías.',
                  '- DBSCAN fuera de la muestra se asigna mediante proximidad a puntos núcleo (aproximación).',
                  '- SHAP del código 12 explica puntuación; TreeSHAP del 19 explica longitud de aislamiento. Sus magnitudes no se comparan.',
                  '- La prueba externa fue consultada en las comparaciones previas; cualquier reajuste exige una nueva evaluación independiente.',
                  '- SMP identifica posición de adquisición, no segundos.', '',
                  '## Estado automático de comprobaciones',
                  f'- Errores detectados: {len(issues)}.',
                  f'- Advertencias / no verificado: {len(warn)}.',
                  '- Ver `00_lista_comprobaciones.csv` y `02_consistencia_salidas_modelos.csv`.', '',
                  '## Archivos más importantes',
                  '- `04_variables_que_distinguen_clusters.csv`: ejes que más diferencian las medias de ambos clusters (en unidades comparables).',
                  '- `05_k_seleccion_silueta_original.csv`: siluetas de los K explorados.',
                  '- `07_cambios_variables_por_corte.csv`: desplazamiento por sensor y por corte.',
                  '- `09_diagnostico_corte_4_ordenado.csv`: ranking del corte 4.',
                  '- `10_auditoria_documentos_imputacion.csv` y `11_comprobacion_valores_terminales.csv`: trazabilidad parcial de dos imputaciones.',
                  '- `12_shap_importancias_separadas.csv`: sin mezclar tipos distintos de SHAP.',
                  '- `13_auditoria_preprocesamiento_entrenamiento.csv`: comprueba que mediana y escalado proceden solo del 70 % inicial.',
                  '- `14_auditoria_episodios_SMP.csv`: comprueba episodios sin unir saltos SMP.',
                  '',
                  'Este documento presenta diagnósticos de calidad de datos y comportamiento de modelos; **no confirma fallas físicas**.',
    ])
    (salida/'LEER_RESULTADOS_CIERRE.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raiz',type=Path,help='Ruta del repositorio que contiene Resultados_Tarea3')
    parser.add_argument('--esperados',type=int,default=143561,help='Registros esperados; default=143561')
    args=parser.parse_args()
    raiz=(args.raiz if args.raiz is not None else Path(__file__).resolve().parent).resolve()
    results=raiz/'Resultados_Tarea3'
    salida=results/'13_cierre_tecnico_python'
    salida.mkdir(parents=True,exist_ok=True)
    logs=[]
    f=sorted((results/'11_analisis_completo_143561').glob('01_resultados_TODOS_*_registros.csv'))
    if len(f)!=1:
        raise FileNotFoundError('Debes ejecutar primero 19_analisis_completo_143561.py SIN --ajuste-final-todos. Falta el CSV completo en 11_analisis_completo_143561.')
    df=read(f[0]); n=len(df)
    check_columns(df,[SMP,'conjunto','cluster_kmeans','alerta_if','alerta_kmeans','ruido_dbscan_aproximado',*NOMBRES], 'Código 19')
    sensores=list(NOMBRES.keys())
    if n!=args.esperados:
        add(logs,'Cantidad registros','ADVERTENCIA',f'Esperados={args.esperados}, encontrados={n}')
    else: add(logs,'Cantidad registros','OK',n)
    if df[SMP].isna().any() or df[SMP].duplicated().any() or not df[SMP].is_monotonic_increasing:
        add(logs,'SMP orden y unicidad','ERROR','SMP no único, nulo o desordenado')
    else:
        gaps=np.diff(df[SMP].to_numpy(dtype=np.int64))
        add(logs,'SMP orden y unicidad','OK',f'Saltos: {(gaps>1).sum()}; posiciones faltantes: {np.maximum(gaps-1,0).sum()}')
    if set(df.conjunto.unique())!=set(PARTES):
        add(logs,'Tres particiones','ERROR',f'Conjuntos recibidos: {df.conjunto.unique().tolist()}')
    else:
        sizes=df.conjunto.value_counts().reindex(PARTES)
        add(logs,'Tres particiones','OK',sizes.to_dict())
    if df[sensores].isna().sum().sum():
        add(logs,'Nulos físicos','ERROR',f"NaNs en salida corregida: {int(df[sensores].isna().sum().sum())}")
    else: add(logs,'Nulos físicos','OK','No hay nulos en las diez variables del CSV de análisis')
    source=find_source(raiz,results)
    if source:
        raw=read_source(source)
        if len(raw)!=n or not np.array_equal(raw[SMP].to_numpy(),df[SMP].to_numpy()):
            add(logs,'Fuente contra código 19','ERROR','No coinciden SMP o cantidad de registros')
        else:
            difer_max=0
            for s in sensores:
                d1=pd.to_numeric(df[s],errors='coerce').to_numpy(float)
                d2=raw[s].to_numpy(float)
                if not np.isfinite(d1).all() or not np.isfinite(d2).all():
                    add(logs,'Valores fisicos','ERROR',f'No finitos en {s}');continue
                difer_max=max(difer_max,float(np.max(np.abs(d1-d2))))
            add(logs,'Fuente contra codigo 19','OK' if difer_max<1e-7 else 'ERROR', f'Error máximo original/codigo19: {difer_max:.3g}')
    else:
        add(logs,'Fuente corregida','NO_VERIFICADO','No se encontró datos_multisensor_corregidos.csv; se usa CSV de código 19')
        raw=df.copy()
    perfil_grupos(df,sensores,salida,logs)
    comparacion_modelos(df,results,salida,logs)
    auditar_preprocesamiento(raw,results,salida,logs)
    auditar_episodios(df,results,salida,logs)
    ks=silueta_archivos(results,salida,logs)
    cut4=cortes_y_corte4(raw,results,salida,logs,sensores)
    auditar_imputaciones(raiz,raw,salida,logs)
    sap_suficiente(results,salida,logs)
    counts=df.groupby('conjunto').agg(registros=(SMP,'size'),alertas_if=('alerta_if','sum'),alertas_kmeans=('alerta_kmeans','sum'),ruido_dbscan=('ruido_dbscan_aproximado','sum')).reset_index()
    save(counts,salida/'01_resumen_tres_modelos_por_particion.csv')
    save(logs,salida/'00_lista_comprobaciones.csv')
    create_report(salida,logs,n,ks,cut4,counts,source or 'Fuente original no localizada')
    print('\nCIERRE TÉCNICO PYTHON FINALIZADO')
    print(f'Carpeta: {salida}')
    print(counts.to_string(index=False))
    if ks: print(f"K={ks['K']} | silueta en validación={ks['silueta']:.6f}")
    print('Comprobaciones: ',pd.DataFrame(logs).estado.value_counts().to_dict())
    print('Revisa LEER_RESULTADOS_CIERRE.md y 00_lista_comprobaciones.csv')
    if any(x['estado']=='ERROR' for x in logs):
        print('ATENCIÓN: se detectaron inconsistencias. No usar las conclusiones hasta resolverlas.',file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':
    main()
