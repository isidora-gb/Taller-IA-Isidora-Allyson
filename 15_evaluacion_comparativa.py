
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
PARTES = ('entrenamiento', 'validacion', 'prueba')
TAMANOS = {'entrenamiento': 100492, 'validacion': 21534, 'prueba': 21535}
SMP = 'smp_continuo'
COLOR = {'Isolation Forest':'#367FA4', 'DBSCAN':'#CB7A55',
         'HDBSCAN':'#8A72B5', 'K-Means':'#5E9980'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.grid':True, 'grid.alpha':0.15,
                     'savefig.facecolor':'white'})


def leer(ruta:Path) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f'No existe {ruta}. Ejecuta el codigo correspondiente en modo final.')
    return pd.read_csv(ruta,sep=';',encoding='utf-8-sig',low_memory=False)


def exportar(tabla, destino):
    pd.DataFrame(tabla).to_csv(destino,sep=';',index=False,encoding='utf-8-sig',float_format='%.10g')
    print('  [CSV]',destino.name,flush=True)


def grafico(fig, destino):
    fig.savefig(destino,dpi=185,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('  [GRAFICO]',destino.name,flush=True)


def verificar_ids(d:pd.DataFrame,origen:str,parte:str):
    if SMP not in d or d[SMP].isna().any() or d[SMP].duplicated().any():
        raise ValueError(f'{origen}: SMP nulo, repetido o ausente ({parte})')
    if len(d) != TAMANOS[parte]:
        raise ValueError(f'{origen}: {parte}: {len(d)} filas; esperadas {TAMANOS[parte]}')
    if not d[SMP].is_monotonic_increasing:
        raise ValueError(f'{origen}: SMP no ordenado en {parte}')


def binaria(d:pd.DataFrame,col:str,origen:str):
    if col not in d or d[col].isna().any() or not d[col].isin([0,1,False,True]).all():
        raise ValueError(f'{origen}: {col} debe contener solamente 0 o 1')
    return d[col].to_numpy(dtype=np.int8)


def obtener_rutas(base:Path):
    kdir=base/'02_kmeans'
    kfiles=sorted(kdir.glob('01_grupos_todos_registros_todas_*.csv'))
    if not kfiles:
        raise FileNotFoundError(f'Falta K-Means de las DIEZ señales en {kdir}. Ejecuta 10_kmeans.py')
    preferidas=[p for p in kfiles if p.stem.endswith('_todas_estandar')]
    if len(kfiles)>1 and not preferidas:
        raise ValueError('Hay varios escalados K-Means. No voy a elegir arbitrariamente: '+str(kfiles))
    karchivo=preferidas[0] if preferidas else kfiles[0]
    hdir=base/'04_hdbscan'
    hexist=[(hdir/f'01_resultado_{p}.csv').is_file() for p in PARTES]
    if any(hexist) and not all(hexist):
        raise FileNotFoundError('HDBSCAN esta INCOMPLETO: faltan CSV finales de algunas particiones. '
                                'Termina 12_hdbscan.py --modo final antes de comparar.')
    return karchivo, all(hexist)


def cargar(base:Path):
    karchivo,tiene_hdb=obtener_rutas(base)
    k=leer(karchivo)
    for c in [SMP,'conjunto','cluster']:
        if c not in k:raise ValueError(f'K-Means: falta columna {c}')
    if len(k)!=sum(TAMANOS.values()) or k[SMP].duplicated().any():
        raise ValueError('K-Means: base multivariable incompleta o SMP duplicados')
    if set(k['conjunto'])!=set(PARTES):
        raise ValueError('K-Means: conjuntos cronologicos incorrectos')
    if k['cluster'].isna().any() or (k['cluster']<0).any():
        raise ValueError('K-Means: etiquetas de grupo invalidas')
    salidas={}
    for parte in PARTES:
        km=k.loc[k.conjunto==parte,[SMP,'cluster']].sort_values(SMP).copy()
        verificar_ids(km,'K-Means',parte)
        km=km.rename(columns={'cluster':'grupo_kmeans'})
        dd=leer(base/'03_dbscan'/f'01_resultado_{parte}.csv')
        verificar_ids(dd,'DBSCAN',parte)
        db_noise=binaria(dd,'alerta_ruido_dbscan','DBSCAN')
        if 'cluster_dbscan' not in dd or not np.array_equal(
                (dd.cluster_dbscan.to_numpy()==-1).astype(np.int8),db_noise):
            raise ValueError(f'DBSCAN: inconsistencia ruido/cluster en {parte}')
        ff=leer(base/'05_isolation_forest'/f'01_resultado_{parte}.csv')
        verificar_ids(ff,'Isolation Forest',parte)
        _=binaria(ff,'alerta_isolation_forest','Isolation Forest')
        if 'puntaje_anomalia' not in ff or not np.isfinite(ff.puntaje_anomalia).all():
            raise ValueError('Isolation Forest: falta puntaje finito')
        f=km.merge(dd[[SMP,'cluster_dbscan','alerta_ruido_dbscan']],on=SMP,
                   how='inner',validate='one_to_one')
        f=f.merge(ff[[SMP,'alerta_isolation_forest','puntaje_anomalia']],on=SMP,
                  how='inner',validate='one_to_one')
        if tiene_hdb:
            hh=leer(base/'04_hdbscan'/f'01_resultado_{parte}.csv')
            verificar_ids(hh,'HDBSCAN',parte)
            ruido=binaria(hh,'ruido_hdbscan','HDBSCAN')
            if 'cluster_hdbscan' not in hh or not np.array_equal(
                    (hh.cluster_hdbscan.to_numpy()==-1).astype(np.int8),ruido):
                raise ValueError(f'HDBSCAN: inconsistencia ruido/cluster en {parte}')
            f=f.merge(hh[[SMP,'cluster_hdbscan','ruido_hdbscan']],on=SMP,
                      how='inner',validate='one_to_one')
        if len(f)!=TAMANOS[parte] or not np.array_equal(f[SMP].to_numpy(),km[SMP].to_numpy()):
            raise ValueError(f'Las etiquetas no coinciden por SMP en {parte}. NO se alinean por indice.')
        salidas[parte]=f.sort_values(SMP).reset_index(drop=True)
    smps=[salidas[p][SMP].to_numpy(dtype=np.int64) for p in PARTES]
    if not (smps[0][-1]<smps[1][0]<smps[2][0]):
        raise ValueError('No se conserva el orden cronologico de 08')
    return salidas,karchivo,tiene_hdb


def resumen_grupos(tablas,tiene_hdb):
    resultados=[]
    for parte,d in tablas.items():
        n=len(d)
        k=int(d.grupo_kmeans.nunique())
        resultados.append(dict(conjunto=parte,algoritmo='K-Means',registros=n,
                               grupos=k,ruido_n=np.nan,ruido_pct=np.nan,
                               alerta_n=np.nan,alerta_pct=np.nan,
                               asignados_a_grupo=n,asignados_pct=100.0))
        for nombre,col,ruido_col in [('DBSCAN','cluster_dbscan','alerta_ruido_dbscan')]+(
            [('HDBSCAN','cluster_hdbscan','ruido_hdbscan')] if tiene_hdb else []):
            ruido=int(d[ruido_col].sum()); ncl=int(d.loc[d[col]>=0,col].nunique())
            resultados.append(dict(conjunto=parte,algoritmo=nombre,registros=n,
                                   grupos=ncl,ruido_n=ruido,ruido_pct=100*ruido/n,
                                   alerta_n=np.nan,alerta_pct=np.nan,
                                   asignados_a_grupo=n-ruido,asignados_pct=100*(n-ruido)/n))
        alertas=int(d.alerta_isolation_forest.sum())
        resultados.append(dict(conjunto=parte,algoritmo='Isolation Forest',registros=n,
                               grupos=np.nan,ruido_n=np.nan,ruido_pct=np.nan,
                               alerta_n=alertas,alerta_pct=100*alertas/n,
                               asignados_a_grupo=np.nan,asignados_pct=np.nan))
    return pd.DataFrame(resultados)


def coincidencia(a,b):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    inter=int(np.count_nonzero(a&b)); union=int(np.count_nonzero(a|b))
    return inter,union,float(inter/union) if union else np.nan


def concordancias(tablas,tiene_hdb):
    filas=[]
    pair=[('DBSCAN','alerta_ruido_dbscan','Isolation Forest','alerta_isolation_forest')]
    if tiene_hdb:
        pair += [('HDBSCAN','ruido_hdbscan','Isolation Forest','alerta_isolation_forest'),
                 ('DBSCAN','alerta_ruido_dbscan','HDBSCAN','ruido_hdbscan')]
    for parte,d in tablas.items():
        for a,ca,b,cb in pair:
            sa=d[ca].to_numpy(dtype=bool);sb=d[cb].to_numpy(dtype=bool)
            inter,union,jacc=coincidencia(sa,sb)
            filas.append(dict(conjunto=parte,bandera_A=a,bandera_B=b,n=len(d),
                              A_total=int(sa.sum()),B_total=int(sb.sum()),
                              coinciden_ambas=inter,union_total=union,
                              jaccard=jacc,
                              porcentaje_de_A_que_tambien_B=(100*inter/sa.sum() if sa.sum() else np.nan),
                              porcentaje_de_B_que_tambien_A=(100*inter/sb.sum() if sb.sum() else np.nan),
                              advertencia='Coincidencia de criterios estadisticos; NO precision de fallas'))
    return pd.DataFrame(filas)


def tabla_grupos_vs_alertas(tablas):
    arr=[]
    for parte,d in tablas.items():
        for gid,g in d.groupby('grupo_kmeans',sort=True):
            arr.append(dict(conjunto=parte,grupo_kmeans=int(gid),registros=len(g),
                            alertas_IF=int(g.alerta_isolation_forest.sum()),
                            tasa_alertas_IF_pct=100*float(g.alerta_isolation_forest.mean()),
                            ruido_DBSCAN_pct=100*float(g.alerta_ruido_dbscan.mean()),
                            ruido_HDBSCAN_pct=100*float(g.ruido_hdbscan.mean()) if 'ruido_hdbscan' in g else np.nan))
    return pd.DataFrame(arr)


def bloques_smp(tablas,tiene_hdb,ancho=2000):
    filas=[]
    for parte,d in tablas.items():
        # Bloques de filas consecutivas, NO minutos ni segundos.
        for j,inicio in enumerate(range(0,len(d),ancho)):
            g=d.iloc[inicio:inicio+ancho]
            fila=dict(conjunto=parte,bloque_en_conjunto=j+1,indice_global_inicio=int(inicio),
                      smp_inicio=int(g[SMP].iloc[0]),smp_fin=int(g[SMP].iloc[-1]),
                      n=len(g),IF_alertas_pct=100*g.alerta_isolation_forest.mean(),
                      DBSCAN_ruido_pct=100*g.alerta_ruido_dbscan.mean(),
                      mediana_puntaje_IF=float(g.puntaje_anomalia.median()))
            if tiene_hdb:fila['HDBSCAN_ruido_pct']=100*g.ruido_hdbscan.mean()
            filas.append(fila)
    return pd.DataFrame(filas)


def figura_tasas(resumen, destino):
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.9),constrained_layout=True)
    x=np.arange(3);lab=['Entrenamiento','Validación','Prueba']
    d0=resumen[resumen.algoritmo=='Isolation Forest'].set_index('conjunto').loc[list(PARTES)]
    axes[0].plot(x,d0.alerta_pct,marker='o',lw=2.5,color=COLOR['Isolation Forest'])
    axes[0].set(xticks=x,xticklabels=lab,title='Isolation Forest | alertas (%)',ylabel='Porcentaje (%)')
    axes[0].set_ylim(bottom=0)
    for nombre in ['DBSCAN','HDBSCAN']:
        d=resumen[resumen.algoritmo==nombre]
        if len(d):
            d=d.set_index('conjunto').loc[list(PARTES)]
            axes[1].plot(x,d.ruido_pct,marker='o',lw=2.5,color=COLOR[nombre],label=nombre)
    axes[1].set(xticks=x,xticklabels=lab,title='Densidad | registros sin grupo (%)',ylabel='Porcentaje (%)')
    axes[1].set_ylim(bottom=0,top=105)
    axes[1].legend(frameon=False)
    fig.suptitle('Criterios diferentes: las magnitudes NO son tasas de falla',fontsize=12,fontweight='bold')
    grafico(fig,destino)


def figura_jaccard(concordancia,destino):
    parejas=concordancia[['bandera_A','bandera_B']].drop_duplicates().apply(
        lambda r:f'{r.bandera_A} / {r.bandera_B}',axis=1).tolist()
    mat=np.full((len(parejas),3),np.nan)
    for i,par in enumerate(parejas):
        a,b=par.split(' / ')
        for j,p in enumerate(PARTES):
            d=concordancia[(concordancia.bandera_A==a)&(concordancia.bandera_B==b)&(concordancia.conjunto==p)]
            if len(d):mat[i,j]=float(d.jaccard.iloc[0])
    fig,ax=plt.subplots(figsize=(8.5,max(3.4,len(parejas)*1.1+2)),constrained_layout=True)
    im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(3),['Entrenamiento','Validación','Prueba'])
    ax.set_yticks(range(len(parejas)),parejas)
    for i in range(len(parejas)):
        for j in range(3):
            v=mat[i,j]
            ax.text(j,i,'—' if not np.isfinite(v) else f'{v:.3f}',ha='center',va='center',
                    color='white' if np.isfinite(v) and v>0.56 else '#20252A',fontweight='bold')
    fig.colorbar(im,ax=ax,label='Jaccard (coincidencia de banderas)')
    ax.set_title('Concordancia estadística entre ruido y alertas\nNO mide precisión diagnóstica')
    grafico(fig,destino)


def figura_smp(bloques,destino,tiene_hdb):
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharey=True,constrained_layout=True)
    for ax,parte in zip(axs,PARTES):
        d=bloques[bloques.conjunto==parte]
        ax.plot(d.bloque_en_conjunto,d.IF_alertas_pct,label='IF: alertas',lw=2,color=COLOR['Isolation Forest'])
        ax.plot(d.bloque_en_conjunto,d.DBSCAN_ruido_pct,label='DBSCAN: ruido',lw=1.6,color=COLOR['DBSCAN'])
        if tiene_hdb:ax.plot(d.bloque_en_conjunto,d.HDBSCAN_ruido_pct,label='HDBSCAN: ruido',lw=1.6,color=COLOR['HDBSCAN'])
        ax.set(ylabel='Registros (%)',title=parte.capitalize())
        ax.set_ylim(-3,103)
    axs[0].legend(ncol=3,frameon=False,loc='upper center')
    axs[-1].set_xlabel('Bloques consecutivos de 2.000 registros (no tiempo)')
    fig.suptitle('Cambios descriptivos a lo largo de la adquisición\nAlertas y ruido son criterios distintos',fontweight='bold')
    grafico(fig,destino)


def figura_kmeans_cruces(cruces,destino):
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.5),sharey=True,constrained_layout=True)
    for ax,parte in zip(axes,PARTES):
        d=cruces[cruces.conjunto==parte]
        ids=d.grupo_kmeans.to_numpy()
        ax.bar([str(int(x)) for x in ids],d.tasa_alertas_IF_pct,color=COLOR['Isolation Forest'],alpha=.85)
        ax.set_title(parte.capitalize());ax.set_xlabel('Grupo K-Means')
        ax.set_ylim(0,100)
        for j,v in enumerate(d.tasa_alertas_IF_pct):
            ax.text(j,v+2,f'{v:.1f}%',ha='center',fontsize=8)
    axes[0].set_ylabel('Alertas IF dentro del grupo (%)')
    fig.suptitle('Relación descriptiva: agrupamiento K-Means y alertas IF\nNO son etiquetas de falla',fontweight='bold')
    grafico(fig,destino)


def figura_puntajes(tablas,destino):
    fig,ax=plt.subplots(figsize=(10.5,5),constrained_layout=True)
    for parte,color in zip(PARTES,['#367FA4','#C18B49','#886CBA']):
        d=tablas[parte]
        ax.hist(d.puntaje_anomalia.to_numpy(),bins=55,density=True,histtype='step',
                lw=2,color=color,label=parte.capitalize())
    ax.set(xlabel='Puntuación IF: mayor = más atípico',ylabel='Densidad',
           title='Distribuciones de puntuaciones por partición cronológica')
    ax.legend(frameon=False)
    grafico(fig,destino)


def leer_shap(base):
    shapfile=base/'06_shap'/'02_importancias_shap_variables.csv'
    if not shapfile.is_file():return None
    d=leer(shapfile)
    requeridas=['senal','participacion_importancia_pct']
    if any(c not in d for c in requeridas) or len(d)!=10:
        raise ValueError('SHAP: se esperan las 10 señales con importancia relativa')
    if not np.isfinite(d.participacion_importancia_pct.to_numpy(dtype=float)).all():
        raise ValueError('SHAP contiene importancias no finitas')
    return d.sort_values('participacion_importancia_pct',ascending=False).reset_index(drop=True)


def figura_shap(d,destino):
    fig,ax=plt.subplots(figsize=(9.3,5.5),constrained_layout=True)
    y=np.arange(len(d))[::-1]
    nombres=d.sensor_corto if 'sensor_corto' in d else d.senal
    ax.barh(y,d.participacion_importancia_pct,color='#5F88A8')
    ax.set_yticks(y,list(nombres));ax.set_xlabel('Participación en importancia SHAP (%)')
    ax.set_title('Interpretación del mismo Isolation Forest (muestra explicada)')
    grafico(fig,destino)


def informe(resumen,conc,cruces,shap,tiene_hdb,karchivo,base):
    lineas=['# Evaluación comparativa — diez señales en un modelo por método','',
            '## Alcance y trazabilidad',
            '- 143.561 registros: 100.492 entrenamiento, 21.534 validación, 21.535 prueba, en ese orden.',
            '- El código 15 NO entrena ni modifica datos. Une resultados por SMP (identificador, no segundos).',
            f'- K-Means: {karchivo.name} (únicamente escenario de las diez señales).',
            f'- HDBSCAN final incorporado: {"sí" if tiene_hdb else "NO; solo comparación parcial"}.',
            f'- SHAP incorporado: {"sí" if shap is not None else "no disponible"}.','',
            '## Resultados por conjunto (agrupamientos vs anomalías)']
    for p in PARTES:
        lineas.append(f'### {p.capitalize()}')
        for r in resumen[resumen.conjunto==p].itertuples():
            if r.algoritmo=='K-Means':
                lineas.append(f'- K-Means: {int(r.grupos)} grupos; asigna todos los registros.')
            elif r.algoritmo=='Isolation Forest':
                lineas.append(f'- Isolation Forest: {int(r.alerta_n):,} alertas ({r.alerta_pct:.2f} %).')
            else:
                lineas.append(f'- {r.algoritmo}: {int(r.grupos)} grupos; {int(r.ruido_n):,} '
                              f'en ruido ({r.ruido_pct:.2f} %).')
        lineas.append('')
    lineas+=['## Coincidencia de banderas',
             'El índice Jaccard mide la intersección / unión de dos banderas. No existe verdad de fallas.']
    for row in conc.itertuples():
        lineas.append(f'- {row.conjunto}: {row.bandera_A} vs {row.bandera_B}: '
                      f'Jaccard {row.jaccard:.3f} ({int(row.coinciden_ambas):,} coincidencias).')
    lineas+=['','## Interpretabilidad SHAP']
    if shap is None:lineas.append('No se encontró el CSV SHAP final; ejecutar 14_shap.py --modo final.')
    else:
        for r in shap.head(5).itertuples():
            lineas.append(f'- {r.senal}: {r.participacion_importancia_pct:.2f} % de importancia absoluta ponderada.')
        lineas.append('Las importancias provienen de una muestra estratificada, no de 143.561 explicaciones.')
    lineas+=['','## Límites que deben presentarse al profesor',
             '- Sin etiquetas verificadas de falla no se calculan precisión, recall, F1, ROC-AUC ni exactitud diagnóstica.',
             '- Los métodos no comparten una definición de anomalía: no escoger por el porcentaje más pequeño.',
             '- El ruido de DBSCAN/HDBSCAN no es una avería confirmada ni se iguala a la alerta de IF.',
             '- La silueta de K-Means mide cohesión/separación geométrica, no detección de fallas.',
             '- DBSCAN extrapola a nuevos registros desde núcleos fijados en entrenamiento; '
             'HDBSCAN utiliza approximate_predict; estas asignaciones no reentrenan la jerarquía.',
             '- La evaluación temporal es descriptiva; una frecuencia SMP no identifica minutos, horas ni periodos de mantenimiento.',
             '- SHAP explica predicciones de IF, no causalidad mecánica.',
             '- No se establecen umbrales de mantenimiento sin especificaciones del equipo y verificación de eventos.',
             '','## Próxima validación experimental',
             'Solicitar bitácora de mantenimiento, fallas verificadas y condiciones de operación; '
             'validar si las alertas anticipan eventos reales y evaluar falsas alarmas y costos.']
    (base/'00_lectura_para_informe.md').write_text('\n'.join(lineas)+'\n',encoding='utf-8')
    print('  [INFORME] 00_lectura_para_informe.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modo',choices=['comprobar','final'],default='comprobar')
    p.add_argument('--resultados',type=Path,default=RAIZ/'Resultados_Tarea3_Nuevos')
    p.add_argument('--salida',type=Path,default=None)
    args=p.parse_args()
    base=args.resultados.expanduser()
    destino=args.salida or base/'07_evaluacion_comparativa'
    print('15 | Comparacion de resultados SIN etiquetas de falla',flush=True)
    print('Leyendo:',base.resolve(),flush=True)
    tablas,karchivo,tiene_hdb=cargar(base)
    shap=leer_shap(base)
    print('Datos consistentes por SMP:',', '.join(f'{k}={len(v):,}' for k,v in tablas.items()),flush=True)
    print('K-Means: diez señales simultáneas, fuente:',karchivo.name,flush=True)
    print('HDBSCAN final:', 'ENCONTRADO' if tiene_hdb else 'NO ENCONTRADO (comparación parcial)',flush=True)
    print('SHAP:', 'ENCONTRADO' if shap is not None else 'NO ENCONTRADO',flush=True)
    if args.modo=='comprobar':
        print('COMPROBACION OK. Para generar: python -u 15_evaluacion_comparativa.py --modo final',flush=True)
        return
    destino.mkdir(parents=True,exist_ok=True)
    resumen=resumen_grupos(tablas,tiene_hdb)
    concordancia=concordancias(tablas,tiene_hdb)
    cruces=tabla_grupos_vs_alertas(tablas)
    bloques=bloques_smp(tablas,tiene_hdb)
    exportar(resumen,destino/'00_resumen_modelos_por_conjunto.csv')
    exportar(concordancia,destino/'01_concordancia_estadistica.csv')
    exportar(cruces,destino/'02_alertas_por_grupo_kmeans.csv')
    exportar(bloques,destino/'03_evolucion_por_bloques_smp.csv')
    for parte,d in tablas.items():
        exportar(d,destino/f'04_resultados_alineados_{parte}.csv')
    figura_tasas(resumen,destino/'01_comparacion_criterios_por_conjunto.png')
    figura_jaccard(concordancia,destino/'02_mapa_concordancias_jaccard.png')
    figura_smp(bloques,destino/'03_evolucion_alertas_ruido_smp.png',tiene_hdb)
    figura_kmeans_cruces(cruces,destino/'04_kmeans_vs_alertas_if.png')
    figura_puntajes(tablas,destino/'05_distribucion_puntajes_if.png')
    if shap is not None:
        exportar(shap,destino/'05_resumen_importancia_shap.csv')
        figura_shap(shap,destino/'06_importancia_shap_para_comparacion.png')
    informe(resumen,concordancia,cruces,shap,tiene_hdb,karchivo,destino)
    config={'division':TAMANOS, 'tiene_hdbscan_final':tiene_hdb,
            'tiene_shap_final':shap is not None, 'kmeans_archivo':karchivo.name,
            'registros':sum(TAMANOS.values()),'variables_por_modelo':10,
            'limite':'sin etiquetas de fallas; coincidencia estadistica, no rendimiento diagnostico'}
    (destino/'00_metodologia_comparativa.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nCOMPARACION TERMINADA |',destino.resolve(),flush=True)
    print(resumen[['conjunto','algoritmo','grupos','ruido_pct','alerta_pct']].to_string(index=False),flush=True)
    if not tiene_hdb:
        print('ADVERTENCIA: falta HDBSCAN final. El archivo de barrido NO sustituye las etiquetas por registro.',flush=True)
    print('ADVERTENCIA: ruido, alertas y grupos NO son averias verificadas.',flush=True)


if __name__=='__main__':
    try:main()
    except (FileNotFoundError,ValueError) as e:
        print('ERROR DE VALIDACION:',e,file=sys.stderr,flush=True)
        sys.exit(2)
