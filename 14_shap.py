# -*- coding: utf-8 -*-
"""14 | Explicabilidad SHAP para el UNICO Isolation Forest multivariable del codigo 13.

REQUISITOS:
    python -m pip install numpy pandas matplotlib scikit-learn joblib shap

EJECUCION (carpeta junto a los scripts 08, 13):
    python -u 14_shap.py                         # solo comprueba, NO inicia un calculo largo
    python -u 14_shap.py --modo final            # tres conjuntos, 10 senales a la vez
    python -u 14_shap.py --modo final --alertas 40 --normales 40 --fondo 40 --permutaciones 5
    python -u 14_shap.py --modo autotest

METODOLOGIA:
* NO reentrena Isolation Forest; carga el modelo ajustado en el codigo 13.
* Confirma que las 10 variables, el escalado, el orden SMP y los puntajes
  guardados coinciden con el modelo de la carpeta 05_isolation_forest.
* Explica directamente el score de anomalia = -modelo.score_samples(X).
  SHAP positivo significa que esa senal aumenta el score respecto al fondo;
  SHAP negativo lo disminuye. NO es una probabilidad ni causalidad fisica.
* El modelo 13 ya uso todos los 100492 registros de entrenamiento. SHAP usa
  una MUESTRA ESTRATIFICADA de cada conjunto por costo de computo.
* El fondo proviene solo de ENTRENAMIENTO; no se usa prueba para construirlo.
* El muestreo de explicaciones incluye el mayor score de prueba para el
  ejemplo individual; un porcentaje de muestras balanceado NO representa
  automaticamente el porcentaje de alertas del universo. La importancia
  general se estima reponderando por prevalencia real de alertas en cada
  particion (y numero de registros entre particiones).
* SMP es un contador de adquisiciones, NO tiempo en segundos ni predictor.
* SHAP explica la prediccion del modelo, NO diagnostica averias.

SALIDA: Resultados_Tarea3_Nuevos/06_shap
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except ImportError as exc:
    raise SystemExit('Falta shap: ejecuta python -m pip install shap') from exc

RAIZ = Path(__file__).resolve().parent
PARTES = ('entrenamiento', 'validacion', 'prueba')
SMP = 'smp_continuo'
SEMILLA = 42
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z', 'Ma X', 'Ma Y', 'Ma Z', 'Temp']
FAMILIAS = ['Acelerometro'] * 3 + ['Giroscopio'] * 3 + ['Magnetometro'] * 3 + ['Temperatura']
COLORES = {'entrenamiento': '#2679A7', 'validacion': '#9C7AB6', 'prueba': '#D98043'}

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'figure.facecolor':'white', 'savefig.facecolor':'white',
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.17})


def leer_csv(ruta: Path):
    if not ruta.is_file():
        raise FileNotFoundError(f'No se encuentra: {ruta}')
    return pd.read_csv(ruta, sep=';', encoding='utf-8-sig')


def csv(tabla, ruta: Path):
    pd.DataFrame(tabla).to_csv(ruta, sep=';', encoding='utf-8-sig', index=False,
                               float_format='%.12g')
    print(f'  [CSV] {ruta.name}',flush=True)


def fig_guardar(fig, ruta: Path):
    fig.savefig(ruta, dpi=205, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [GRAFICO] {ruta.name}',flush=True)


def preflight(preparados: Path, if_dir: Path):
    path_conf = if_dir/'00_configuracion_modelo.json'
    path_meta = preparados/'00_metadatos.json'
    if not path_conf.is_file() or not path_meta.is_file():
        raise FileNotFoundError('Falta la configuracion 13 o los metadatos 08. Ejecuta 08 y luego 13 en modo final.')
    conf = json.loads(path_conf.read_text(encoding='utf-8'))
    meta = json.loads(path_meta.read_text(encoding='utf-8'))
    variables = meta['variables']
    if len(variables)!=10 or len(set(variables))!=10 or conf.get('variables') != variables:
        raise ValueError('Las 10 variables del modelo 13 no coinciden con las preparadas en 08.')
    escalado = conf.get('escalado')
    if escalado not in ('estandar', 'robusto'):
        raise ValueError('Escalado no reconocido en el modelo 13.')
    if conf.get('algoritmo') != 'IsolationForest' or conf.get('entrenado_unicamente_en') != 'entrenamiento cronologico':
        raise ValueError('La configuracion de 13 no corresponde al modelo de este trabajo.')
    ruta_modelo = if_dir/'00_modelo_isolation_forest.joblib'
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f'No existe {ruta_modelo}. Ejecuta el codigo 13 con --modo final.')
    # Solamente cargar archivos .joblib creados por ti: joblib usa pickle.
    modelo = joblib.load(ruta_modelo)
    if not hasattr(modelo,'score_samples') or getattr(modelo,'n_features_in_',None)!=10:
        raise ValueError('El modelo guardado no corresponde a IsolationForest de diez variables.')
    if len(modelo.estimators_)!=conf['n_estimators'] or int(modelo.max_samples_) != int(conf['max_samples_real_por_arbol']):
        raise ValueError('El numero de arboles/submuestras del modelo no coincide con la configuracion.')

    cols = ['z_'+nombre for nombre in variables]
    grupos = {}
    for p in PARTES:
        archivo = leer_csv(preparados/f'01_{p}_{escalado}.csv')
        tabla = leer_csv(if_dir/f'01_resultado_{p}.csv')
        if list(archivo.columns) != [SMP,*cols]:
            raise ValueError(f'Las columnas de {p} no corresponden a 08 y estan desordenadas.')
        if any(c not in tabla.columns for c in [SMP,'puntaje_anomalia','alerta_isolation_forest']):
            raise ValueError(f'Faltan resultados de 13 en {p}.')
        s = archivo[SMP].to_numpy(np.int64)
        s_if = tabla[SMP].to_numpy(np.int64)
        if len(s)!=conf['division'][p] or not np.array_equal(s,s_if):
            raise ValueError(f'SMP / cantidad de filas de {p} no coinciden entre 08 y 13.')
        X = np.ascontiguousarray(archivo[cols].to_numpy(dtype=np.float32))
        if X.shape[1]!=10 or not np.isfinite(X).all():
            raise ValueError(f'Senales ausentes/no finitas en {p}.')
        pts = tabla['puntaje_anomalia'].to_numpy(np.float64)
        alertas = tabla['alerta_isolation_forest'].to_numpy(np.int8).astype(bool)
        if not set(np.unique(tabla['alerta_isolation_forest'])).issubset({0,1}):
            raise ValueError(f'Etiquetas de alerta incorrectas en {p}.')
        # Detecta codigo 13 regenerado/desincronizado respecto a su modelo.
        check = np.unique(np.linspace(0,len(s)-1,min(15,len(s)),dtype=int))
        pred = -modelo.score_samples(X[check])
        err = float(np.max(np.abs(pred-pts[check])))
        umbral = float(conf['umbral_score_mayor_mas_atipico'])
        if err > 2e-6 or not np.array_equal(pred>umbral,alertas[check]):
            raise ValueError(f'Modelo y puntuaciones del codigo 13 NO coinciden en {p} (error={err:.3g}). '
                             'Vuelve a ejecutar 13 --modo final con la configuracion elegida.')
        grupos[p] = {'X':X, 'smp':s, 'score':pts, 'alerta':alertas}
    if not (grupos['entrenamiento']['smp'][-1] < grupos['validacion']['smp'][0]
            < grupos['prueba']['smp'][0]):
        raise ValueError('La particion cronologica se ha roto.')
    if tuple(len(grupos[k]['smp']) for k in PARTES)!=(100492,21534,21535):
        raise ValueError('No se conserva la division completa de 143561 registros.')
    print(f'[DATOS] 143.561 registros, 10 senales; escalado {escalado}.',flush=True)
    print(f'[MODELO 13] {len(modelo.estimators_)} arboles; {modelo.max_samples_} muestras internas/arbol.', flush=True)
    print('[VERIFICACION] Orden SMP, columnas, clasificaciones y puntuaciones: OK.',flush=True)
    return grupos, conf, meta, modelo


def seleccion_por_conjunto(grupo, n_alertas, n_normales, rng, priorizar_extremo=False):
    alerta=np.flatnonzero(grupo['alerta'])
    normal=np.flatnonzero(~grupo['alerta'])
    seleccion=[]
    for indices,n,incluir_extremo in [(alerta,n_alertas,priorizar_extremo),
                                       (normal,n_normales,priorizar_extremo)]:
        n=min(n,len(indices))
        if n<=0: continue
        obligatorio = None
        if incluir_extremo:
            if np.array_equal(indices,alerta) and len(alerta):
                obligatorio=int(indices[np.argmax(grupo['score'][indices])])
            elif len(indices):
                orden=np.argsort(grupo['score'][indices],kind='stable')
                obligatorio=int(indices[orden[len(orden)//2]])
        if obligatorio is None:
            elegidos=rng.choice(indices,size=n,replace=False)
        else:
            otros=indices[indices!=obligatorio]
            elegidos=np.r_[obligatorio, rng.choice(otros,size=n-1,replace=False)]
        seleccion.extend(int(x) for x in elegidos)
    if not seleccion: raise ValueError('No se pudo obtener ninguna fila para SHAP.')
    return np.array(sorted(seleccion),dtype=np.int64)


def explicar(grupos, modelo, vars_orig, alertas_objetivo, normales_objetivo,
              fondo_n, lote, semillas, salida, permutaciones):
    rng=np.random.default_rng(semillas)
    train=grupos['entrenamiento']['X']
    fondo_idx=rng.choice(len(train),size=min(fondo_n,len(train)),replace=False)
    fondo=train[fondo_idx].astype(np.float64)
    fondo_smp=grupos['entrenamiento']['smp'][fondo_idx]
    # Se usan fondos del entrenamiento; NO datos de validacion o prueba.
    csv(pd.DataFrame({SMP:fondo_smp}),salida/'00_smp_referencias_fondo.csv')
    def funcion(matriz):
        matriz=np.asarray(matriz,dtype=np.float32)
        if matriz.ndim==1: matriz=matriz.reshape(1,-1)
        return -modelo.score_samples(matriz)
    print(f'[SHAP] Fondo de {len(fondo)} filas SOLO del entrenamiento.',flush=True)
    mascara=shap.maskers.Independent(fondo,max_samples=len(fondo))
    explainer=shap.Explainer(funcion, mascara, algorithm='permutation',
                             feature_names=vars_orig, seed=semillas)
    max_evals=(2*len(vars_orig)+1)*permutaciones  # varias permutaciones; aproximacion mas estable
    rows=[]
    inicio=time.perf_counter()
    for p in PARTES:
        G=grupos[p]
        idx=seleccion_por_conjunto(G,alertas_objetivo,normales_objetivo,rng,
                                   priorizar_extremo=p=='prueba')
        print(f'\n[EXPLICACION] {p.upper()}: {len(idx)} filas de {len(G["X"]):,} '
              f'({G["alerta"][idx].sum()} alertas y {(~G["alerta"][idx]).sum()} sin alerta).',flush=True)
        for lo in range(0,len(idx),lote):
            sub=idx[lo:lo+lote]
            t=time.perf_counter()
            exp=explainer(G['X'][sub].astype(np.float64),max_evals=max_evals,
                           batch_size=128,silent=True)
            valor=np.asarray(exp.values,dtype=np.float64)
            base=np.broadcast_to(np.asarray(exp.base_values,dtype=np.float64),len(sub))
            reconstruida=base+valor.sum(axis=1)
            if valor.shape!=(len(sub),10) or not np.isfinite(valor).all():
                raise ValueError('SHAP entrego dimensiones o contribuciones invalidas.')
            actual=G['score'][sub]
            maxerr=float(np.max(np.abs(reconstruida-actual)))
            if maxerr>2e-5:
                raise ValueError(f'Falla de aditividad SHAP: error maximo {maxerr:.3g}')
            for j,idxreal in enumerate(sub):
                row={'conjunto':p,'indice_en_conjunto':int(idxreal),
                     SMP:int(G['smp'][idxreal]),'alerta_isolation_forest':int(G['alerta'][idxreal]),
                     'puntaje_anomalia':float(actual[j]),
                     'shap_base':float(base[j]),
                     'shap_suma':float(valor[j].sum()),
                     'shap_reconstruido':float(reconstruida[j]),
                     'error_aditividad':float(abs(reconstruida[j]-actual[j]))}
                for k,nom in enumerate(vars_orig):
                    row[f'shap_{nom}']=float(valor[j,k])
                    row[f'z_{nom}']=float(G['X'][idxreal,k])
                rows.append(row)
            print(f'  {min(lo+len(sub),len(idx))}/{len(idx)} | '
                  f'{time.perf_counter()-t:.1f}s lote | '
                  f'{time.perf_counter()-inicio:.1f}s acumulados | '
                  f'error aditividad {maxerr:.2e}',flush=True)
            # Recuperacion en caso de interrupcion: resultado parcial (NO analisis final).
            pd.DataFrame(rows).to_csv(salida/'01_explicaciones_parciales.csv',sep=';',
                                      index=False,encoding='utf-8-sig')
    df=pd.DataFrame(rows)
    csv(df,salida/'01_explicaciones_shap.csv')
    parcial=salida/'01_explicaciones_parciales.csv'
    if parcial.exists(): parcial.unlink()
    return df, fondo, funcion


def calcular_importancias(df,grupos,variables,salida):
    cols=['shap_'+n for n in variables]
    vals=df[cols].to_numpy(np.float64)
    alertas=df['alerta_isolation_forest'].to_numpy(bool)
    conjunto=df['conjunto'].to_numpy()
    filas=[]
    pesos_generales=np.zeros(len(df),dtype=np.float64)
    por_conjunto=[]
    # Muestreo estratificado deliberado: reponderar por fraccion de alertas
    # de TODO el conjunto y, entre particiones, por su tamano real.
    Ntotal=sum(len(grupos[p]['X']) for p in PARTES)
    for p in PARTES:
        target=conjunto==p
        N=len(grupos[p]['X'])
        prev=float(grupos[p]['alerta'].mean())
        estimador=np.zeros(10)
        for estado,proporcion in [(True,prev),(False,1.0-prev)]:
            mask=target & (alertas==estado)
            if not np.any(mask) and proporcion>0:
                raise ValueError(f'Falta estrato requerido en {p} estado alerta={estado}')
            if mask.any():
                # Cada registro muestral representa la misma fraccion de su estrato.
                pesos_generales[mask]=(N/Ntotal)*proporcion/mask.sum()
        pesos_p=pesos_generales.copy()
        pesos_p[~target]=0
        if pesos_p.sum() <= 0: raise ValueError('Error de pesos estratificados.')
        pesos_p/=pesos_p.sum()
        abs_p=(np.abs(vals)*pesos_p[:,None]).sum(axis=0)
        for j,variable in enumerate(variables):
            por_conjunto.append({'conjunto':p,'senal':variable,'sensor_corto':CORTOS[j],
                'familia':FAMILIAS[j], 'importancia_abs_ponderada':float(abs_p[j]),
                'tasa_alertas_real_pct':100*prev,
                'registros_totales':N,'registros_explicados':int(target.sum())})
    pesos_generales/=pesos_generales.sum()
    media_abs=(np.abs(vals)*pesos_generales[:,None]).sum(axis=0)
    media_firmada=(vals*pesos_generales[:,None]).sum(axis=0)
    suma=float(media_abs.sum())
    for j,variable in enumerate(variables):
        mask_a=alertas
        mask_n=~alertas
        filas.append({'senal':variable,'sensor_corto':CORTOS[j], 'familia':FAMILIAS[j],
                      'importancia_abs_ponderada':float(media_abs[j]),
                      'aporte_firmado_ponderado':float(media_firmada[j]),
                      'participacion_importancia_pct':float(100*media_abs[j]/suma) if suma else 0.0,
                      'media_abs_alertas_muestra':float(np.mean(np.abs(vals[mask_a,j]))) if mask_a.any() else np.nan,
                      'media_abs_sin_alerta_muestra':float(np.mean(np.abs(vals[mask_n,j]))) if mask_n.any() else np.nan})
    ranking=pd.DataFrame(filas).sort_values('importancia_abs_ponderada',ascending=False)
    por_conjunto=pd.DataFrame(por_conjunto)
    familias=ranking.groupby('familia',as_index=False).agg(
        importancia_abs_ponderada=('importancia_abs_ponderada','sum'),
        participacion_importancia_pct=('participacion_importancia_pct','sum'))
    familias=familias.sort_values('importancia_abs_ponderada',ascending=False)
    csv(ranking,salida/'02_importancias_shap_variables.csv')
    csv(por_conjunto,salida/'02_importancias_shap_por_conjunto.csv')
    csv(familias,salida/'03_resumen_familias_en_modelo_unico.csv')
    resumen=[]
    for p in PARTES:
        n=len(grupos[p]['X']); nn=int(grupos[p]['alerta'].sum()); exp=df[df.conjunto==p]
        resumen.append({'conjunto':p,'registros_totales':n,'alertas_totales':nn,
                        'alertas_pct':100*nn/n,'registros_explicados':len(exp),
                        'alertas_explicadas':int(exp.alerta_isolation_forest.sum()),
                        'sin_alerta_explicados':int((1-exp.alerta_isolation_forest).sum()),
                        'error_maximo_aditividad':float(exp.error_aditividad.max())})
    csv(resumen,salida/'00_resumen_shap.csv')
    return ranking,por_conjunto,familias,pesos_generales


def graficar(df,ranking,por_conjunto,familias,variables,pesos,salida):
    cols=['shap_'+n for n in variables]
    vals=df[cols].to_numpy(float)
    X=df[['z_'+n for n in variables]].to_numpy(float)
    orden=ranking['sensor_corto'].to_list()
    valores=ranking['importancia_abs_ponderada'].to_numpy(float)

    # 01 - importancias ponderadas globales de DIEZ variables del MISMO modelo.
    fig,ax=plt.subplots(figsize=(10.0,6.0))
    pos=np.arange(len(orden))
    ax.barh(pos,valores[::-1],color='#376F99')
    ax.set_yticks(pos,orden[::-1])
    ax.set(xlabel='Media ponderada de |SHAP| (unidades de puntuacion)',
           title='Isolation Forest unico | importancia aproximada por senal')
    ax.text(.01,.01,'Muestra estratificada reponderada; NO causalidad ni fallas confirmadas',
            transform=ax.transAxes,fontsize=8,color='#565656')
    fig_guardar(fig,salida/'01_importancia_shap_10_senales.png')

    # 02 - beeswarm manual: contribucion firmada por senal, color=valor escalado.
    fig,ax=plt.subplots(figsize=(11.2,6.9))
    rng=np.random.default_rng(SEMILLA)
    lim_color=3.0
    for row,j in enumerate(np.argsort(-np.average(np.abs(vals),axis=0,weights=pesos))):
        jitter=rng.normal(0,.10,size=len(vals))
        ax.scatter(vals[:,j],np.full(len(vals),row)+jitter,
                   c=np.clip(X[:,j],-lim_color,lim_color),vmin=-lim_color,vmax=lim_color,
                   cmap='coolwarm',s=15,alpha=.68,edgecolors='none',rasterized=True)
    ax.axvline(0,color='#555555',lw=1)
    ax.set_yticks(np.arange(10),np.array(CORTOS)[np.argsort(-np.average(np.abs(vals),axis=0,weights=pesos))])
    ax.invert_yaxis()
    ax.set(xlabel='SHAP firmado (positivo = mayor anomalia respecto del fondo)',
           title='Distribucion local de SHAP | solo registros explicados')
    norm=plt.Normalize(-lim_color,lim_color)
    sm=plt.cm.ScalarMappable(cmap='coolwarm',norm=norm);sm.set_array([])
    cbar=fig.colorbar(sm,ax=ax,pad=.01,shrink=.74)
    cbar.set_label('Valor de senal escalado (recorte visual ±3)')
    fig_guardar(fig,salida/'02_distribucion_shap_tipo_beeswarm.png')

    # 03 - por conjunto, SIN cambiar de modelo.
    data=por_conjunto.pivot(index='sensor_corto',columns='conjunto',values='importancia_abs_ponderada')
    ordlabels=ranking['sensor_corto'].tolist()[::-1]
    ypos=np.arange(len(ordlabels)); alto=.25
    fig,ax=plt.subplots(figsize=(11.2,6.0))
    for i,p in enumerate(PARTES):
        ax.barh(ypos+(i-1)*alto,data.loc[ordlabels,p].to_numpy(),height=alto,
                label=p.capitalize(),color=COLORES[p])
    ax.set_yticks(ypos,ordlabels)
    ax.set(xlabel='Media |SHAP| reponderada dentro de cada conjunto',
           title='Contribuciones por particion | un mismo modelo ajustado en entrenamiento')
    ax.legend(frameon=False)
    fig_guardar(fig,salida/'03_importancia_por_conjunto.png')

    # 04 - alertas vs sin alerta (solo muestra; NO prevalencia natural).
    ypos=np.arange(len(ordlabels));alto=.35
    ref=ranking.set_index('sensor_corto').loc[ordlabels]
    fig,ax=plt.subplots(figsize=(11.1,6.0))
    ax.barh(ypos-alto/2,ref['media_abs_alertas_muestra'],height=alto,
            color='#C65E61',label='Alertas explicadas (muestra)')
    ax.barh(ypos+alto/2,ref['media_abs_sin_alerta_muestra'],height=alto,
            color='#477B9D',label='Sin alerta explicadas (muestra)')
    ax.set_yticks(ypos,ordlabels)
    ax.set(xlabel='Media |SHAP| en cada estrato de muestra',
           title='Intensidad de contribuciones por etiqueta del modelo')
    ax.legend(frameon=False)
    fig_guardar(fig,salida/'04_shap_alertas_vs_sin_alerta.png')

    # 05 - mapa de calor firmado (alertas de prueba mas extremas entre las explicadas).
    subset=df[(df.conjunto=='prueba')&(df.alerta_isolation_forest==1)]\
           .sort_values('puntaje_anomalia',ascending=False).head(25)
    if len(subset):
        matriz=subset[cols].to_numpy(float)
        v=max(float(np.quantile(np.abs(matriz),.97)),.005)
        fig,ax=plt.subplots(figsize=(12.1,max(4.4, min(8.5,len(subset)*.29+2.8))))
        im=ax.imshow(matriz,aspect='auto',cmap='RdBu_r',vmin=-v,vmax=v,interpolation='nearest')
        ax.set_xticks(np.arange(10),CORTOS,rotation=35,ha='right')
        ax.set_yticks(np.arange(len(subset)),[str(int(s)) for s in subset[SMP]])
        ax.set(xlabel='Las 10 senales en el modelo unico',ylabel='SMP de prueba',
               title='Mapa de SHAP firmado | alertas explicadas mas extremas (prueba)')
        fig.colorbar(im,ax=ax,pad=.015,label='Contribucion SHAP a score de anomalia')
        fig_guardar(fig,salida/'05_mapa_calor_alertas_prueba.png')

    # 06 - waterfall manual, registro de prueba con mayor score incluido en la muestra.
    pt=df[df.conjunto=='prueba'].copy()
    if len(pt):
        for nombre,sub in [('alerta',pt.sort_values('puntaje_anomalia',ascending=False).head(1)),
                           ('sin_alerta',pt[pt.alerta_isolation_forest==0]
                            .sort_values('puntaje_anomalia').iloc[len(pt[pt.alerta_isolation_forest==0])//2:
                                                                 len(pt[pt.alerta_isolation_forest==0])//2+1])]:
            if sub.empty:continue
            r=sub.iloc[0]
            sv=np.array([r['shap_'+n] for n in variables],float)
            order=np.argsort(np.abs(sv))[::-1]
            fig,ax=plt.subplots(figsize=(10.4,6.0))
            labs=np.array(CORTOS)[order]
            ypos=np.arange(10)
            ax.barh(ypos,sv[order],color=['#C75E60' if v>0 else '#357FB2' for v in sv[order]])
            ax.set_yticks(ypos,labs)
            ax.invert_yaxis()
            ax.axvline(0,color='#333333',lw=1)
            ax.set(xlabel='Contribucion SHAP a -score_samples',
                   title=f'SMP {int(r[SMP])} | {nombre.replace("_"," ")} | '\
                         f'score {r["puntaje_anomalia"]:.4f}')
            ax.text(.99,.02,f'Base: {r["shap_base"]:.4f} | suma: {r["shap_suma"]:+.4f} | '
                    f'score: {r["shap_reconstruido"]:.4f}',
                    transform=ax.transAxes,ha='right',fontsize=9)
            fig_guardar(fig,salida/f'06_explicacion_individual_{nombre}.png')

    # 08 - reconstruccion aditiva observada (modelo exacto en registros explicados).
    fig,ax=plt.subplots(figsize=(7.3,6.4))
    for p in PARTES:
        sub=df[df.conjunto==p]
        ax.scatter(sub['puntaje_anomalia'],sub['shap_reconstruido'],
                   label=p.capitalize(),c=COLORES[p],s=24,alpha=.72)
    lim=[min(df['puntaje_anomalia'].min(),df['shap_reconstruido'].min())-.005,
         max(df['puntaje_anomalia'].max(),df['shap_reconstruido'].max())+.005]
    ax.plot(lim,lim,ls='--',c='#222222',lw=1.1)
    ax.set(xlim=lim,ylim=lim,xlabel='Score real del modelo 13: -score_samples',
           ylabel='Base SHAP + contribuciones',
           title='Validacion: reconstruccion de puntuaciones del modelo')
    ax.legend(frameon=False)
    fig_guardar(fig,salida/'08_validacion_aditividad_shap.png')

    # 09 - familias: SOLO resumen de contribuciones de 10 variables juntas.
    fam=familias.sort_values('importancia_abs_ponderada')
    fig,ax=plt.subplots(figsize=(9.1,4.7))
    ax.barh(fam.familia,fam.participacion_importancia_pct,color='#667D9F')
    ax.set(xlabel='Participacion del total de media |SHAP| ponderada (%)',
           title='Resumen por familia del MISMO Isolation Forest de 10 variables')
    ax.text(.01,.01,'Agrupacion para comunicar, NO modelos independientes ni causa fisica',
            transform=ax.transAxes,fontsize=8,color='#555555')
    fig_guardar(fig,salida/'09_aporte_agregado_por_familia.png')

    # 10 - importancia relativa vs valor escalado (tres variables top).
    j_top=[variables.index(v) for v in ranking.senal.iloc[:3]]
    for rank,j in enumerate(j_top,1):
        fig,ax=plt.subplots(figsize=(8.3,5.0))
        for p in PARTES:
            mask=(df['conjunto'].to_numpy()==p)
            ax.scatter(X[mask,j],vals[mask,j],s=19,alpha=.57,
                       c=COLORES[p],label=p.capitalize(),edgecolors='none')
        ax.axhline(0,color='#555555',lw=1)
        ax.set(xlabel=f'{CORTOS[j]}: valor escalado',ylabel='SHAP firmado',
               title=f'Variable destacada {rank}: {variables[j]} | muestra explicada')
        ax.legend(frameon=False)
        fig_guardar(fig,salida/f'10_dependencia_shap_top{rank}.png')


def autotest():
    from sklearn.ensemble import IsolationForest
    rng=np.random.default_rng(SEMILLA)
    X=rng.normal(size=(120,10)).astype(np.float32)
    mod=IsolationForest(n_estimators=15,max_samples=40,random_state=SEMILLA).fit(X)
    exp=shap.Explainer(lambda a: -mod.score_samples(np.asarray(a,dtype=np.float32)),
                       shap.maskers.Independent(X[:12],max_samples=12),
                       algorithm='permutation',seed=SEMILLA)
    sv=exp(X[52:55],max_evals=21,silent=True)
    res=sv.base_values+sv.values.sum(axis=1)
    calc=-mod.score_samples(X[52:55])
    assert np.allclose(res,calc,atol=1e-8), (res,calc)
    print('[AUTOTEST] OK: SHAP explica directamente -score_samples, 10 variables y aditividad.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modo',choices=['comprobar','final','autotest'],default='comprobar')
    parser.add_argument('--preparados',type=Path,default=RAIZ/'Resultados_Tarea3_Nuevos'/'00_preparacion')
    parser.add_argument('--isolation',type=Path,default=RAIZ/'Resultados_Tarea3_Nuevos'/'05_isolation_forest')
    parser.add_argument('--salida',type=Path,default=RAIZ/'Resultados_Tarea3_Nuevos'/'06_shap')
    parser.add_argument('--alertas',type=int,default=20,help='Muestra de alertas POR CONJUNTO para SHAP')
    parser.add_argument('--normales',type=int,default=20,help='Muestra de registros SIN alerta POR CONJUNTO para SHAP')
    parser.add_argument('--fondo',type=int,default=25,help='Referencias del entrenamiento solamente')
    parser.add_argument('--lote',type=int,default=8,help='Cada cuantas explicaciones se informa progreso')
    parser.add_argument('--permutaciones',type=int,default=3,help='Permutaciones SHAP por registro: 1=rapido, 3=predeterminado')
    parser.add_argument('--semilla',type=int,default=SEMILLA)
    args=parser.parse_args()
    if args.modo=='autotest':
        autotest();return
    if min(args.alertas,args.normales,args.fondo,args.lote,args.permutaciones)<1:
        parser.error('alertas, normales, fondo y lote deben ser mayores que cero.')
    print('[COMPROBACION] SHAP de Isolation Forest 13: carga y verifica modelo ya entrenado.',flush=True)
    grupos,config,meta,modelo=preflight(args.preparados,args.isolation)
    if args.modo=='comprobar':
        print('[OK] Ejecuta: python -u 14_shap.py --modo final',flush=True)
        print('SHAP explica muestras estratificadas; NO todos los registros. Modelo 13: uno solo y 10 senales.',flush=True)
        return
    salida=args.salida;salida.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    df,fondo,fn=explicar(grupos,modelo,meta['variables'],args.alertas,
                        args.normales,args.fondo,args.lote,args.semilla,salida,args.permutaciones)
    ranking,por_conjunto,familias,pesos=calcular_importancias(
        df,grupos,meta['variables'],salida)
    print('\n[GRAFICOS] Generando interpretaciones de muestras de los tres conjuntos.',flush=True)
    graficar(df,ranking,por_conjunto,familias,meta['variables'],pesos,salida)
    duracion=time.perf_counter()-t0
    info={'modelo_origen':'05_isolation_forest/00_modelo_isolation_forest.joblib',
          'reentrenamiento':False,'predictoras':meta['variables'],
          'numero_predictoras':10,'escalado':config['escalado'],
          'algoritmo_SHAP':'PermutationExplainer; masker Independent',
          'objetivo_explicado':'-IsolationForest.score_samples (mayor=mas anomalo)',
          'max_evals_por_explicacion':21*args.permutaciones,'permutaciones_por_registro':args.permutaciones,'fondo_muestras_entrenamiento':len(fondo),
          'muestreo':'aleatorio estratificado segun alerta y particion, semilla fija; '
                     'incluye el score mas alto en prueba',
          'registros_totales':sum(len(grupos[p]['X']) for p in PARTES),
          'registros_explicados':len(df),'registros_explicados_por_conjunto':
              {p:int((df.conjunto==p).sum()) for p in PARTES},
          'importancia':'estimacion estratificada reponderada por prevalencia real en '
                         'cada particion y volumen de la particion (NO exhaustiva)',
          'error_aditividad_maximo':float(df.error_aditividad.max()),
          'duracion_segundos':float(duracion),
          'limitaciones':['Las importancias se estiman en una muestra, no en todos los registros.',
                          'SHAP depende de la distribucion del fondo y del modelo.',
                          'El masker independiente puede crear combinaciones no observadas '
                          'si hay dependencia entre variables.',
                          'El signo SHAP es relativo al score del modelo, no indica gravedad fisica.',
                          'No hay etiquetas de falla; no equivale a averias confirmadas.',
                          'SMP ordena adquisiciones, no representa segundos.']}
    (salida/'00_metadatos_shap.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    texto=(
        'CODIGO 14 - SHAP PARA EL MISMO ISOLATION FOREST DE 10 SENALES\n'
        f'Modelo 13: {len(modelo.estimators_)} arboles / max_samples {modelo.max_samples_}. NO REENTRENADO.\n'
        f'Entrenamiento: {len(grupos["entrenamiento"]["X"])}; validacion: {len(grupos["validacion"]["X"])}; prueba: {len(grupos["prueba"]["X"])}.\n'
        f'SHAP explicado: {len(df)} registros MUESTREADOS; no confundir con 143561 procesados por el modelo 13.\n'
        'Fondo SHAP aleatorio del entrenamiento; muestra de explicaciones estratificada.\n'
        'Interpretacion: SHAP positivo incrementa puntuacion de anomalia relativa al fondo;\n'
        'SHAP negativo disminuye puntuacion de anomalia relativa al fondo.\n'
        'Importancia = media |SHAP| estimada reponderando muestra por prevalencia real y tamano de cada conjunto.\n'
        'Familias del grafico 09 son agregaciones de 10 variables del modelo UNICO, NO modelos separados.\n'
        'No son causalidad mecanica, etiquetas confirmadas, ni criterios ISO de mantenimiento.\n'
        'SMP NO es tiempo en segundos.\n')
    (salida/'00_LEER_INTERPRETACION.txt').write_text(texto,encoding='utf-8')
    print('\n=== RESULTADO SHAP VERIFICADO ===',flush=True)
    print('Las 10 variables forman un UNICO modelo, sin reentrenamiento.',flush=True)
    print(f'Registros explicados SHAP: {len(df)} (de 143.561 analizados por el modelo 13).',flush=True)
    print(f'Error maximo reconstruccion: {float(df.error_aditividad.max()):.2e}',flush=True)
    print('Ranking estimado por media |SHAP| ponderada en muestra:')
    print(ranking[['sensor_corto','importancia_abs_ponderada','participacion_importancia_pct']].to_string(index=False),flush=True)
    print(f'\nArchivos guardados en {salida}',flush=True)
    print('ADVERTENCIA: SHAP explica predicciones; NO confirma fallas ni causas fisicas.',flush=True)


if __name__=='__main__':
    main()
