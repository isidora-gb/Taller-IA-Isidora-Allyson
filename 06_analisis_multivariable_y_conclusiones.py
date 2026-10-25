# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv
import math
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
COLUMNAS = [
    'AcelX1_SisGenEol', 'AcelY1_SisGenEol', 'AcelZ1_SisGenEol',
    'GiroX_SisGenEol', 'GiroY_SisGenEol', 'GiroZ_SisGenEol',
    'MagneX_SisGenEol', 'MagneY_SisGenEol', 'MagneZ_SisGenEol',
    'Ter_SisGenEol'
]
ETIQUETAS = ['Ac-X','Ac-Y','Ac-Z','Gi-X','Gi-Y','Gi-Z','Ma-X','Ma-Y','Ma-Z','Temp']
UNIDADES = ['m/s²']*3 + ['°/s']*3 + ['unidad no verificada']*3 + ['°C']
GRUPOS = {
    'acelerometro': range(0, 3), 'giroscopio': range(3, 6),
    'magnetometro': range(6, 9), 'temperatura': range(9, 10)
}
COLORES = {'acelerometro': '#256a91', 'giroscopio': '#7a5ab2',
           'magnetometro': '#209a89', 'temperatura': '#c98136'}
NIST = 'https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm'
IEC = 'https://webstore.iec.ch/en/publication/111042'

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.16,'savefig.facecolor':'white',
    'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.dpi':300,'xtick.labelsize':9,'ytick.labelsize':9
})


def numero(s: str) -> float:
    s = s.strip().replace(' ', '')
    if not s: return float('nan')
    return float(s.replace(',', '.'))


def leer_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'No existe: {path}. Primero ejecuta 00_validar_y_separar_excel.py')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        headers = reader.fieldnames or []
        needed = COLUMNAS + ['smp_original','smp_continuo','smp_desde_cero','fila_excel',
                              'smp_estimado','medicion_imputada']
        faltan = [x for x in needed if x not in headers]
        if faltan: raise ValueError(f'Faltan columnas en archivo multisensor: {faltan}')
        rows = list(reader)
    if not rows: raise ValueError('CSV multisensor está vacío.')
    valores = np.asarray([[numero(r[c]) for c in COLUMNAS] for r in rows], dtype=float)
    smp = np.asarray([numero(r['smp_continuo']) for r in rows], dtype=float)
    original = np.asarray([numero(r['smp_original']) for r in rows], dtype=float)
    orden = np.asarray([int(r['fila_excel']) for r in rows], dtype=int)
    imputado = np.asarray([r['medicion_imputada'].strip().lower() in ('true','1','si') for r in rows],dtype=bool)
    smp_estimado = np.asarray([r['smp_estimado'].strip().lower() in ('true','1','si') for r in rows],dtype=bool)
    if not np.isfinite(smp).all() or not (np.diff(smp) > 0).all():
        raise ValueError('SMP continuo no es numérico y estrictamente creciente.')
    if np.isnan(valores).any() or not np.isfinite(valores).all():
        print('[AVISO] Hay valores no numéricos en las mediciones; se documentarán como faltantes.')
    return valores, smp, original, orden, imputado, smp_estimado


def guardar_csv(ruta, header, rows):
    ruta.parent.mkdir(exist_ok=True, parents=True)
    with ruta.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(header)
        for row in rows:
            w.writerow([str(v).replace('.',',') if isinstance(v, (float, np.floating)) and np.isfinite(v)
                        else ('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v)
                        for v in row])


def stats(x):
    y=x[np.isfinite(x)]
    if len(y)==0: return [0, len(x), *([float('nan')]*12)]
    mu=float(y.mean()); sd=float(y.std(ddof=1)) if len(y)>1 else float('nan')
    q1, med, q3 = np.percentile(y,[25,50,75]); mad=float(np.median(np.abs(y-med)))
    return [int(len(y)),int(len(x)-len(y)),mu,sd,float(y.min()),float(y.max()),
            float(med),float(q1),float(q3),float(q3-q1),mad,
            100*sd/abs(mu) if mu!=0 else float('nan'),float(y.max()-y.min()),float(y.var(ddof=1))]


def robust(x):
    ok=np.isfinite(x); classes=np.full(len(x),-1,dtype=np.int8)
    if np.sum(ok)<10: return classes, float('nan'),float('nan')
    med=np.median(x[ok]); mad=np.median(np.abs(x[ok]-med))
    if mad<=0 or not np.isfinite(mad): return classes,float(med),float(mad)
    z=np.abs(0.6745*(x[ok]-med)/mad)
    classes[ok]=0
    classes[np.flatnonzero(ok)[z>3.5]]=1
    classes[np.flatnonzero(ok)[z>6]]=2
    return classes,float(med),float(mad)


def pearson_matriz(x):
    n=x.shape[1]
    mat=np.full((n,n),np.nan)
    count=np.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i,n):
            ok=np.isfinite(x[:,i]) & np.isfinite(x[:,j]); count[i,j]=count[j,i]=int(ok.sum())
            if ok.sum()>=3:
                a=x[ok,i]; b=x[ok,j]
                if np.std(a)>0 and np.std(b)>0:
                    mat[i,j]=mat[j,i]=float(np.corrcoef(a,b)[0,1])
    return mat,count


def rango_empates(x):
    """Rangos promedio para empates; permite Spearman sin scipy."""
    orden=np.argsort(x,kind='mergesort')
    ordenados=x[orden]
    puntos=np.r_[0, np.flatnonzero(ordenados[1:] != ordenados[:-1])+1, len(x)]
    rank=np.empty(len(x),dtype=float)
    for a,b in zip(puntos[:-1],puntos[1:]): rank[orden[a:b]]=(a+b+1)/2
    return rank


def spearman_matriz(x):
    n=x.shape[1]; rangos=np.column_stack([rango_empates(x[:,i]) for i in range(n)])
    return np.corrcoef(rangos,rowvar=False)


def heatmap(matrix,ruta,titulo):
    fig,ax=plt.subplots(figsize=(10.5,8.4))
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
    ax.set_xticks(range(len(ETIQUETAS)),ETIQUETAS,rotation=40,ha='right')
    ax.set_yticks(range(len(ETIQUETAS)),ETIQUETAS)
    ax.grid(False)
    for i in range(len(ETIQUETAS)):
        for j in range(len(ETIQUETAS)):
            v=matrix[i,j]; txt=f'{v:.2f}' if np.isfinite(v) else 'N/D'
            ax.text(j,i,txt,ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(v) and abs(v)>.55 else '#202e3b')
    fig.colorbar(im, ax=ax, shrink=.85,label='Coeficiente [-1,1]')
    ax.set_title(titulo,weight='bold',pad=18)
    fig.tight_layout();fig.savefig(ruta,dpi=300,bbox_inches='tight');plt.close(fig)


def grafico_evolucion(x,smp,ruta, hz=None):
    """Cuantiles por bloques, no se confunde número de muestra con segundos."""
    block=1000; n=len(smp)
    grupos=[np.arange(a,min(a+block,n)) for a in range(0,n,block)]
    if hz is None:
        xs=np.array([np.median(smp[g]-smp[0]) for g in grupos])
        xlabel='SMP desde cero (índice de muestra; NO segundos)'
    else:
        xs=np.array([np.median((smp[g]-smp[0])/hz) for g in grupos])
        xlabel=f'Tiempo inferido (segundos, usando frecuencia confirmada {hz:g} Hz)'
    fig,axs=plt.subplots(4,1,figsize=(15,12.5),sharex=True)
    for ax,(name,cols) in zip(axs,GRUPOS.items()):
        for k in cols:
            v=x[:,k]; med=np.median(v[np.isfinite(v)])
            mad=np.median(np.abs(v[np.isfinite(v)]-med))
            escala=mad/0.6745 if mad>0 else np.std(v[np.isfinite(v)])
            if not np.isfinite(escala) or escala==0: escala=1
            norm=(v-med)/escala
            central=np.array([np.nanmedian(norm[g]) for g in grupos])
            p10=np.array([np.nanpercentile(norm[g],10) for g in grupos])
            p90=np.array([np.nanpercentile(norm[g],90) for g in grupos])
            color=plt.get_cmap('tab10')(k%10)
            ax.plot(xs, central,label=ETIQUETAS[k],color=color,lw=1.3)
            ax.fill_between(xs,p10,p90,color=color,alpha=.08)
        ax.axhline(0,lw=.7,c='#555')
        ax.set_ylabel(name.capitalize()+'\n(unid. robustas)')
        ax.legend(loc='upper right',frameon=False,ncol=3,fontsize=8)
    axs[0].set_title('Evolución por bloques de 1.000 registros | mediana y percentiles 10–90',weight='bold',pad=13)
    axs[-1].set_xlabel(xlabel)
    fig.text(.01,.01,'Cada variable está centrada por su mediana y dividida por MAD/0,6745; NO son unidades físicas.\n'
                     'Las áreas muestran variabilidad por bloque. No equivalen a bandas de seguridad.',size=8,color='#444')
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(ruta,dpi=300,bbox_inches='tight');plt.close(fig)


def grafico_dispersion(x,corr,ruta):
    candidatos=sorted([(abs(corr[i,j]),i,j) for i in range(10) for j in range(i+1,10)
                       if i//3!=j//3 and np.isfinite(corr[i,j])],reverse=True)
    # Excluimos pares dentro del mismo sensor mediante grupos reales.
    grupo_de={k:n for n,keys in GRUPOS.items() for k in keys}
    candidatos=sorted([(abs(corr[i,j]),i,j) for i in range(10) for j in range(i+1,10)
                       if grupo_de[i]!=grupo_de[j] and np.isfinite(corr[i,j])],reverse=True)[:3]
    if not candidatos: return
    rng=np.random.default_rng(2026); idx=rng.choice(len(x),min(4000,len(x)),replace=False)
    fig,axs=plt.subplots(1,len(candidatos),figsize=(5.4*len(candidatos),4.7),squeeze=False)
    for ax,(_,i,j) in zip(axs[0],candidatos):
        ax.scatter(x[idx,i],x[idx,j],s=4,alpha=.2,color='#205c80',rasterized=True)
        ax.set_xlabel(ETIQUETAS[i]+' ('+UNIDADES[i]+')')
        ax.set_ylabel(ETIQUETAS[j]+' ('+UNIDADES[j]+')')
        ax.set_title(f'Pearson r={corr[i,j]:+.3f}',weight='semibold')
    fig.suptitle('Tres relaciones lineales más altas entre sensores distintos | muestra aleatoria reproducible',
                 fontsize=12,weight='bold')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ruta,dpi=300,bbox_inches='tight');plt.close(fig)


def grafico_porcentajes(anoms,n,ruta):
    nombres=list(GRUPOS)
    p_alerta=[100*(anoms[g]==1).sum()/n for g in nombres]
    p_crit=[100*(anoms[g]==2).sum()/n for g in nombres]
    fig,ax=plt.subplots(figsize=(9,4.8)); xpos=np.arange(4)
    ax.bar(xpos,p_alerta,label='Alerta: 3,5 < |Zmod| ≤ 6',color='#dda250')
    ax.bar(xpos,p_crit,bottom=p_alerta,label='Extremo: |Zmod| > 6',color='#b24857')
    ax.set_xticks(xpos,['Acelerómetro','Giroscopio','Magnetómetro','Temperatura'])
    ax.set_ylabel('% de registros evaluados');ax.set_title('Anomalías exploratorias por sensor',weight='bold')
    ax.legend(frameon=False,fontsize=8)
    for i,(a,b) in enumerate(zip(p_alerta,p_crit)):
        ax.text(i,a+b+.01,f'{a+b:.3f} %',ha='center',fontsize=9)
    ax.set_ylim(0,max(.3,max(a+b for a,b in zip(p_alerta,p_crit))*1.22))
    fig.tight_layout();fig.savefig(ruta,dpi=300,bbox_inches='tight');plt.close(fig)


def grafico_anomalias_bloques(anoms,smp,ruta,hz=None):
    block=2500; n=len(smp)
    grupos=[slice(a,min(a+block,n)) for a in range(0,n,block)]
    xs=np.array([np.nanmean(smp[s]-smp[0]) for s in grupos])
    if hz is not None: xs=xs/hz
    fig,ax=plt.subplots(figsize=(13.5,4.8))
    for sensor in GRUPOS:
        yy=[100*np.mean(anoms[sensor][g]>0) for g in grupos]
        ax.plot(xs,yy,label=sensor.capitalize(),lw=1.5,color=COLORES[sensor])
    ax.set_title('Evolución de alertas por bloque de 2.500 registros',weight='bold')
    ax.set_xlabel('Tiempo estimado (segundos)' if hz else 'SMP desde cero (NO tiempo)')
    ax.set_ylabel('% alerta o extremo en cada bloque')
    ax.legend(frameon=False,ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(ruta,dpi=300,bbox_inches='tight');plt.close(fig)


def principal(entrada, salida,hz=None):
    salida.mkdir(parents=True,exist_ok=True)
    graficos=salida/'07_graficos_multivariable';graficos.mkdir(exist_ok=True)
    x,smp,original,fila_excel,imputado,smp_estimado=leer_csv(entrada)
    n=len(x)
    print(f'Análisis multivariable: {n:,} filas, {len(COLUMNAS)} variables')
    if n!=143561: print('[AVISO] Cantidad distinta de 143.561; revisa la auditoría de origen.')
    validez=[]; estad=[]; seis_mask=np.zeros(x.shape,dtype=bool)
    clases={}
    for k,c in enumerate(COLUMNAS):
        v=x[:,k]; st=stats(v)
        mu,sd=st[2],st[3]
        mascaras=(np.abs(v-mu)>6*sd) if np.isfinite(sd) and sd>0 else np.zeros(n,dtype=bool)
        seis_mask[:,k]=mascaras
        estad.append([c, ETIQUETAS[k], *st, int(mascaras.sum()),
                      'Estimado 1 dato final' if k in (8,9) and np.any(imputado) else 'Solo observado'])
        validez.append([c,'Unidad instrumental no confirmada',float(np.nanmin(v)),float(np.nanmax(v)),
                        'Sin especificaciones del fabricante ni condiciones de operación',
                        'NO evaluable físicamente (solo consistencia y estadística)',
                        'Revisar escala, calibración, ubicación, potencia y operación'])
    st_heads=['variable','alias','validos','nulos','media','desviacion_estandar_muestral',
              'minimo','maximo','mediana','q1','q3','iqr','MAD','coeficiente_variacion_pct','rango',
              'varianza','atipicos_fuera_6sigma','origen']
    guardar_csv(salida/'01_estadisticos_10_variables.csv',st_heads,estad)
    guardar_csv(salida/'02_validez_fisica_pendiente.csv',
                ['variable','unidad','minimo_en_base','maximo_en_base','limitacion','dictamen','accion'],validez)
    pairwise,cuentas=pearson_matriz(x)
    sp=spearman_matriz(x) if np.isfinite(x).all() else np.full((10,10),np.nan)
    tratado=x.copy();tratado[seis_mask]=np.nan
    despues,cuentas_despues=pearson_matriz(tratado)
    for filename,mat in [('03_correlacion_pearson_10.csv',pairwise),('04_correlacion_spearman_10.csv',sp),
                         ('05_correlacion_pearson_despues_6sigma.csv',despues),
                         ('06_cambio_correlacion_despues_menos_antes.csv',despues-pairwise)]:
        guardar_csv(salida/filename,['variable',*ETIQUETAS],[[ETIQUETAS[i],*mat[i]] for i in range(10)])
    pares=[]
    for i in range(10):
        for j in range(i+1,10):
            par=[COLUMNAS[i],COLUMNAS[j],ETIQUETAS[i],ETIQUETAS[j],pairwise[i,j],sp[i,j],
                 despues[i,j],cuentas[i,j],cuentas_despues[i,j]]
            pares.append(par)
    pares.sort(key=lambda p:(0 if np.isfinite(p[4]) else 1,-abs(p[4]) if np.isfinite(p[4]) else 0))
    guardar_csv(salida/'07_pares_correlacion_ordenados.csv',
                ['var1','var2','alias_1','alias_2','pearson_antes','spearman_antes',
                 'pearson_despues_6sigma','pares_validos_antes','pares_validos_despues'],pares)

    for name,cols in GRUPOS.items():
        cls=np.stack([robust(x[:,k])[0] for k in cols],axis=1)
        # -1 en una variable no significa normal: no declarar estado válido si hay una variable sin referencia.
        estado=np.max(cls,axis=1)
        estado[np.any(cls<0,axis=1)]=-1
        clases[name]=estado
    recs=[]
    for name in GRUPOS:
        cls=clases[name];ne=int(np.sum(cls>=0));a=int(np.sum(cls==1));b=int(np.sum(cls==2))
        recs.append([name,n,ne,int(np.sum(cls==0)),a,b,int(np.sum(cls<0)),
                     100*a/ne if ne else float('nan'),100*b/ne if ne else float('nan')])
    guardar_csv(salida/'08_resumen_anomalias_multisensor.csv',
                ['sensor','filas','evaluables','normal','alerta','extremo_estadistico','sin_referencia',
                 'porcentaje_alerta','porcentaje_extremo'],recs)
    matriz_est=np.column_stack([clases[k] for k in GRUPOS]);colisiones=np.sum(matriz_est>0,axis=1)
    eventos=np.flatnonzero(colisiones>=2)
    guardar_csv(salida/'09_coincidencias_anomalias_entre_sensores.csv',
                ['fila_excel','smp_continuo','smp_original','smp_estimado','sensores_con_anomalia',*GRUPOS],
                [[int(fila_excel[i]),int(smp[i]),'' if not np.isfinite(original[i]) else int(original[i]),
                  bool(smp_estimado[i]),int(colisiones[i]), *[int(clases[g][i]) for g in GRUPOS]]
                 for i in eventos])
    calidad=[
        ('registros_conservados',n,'Después de corrección estructural'),
        ('variables_de_medicion',len(COLUMNAS),'Acelerómetro 3, giroscopio 3, magnetómetro 3, temperatura 1'),
        ('mediciones_nulas_tras_correccion',int(np.sum(~np.isfinite(x))),'Sobre 10 variables; no confundir con nulos originales'),
        ('filas_con_alguna_medicion_nula',int(np.sum(~np.isfinite(x).all(axis=1))),'Después de imputar'),
        ('filas_con_mediciones_imputadas',int(imputado.sum()),'La fila final tiene 2 valores de sensor estimados'),
        ('total_celdas_de_sensor_imputadas',int(imputado.sum())*2,'MagneZ y temperatura: promedio de 2 anteriores'),
        ('filas_con_smp_original_vacio',int((~np.isfinite(original)).sum()),'SMP final no observado'),
        ('filas_con_smp_estimado',int(smp_estimado.sum()),'Reconstrucción +1, no medida'),
        ('saltos_positivos_smp',int(np.sum(np.diff(smp)>1)),'No equivalen necesariamente a pérdidas de tiempo'),
        ('muestras_smp_ausentes_por_saltos',int(np.sum(np.maximum(np.diff(smp)-1,0))),'351 en archivo original auditado'),
        ('coincidencias_de_anomalias_en_2_o_mas_sensores',len(eventos),'Simultaneidad por fila, no validación de falla'),
        ('frecuencia_muestreo_hz', hz if hz is not None else '',
          'Sin frecuencia comprobada, NO se convierte SMP a tiempo real'),
    ]
    guardar_csv(salida/'10_auditoria_nulos_integridad.csv',['indicador','valor','interpretacion'],calidad)
    # Cuantificar evolución como comparación entre cuartiles de la adquisición.
    t=int(n//4)
    comparacion=[]
    for i,c in enumerate(COLUMNAS):
        a=x[:t,i];b=x[-t:,i]
        comparacion.append([c,float(np.nanmedian(a)),float(np.nanmedian(b)),
                            float(np.nanmedian(b)-np.nanmedian(a)),
                            'Diferencia descriptiva entre tramos; no prueba tendencia temporal física'])
    guardar_csv(salida/'11_comparacion_tramos_inicial_final.csv',
                ['variable','mediana_primer_cuartil','mediana_ultimo_cuartil',
                 'diferencia','interpretacion'],comparacion)

    heatmap(pairwise,graficos/'01_heatmap_pearson_10_variables.png','Correlaciones Pearson | 10 variables, antes de 6σ')
    heatmap(sp,graficos/'02_heatmap_spearman_10_variables.png','Correlaciones Spearman | 10 variables')
    heatmap(despues,graficos/'03_heatmap_pearson_despues_6sigma.png',
            'Correlaciones Pearson | valores >6σ vacíos solo en copia')
    grafico_evolucion(x,smp,graficos/'04_evolucion_por_sensor_y_smp.png',hz)
    grafico_dispersion(x,pairwise,graficos/'05_dispersion_entre_sensores.png')
    grafico_porcentajes(clases,n,graficos/'06_porcentaje_anomalias.png')
    grafico_anomalias_bloques(clases,smp,graficos/'07_anomalias_por_bloques_smp.png',hz)

    best=pares[0]; cross=[p for p in pares if p[2].split('-')[0]!=p[3].split('-')[0]]
    # 'Ac' vs 'Gi' etc.  Si existe par Temp, también queda correctamente separado.
    topcross=cross[0] if cross else None
    mag=next(r for r in recs if r[0]=='magnetometro')
    temp=next(r for r in recs if r[0]=='temperatura')
    uni=[int(seis_mask[:, k].sum()) for k in range(10)]
    informe=f'''# Taller IA 2 — Punto 1. Análisis exploratorio de datos

**Base:** `{entrada.name}`. **Registros:** {n:,}. **Variables:** 10 en cuatro sensores.

## 1. Visualización y estadísticas

Los estadísticos descriptivos por variable (media, desviación estándar, mínimo, máximo,
mediana, cuartiles y MAD) están en `01_estadisticos_10_variables.csv`.
Los gráficos individuales de cada eje están en `Resultados_acelerometro`,
`Resultados_giroscopio`, `Resultados_magnetometro` y `Resultados_temperatura`.
Los gráficos 01–07 de esta carpeta complementan los análisis individuales.

**Validez física:** el CSV no trae fabricante, calibración, ubicación instrumental,
unidades verificadas para todos los sensores, rangos admisibles ni estado operativo
nominal. Por ello, no se declara que todas las mediciones sean físicamente normales
ni se aplican límites de otros aerogeneradores. Solo se ha verificado su consistencia
interna. Ver `02_validez_fisica_pendiente.csv`.

## 2. Nulos y corrección estructural

El Excel de origen tiene 352 filas con fragmentos concatenados y una última fila
incompleta. El código 00 recupera el bloque íntegro de cada fila afectada, mantiene
la última fila y documenta las imputaciones. En la última se imputan exclusivamente
MagneZ y temperatura con la media de las dos últimas observaciones previas.
`SMP original` final no se inventa: queda vacío y se usa un SMP continuo estimado.
Se conservan **{n:,} registros**; hay **{int(np.sum(~np.isfinite(x)))} mediciones nulas**
en las 10 variables después del proceso. El origen SÍ contenía datos faltantes.
La auditoría está en `Datos_corregidos/registro_imputaciones.csv` y
`10_auditoria_nulos_integridad.csv`. Existen **{int(np.sum(np.maximum(np.diff(smp)-1,0)))}**
identificadores de muestra no representados en registros completos, que no se interpolan.

## 3. Criterios de valores atípicos

**6σ clásico:** media ±6 desviaciones estándar muestrales, por variable.
Los valores fuera se registran como sospechosos y se dejan vacíos solo en la copia
tratada. Recuentos por variable: {', '.join(f'{ETIQUETAS[i]}={v}' for i,v in enumerate(uni))}.

**Z modificado (NIST/Iglewicz-Hoaglin):** z=0,6745·(x−mediana)/MAD. Un valor
con |z| >3,5 es un posible atípico. El proyecto separa alertas (3,5 <|z|≤6)
de extremos estadísticos (|z| >6), límite este último elegido para explorar.
**No son criterios físicos de seguridad ni diagnósticos de falla.**

Acelerómetro: {recs[0][4]} alertas y {recs[0][5]} extremos; giroscopio:
{recs[1][4]} y {recs[1][5]}; magnetómetro: {mag[4]} y {mag[5]};
temperatura: {temp[4]} y {temp[5]}. Se identificaron {len(eventos)}
filas con anomalías en al menos dos sensores (no necesariamente fallas).

Fuente metodológica: {NIST}

## 4. Relación entre variables y evolución en la adquisición

Se calculó correlación Pearson y Spearman para las diez variables, y Pearson
después de dejar vacíos valores >6σ en la copia (matrices 03–06 y gráficos 01–03).
El mayor valor absoluto Pearson entre pares distintos de variables fue
**r={best[4]:+.3f}** para **{best[2]} y {best[3]}**.
'''
    if topcross is not None:
        informe+=f'Entre sensores diferentes, el mayor |r| fue **{topcross[4]:+.3f}** para **{topcross[2]} y {topcross[3]}**.\n'
    informe+='''
Correlación no demuestra causalidad ni descarta relaciones no lineales o dependientes
de condiciones operativas. La evolución por bloques y el porcentaje de alertas por
SMP se presentan en los gráficos 04 y 07. Se incluyen dispersiones entre sensores
(gráfico 05) y comparación del primer y último cuartil de registros (tabla 11).

**Advertencia metodológica:** el contador SMP permite ordenar observaciones, pero
no equivale a segundos. Sin frecuencia de muestreo y su estabilidad comprobadas
no se puede estimar duración, velocidad de degradación ni periodicidad de
mantenimiento. Los saltos de SMP pueden afectar análisis de episodios.
'''
    if hz is not None:
        informe+=f'\nPara este cálculo se proporcionó externamente una frecuencia de {hz:g} Hz. Su respaldo debe adjuntarse al informe.\n'
    informe+='''
## 5. Conclusión para el informe

Los datos corregidos permiten describir el comportamiento de las diez variables,
revisar su completitud y detectar desviaciones estadísticas. Tras la corrección de
las filas desplazadas, las anomalías extremas previamente observadas en temperatura
y acelerómetro desaparecen bajo el criterio robusto aplicado a esta base. El
magnetómetro concentra la mayor proporción de alertas y el giroscopio presenta
algunos valores extremos aislados; estos hallazgos aconsejan una **revisión dirigida
de las señales**, no una orden automática de mantenimiento.

Las correlaciones lineales observadas son bajas o moderadas y, por sí mismas, no
permiten inferir averías. La falta de especificaciones de sensores, escala
instrumental, frecuencia de muestreo y registros etiquetados de fallas impide
validar la seguridad física de las mediciones, fechar eventos o predecir la fecha
de mantenimiento. Se recomienda obtener la ficha técnica, validar unidades y
operación normal, y comparar las anomalías con inspecciones reales.

## Relación con los cinco requisitos del profesor

1. **Estadísticas y gráficos:** completo como análisis descriptivo; validez
   física condicionada a información técnica ausente.
2. **Nulos:** auditados antes/después y registradas las imputaciones.
3. **Atípicos:** definidos por dos criterios documentados, sin confundirlos.
4. **Relaciones y evolución:** matrices conjuntas + dispersión + SMP;
   evolución en tiempo real pendiente de frecuencia de adquisición verificada.
5. **Conclusión:** redactada aquí con alcances y limitaciones explícitos.

## Bibliografía metodológica y normativa

- NIST/SEMATECH, *Engineering Statistics Handbook – Detection of Outliers*:
  https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- IEC, *IEC 61400-1:2019+AMD1:2025*, requisitos generales de diseño:
  https://webstore.iec.ch/en/publication/111042

Las normas citadas no proporcionan en la vista pública límites aplicables directamente
al acelerómetro, giroscopio o magnetómetro de esta base.
'''
    (salida/'12_texto_para_informe_punto1.md').write_text(informe,encoding='utf-8')
    instrucciones='''ÍNDICE DE RESULTADOS — ANÁLISIS MULTIVARIABLE\n\n'''+ '\n'.join(
        str(p.relative_to(salida)) for p in sorted(salida.rglob('*')) if p.is_file()) + '\n'
    (salida/'LEEME_resultados.txt').write_text(instrucciones,encoding='utf-8')
    print('Guardados',len(list(graficos.glob('*.png'))),'gráficos a 300 dpi y 12 tablas/informe.')
    print('Correlación máxima',best[2],best[3],round(float(best[4]),4))
    print('Resultados:',salida)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrada',type=Path,default=BASE/'Datos_corregidos'/'datos_multisensor_corregidos.csv')
    p.add_argument('--salida',type=Path,default=BASE/'Resultados_multivariable')
    p.add_argument('--frecuencia-hz',type=float,default=None,
                   help='SOLO si la frecuencia de adquisición está documentada.')
    a=p.parse_args()
    if a.frecuencia_hz is not None and (not np.isfinite(a.frecuencia_hz) or a.frecuencia_hz<=0):
        p.error('--frecuencia-hz debe ser positiva y finita.')
    principal(a.entrada,a.salida,a.frecuencia_hz)

if __name__=='__main__':main()
