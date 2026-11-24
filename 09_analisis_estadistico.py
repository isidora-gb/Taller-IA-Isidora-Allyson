
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SMP = 'smp_continuo'
SENSORES = [
    'AcelX1_SisGenEol', 'AcelY1_SisGenEol', 'AcelZ1_SisGenEol',
    'GiroX_SisGenEol', 'GiroY_SisGenEol', 'GiroZ_SisGenEol',
    'MagneX_SisGenEol', 'MagneY_SisGenEol', 'MagneZ_SisGenEol',
    'Ter_SisGenEol',
]
CORTOS = ['Ac X', 'Ac Y', 'Ac Z', 'Gi X', 'Gi Y', 'Gi Z',
          'Ma X', 'Ma Y', 'Ma Z', 'Temp']
ARCHIVOS = ['acel_x', 'acel_y', 'acel_z', 'giro_x', 'giro_y', 'giro_z',
            'magne_x', 'magne_y', 'magne_z', 'temperatura']
UNIDADES = ['Aceleracion (unidad por verificar)'] * 3 + [
    'Giro (unidad por verificar)'] * 3 + [
    'Magnetometro (unidad por verificar)'] * 3 + [
    'Temperatura (grados C, segun base)']
PARTES = ['entrenamiento', 'validacion', 'prueba']
COLORES = {'entrenamiento': '#2C6B94', 'validacion': '#D18A48', 'prueba': '#8161A8'}
COLOR_1 = '#31A3A1'
COLOR_2 = '#DF9A37'
COLOR_3 = '#C75165'


def estilo():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.2,
        'axes.titlesize': 11, 'axes.titleweight': 'medium',
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.grid': False,
    })


def guardar(fig, ruta: Path):
    fig.savefig(ruta, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def csv_guardar(tabla, ruta: Path):
    pd.DataFrame(tabla).to_csv(ruta, sep=';', index=False,
                              encoding='utf-8-sig', float_format='%.12g')


def convert_numeric(serie):
    return pd.to_numeric(serie.astype('string').str.strip().str.replace(',', '.', regex=False),
                         errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def origen_por_defecto(raiz, entrada):
    if entrada:
        p = Path(entrada).expanduser()
        p = p if p.is_absolute() else raiz / p
        if not p.is_file():
            raise FileNotFoundError(f'No existe el CSV: {p}')
        return p.resolve()
    for p in [raiz / 'datos_multisensor_corregidos.csv',
              raiz / 'Datos_corregidos' / 'datos_multisensor_corregidos.csv',
              raiz / 'Datos_corregidos_desde_excel' / 'datos_multisensor_corregidos.csv']:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError('No encuentro datos_multisensor_corregidos.csv: usa --entrada RUTA')


def cargar_validado(origen: Path, prep: Path):
    for nombre in ('00_division_cronologica.csv', '00_metadatos.json',
                   '00_parametros_entrenamiento.csv'):
        if not (prep / nombre).is_file():
            raise FileNotFoundError(f'Falta {prep / nombre}. Ejecuta primero el NUEVO 08_preparacion_datos.py')
    meta = json.loads((prep / '00_metadatos.json').read_text(encoding='utf-8'))
    if meta.get('variables') != SENSORES or meta.get('regla_division') != '70/15/15 en orden de SMP, sin mezclas':
        raise ValueError('La preparacion encontrada no corresponde al nuevo codigo 08. Reejecuta el 08.')
    sha = hashlib.sha256()
    with origen.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha.update(chunk)
    if sha.hexdigest() != meta.get('sha256_origen'):
        raise ValueError('El CSV cambio desde que corriste el 08: vuelve a ejecutar 08_preparacion_datos.py')

    with origen.open('r', encoding='utf-8-sig') as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    datos = pd.read_csv(origen, sep=sep, encoding='utf-8-sig', dtype='string')
    faltan = [c for c in [SMP, *SENSORES] if c not in datos.columns]
    if faltan:
        raise ValueError(f'Faltan columnas de sensores: {faltan}')
    datos[SMP] = convert_numeric(datos[SMP])
    if datos[SMP].isna().any() or (datos[SMP] % 1 != 0).any():
        raise ValueError('SMP con valores ausentes o no enteros')
    datos[SMP] = datos[SMP].astype('int64')
    if not datos[SMP].is_monotonic_increasing or datos[SMP].duplicated().any():
        raise ValueError('SMP no esta ordenado o es duplicado')
    for c in SENSORES:
        datos[c] = convert_numeric(datos[c])

    division = pd.read_csv(prep / '00_division_cronologica.csv', sep=';', encoding='utf-8-sig')
    if len(division) != len(datos) or not np.array_equal(division[SMP].to_numpy(dtype='int64'),
                                                        datos[SMP].to_numpy(dtype='int64')):
        raise ValueError('Division del 08 no coincide con el CSV. Reejecuta el 08 con la misma base.')
    if not division['conjunto'].isin(PARTES).all():
        raise ValueError('Conjuntos desconocidos en la division del 08')
    datos['conjunto'] = division['conjunto'].to_numpy()
    for i, p in enumerate(PARTES):
        mask = datos['conjunto'] == p
        idx = np.flatnonzero(mask)
        if not len(idx) or not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
            raise ValueError(f'La particion {p} no es cronologica y contigua')
    for col in ('smp_estimado', 'medicion_imputada'):
        if col not in datos.columns:
            datos[col] = pd.Series(['False']*len(datos), dtype='string')
        datos[col] = datos[col].astype('string').str.lower().str.strip().isin(['true','1','si','sí'])
    if len(datos) != int(meta['registros']):
        raise ValueError('Cantidad de registros diferente de la declarada por codigo 08')
    return datos, meta


def referencia_y_estadisticas(datos):
    train = datos.loc[datos.conjunto == 'entrenamiento', SENSORES]
    ref, estad, exceso = {}, [], []
    for c, corto in zip(SENSORES, CORTOS):
        xtr = train[c].dropna().to_numpy(float)
        mu = float(np.mean(xtr))
        sigma = float(np.std(xtr, ddof=1))
        ref[c] = {'media': mu, 'sigma': sigma,
                  'valida': bool(np.isfinite(sigma) and sigma > 0)}
        for p in PARTES:
            x = datos.loc[datos.conjunto == p, c]
            y = x.dropna().to_numpy(float)
            registro = {
                'variable': c, 'sensor': corto, 'conjunto': p,
                'registros': len(x), 'validos': len(y), 'faltantes': int(x.isna().sum()),
                'distintos': int(len(np.unique(y))),
                'media': float(np.mean(y)) if len(y) else np.nan,
                'mediana': float(np.median(y)) if len(y) else np.nan,
                'desviacion_estandar': float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                'q01': float(np.quantile(y, .01)) if len(y) else np.nan,
                'q25': float(np.quantile(y, .25)) if len(y) else np.nan,
                'q75': float(np.quantile(y, .75)) if len(y) else np.nan,
                'q99': float(np.quantile(y, .99)) if len(y) else np.nan,
                'minimo': float(np.min(y)) if len(y) else np.nan,
                'maximo': float(np.max(y)) if len(y) else np.nan,
                'media_ref_train': mu, 'sigma_ref_train': sigma,
                'media_desplazamiento_en_sigma': (float(np.mean(y))-mu)/sigma if len(y) and sigma > 0 else np.nan,
            }
            estad.append(registro)
            for k in (1, 2, 3):
                if sigma > 0 and np.isfinite(sigma):
                    arriba = int((y > mu + k*sigma).sum())
                    abajo = int((y < mu - k*sigma).sum())
                    estado = 'CALCULADO'
                else:
                    arriba = abajo = 0
                    estado = 'SIN_REFERENCIA_SIGMA'
                exceso.append({
                    'variable': c, 'sensor': corto, 'conjunto': p,
                    'sigma': k, 'n_validos': len(y), 'n_fuera': arriba + abajo,
                    'n_debajo': abajo, 'n_encima': arriba,
                    'porcentaje_fuera': 100*(arriba+abajo)/len(y) if len(y) and estado == 'CALCULADO' else np.nan,
                    'limite_inferior': mu-k*sigma if estado == 'CALCULADO' else np.nan,
                    'limite_superior': mu+k*sigma if estado == 'CALCULADO' else np.nan,
                    'estado': estado,
                })
    return ref, pd.DataFrame(estad), pd.DataFrame(exceso)


def preparar_visualizacion_por_bloques(x, y, bloques_max=1250):
    """Usa TODAS las observaciones; minimo/maximo preservan extremos."""
    ok = np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if not len(y):
        return np.array([]), np.array([]), np.array([]), np.array([])
    n_b = min(bloques_max, len(y))
    indices = np.array_split(np.arange(len(y)), n_b)
    xx = np.asarray([float(np.median(x[g])) for g in indices])
    med = np.asarray([float(np.median(y[g])) for g in indices])
    minimo = np.asarray([float(np.min(y[g])) for g in indices])
    maximo = np.asarray([float(np.max(y[g])) for g in indices])
    return xx, med, minimo, maximo


def dibujar_bandas(ax, mu, sig, anotar=False):
    """Dibuja siete referencias (media, +/-1, +/-2, +/-3 DS) de entrenamiento."""
    if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0):
        ax.axhline(mu, color='#21252D', lw=1.6, zorder=5)
        return
    # Regiones transparentes: +/−1 DS = verde; +/−2 = ámbar; +/−3 = rojo.
    # Bandas de mayor amplitud debajo para que el centro destaque.
    for n, color, alpha in ((3, COLOR_3, .055), (2, COLOR_2, .060), (1, COLOR_1, .10)):
        ax.axhspan(mu - n*sig, mu + n*sig, color=color, alpha=alpha, zorder=0)
    for n, color, linestyle, width in ((3, COLOR_3, ':', 1.6),
                                        (2, COLOR_2, '--', 1.65),
                                        (1, COLOR_1, '-', 1.85)):
        for signo in (-1, 1):
            y = mu + signo*n*sig
            ax.axhline(y, color=color, lw=width, ls=linestyle, alpha=.95, zorder=4)
            if anotar:
                # Signo y valor exactos del umbral; se ven separados a la derecha.
                ax.text(1.007, y, f'{"+" if signo>0 else "−"}{n}σ',
                        transform=ax.get_yaxis_transform(), va='center', ha='left',
                        fontsize=8.5, color=color, fontweight='bold', clip_on=False,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.78,'pad':.4})
    ax.axhline(mu, color='#22252D', lw=1.9, zorder=5)
    if anotar:
        ax.text(1.007, mu, 'Media', transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=8.4, color='#22252D',
                fontweight='bold', clip_on=False,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.78,'pad':.4})


def leyenda_bandas(fig):
    """Leyenda que identifica explícitamente las siete líneas estadísticas."""
    elementos = [Line2D([0],[0],color=COLORES[p],lw=2,label=p.capitalize()) for p in PARTES]
    elementos += [Line2D([0],[0],color='#22252D',lw=2,label='Media (entrenamiento)'),
                  Line2D([0],[0],color=COLOR_1,lw=2,label='±1 DS'),
                  Line2D([0],[0],color=COLOR_2,lw=2,ls='--',label='±2 DS'),
                  Line2D([0],[0],color=COLOR_3,lw=2,ls=':',label='±3 DS')]
    fig.legend(handles=elementos, loc='upper center', bbox_to_anchor=(.5,.929),
               ncol=7, fontsize=8.5, frameon=False, columnspacing=1.6,
               handlelength=2.6)


def dibujar_serie(fig, ax, pares, mu, sig, titulo, unidad, vista_zoom):
    fig.suptitle(titulo, x=.074, y=.981, ha='left', fontsize=12,
                 fontweight='semibold')
    leyenda_bandas(fig)
    for p in PARTES:
        x, med, bajo, alto = pares[p]
        if not len(x):
            continue
        # TODOS los valores quedan representados por el rango min-max del bloque.
        ax.fill_between(x, bajo, alto, color=COLORES[p], alpha=.16,
                        linewidth=0, zorder=1, rasterized=True)
        ax.plot(x, med, color=COLORES[p], lw=.78, alpha=.95,
                zorder=3, rasterized=True)
    dibujar_bandas(ax, mu, sig, anotar=vista_zoom)
    ax.set_xlabel('SMP continuo (contador de muestras; no segundos)', fontsize=10)
    ax.set_ylabel(unidad, fontsize=10)
    ax.grid(axis='y', color='#BDC5CE', alpha=.33, lw=.7)
    ax.margins(x=.004)
    fig.subplots_adjust(left=.075, right=.900, top=.805, bottom=.19)
    if vista_zoom and np.isfinite(sig) and sig>0:
        ax.set_ylim(mu - 3.45*sig, mu + 3.45*sig)
    elif not vista_zoom:
        # Sin zoom: rango de TODOS los registros + las bandas estadísticas.
        v = np.concatenate([np.array(pares[p][2], dtype=float) for p in PARTES] +
                           [np.array(pares[p][3], dtype=float) for p in PARTES])
        v = v[np.isfinite(v)]
        if len(v):
            bajo, alto = float(v.min()), float(v.max())
            if np.isfinite(sig) and sig>0:
                bajo = min(bajo, mu - 3*sig)
                alto = max(alto, mu + 3*sig)
            ancho = max(alto-bajo, np.finfo(float).eps*max(abs(mu),1.))
            ax.set_ylim(bajo-.035*ancho, alto+.035*ancho)
    nota = ('VISTA ZOOM: escala vertical de media ±3,45 DS. Los valores extremos '
            'permanecen en la base y en el grafico completo.' if vista_zoom else
            'VISTA COMPLETA: sin recorte de valores. Todos los registros '
            'contribuyen al minimo/maximo y mediana por bloque visual.')
    fig.text(.075, .074, nota, fontsize=8.7, color='#404C58')
    fig.text(.075, .045,
             'DS calculada SOLO con entrenamiento. Bandas estadísticas ≠ umbrales '
             'de falla. SMP no representa segundos.', fontsize=8.4, color='#56626D')


def grafico_serie(datos, ref, c, corto, unidad, archivo, salida):
    mu, sig = ref[c]['media'], ref[c]['sigma']
    pares = {}
    for p in PARTES:
        dfp = datos.loc[datos.conjunto == p, [SMP, c]]
        pares[p] = preparar_visualizacion_por_bloques(
            dfp[SMP].to_numpy(), dfp[c].to_numpy(float))

    # Figura 1: TODA la amplitud de la señal, incluidos valores extremos.
    f1, ax1 = plt.subplots(figsize=(14.4, 5.6))
    dibujar_serie(f1, ax1, pares, mu, sig,
                  f'{corto} | Serie completa — ±1, ±2 y ±3 desviaciones estándar',
                  unidad, vista_zoom=False)
    guardar(f1, salida / f'01_senal_{archivo}_completa.png')

    # Figura 2: misma señal y mismas referencias, escala centrada en +/-3.45 DS.
    f2, ax2 = plt.subplots(figsize=(14.4, 5.6))
    dibujar_serie(f2, ax2, pares, mu, sig,
                  f'{corto} | Zoom de desviaciones estándar (DS)',
                  unidad, vista_zoom=True)
    guardar(f2, salida / f'01_senal_{archivo}_zoom_ds.png')

    # No dejar versiones obsoletas mezcladas con las figuras nuevas.
    anterior = salida / f'01_senal_{archivo}_bandas_sigma.png'
    if anterior.is_file():
        anterior.unlink()


def grafico_distribucion(datos, ref, c, corto, unidad, archivo, salida):
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.55, 1]})
    validos = datos[c].dropna().to_numpy(float)
    lo, hi = np.quantile(validos, [.01, .99])
    if hi <= lo:
        lo, hi = float(np.min(validos))-1e-6, float(np.max(validos))+1e-6
    for p in PARTES:
        d = datos.loc[datos.conjunto == p, c].dropna().to_numpy(float)
        visibles = d[(d >= lo) & (d <= hi)]
        axs[0].hist(visibles, bins=45, density=True, histtype='step', lw=1.5,
                    color=COLORES[p], label=f'{p.capitalize()} (n={len(d):,})')
    mu, sig = ref[c]['media'], ref[c]['sigma']
    axs[0].axvline(mu, lw=1, color='#333333', ls='--', label='Media train')
    axs[0].set_xlabel(f'{unidad} | ventana P1 a P99 (solo visual)')
    axs[0].set_ylabel('Densidad (subconjunto visible)')
    axs[0].set_title('Histograma comparativo')
    axs[0].legend(fontsize=7.4, frameon=False)
    muestras = [datos.loc[datos.conjunto == p,c].dropna().to_numpy(float) for p in PARTES]
    bp = axs[1].boxplot(muestras, patch_artist=True, widths=.55, showfliers=False,
                       tick_labels=['Ent.','Val.','Prueb.'],
                       medianprops={'color':'#222222','linewidth':1.6})
    for patch, p in zip(bp['boxes'], PARTES):
        patch.set_facecolor(COLORES[p]);patch.set_alpha(.35)
    axs[1].set_title('Cajas (sin dibujar puntos extremos)')
    axs[1].set_ylabel(unidad)
    axs[1].grid(axis='y', alpha=.15)
    fig.suptitle(f'{corto} | Distribuciones de los 3 conjuntos', fontsize=12)
    fig.text(.01, -.035, 'Se usan todas las observaciones en los estadisticos. '
             'El histograma tiene zoom P1-P99; el boxplot oculta solo los marcadores atipicos.',
             fontsize=8, color='#555555')
    guardar(fig, salida / f'02_distribucion_{archivo}.png')


def grafico_heatmap_correlaciones(datos, metodo, salida):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.65), constrained_layout=True)
    ims = []
    for ax, p in zip(axes, PARTES):
        matriz = datos.loc[datos.conjunto == p, SENSORES].corr(method=metodo).to_numpy(float)
        im = ax.imshow(matriz, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ims.append(im)
        ax.set_xticks(range(10), CORTOS, rotation=60, ha='right')
        ax.set_yticks(range(10), CORTOS)
        ax.set_title(p.capitalize())
        for i in range(10):
            for j in range(10):
                valor = matriz[i,j]
                ax.text(j,i,f'{valor:.2f}' if np.isfinite(valor) else '—',ha='center',va='center',
                        fontsize=6.8, color='white' if np.isfinite(valor) and abs(valor)>.55 else '#222222')
    fig.colorbar(ims[-1], ax=axes, shrink=.75, label=f'Correlacion de {metodo.capitalize()}')
    fig.suptitle(f'Correlaciones {metodo.capitalize()} por conjunto (10 variables)', fontsize=13)
    guardar(fig, salida / f'03_correlaciones_{metodo}.png')


def grafico_excesos(excesos, salida):
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 5.2), constrained_layout=True)
    heat = None
    for ax, n in zip(axes, (1,2,3)):
        d = excesos.loc[excesos.sigma == n]
        mat = d.pivot(index='sensor',columns='conjunto',values='porcentaje_fuera').reindex(
            index=CORTOS,columns=PARTES).to_numpy(float)
        heat = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(3), ['Ent.','Val.','Prueba'])
        ax.set_yticks(range(10), CORTOS)
        ax.set_title(f'Fuera de ±{n}σ')
        for i in range(10):
            for j in range(3):
                v=mat[i,j]
                ax.text(j,i, f'{v:.1f}%' if np.isfinite(v) else 'N/D',
                        ha='center',va='center',fontsize=8,
                        color='white' if np.isfinite(v) and v>55 else '#242424')
    fig.colorbar(heat, ax=axes, shrink=.78, label='Porcentaje de mediciones')
    fig.suptitle('Desviaciones respecto a la referencia de entrenamiento', fontsize=13)
    guardar(fig, salida / '04_porcentaje_fuera_de_sigma.png')


def grafico_desplazamiento(estad, salida):
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    piv = estad.pivot(index='sensor', columns='conjunto', values='media_desplazamiento_en_sigma')
    piv=piv.reindex(index=CORTOS,columns=PARTES)
    x=np.arange(10)
    for i, p in enumerate(PARTES):
        ax.bar(x+(i-1)*.25, piv[p].to_numpy(float), width=.23,
               label=p.capitalize(), color=COLORES[p])
    ax.set_xticks(x,CORTOS)
    ax.axhline(0,color='#333333',lw=1)
    ax.set_ylabel('(Media conjunto - media train) / DS train')
    ax.set_title('Cambio de nivel medio por sensor entre conjuntos')
    ax.legend(frameon=False,ncol=3)
    ax.grid(axis='y',alpha=.16)
    fig.text(.01,-.02,'Cambios estadisticos en escala de entrenamiento; no indican por si solos averias.',
             fontsize=8,color='#666666')
    guardar(fig, salida/'05_cambio_medias_por_sensor.png')


def grafico_dispersion(datos, a, b, nombre, salida):
    # Todas las observaciones: recorte solamente visual de ejes para destacar zona central.
    xx=datos[a].to_numpy(float)
    yy=datos[b].to_numpy(float)
    ok=np.isfinite(xx)&np.isfinite(yy)
    qx=np.quantile(xx[ok],[.005,.995]);qy=np.quantile(yy[ok],[.005,.995])
    fig,ax=plt.subplots(figsize=(8.3,5.4),constrained_layout=True)
    for p in PARTES:
        d=datos.loc[datos.conjunto==p,[a,b]].dropna()
        ax.scatter(d[a],d[b],s=1.6,alpha=.09,color=COLORES[p],
                   linewidths=0,rasterized=True,label=f'{p.capitalize()} (n={len(d):,})')
    if qx[1]>qx[0] and qy[1]>qy[0]:
        rx,ry=np.diff(qx)[0],np.diff(qy)[0]
        ax.set_xlim(qx[0]-.08*rx,qx[1]+.08*rx)
        ax.set_ylim(qy[0]-.08*ry,qy[1]+.08*ry)
    ax.set_xlabel(a);ax.set_ylabel(b)
    ax.set_title(f'Dispersion entre sensores | {nombre}')
    ax.legend(markerscale=4,loc='best',fontsize=8,frameon=False)
    ax.grid(alpha=.1)
    fig.text(.01,-.018,'Zoom visual P0.5-P99.5; todas las mediciones incluidas. '
             'No son clusters ni estados de falla.',fontsize=8,color='#555555')
    guardar(fig, salida/f'06_dispersion_{nombre}.png')


def episodios_por_smp(datos, mascara, n3):
    x = datos[SMP].to_numpy(np.int64)
    conj = datos['conjunto'].to_numpy()
    activo = np.asarray(mascara,dtype=bool)
    cortes = np.r_[True,(np.diff(x)!=1)|(conj[1:]!=conj[:-1])|(~activo[:-1])]
    inicia = np.flatnonzero(activo & cortes)
    termina = np.flatnonzero(activo & np.r_[(np.diff(x)!=1)|(conj[1:]!=conj[:-1])|(~activo[1:]),True])
    if len(inicia)!=len(termina):
        raise AssertionError('Error al construir episodios')
    registros = []
    for i,(a,b) in enumerate(zip(inicia,termina),1):
        registros.append({'episodio':i,'conjunto':str(conj[a]),
                          'smp_inicio':int(x[a]),'smp_fin':int(x[b]),
                          'longitud_muestras':int(b-a+1),
                          'max_sensores_fuera_3sigma':int(np.max(n3[a:b+1]))})
    return pd.DataFrame(registros,columns=['episodio','conjunto','smp_inicio',
                 'smp_fin','longitud_muestras','max_sensores_fuera_3sigma'])


def graficar_coincidencias(datos, n3, salida):
    fig, axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True,
                           gridspec_kw={'height_ratios':[1.5,1]})
    for p in PARTES:
        d=datos.loc[datos.conjunto==p]
        indices=d.index.to_numpy()
        # Toda fila aporta al maximo por bloque visual.
        cantidad=n3[indices]
        x=d[SMP].to_numpy(np.int64)
        bloques=np.array_split(np.arange(len(x)),min(1250,len(x)))
        bx=np.asarray([np.median(x[g]) for g in bloques])
        by=np.asarray([np.max(cantidad[g]) for g in bloques])
        axes[0].plot(bx,by,color=COLORES[p],lw=.65,alpha=.85,label=p.capitalize())
    axes[0].set_ylabel('Maximo de sensores fuera ±3σ / bloque')
    axes[0].set_xlabel('SMP continuo (NO segundos)')
    axes[0].set_ylim(-.2,10.5)
    axes[0].set_title('Coincidencias de desviaciones estadisticas (referencia train)')
    axes[0].legend(frameon=False,ncol=3)
    axes[0].grid(axis='y',alpha=.15)
    x=np.arange(11)
    for i,p in enumerate(PARTES):
        val=n3[(datos.conjunto==p).to_numpy()]
        counts=np.bincount(val,minlength=11)
        axes[1].bar(x+(i-1)*.26, 100*counts/len(val),width=.24,color=COLORES[p],label=p.capitalize())
    axes[1].set_xticks(x)
    axes[1].set_xlabel('Cantidad de variables fuera de ±3σ en la misma muestra')
    axes[1].set_ylabel('Porcentaje de registros')
    axes[1].set_title('Distribucion de coincidencias por conjunto')
    axes[1].grid(axis='y',alpha=.15)
    fig.text(.01,-.02,'Anomalias ESTADISTICAS candidatas, no fallas. '
             'Los bloques visuales conservan el maximo de TODOS los datos.',
             fontsize=8,color='#555555')
    guardar(fig,salida/'07_coincidencias_multisensor.png')


def ejecutar(origen: Path, prep: Path, salida: Path):
    estilo()
    datos, meta = cargar_validado(origen,prep)
    if datos[SMP].duplicated().any():
        raise AssertionError('SMP duplicado')
    salida.mkdir(parents=True,exist_ok=True)
    ref, resumen, excesos = referencia_y_estadisticas(datos)
    csv_guardar(resumen,salida/'00_resumen_estadistico.csv')
    csv_guardar(excesos,salida/'01_excesos_sigma.csv')
    csv_guardar(pd.DataFrame([{'variable':c,'sensor':corto,**ref[c]}
                for c,corto in zip(SENSORES,CORTOS)]),salida/'02_referencia_entrenamiento.csv')

    print('09 | ANALISIS ESTADISTICO DE TODAS LAS SENALES',flush=True)
    print(f'Registros: {len(datos):,} | Variables: {len(SENSORES)}',flush=True)
    for j,(c,corto,unidad,archivo) in enumerate(zip(SENSORES,CORTOS,UNIDADES,ARCHIVOS),1):
        grafico_serie(datos,ref,c,corto,unidad,archivo,salida)
        grafico_distribucion(datos,ref,c,corto,unidad,archivo,salida)
        print(f' [{j}/10] {corto}: serie COMPLETA + zoom DS (2 archivos) + histograma y cajas',flush=True)
    for method in ('pearson','spearman'):
        grafico_heatmap_correlaciones(datos,method,salida)
    grafico_excesos(excesos,salida)
    grafico_desplazamiento(resumen,salida)
    dispersion = [
        (SENSORES[0],SENSORES[1],'acelerometro_xy'),
        (SENSORES[3],SENSORES[4],'giroscopio_xy'),
        (SENSORES[6],SENSORES[7],'magnetometro_xy'),
        (SENSORES[5],SENSORES[9],'giro_z_temperatura'),
    ]
    for a,b,nombre in dispersion:
        grafico_dispersion(datos,a,b,nombre,salida)

    arr = datos[SENSORES].to_numpy(float)
    contadores={}
    for k in (1,2,3):
        z=[]
        for j,c in enumerate(SENSORES):
            r=ref[c]
            good=np.isfinite(arr[:,j])
            if r['valida']:
                z.append((good & (np.abs(arr[:,j]-r['media'])>k*r['sigma'])).astype(np.int8))
            else:
                z.append(np.zeros(len(datos),dtype=np.int8))
        contadores[k]=np.column_stack(z).sum(axis=1)
    traza=pd.DataFrame({SMP:datos[SMP].to_numpy(np.int64),
                       'conjunto':datos['conjunto'],
                       'sensores_fuera_1sigma':contadores[1],
                       'sensores_fuera_2sigma':contadores[2],
                       'sensores_fuera_3sigma':contadores[3],
                       'candidato_exploratorio_2_sensores_3sigma':(contadores[3]>=2).astype('int8'),
                       'medicion_imputada_origen':datos['medicion_imputada'].astype('int8'),
                       'smp_estimado_origen':datos['smp_estimado'].astype('int8')})
    csv_guardar(traza,salida/'03_coincidencias_por_smp.csv')
    eventos=episodios_por_smp(datos,contadores[3]>=2,contadores[3])
    csv_guardar(eventos,salida/'04_episodios_estadisticos_exploratorios.csv')
    graficar_coincidencias(datos,contadores[3],salida)

    informe = {
        'registros':int(len(datos)), 'variables':len(SENSORES),
        'conjuntos':{p:int((datos.conjunto==p).sum()) for p in PARTES},
        'fila_imputada_en_origen':int(datos.medicion_imputada.sum()),
        'smp_estimados_en_origen':int(datos.smp_estimado.sum()),
        'n_eventos_2_sensores_fuera_3sigma':len(eventos),
        'registros_2_sensores_fuera_3sigma':int((contadores[3]>=2).sum()),
        'fuente_sha256':meta['sha256_origen'],
        'metodologia':'referencias solo train; cortes sigma estrictos; no son fallas',
        'nota_visualizacion':'cada variable: grafico de serie completa sin recorte y zoom +/-3.45 DS; todos los datos en estadisticos; series resumidas por bloques min/max/mediana para dibujar',
        'nota_smp':'contador de muestras, frecuencia temporal no confirmada',
        'normas':'sin umbrales ISO de falla',
    }
    (salida/'00_metadatos_analisis.json').write_text(
        json.dumps(informe,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nCompletado: 30 figuras por senal + 2 mapas correlacion + 2 resúmenes +'
          ' 4 dispersiones + 1 grafico multisensor = 39 graficos.',flush=True)
    print('Episodios estadisticos exploratorios:',len(eventos),flush=True)
    print('Salida:',salida.resolve(),flush=True)
    print('IMPORTANTE: desviaciones estadisticas no son diagnosticos de falla.',flush=True)


def main():
    ap=argparse.ArgumentParser(description='09 - Exploracion estadistica multisensor completa')
    ap.add_argument('--entrada',help='Ruta al CSV original corregido')
    ap.add_argument('--preparacion',help='Carpeta salida del 08, si fue personalizada')
    ap.add_argument('--salida',help='Carpeta salida de este codigo')
    args=ap.parse_args()
    raiz=Path(__file__).resolve().parent
    origen=origen_por_defecto(raiz,args.entrada)
    prep=Path(args.preparacion).expanduser() if args.preparacion else \
         raiz/'Resultados_Tarea3_Nuevos'/'00_preparacion'
    out=Path(args.salida).expanduser() if args.salida else \
        raiz/'Resultados_Tarea3_Nuevos'/'01_analisis_estadistico'
    if not prep.is_absolute():prep=raiz/prep
    if not out.is_absolute():out=raiz/out
    ejecutar(origen,prep,out)


if __name__=='__main__':
    main()
