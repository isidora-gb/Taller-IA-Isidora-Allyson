# -*- coding: utf-8 -*-
r"""
ANALISIS DE TEMPERATURA - TALLER IA 2

Entrada por defecto:
 Taller ia 2\Datos_corregidos\04_temperatura_smp_ordenado.csv
Salida por defecto:
 Taller ia 2\Resultados_temperatura

Instalacion (en tu .venv):
    python -m pip install numpy matplotlib

Ejecutar:
    python "04_analisis_temperatura.py"

Para indicar otras rutas sin editar el script:
    python "04_analisis_temperatura.py" --entrada "C:\ruta\archivo.csv" --salida "C:\ruta\salida"

METODO:
 * Conserva todos los registros completos recuperados por el codigo 00,
   con las 3 columnas de SMP sin modificarlas.
 * Aplica 6 sigma global: media +/- 6 desviaciones estandar muestrales.
 * Marca los atipicos y solo en la copia TRATADA deja vacio ese valor.
 * No supone que la temperatura este en °C ni que SMP indique segundos.
 * Verifica cuantos atipicos tambien presentan SMP original invalido.
 * Como hay una sola variable, no se calcula correlacion entre sensores.
   En su lugar se informa autocorrelacion descriptiva con la muestra siguiente.
 * 15 figuras cientificas a 300 dpi: 9 de temperatura + 6 de SMP.
 * No altera el CSV de entrada y no abre ventanas graficas.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:
    sys.exit(f"ERROR: falta una biblioteca: {exc}\nEjecuta: python -m pip install numpy matplotlib")

ENTRADA = Path(__file__).resolve().parent / "Datos_corregidos" / "04_temperatura_smp_ordenado.csv"
SALIDA = Path(__file__).resolve().parent / "Resultados_temperatura"
COLUMNAS = ["smp_original", "smp_continuo", "smp_desde_cero", "Ter_SisGenEol"]
VARIABLE = "Ter_SisGenEol"
COLOR = "#186789"
COLOR_OTRO = "#279A91"
COLOR_ATIP = "#C64B50"
COLOR_REINICIO = "#BE495A"
COLOR_SALTO = "#C28B31"
VENTANA = 1000  # promedio de 1.000 muestras; NO equivale a tiempo fijo

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 300,
})


def guardar_csv(path: Path, columnas: list[str], filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columnas)
        writer.writerows(filas)


def guardar_png(fig, destino: Path):
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {destino.relative_to(destino.parents[1])}")


def numero(valor):
    try:
        v = float(str(valor).strip().replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def cargar_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo: {path}\n"
            "Ejecuta 00_validar_y_separar_excel.py para generar Datos_corregidos/04_temperatura_smp_ordenado.csv."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=';')
        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene encabezados.")
        cabecera = [x.strip().replace("\ufeff", "") for x in reader.fieldnames]
        faltantes = [k for k in COLUMNAS if k not in cabecera]
        if faltantes:
            raise ValueError(f"Faltan columnas: {faltantes}. Disponibles: {cabecera}")
        filas = []
        for row in reader:
            fila = {k: row.get(k, "") or "" for k in COLUMNAS}
            # CSV de entrada usa decimal ',' para Excel en español;
            # convertimos a decimal '.' al escribir los CSV de resultados.
            fila[VARIABLE] = fila[VARIABLE].replace(",", ".")
            filas.append(fila)
    if not filas:
        raise ValueError("El CSV esta vacio.")

    original = np.array([numero(r["smp_original"]) for r in filas], dtype=float)
    continuo = np.array([numero(r["smp_continuo"]) for r in filas], dtype=float)
    cero = np.array([numero(r["smp_desde_cero"]) for r in filas], dtype=float)
    temperatura = np.array([numero(r[VARIABLE]) for r in filas], dtype=float)

    for nombre, datos in (("smp_continuo", continuo), ("smp_desde_cero", cero)):
        if not np.isfinite(datos).all() or not np.all(datos == np.floor(datos)):
            raise ValueError(f"{nombre} debe contener unicamente enteros finitos.")
        if not np.all(np.diff(datos) > 0):
            raise ValueError(f"{nombre} debe ser estrictamente creciente.")
    if not np.allclose(cero, continuo - continuo[0], rtol=0, atol=1e-9):
        raise ValueError("smp_desde_cero no coincide con smp_continuo - primer registro.")
    if cero[0] != 0:
        raise ValueError("smp_desde_cero debe comenzar en 0.")

    smp_valido = (np.isfinite(original) & (original >= 0) &
                  (original <= 65535) & (original == np.floor(original)))
    pos = np.flatnonzero(smp_valido)
    reinicios = np.array([], dtype=int)
    if len(pos) > 1:
        a = original[pos[:-1]]
        b = original[pos[1:]]
        reinicios = pos[1:][(a >= 65000) & (b <= 500) & (b < a)]

    incrementos = np.diff(continuo)
    saltos = np.flatnonzero(incrementos > 1) + 1
    return filas, original, continuo, cero, temperatura, smp_valido, reinicios, saltos, incrementos


def estadisticos(y: np.ndarray):
    a = y[np.isfinite(y)]
    n = int(a.size)
    if n == 0:
        return {"variable": VARIABLE, "n_validos": 0, "n_nulos": len(y),
                **{k: "" for k in ("media", "mediana", "desviacion_estandar",
                                      "varianza", "minimo", "q1_25pct", "q3_75pct",
                                      "maximo", "rango", "rango_intercuartil", "cv_pct")}}
    media = float(a.mean())
    desviacion = float(a.std(ddof=1)) if n > 1 else float("nan")
    q1, mediana, q3 = np.percentile(a, [25, 50, 75])
    return {
        "variable": VARIABLE, "n_validos": n, "n_nulos": len(y) - n,
        "media": media, "mediana": float(mediana),
        "desviacion_estandar": desviacion,
        "varianza": desviacion ** 2, "minimo": float(a.min()),
        "q1_25pct": float(q1), "q3_75pct": float(q3),
        "maximo": float(a.max()), "rango": float(a.max() - a.min()),
        "rango_intercuartil": float(q3 - q1),
        "cv_pct": 100 * desviacion / abs(media) if media != 0 else float("nan"),
    }


def media_movil(y: np.ndarray, ventana: int = VENTANA):
    """Promedio de las ultimas 'ventana' filas, ignorando valores ausentes."""
    valido = np.isfinite(y)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valido, y, 0))))
    counts = np.concatenate(([0], np.cumsum(valido.astype(int))))
    fin = np.arange(1, len(y) + 1)
    ini = np.maximum(0, fin - ventana)
    delta_n = counts[fin] - counts[ini]
    delta_s = sums[fin] - sums[ini]
    return np.divide(delta_s, delta_n, out=np.full(len(y), np.nan), where=delta_n > 0)


def autocorrelacion_consecutiva(y, continuo):
    """Pearson entre filas consecutivas SOLO si contador avanza 1."""
    par = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (np.diff(continuo) == 1)
    if int(par.sum()) < 3:
        return float("nan"), int(par.sum())
    a, b = y[:-1][par], y[1:][par]
    if a.std() == 0 or b.std() == 0:
        return float("nan"), int(par.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(par.sum())


def estilizar(ax, xlabel="smp_desde_cero (muestra, no segundos)", ylabel="Temperatura (unidad no verificada)"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.margins(x=0)


def grafico_serie(x, y, mascara, titulo, archivo, mostrar_atip=True):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, y, color=COLOR, lw=0.75, alpha=.8, rasterized=True)
    if mostrar_atip and np.any(mascara):
        ax.scatter(x[mascara], y[mascara], color=COLOR_ATIP,
                   s=13, alpha=.85, label="Fuera de 6σ", zorder=3, rasterized=True)
        ax.legend(frameon=False, loc="lower left")
    estilizar(ax)
    ax.set_title(titulo, loc="left", fontweight="bold")
    fig.tight_layout()
    guardar_png(fig, archivo)


def grafico_histograma(y, archivo, titulo):
    a = y[np.isfinite(y)]
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    if len(a):
        ax.hist(a, bins=min(65, max(10, int(np.sqrt(len(np.unique(a)))))),
                color=COLOR, edgecolor="white", linewidth=.45)
        ax.axvline(a.mean(), ls="--", lw=1.3, color=COLOR_ATIP,
                   label=f"Media = {a.mean():.4f}")
        ax.legend(frameon=False)
    ax.set_xlabel("Temperatura (unidad no verificada)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(titulo, loc="left", fontweight="bold")
    fig.tight_layout()
    guardar_png(fig, archivo)


def grafico_caja(y, archivo, titulo):
    a = y[np.isfinite(y)]
    fig, ax = plt.subplots(figsize=(5.5, 6.2))
    if len(a):
        # Boxplot usa IQR; no equivale al criterio de 6 sigma.
        ax.boxplot([a], patch_artist=True, showfliers=False, widths=.38,
                   tick_labels=["Temperatura"],
                   boxprops={"facecolor": COLOR, "alpha": .45},
                   medianprops={"color": COLOR_ATIP, "linewidth": 2})
    ax.set_ylabel("Temperatura (unidad no verificada)")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.text(.5, .01, "Atencion: los limites IQR no son limites 6σ.",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=8,
            color="#5A636D")
    fig.tight_layout()
    guardar_png(fig, archivo)


def grafico_tendencia(x, y, archivo, titulo):
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.plot(x, y, lw=.45, alpha=.28, color=COLOR, label="Medicion", rasterized=True)
    ax.plot(x, media_movil(y), lw=1.3, color=COLOR_ATIP,
            label=f"Media movil: {VENTANA:,} filas".replace(",", "."), rasterized=True)
    estilizar(ax)
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    guardar_png(fig, archivo)


def grafico_comparar(y_antes, y_despues, archivo):
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.6))
    a = y_antes[np.isfinite(y_antes)]
    b = y_despues[np.isfinite(y_despues)]
    for ax, y, nombre, color in zip(axs, [a, b], ["Antes de 6σ", "Despues de 6σ"], [COLOR, COLOR_OTRO]):
        if len(y):
            ax.hist(y, bins=50, color=color, alpha=.82,
                    edgecolor="white", linewidth=.4)
            ax.axvline(y.mean(), color=COLOR_ATIP, lw=1.2, ls="--", label=f"Media: {y.mean():.4f}")
            ax.legend(frameon=False)
        ax.set_title(nombre, fontweight="bold")
        ax.set_xlabel("Temperatura (unidad no verificada)")
        ax.set_ylabel("Frecuencia")
    fig.suptitle("Comparacion de distribuciones: antes y despues del filtrado 6σ",
                 weight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    guardar_png(fig, archivo)


def graficos_smp(original, continuo, cero, temperatura, valido, reinicios, saltos, incrementos, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    n = len(temperatura)
    filas = np.arange(n)
    original_para_grafico = original.copy()
    original_para_grafico[~valido] = np.nan

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, y, titulo, color in zip(axs,
             [original_para_grafico, continuo, cero],
             ["SMP original", "SMP continuo", "SMP desde cero"],
             ["#787F94", COLOR, COLOR_OTRO]):
        ax.plot(filas, y, lw=.75, color=color, rasterized=True)
        for pos in reinicios:
            ax.axvline(pos, color=COLOR_REINICIO, lw=.8, ls="--", alpha=.8)
        ax.set_title(titulo, loc="left", fontweight="semibold")
        ax.set_ylabel(titulo)
        ax.margins(x=0)
    axs[-1].set_xlabel("Posicion de fila (no segundos)")
    fig.suptitle("Comparacion de los tres contadores SMP", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    guardar_png(fig, carpeta / "01_comparacion_tres_smp.png")

    def graficar_con_smp(x, nombre, titulo, filename, limit=None, es_original=False):
        xx = x.copy() if limit is None else x[:limit].copy()
        yy = temperatura if limit is None else temperatura[:limit]
        fig, ax = plt.subplots(figsize=(14, 4.8))
        if es_original:
            xx[~valido[:len(xx)]] = np.nan
            # Cortar el trazo en el paso 65535 -> 0.
            for pos in reinicios:
                if pos < len(xx):
                    xx[pos] = np.nan
        ax.plot(xx, yy, lw=.65, color=COLOR, alpha=.85, rasterized=True)
        estilizar(ax, xlabel=f"{nombre} (contador de muestras, NO segundos)")
        ax.set_title(titulo, loc="left", fontweight="bold")
        if es_original:
            ax.text(.01, .02, "Contador original: segmentos superpuestos despues del reinicio",
                    transform=ax.transAxes, fontsize=8, color="#5F6875")
        fig.tight_layout()
        guardar_png(fig, carpeta / filename)

    graficar_con_smp(original_para_grafico, "smp_original", "Temperatura segun SMP original",
                    "02_temperatura_smp_original.png", es_original=True)
    graficar_con_smp(continuo, "smp_continuo", "Temperatura segun SMP continuo",
                    "03_temperatura_smp_continuo.png")
    graficar_con_smp(cero, "smp_desde_cero", "Temperatura segun SMP desde cero",
                    "04_temperatura_smp_desde_cero.png")
    graficar_con_smp(cero, "smp_desde_cero", "Detalle: primeras 3.000 muestras",
                    "05_detalle_primeras_3000_muestras.png", limit=min(n, 3000))

    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    axs[0].plot(filas, continuo, color=COLOR, lw=.85, rasterized=True)
    for pos in reinicios:
        axs[0].axvline(pos, color=COLOR_REINICIO, ls="--", lw=.8)
    axs[0].set_ylabel("SMP continuo")
    axs[0].set_title("Secuencia reconstruida", loc="left", fontweight="semibold")
    if len(saltos):
        axs[1].scatter(saltos, incrementos[saltos-1], s=11, color=COLOR_SALTO,
                       alpha=.8, label="Saltos > 1", rasterized=True)
    invalidos = np.flatnonzero(~valido)
    if len(invalidos):
        axs[1].scatter(invalidos, np.zeros(len(invalidos)), marker="|", s=12,
                       color=COLOR_REINICIO, label="SMP original invalido", rasterized=True)
    axs[1].axhline(1, color="#858D97", ls="--", lw=.8)
    axs[1].set_xlabel("Posicion de fila (no segundos)")
    axs[1].set_ylabel("Incremento SMP")
    axs[1].set_title("Saltos y registros sospechosos", loc="left", fontweight="semibold")
    axs[1].legend(frameon=False)
    axs[1].set_ylim(bottom=-.15)
    fig.suptitle("Auditoria de la reconstruccion de SMP", weight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    guardar_png(fig, carpeta / "06_revision_saltos_y_registros_invalidos.png")


def ejecutar(entrada: Path, salida: Path):
    filas, orig, cont, cero, temp, smp_valido, reinicios, saltos, incrementos = cargar_csv(entrada)
    n = len(filas)
    grupos = {
        "estadisticos": salida / "01_estadisticos",
        "atipicos": salida / "02_atipicos_6sigma",
        "antes": salida / "03_graficos_antes",
        "despues": salida / "04_graficos_despues",
        "comparacion": salida / "05_comparaciones",
        "tratados": salida / "06_datos_tratados",
        "smp": salida / "07_graficos_smp",
    }
    for p in grupos.values():
        p.mkdir(parents=True, exist_ok=True)

    antes = estadisticos(temp)
    mu = antes["media"]
    sigma = antes["desviacion_estandar"]
    if not np.isfinite(numero(mu)):
        raise ValueError("Temperatura no contiene valores numericos suficientes.")
    if not np.isfinite(numero(sigma)) or sigma == 0:
        inferior = superior = mu
        atipicos = np.zeros(n, dtype=bool)
    else:
        inferior = mu - 6 * sigma
        superior = mu + 6 * sigma
        atipicos = np.isfinite(temp) & ((temp < inferior) | (temp > superior))
    tratado = temp.copy()
    tratado[atipicos] = np.nan
    despues = estadisticos(tratado)
    n_atip = int(atipicos.sum())
    total_invalidos = int((~smp_valido).sum())
    atip_smp_invalido = int((atipicos & ~smp_valido).sum())
    atip_smp_valido = int((atipicos & smp_valido).sum())

    # 1. Estadisticas y calidad
    lista_indicadores = [
        ("n_registros", n),
        ("smp_original_invalidos", total_invalidos),
        ("smp_original_nulos", sum(not r["smp_original"].strip() for r in filas)),
        ("smp_original_reinicios_probables_16bits", len(reinicios)),
        ("smp_continuo_incrementos_mayores_a_uno", len(saltos)),
        ("smp_continuo_muestras_omitidas_estimadas", int(np.sum(incrementos - 1))),
        ("smp_desde_cero_inicial", int(cero[0])),
        ("smp_desde_cero_final", int(cero[-1])),
        ("temperatura_nulos_o_no_numericos", int((~np.isfinite(temp)).sum())),
        ("temperatura_atipicos_6sigma", n_atip),
        ("temperatura_atipicos_6sigma_con_smp_invalido", atip_smp_invalido),
    ]
    guardar_csv(grupos["estadisticos"] / "calidad_datos.csv",
                ["indicador", "valor"], lista_indicadores)
    claves = list(antes.keys())
    guardar_csv(grupos["estadisticos"] / "estadisticos_antes.csv",
                claves, [[antes[k] for k in claves]])
    guardar_csv(grupos["estadisticos"] / "estadisticos_despues.csv",
                claves, [[despues[k] for k in claves]])

    # 2. Anomalias 6sigma y coincidencia con SMP invalido.
    guardar_csv(grupos["atipicos"] / "limites_y_conteos_6sigma.csv",
                ["variable", "media", "desviacion_estandar_muestral", "limite_inferior",
                 "limite_superior", "n_atipicos", "porcentaje_atipicos"],
                [[VARIABLE, mu, sigma, inferior, superior, n_atip, 100*n_atip/n]])
    guardar_csv(grupos["atipicos"] / "cruce_atipicos_y_smp.csv",
                ["grupo", "n_registros"], [
                    ["atipicos_6sigma_y_smp_original_invalido", atip_smp_invalido],
                    ["atipicos_6sigma_y_smp_original_valido", atip_smp_valido],
                    ["no_atipicos_y_smp_original_invalido", total_invalidos-atip_smp_invalido],
                    ["no_atipicos_y_smp_original_valido", n-total_invalidos-atip_smp_valido],
                ])
    guardar_csv(grupos["atipicos"] / "registro_atipicos_6sigma.csv",
                ["fila_original", *COLUMNAS[:3], "variable", "valor_original",
                 "limite_inferior", "limite_superior", "smp_original_valido",
                 "tratamiento"],
                ([i+1, *(filas[i][c] for c in COLUMNAS[:3]), VARIABLE,
                  filas[i][VARIABLE], inferior, superior, bool(smp_valido[i]),
                  "Marcado; vacio solo en CSV tratado"]
                 for i in np.flatnonzero(atipicos)))

    # 3. Conservar original y colocar VACIO solo para valores atipicos en tratado.
    guardar_csv(grupos["tratados"] / "temperatura_marcada_6sigma.csv",
                ["fila_original", *COLUMNAS, "atipico_temperatura_6sigma", "smp_original_valido"],
                ([i+1, *(fila[c] for c in COLUMNAS), bool(atipicos[i]), bool(smp_valido[i])]
                 for i, fila in enumerate(filas)))
    guardar_csv(grupos["tratados"] / "temperatura_tratada_6sigma.csv",
                COLUMNAS,
                ([*(fila[c] for c in COLUMNAS[:3]),
                  "" if atipicos[i] else fila[VARIABLE]]
                 for i, fila in enumerate(filas)))

    # 4. Comparaciones y autocorrelacion entre muestras consecutivas validas.
    campos_comparacion = ["indicador", "antes", "despues", "diferencia_despues_menos_antes"]
    indicadores = ["n_validos", "n_nulos", "media", "mediana", "desviacion_estandar",
                   "varianza", "minimo", "q1_25pct", "q3_75pct", "maximo", "rango",
                   "rango_intercuartil", "cv_pct"]
    comparaciones = []
    for k in indicadores:
        a, b = antes[k], despues[k]
        d = b-a if isinstance(a, (int,float)) and isinstance(b, (int,float)) else ""
        comparaciones.append([k, a, b, d])
    guardar_csv(grupos["comparacion"] / "comparacion_estadisticos.csv",
                campos_comparacion, comparaciones)
    ac_antes, pares_antes = autocorrelacion_consecutiva(temp, cont)
    ac_despues, pares_despues = autocorrelacion_consecutiva(tratado, cont)
    guardar_csv(grupos["comparacion"] / "autocorrelacion_lag1.csv",
                ["etapa", "autocorrelacion_lag1", "pares_consecutivos_validos", "nota"],
                [["antes", ac_antes, pares_antes, "Entre muestras consecutivas; no implica escala de tiempo"],
                 ["despues", ac_despues, pares_despues, "Entre muestras consecutivas; no implica escala de tiempo"]])

    # 5. Figuras cientificas a 300 dpi.
    print("Generando 15 graficos PNG a 300 dpi...")
    grafico_serie(cero, temp, atipicos, "Temperatura | mediciones originales (6σ destacado)",
                  grupos["antes"] / "01_serie_antes.png")
    grafico_histograma(temp, grupos["antes"] / "02_histograma_antes.png",
                       "Distribucion de temperatura | antes de 6σ")
    grafico_caja(temp, grupos["antes"] / "03_boxplot_antes.png",
                 "Temperatura | diagrama de caja antes de 6σ")
    grafico_tendencia(cero, temp, grupos["antes"] / "04_tendencia_movil_antes.png",
                      "Temperatura | tendencia antes de 6σ")
    grafico_serie(cero, tratado, atipicos, "Temperatura | despues de 6σ (atipicos vacios)",
                  grupos["despues"] / "01_serie_despues.png", mostrar_atip=False)
    grafico_histograma(tratado, grupos["despues"] / "02_histograma_despues.png",
                       "Distribucion de temperatura | despues de 6σ")
    grafico_caja(tratado, grupos["despues"] / "03_boxplot_despues.png",
                 "Temperatura | diagrama de caja despues de 6σ")
    grafico_tendencia(cero, tratado, grupos["despues"] / "04_tendencia_movil_despues.png",
                      "Temperatura | tendencia despues de 6σ")
    grafico_comparar(temp, tratado, grupos["comparacion"] / "01_comparacion_distribuciones.png")
    graficos_smp(orig, cont, cero, temp, smp_valido, reinicios, saltos,
                 incrementos, grupos["smp"])

    resumen = (
        "ANALISIS DE TEMPERATURA - METODO 6 SIGMA\n"
        f"Entrada: {entrada}\nSalida: {salida}\n"
        f"Total registros: {n}\n"
        f"No numericos/vacios en temperatura: {int((~np.isfinite(temp)).sum())}\n"
        f"Media original: {mu:.8f}; desv. muestral: {sigma:.8f}\n"
        f"Limites 6σ: [{inferior:.8f}, {superior:.8f}]\n"
        f"Atipicos detectados: {n_atip} ({n_atip / n * 100:.4f}%)\n"
        f"Coinciden con SMP original invalido: {atip_smp_invalido}\n"
        f"SMP original invalidos: {total_invalidos}\n"
        f"Reinicios 16 bits probables: {len(reinicios)}\n"
        f"Saltos de contador > 1: {len(saltos)}\n"
        "\nNOTAS METODOLOGICAS:\n"
        "- Valores fuera de 6σ no prueban por si solos una falla real.\n"
        "- El cruce con SMP original invalido debe investigarse antes de inferir anomalias termicas.\n"
        "- El CSV original no se modifica y no se eliminan filas.\n"
        "- En el CSV tratado, los valores atipicos se reemplazan por celdas vacias.\n"
        "- Los 3 SMP permanecen inalterados, incluido el original sospechoso.\n"
        "- El eje X representa muestras, NO tiempo real: falta frecuencia de muestreo.\n"
        "- Las unidades de Ter_SisGenEol no estan confirmadas en los metadatos.\n"
        "- El criterio de boxplot es IQR (no 6sigma).\n"
        "- La autocorrelacion lag1 describe pares de muestras consecutivas; no es correlacion entre sensores.\n"
        "- Los promedios moviles tienen una ventana de 1000 filas, no de segundos.\n"
        "- No se infieren ciclos ni periodicidad de mantenimiento a partir de SMP por si solo.\n"
    )
    (salida / "LEEME_resultados.txt").write_text(resumen, encoding="utf-8-sig")
    print("\n" + resumen)
    print("Listo: archivos generados y CSV de entrada conservado.")


def main():
    parser = argparse.ArgumentParser(description="Analisis de temperatura: 6 sigma y tres SMP")
    parser.add_argument("--entrada", type=Path, default=ENTRADA, help="CSV de entrada")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="Carpeta de resultados")
    args = parser.parse_args()
    try:
        ejecutar(args.entrada, args.salida)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
