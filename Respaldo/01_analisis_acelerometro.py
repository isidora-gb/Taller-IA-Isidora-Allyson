# -*- coding: utf-8 -*-
r"""
Analisis del acelerometro - Taller IA 2.

Entrada por defecto:
 Taller ia 2\Datos_corregidos\01_acelerometro_smp_ordenado.csv
Salida por defecto:
 Taller ia 2\Resultados_acelerometro

Instalar dependencias una sola vez:
    python -m pip install pandas numpy matplotlib

Ejecutar (PowerShell):
    python "01_analisis_acelerometro.py"

Para probar con otras rutas, sin modificar este script:
    python "01_analisis_acelerometro.py" --entrada "C:\ruta\archivo.csv" --salida "C:\ruta\Resultados"

Metodologia: limite por eje = media +/- 6 * desviacion estandar muestral.
Los extremos se identifican; en el CSV tratado solo esos valores se
sustituyen por NaN. No se eliminan filas ni se interpolan datos.

Se conservan los tres SMP: smp_original, smp_continuo, smp_desde_cero.
Los graficos de series usan smp_desde_cero como eje horizontal (no segundos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # Evita ventanas emergentes de graficos.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:
    print("Falta una biblioteca:", exc)
    print("Instalala con: python -m pip install pandas numpy matplotlib")
    sys.exit(1)


ENTRADA = Path(__file__).resolve().parent / "Datos_corregidos" / "01_acelerometro_smp_ordenado.csv"
SALIDA = Path(__file__).resolve().parent / "Resultados_acelerometro"
EJES = {
    "X": "AcelX1_SisGenEol",
    "Y": "AcelY1_SisGenEol",
    "Z": "AcelZ1_SisGenEol",
}
COLORES = {"X": "#166088", "Y": "#28a59a", "Z": "#7953a9"}
COLOR_ATIPICO = "#ca3e47"
SMPS = ["smp_original", "smp_continuo", "smp_desde_cero"]


# Estilo consistente con figuras para informes academicos.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 300,
})


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, encoding="utf-8-sig")


def guardar_figura(fig, ruta: Path) -> None:
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Grafico:", ruta.name)


def estadisticos(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for eje, col in EJES.items():
        serie = df[col].dropna()
        if serie.empty:
            filas.append({"eje": eje, "variable": col, "n_validos": 0,
                           "n_nulos": len(df)})
            continue
        media = serie.mean()
        desv = serie.std(ddof=1)
        filas.append({
            "eje": eje, "variable": col,
            "n_validos": int(serie.size),
            "n_nulos": int(df[col].isna().sum()),
            "media": media, "mediana": serie.median(),
            "desviacion_estandar": desv,
            "varianza": serie.var(ddof=1),
            "minimo": serie.min(), "q1_25pct": serie.quantile(.25),
            "q3_75pct": serie.quantile(.75), "maximo": serie.max(),
            "rango": serie.max() - serie.min(),
            "rango_intercuartil": serie.quantile(.75) - serie.quantile(.25),
            "cv_pct": (100 * desv / abs(media)) if media != 0 else np.nan,
        })
    return pd.DataFrame(filas)


def matriz_correlaciones(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(EJES.values())
    c = df[cols].corr(method="pearson", min_periods=2)
    c.index = list(EJES)
    c.columns = list(EJES)
    c.index.name = "eje"
    return c


def grafico_serie(df, ruta, titulo, mascaras=None):
    # SMP reconstruido: preserva los posibles saltos de muestra; NO son segundos.
    x = df["smp_desde_cero"].to_numpy(dtype=float)
    fig, axs = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    for ax, (eje, col) in zip(axs, EJES.items()):
        ax.plot(x, df[col].to_numpy(dtype=float), color=COLORES[eje],
                lw=.55, alpha=.9, rasterized=True)
        if mascaras is not None and mascaras[eje].any():
            selec = mascaras[eje].to_numpy(dtype=bool)
            ax.scatter(x[selec], df.loc[selec, col], s=8,
                       color=COLOR_ATIPICO, alpha=.8, zorder=3,
                       label="Fuera de 6σ", rasterized=True)
            ax.legend(loc="upper right", frameon=False)
        ax.set_ylabel(f"Eje {eje}\n(ud. no especificada)")
        ax.set_title(f"{eje}: {col}", loc="left", fontweight="semibold")
        ax.margins(x=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axs[-1].set_xlabel("smp_desde_cero (contador de muestras, no segundos)")
    fig.suptitle(titulo, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .955])
    guardar_figura(fig, ruta)


def grafico_histogramas(df, ruta, titulo):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, (eje, col) in zip(axs, EJES.items()):
        s = df[col].dropna()
        if not s.empty:
            ax.hist(s, bins=70, color=COLORES[eje], edgecolor="white", linewidth=.3)
            ax.axvline(s.mean(), color="#333333", lw=1.15, ls="--", label="Media")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Acelerometro: eje {eje}", fontweight="semibold")
        ax.set_xlabel("Medicion (unidad no especificada)")
        ax.set_ylabel("Frecuencia")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    fig.suptitle(titulo, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .92])
    guardar_figura(fig, ruta)


def grafico_cajas(df, ruta, titulo):
    # Un panel por eje: las magnitudes de X/Y/Z son diferentes.
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (eje, col) in zip(axs, EJES.items()):
        s = df[col].dropna().to_numpy()
        if len(s):
            ax.boxplot([s], tick_labels=[eje], widths=.4, patch_artist=True,
                       boxprops={"facecolor": COLORES[eje], "alpha": .55},
                       medianprops={"color": "black", "linewidth": 1.4},
                       flierprops={"marker": ".", "markersize": 1.5,
                                   "markeredgecolor": COLOR_ATIPICO, "alpha": .55})
        ax.set_ylabel("Medicion (unidad no especificada)")
        ax.set_title(f"Eje {eje}")
    fig.suptitle(titulo, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .92])
    guardar_figura(fig, ruta)


def grafico_corr(matriz, ruta, titulo):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    datos = matriz.to_numpy(dtype=float)
    im = ax.imshow(datos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3), labels=list(EJES))
    ax.set_yticks(range(3), labels=list(EJES))
    for i in range(3):
        for j in range(3):
            val = datos[i, j]
            etiqueta = f"{val:.3f}" if np.isfinite(val) else "N/D"
            ax.text(j, i, etiqueta, ha="center", va="center",
                    color="white" if np.isfinite(val) and abs(val) > .55 else "black",
                    fontweight="semibold")
    fig.colorbar(im, ax=ax, shrink=.8, label="Correlacion de Pearson")
    ax.set_title(titulo, fontweight="bold", pad=15)
    fig.tight_layout()
    guardar_figura(fig, ruta)


def grafico_diferencias(tabla, ruta):
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4))
    for ax, eje in zip(axs, EJES):
        s = tabla.loc[tabla["eje"] == eje].iloc[0]
        vals = [s["media_antes"], s["media_despues"]]
        ax.bar(["Antes", "Despues"], vals,
               color=["#8796a3", COLORES[eje]], width=.58)
        ax.set_title(f"Eje {eje}")
        ax.set_ylabel("Media (unidad no especificada)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Comparacion de medias antes y despues de 6σ",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .92])
    guardar_figura(fig, ruta)


def ejecutar(entrada: Path, salida: Path):
    if not entrada.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo:\n{entrada}\n"
            "Verifica el nombre o indica otra ruta con --entrada."
        )
    carpetas = {
        "estadisticos": salida / "01_estadisticos",
        "atipicos": salida / "02_atipicos_6sigma",
        "antes": salida / "03_graficos_antes",
        "despues": salida / "04_graficos_despues",
        "comparaciones": salida / "05_comparaciones",
        "tratados": salida / "06_datos_tratados",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(entrada, encoding="utf-8-sig", sep=";", decimal=",", low_memory=False)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    requeridas = [*SMPS, *EJES.values()]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}. Columnas disponibles: {df.columns.tolist()}\n"
            "Debes ejecutar 00_validar_y_separar_excel.py para generar Datos_corregidos/01_acelerometro_smp_ordenado.csv"
        )
    # Nunca se reordenan ni eliminan registros: conservamos el orden original.
    df = df[requeridas].copy()
    if df.empty:
        raise ValueError("El archivo CSV esta vacio.")
    conversiones_invalidas = {}
    for col in EJES.values():
        nulos_originales = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        conversiones_invalidas[col] = int(df[col].isna().sum() - nulos_originales)

    # SMP original: se analiza para control de calidad, SIN corregirlo aqui.
    smp_original = pd.to_numeric(df["smp_original"], errors="coerce")
    es_valido_original = (
        smp_original.notna()
        & smp_original.between(0, 65535)
        & (smp_original % 1 == 0)
    )
    smp_invalidos = ~es_valido_original
    smp_no_enteros = smp_original.notna() & (smp_original % 1 != 0)
    smp_fuera_rango = smp_original.notna() & ~smp_original.between(0, 65535)
    delta_original = smp_original.diff()

    # Los SMP reconstruidos deben conservar la secuencia del CSV.
    for col in ("smp_continuo", "smp_desde_cero"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"La columna {col} tiene valores nulos o no numericos.")
        if not (df[col].diff().iloc[1:] > 0).all():
            raise ValueError(f"La columna {col} no es estrictamente creciente.")
    if not np.allclose(
        df["smp_continuo"].to_numpy(dtype=float) - float(df["smp_continuo"].iloc[0]),
        df["smp_desde_cero"].to_numpy(dtype=float), rtol=0, atol=1e-9
    ):
        raise ValueError("smp_desde_cero no coincide con smp_continuo menos su valor inicial.")

    delta_continuo = df["smp_continuo"].diff()
    # El paso de 65535 a 0 es consistente con el desbordamiento de 16 bits;
    # no prueba por si solo que la adquisicion haya sido continua.
    posiciones_validas = np.flatnonzero(es_valido_original.to_numpy())
    valores_validos = smp_original.iloc[posiciones_validas].to_numpy(dtype=float)
    reinicios = int(((valores_validos[:-1] >= 65000)
                     & (valores_validos[1:] <= 500)
                     & (valores_validos[1:] < valores_validos[:-1])).sum())
    calidad = [
        ("total_registros", len(df)),
        ("columnas_analizadas", len(df.columns)),
        ("filas_duplicadas_exactas", int(df.duplicated().sum())),
        ("smp_original_nulos", int(df["smp_original"].isna().sum())),
        ("smp_original_no_numericos_adicionales", int(smp_original.isna().sum() - df["smp_original"].isna().sum())),
        ("smp_original_valores_no_enteros", int(smp_no_enteros.sum())),
        ("smp_original_fuera_rango_0_65535", int(smp_fuera_rango.sum())),
        ("smp_original_invalidos_total", int(smp_invalidos.sum())),
        ("smp_original_saltos_negativos", int((delta_original < 0).sum())),
        ("smp_original_reinicios_probables_16bits", reinicios),
        ("smp_continuo_incrementos_mayores_a_uno", int((delta_continuo > 1).sum())),
        ("smp_continuo_muestras_omitidas_estimadas", int((delta_continuo - 1).clip(lower=0).sum())),
        ("smp_desde_cero_valor_inicial", int(df["smp_desde_cero"].iloc[0])),
        ("smp_desde_cero_valor_final", int(df["smp_desde_cero"].iloc[-1])),
    ]
    for eje, col in EJES.items():
        calidad.append((f"{eje}_nulos", int(df[col].isna().sum())))
        calidad.append((f"{eje}_no_numericos_convertidos_a_nulos", conversiones_invalidas[col]))
    guardar_csv(pd.DataFrame(calidad, columns=["indicador", "valor"]),
                carpetas["estadisticos"] / "calidad_datos.csv")

    antes = estadisticos(df)
    guardar_csv(antes, carpetas["estadisticos"] / "estadisticos_antes.csv")
    corr_antes = matriz_correlaciones(df)
    guardar_csv(corr_antes.reset_index(), carpetas["comparaciones"] / "correlacion_antes.csv")

    # Se calcula un limite global, independiente para cada eje.
    mascaras = {}
    limites = []
    detalle = []
    marcado = df.copy()
    marcado.insert(0, "fila_original", np.arange(1, len(df) + 1))
    tratado = df.copy()
    for eje, col in EJES.items():
        serie = df[col]
        mu = serie.mean()
        sigma = serie.std(ddof=1)
        if pd.notna(sigma) and sigma > 0:
            li, ls = mu - 6 * sigma, mu + 6 * sigma
            mascara = (serie < li) | (serie > ls)
        else:
            li, ls = mu, mu
            mascara = pd.Series(False, index=df.index)
        mascaras[eje] = mascara.fillna(False)
        marcado[f"atipico_{eje}_6sigma"] = mascaras[eje]
        n = int(mascaras[eje].sum())
        limites.append({
            "eje": eje, "variable": col,
            "media": mu, "desviacion_estandar_muestral": sigma,
            "limite_inferior": li, "limite_superior": ls,
            "n_atipicos": n, "porcentaje_atipicos": n / len(df) * 100,
        })
        if n:
            indices = df.index[mascaras[eje]]
            detalle.append(pd.DataFrame({
                "fila_original": indices.to_numpy() + 1,
                "smp_original": df.loc[indices, "smp_original"].to_numpy(),
                "smp_continuo": df.loc[indices, "smp_continuo"].to_numpy(),
                "smp_desde_cero": df.loc[indices, "smp_desde_cero"].to_numpy(),
                "eje": eje, "variable": col,
                "valor_original": df.loc[indices, col].to_numpy(),
                "limite_inferior": li, "limite_superior": ls,
                "tratamiento": "Marcado como atipico y reemplazado por NaN solo en el CSV tratado",
            }))
        tratado.loc[mascaras[eje], col] = np.nan
    marcado["atipico_cualquier_eje_6sigma"] = pd.concat(mascaras, axis=1).any(axis=1)
    guardar_csv(marcado, carpetas["tratados"] / "acelerometro_marcado_6sigma.csv")
    guardar_csv(tratado, carpetas["tratados"] / "acelerometro_tratado_6sigma.csv")
    columnas_registro = ["fila_original", *SMPS, "eje", "variable",
                         "valor_original", "limite_inferior", "limite_superior", "tratamiento"]
    registro = pd.concat(detalle, ignore_index=True) if detalle else pd.DataFrame(columns=columnas_registro)
    guardar_csv(registro, carpetas["atipicos"] / "registro_atipicos_6sigma.csv")
    guardar_csv(pd.DataFrame(limites), carpetas["atipicos"] / "limites_y_conteos_6sigma.csv")

    despues = estadisticos(tratado)
    guardar_csv(despues, carpetas["estadisticos"] / "estadisticos_despues.csv")
    corr_despues = matriz_correlaciones(tratado)
    guardar_csv(corr_despues.reset_index(), carpetas["comparaciones"] / "correlacion_despues.csv")
    comparacion = antes.merge(despues, on=["eje", "variable"], suffixes=("_antes", "_despues"))
    comparacion["atipicos_6sigma"] = [int(mascaras[eje].sum()) for eje in EJES]
    guardar_csv(comparacion, carpetas["comparaciones"] / "comparacion_estadisticos.csv")
    cambios_corr = (corr_despues - corr_antes).reset_index()
    guardar_csv(cambios_corr, carpetas["comparaciones"] / "cambio_correlaciones_despues_menos_antes.csv")

    print("Generando figuras en PNG (300 dpi)...")
    grafico_serie(df, carpetas["antes"] / "01_series_antes.png",
                  "Acelerometro - mediciones originales", mascaras)
    grafico_histogramas(df, carpetas["antes"] / "02_histogramas_antes.png",
                        "Distribucion por eje - antes de 6σ")
    grafico_cajas(df, carpetas["antes"] / "03_boxplots_antes.png",
                   "Diagramas de caja - antes de 6σ")
    grafico_corr(corr_antes, carpetas["antes"] / "04_correlacion_antes.png",
                 "Pearson: antes de 6σ")
    grafico_serie(tratado, carpetas["despues"] / "01_series_despues.png",
                  "Acelerometro - despues del tratamiento 6σ")
    grafico_histogramas(tratado, carpetas["despues"] / "02_histogramas_despues.png",
                        "Distribucion por eje - despues de 6σ")
    grafico_cajas(tratado, carpetas["despues"] / "03_boxplots_despues.png",
                   "Diagramas de caja - despues de 6σ")
    grafico_corr(corr_despues, carpetas["despues"] / "04_correlacion_despues.png",
                 "Pearson: despues de 6σ")
    grafico_diferencias(comparacion, carpetas["comparaciones"] / "01_comparacion_medias.png")

    n_filas_atipicas = int(marcado["atipico_cualquier_eje_6sigma"].sum())
    total_por_eje = ", ".join(f"{eje}={int(m.sum())}" for eje, m in mascaras.items())
    resumen = (
        "ANALISIS ACELEROMETRO - METODO 6 SIGMA\n"
        f"Archivo de entrada: {entrada}\nCarpeta resultados: {salida}\n"
        f"Registros: {len(df)}\n"
        f"Atipicos por eje: {total_por_eje}\n"
        f"Filas con al menos un eje atipico: {n_filas_atipicas}\n"
        f"smp_original: {int(smp_invalidos.sum())} invalidos; "
        f"{int(smp_no_enteros.sum())} decimales; "
        f"{reinicios} reinicios probables de contador de 16 bits.\n"
        f"smp_continuo: {int((delta_continuo > 1).sum())} saltos superiores a uno.\n\n"
        "METODOLOGIA\n"
        "- Se usa la media global +/- 6 veces la desviacion estandar muestral por eje.\n"
        "- Un valor fuera del intervalo es marcado, NO se considera necesariamente un error.\n"
        "- El CSV tratado reemplaza por NaN solo las celdas identificadas.\n"
        "- Se conservan las 3 columnas: smp_original, smp_continuo y smp_desde_cero.\n"
        "- Los graficos de series usan smp_desde_cero; no es tiempo en segundos.\n"
        "- Los reinicios y saltos del contador se documentan, no se corrigen de nuevo.\n"
        "- La correlacion de Pearson usa los pares de valores disponibles.\n"
        "- El boxplot aplica criterio IQR, que NO equivale a 6 sigma.\n"
        "- No se concluye frecuencia de muestreo, velocidad, mantenimiento ni fallas con estos datos.\n"
        "- El tratamiento 6 sigma no debe reemplazar inspeccion de eventos/anomalias reales.\n"
    )
    (salida / "LEEME_resultados.txt").write_text(resumen, encoding="utf-8-sig")
    print("\n" + resumen)
    print("Proceso terminado. Archivo original intacto.")


def main():
    ap = argparse.ArgumentParser(description="Analisis del acelerometro por 6 sigma")
    ap.add_argument("--entrada", type=Path, default=ENTRADA, help="CSV de entrada")
    ap.add_argument("--salida", type=Path, default=SALIDA, help="Carpeta de resultados")
    args = ap.parse_args()
    try:
        ejecutar(args.entrada, args.salida)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
