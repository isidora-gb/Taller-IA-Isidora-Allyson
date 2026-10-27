# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import warnings
import unicodedata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

VARIABLES = [
    "AcelX1_SisGenEol", "AcelY1_SisGenEol", "AcelZ1_SisGenEol",
    "GiroX_SisGenEol", "GiroY_SisGenEol", "GiroZ_SisGenEol",
    "MagneX_SisGenEol", "MagneY_SisGenEol", "MagneZ_SisGenEol",
    "Ter_SisGenEol",
]
SENSORES = {
    "acelerometro": ["AcelX1_SisGenEol", "AcelY1_SisGenEol", "AcelZ1_SisGenEol"],
    "giroscopio": ["GiroX_SisGenEol", "GiroY_SisGenEol", "GiroZ_SisGenEol"],
    "magnetometro": ["MagneX_SisGenEol", "MagneY_SisGenEol", "MagneZ_SisGenEol"],
    "temperatura": ["Ter_SisGenEol"],
}
LIMITE_ALERTA = 3.5
LIMITE_CRITICO = 6.0
MIN_OBS = 10

COLOR_BAR = "#245F83"
COLOR_LINE = "#CB7940"
COLOR_ARROW = "#355C7D"
COLORES_ESTADO = {
    "NORMAL": "#2A9D55",
    "ALERTA": "#E09A2B",
    "CRITICO_ESTADISTICO": "#C63E4A",
    "SIN_REFERENCIA": "#8A8A8A",
    "SIN_DATO": "#B5B5B5",
}

def configurar_estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })

def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df.to_csv(ruta, index=False, sep=";", decimal=",", encoding="utf-8-sig")

def leer_csv(ruta: Path) -> pd.DataFrame:
    with open(ruta, "r", encoding="utf-8-sig") as f:
        cab = f.readline()
    sep = ";" if cab.count(";") > cab.count(",") else ","
    decimal = "," if sep == ";" else "."
    return pd.read_csv(ruta, sep=sep, decimal=decimal, encoding="utf-8-sig", low_memory=False)

def a_num(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    txt = serie.astype("string").str.strip()
    direct = pd.to_numeric(txt, errors="coerce")
    comma = pd.to_numeric(txt.str.replace(",", ".", regex=False), errors="coerce")
    european = pd.to_numeric(
        txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    mejor = max((direct, comma, european), key=lambda s: int(s.notna().sum()))
    return mejor.astype(float).replace([np.inf, -np.inf], np.nan)

def simplificar(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(texto).lower())
        if not unicodedata.combining(c)
    )

def buscar_entrada(raiz: Path, entrada_usuario: str | None):
    if entrada_usuario:
        ruta = Path(entrada_usuario).expanduser()
        if not ruta.is_absolute():
            ruta = raiz / ruta
        if not ruta.is_file():
            raise SystemExit(f"No existe el archivo indicado: {ruta}")
        return ruta.resolve(), None
    candidatos_tratados = [
        raiz / "Datos_tratados" / "datos_multisensor_tratados_6sigma.csv",
        raiz / "datos_multisensor_tratados_6sigma.csv",
    ]
    candidatos_corregidos = [
        raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv",
        raiz / "Datos_corregidos_desde_excel" / "datos_multisensor_corregidos.csv",
        raiz / "datos_multisensor_corregidos.csv",
        Path("/mnt/data/Datos_corregidos_desde_excel/datos_multisensor_corregidos.csv"),
    ]
    tratado = next((p for p in candidatos_tratados if p.is_file()), None)
    corregido = next((p for p in candidatos_corregidos if p.is_file()), None)
    if tratado:
        return tratado.resolve(), corregido.resolve() if corregido else None
    if corregido:
        return corregido.resolve(), corregido.resolve()
    raise SystemExit("No encontre el CSV multisensor tratado ni el corregido. "
                     "Ubicalo en Datos_tratados/ o Datos_corregidos/.")

def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in VARIABLES:
        x = a_num(df[c])
        mu = x.mean()
        sd = x.std(ddof=1)
        filas.append({
            "variable": c,
            "registros": len(x),
            "validos": int(x.notna().sum()),
            "nulos": int(x.isna().sum()),
            "media": mu,
            "DS": sd,
            "minimo": x.min(),
            "maximo": x.max(),
            "limite_inferior_6sigma": mu - 6 * sd if np.isfinite(sd) else np.nan,
            "limite_superior_6sigma": mu + 6 * sd if np.isfinite(sd) else np.nan,
        })
    return pd.DataFrame(filas)

def graficar_matriz(corr: pd.DataFrame, titulo: str, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9))
    mat = corr.to_numpy(dtype=float)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(VARIABLES)), labels=VARIABLES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(VARIABLES)), labels=VARIABLES, fontsize=8)
    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            valor = mat[i, j]
            ax.text(j, i, f"{valor:.2f}" if np.isfinite(valor) else "—",
                    ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.83)
    cbar.set_label("Coeficiente")
    ax.set_title(titulo, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)

def tratar_6sigma_en_memoria(df_corregido: pd.DataFrame) -> pd.DataFrame:
    d = df_corregido.copy()
    for c in VARIABLES:
        x = a_num(d[c])
        mu, ds = x.mean(), x.std(ddof=1)
        if np.isfinite(ds) and ds > 0:
            mascara = (x - mu).abs() > 6 * ds
            d.loc[mascara, c] = np.nan
        d[c] = a_num(d[c])
    return d

def calcular_estado_variable(serie: pd.Series) -> tuple[np.ndarray, dict]:
    x = a_num(serie).to_numpy(dtype=float)
    clases = np.full(len(x), "SIN_DATO", dtype=object)
    validas = np.isfinite(x)
    n = int(validas.sum())
    meta = {"n_validos": n, "mediana": np.nan, "mad": np.nan, "escala": np.nan, "referencia": "APTA"}
    if n < MIN_OBS:
        clases[validas] = "SIN_REFERENCIA"
        meta["referencia"] = "SIN_REFERENCIA"
        return clases, meta
    med = float(np.median(x[validas]))
    mad = float(np.median(np.abs(x[validas] - med)))
    meta.update({"mediana": med, "mad": mad})
    if (not np.isfinite(mad)) or mad <= 0:
        clases[validas] = "SIN_REFERENCIA"
        meta["referencia"] = "SIN_REFERENCIA"
        return clases, meta
    escala = mad / 0.6745
    meta["escala"] = escala
    z = np.full(len(x), np.nan)
    z[validas] = np.abs((x[validas] - med) / escala)
    clases[validas] = "NORMAL"
    clases[validas & (z > LIMITE_ALERTA)] = "ALERTA"
    clases[validas & (z > LIMITE_CRITICO)] = "CRITICO_ESTADISTICO"
    return clases, meta

def combinar_estados(df_base: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_base.index)
    for sensor, cols in SENSORES.items():
        for c in cols:
            clases, _ = calcular_estado_variable(df_base[c])
            out[f"estado__{c}"] = clases
        estados_sensor = np.full(len(df_base), "NORMAL", dtype=object)
        sub = out[[f"estado__{c}" for c in cols]].to_numpy(dtype=object)
        for i in range(len(df_base)):
            fila = list(sub[i])
            if "CRITICO_ESTADISTICO" in fila:
                estados_sensor[i] = "CRITICO_ESTADISTICO"
            elif "ALERTA" in fila:
                estados_sensor[i] = "ALERTA"
            elif all(v == "SIN_DATO" for v in fila):
                estados_sensor[i] = "SIN_DATO"
            elif "SIN_REFERENCIA" in fila:
                estados_sensor[i] = "SIN_REFERENCIA"
            else:
                estados_sensor[i] = "NORMAL"
        out[f"estado_sensor__{sensor}"] = estados_sensor
    globales = np.full(len(df_base), "NORMAL", dtype=object)
    cols_sensor = [f"estado_sensor__{s}" for s in SENSORES]
    matriz = out[cols_sensor].to_numpy(dtype=object)
    for i in range(len(df_base)):
        fila = list(matriz[i])
        if "CRITICO_ESTADISTICO" in fila:
            globales[i] = "CRITICO_ESTADISTICO"
        elif "ALERTA" in fila:
            globales[i] = "ALERTA"
        elif all(v == "SIN_DATO" for v in fila):
            globales[i] = "SIN_DATO"
        elif "SIN_REFERENCIA" in fila:
            globales[i] = "SIN_REFERENCIA"
        else:
            globales[i] = "NORMAL"
    out["estado_global"] = globales
    return out

def graficar_cargas_flechas(cargas_pc12: np.ndarray, variables: list[str], ruta: Path) -> pd.DataFrame:
    x = cargas_pc12[:, 0]
    y = cargas_pc12[:, 1]
    ang = np.degrees(np.arctan2(y, x))
    mag = np.sqrt(x**2 + y**2)
    tabla = pd.DataFrame({
        "variable": variables,
        "carga_PC1": x,
        "carga_PC2": y,
        "angulo_grados": ang,
        "magnitud": mag,
    })

    fig, ax = plt.subplots(figsize=(9, 9))
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="#999999", linewidth=1)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.axvline(0, color="#777777", linewidth=1)
    for xi, yi, nombre, angulo in zip(x, y, variables, ang):
        ax.arrow(0, 0, xi, yi, length_includes_head=True,
                 head_width=0.035, head_length=0.05, linewidth=1.4,
                 color=COLOR_ARROW, alpha=0.9)
        ax.scatter([xi], [yi], s=25, color=COLOR_ARROW)
        dx = 0.03 if xi >= 0 else -0.03
        dy = 0.03 if yi >= 0 else -0.03
        ha = "left" if xi >= 0 else "right"
        va = "bottom" if yi >= 0 else "top"
        ax.text(xi + dx, yi + dy, f"{nombre}\n{angulo:.1f}°", fontsize=8, ha=ha, va=va)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA | Cargas con flechas y angulos (PC1-PC2)", weight="bold")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return tabla

def principal():
    parser = argparse.ArgumentParser(description="PCA y analisis de datos procesados")
    parser.add_argument("--carpeta", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--entrada", default=None, help="Ruta opcional del CSV multisensor")
    args = parser.parse_args()

    raiz = Path(args.carpeta).expanduser().resolve()
    salida = raiz / "Resultados_07_PCA_y_procesados"
    graficos = salida / "Graficos"
    salida.mkdir(parents=True, exist_ok=True)
    graficos.mkdir(exist_ok=True)
    configurar_estilo()

    ruta_entrada, ruta_corregido = buscar_entrada(raiz, args.entrada)
    base = leer_csv(ruta_entrada)
    faltantes = [c for c in VARIABLES if c not in base.columns]
    if faltantes:
        raise SystemExit("Faltan columnas requeridas: " + ", ".join(faltantes))
    for c in VARIABLES:
        base[c] = a_num(base[c])

    if "tratados" in ruta_entrada.name.lower():
        d_tratado = base.copy()
        fuente_texto = "CSV tratado 6sigma"
    else:
        d_tratado = tratar_6sigma_en_memoria(base)
        fuente_texto = "CSV corregido con tratamiento 6sigma en memoria"
    guardar_csv(d_tratado, salida / "00_copia_tratada_utilizada_por_07.csv")

    estad = resumen_estadistico(d_tratado)
    guardar_csv(estad, salida / "01_estadisticos_datos_tratados.csv")

    pearson = d_tratado[VARIABLES].corr(method="pearson")
    spearman = d_tratado[VARIABLES].corr(method="spearman")
    guardar_csv(pearson.reset_index().rename(columns={"index": "variable"}),
                salida / "02_correlacion_pearson_tratados.csv")
    guardar_csv(spearman.reset_index().rename(columns={"index": "variable"}),
                salida / "03_correlacion_spearman_tratados.csv")
    graficar_matriz(pearson, "Correlacion Pearson | Datos tratados 6sigma",
                    graficos / "01_matriz_pearson.png")
    graficar_matriz(spearman, "Correlacion Spearman | Datos tratados 6sigma",
                    graficos / "02_matriz_spearman.png")

    base_corregida = None
    if ruta_corregido and Path(ruta_corregido).is_file():
        base_corregida = leer_csv(Path(ruta_corregido))
        for c in VARIABLES:
            base_corregida[c] = a_num(base_corregida[c])
        antes = resumen_estadistico(base_corregida)
        comp = antes.merge(estad, on="variable", suffixes=("_antes", "_despues"))
        for nombre in ("media", "DS", "minimo", "maximo", "nulos"):
            comp[f"cambio_{nombre}"] = comp[f"{nombre}_despues"] - comp[f"{nombre}_antes"]
        guardar_csv(comp, salida / "04_comparacion_antes_despues.csv")
        corr_antes = base_corregida[VARIABLES].corr(method="pearson")
        delta = pearson - corr_antes
        guardar_csv(delta.reset_index().rename(columns={"index": "variable"}),
                    salida / "05_cambio_correlacion_pearson.csv")
        graficar_matriz(delta, "Cambio de correlacion Pearson | Despues - Antes",
                        graficos / "03_cambio_correlaciones.png")
        fig, ax = plt.subplots(figsize=(11, 5))
        pos = np.arange(len(VARIABLES))
        ax.bar(pos - 0.19, antes["DS"], width=0.38, label="Antes", color=COLOR_BAR)
        ax.bar(pos + 0.19, estad["DS"], width=0.38, label="Despues", color=COLOR_LINE)
        ax.set_xticks(pos, labels=VARIABLES, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("Desviacion estandar (unidad original de cada variable)")
        ax.set_title("Comparacion de dispersion antes y despues de 6sigma", weight="bold")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(graficos / "04_DS_antes_despues.png", bbox_inches="tight")
        plt.close(fig)

    X = d_tratado[VARIABLES].to_numpy(dtype=float)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_std = StandardScaler().fit_transform(X_imp)

    pca = PCA(n_components=len(VARIABLES), svd_solver="full")
    scores = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_
    acum = np.cumsum(var)
    autoval = pca.explained_variance_

    tabla_var = pd.DataFrame({
        "componente": [f"PC{i}" for i in range(1, len(VARIABLES) + 1)],
        "varianza_explicada": var,
        "porcentaje_varianza": var * 100,
        "varianza_acumulada": acum,
        "porcentaje_acumulado": acum * 100,
        "autovalor": autoval,
    })
    guardar_csv(tabla_var, salida / "06_varianza_explicada_PCA.csv")

    cargas_componentes = pd.DataFrame(pca.components_.T, columns=[f"PC{i}" for i in range(1, len(VARIABLES) + 1)])
    cargas_componentes.insert(0, "variable", VARIABLES)
    guardar_csv(cargas_componentes, salida / "07_cargas_PCA.csv")

    # Loadings para grafico con flechas (correlation circle)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    tabla_flechas = graficar_cargas_flechas(loadings[:, :2], VARIABLES, graficos / "07b_cargas_flechas_angulos_PC1_PC2.png")
    guardar_csv(tabla_flechas, salida / "07b_cargas_flechas_angulos_PC1_PC2.csv")

    base_para_estado = base_corregida if base_corregida is not None else base
    estados = combinar_estados(base_para_estado)
    idx = None
    for cand in ("smp_continuo", "smp_desde_cero", "smp_original"):
        if cand in base_para_estado.columns:
            idx = a_num(base_para_estado[cand])
            break
    if idx is None:
        idx = pd.Series(np.arange(len(base_para_estado)), name="indice_fila")

    tabla_scores = pd.DataFrame({
        "indice_referencia": idx,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "PC3": scores[:, 2],
        "estado_global": estados["estado_global"],
    })
    for sensor in SENSORES:
        tabla_scores[f"estado_sensor__{sensor}"] = estados[f"estado_sensor__{sensor}"]
    guardar_csv(tabla_scores, salida / "08_scores_PCA_con_estados.csv")

    pcs = np.arange(1, len(VARIABLES) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(pcs, var * 100, color=COLOR_BAR)
    ax.set_xticks(pcs)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada individual (%)")
    ax.set_title("PCA | Varianza explicada individual", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "05_varianza_individual_PCA.png", bbox_inches="tight")
    plt.close(fig)

    # Grafico de codo solicitado
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(pcs, var * 100, marker="o", color=COLOR_LINE, linewidth=2)
    ax.set_xticks(pcs)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title("PCA | Grafico de codo", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "05b_grafico_de_codo_PCA.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(pcs, acum * 100, marker="o", color=COLOR_LINE)
    ax.axhline(90, linestyle="--", linewidth=1.2, color=COLOR_LINE, alpha=0.85)
    ax.set_xticks(pcs)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada acumulada (%)")
    ax.set_title("PCA | Varianza explicada acumulada", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "06_varianza_acumulada_PCA.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    mat_cargas = pca.components_.T[:, :5]
    im = ax.imshow(mat_cargas, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(len(VARIABLES)), labels=VARIABLES, fontsize=8)
    ax.set_xticks(range(5), labels=[f"PC{i}" for i in range(1, 6)])
    for i in range(len(VARIABLES)):
        for j in range(5):
            ax.text(j, i, f"{mat_cargas[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Carga")
    ax.set_title("PCA | Cargas de las variables", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "07_cargas_PCA.png", bbox_inches="tight")
    plt.close(fig)

    paso = max(1, len(tabla_scores) // 9000)
    ix = np.arange(0, len(tabla_scores), paso)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(scores[ix, 0], scores[ix, 1], c=ix, cmap="viridis",
                    s=5, alpha=0.45, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Posicion de registro (no segundos)")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.2f}%)")
    ax.set_title("PCA | Proyeccion PC1-PC2", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "08_proyeccion_PC1_PC2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 7))
    sub = tabla_scores.iloc[ix].copy()
    orden_estados = ["NORMAL", "ALERTA", "CRITICO_ESTADISTICO", "SIN_REFERENCIA", "SIN_DATO"]
    for estado in orden_estados:
        parte = sub[sub["estado_global"] == estado]
        if len(parte) == 0:
            continue
        ax.scatter(
            parte["PC1"], parte["PC2"],
            s=7 if estado == "NORMAL" else 11,
            alpha=0.42 if estado == "NORMAL" else 0.75,
            c=COLORES_ESTADO[estado],
            label=f"{estado} (n={len(parte)})",
            rasterized=True
        )
    ax.set_xlabel(f"PC1 ({var[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.2f}%)")
    ax.set_title("PCA | Proyeccion PC1-PC2 segun estado de alerta", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(graficos / "09_proyeccion_PC1_PC2_por_estado.png", bbox_inches="tight")
    plt.close(fig)

    cuenta_estados = tabla_scores["estado_global"].value_counts(dropna=False).rename_axis("estado").reset_index(name="registros")
    cuenta_estados["porcentaje"] = 100 * cuenta_estados["registros"] / len(tabla_scores)
    guardar_csv(cuenta_estados, salida / "09b_resumen_estados_globales.csv")

    n90 = int(np.searchsorted(acum, 0.90) + 1)
    n95 = int(np.searchsorted(acum, 0.95) + 1)
    top_pc1 = cargas_componentes.set_index("variable")["PC1"].abs().sort_values(ascending=False).head(4)
    texto = (
        "ANALISIS DE DATOS PROCESADOS Y PCA\n\n"
        f"Fuente de trabajo: {ruta_entrada}\n"
        f"Modo de analisis: {fuente_texto}\n"
        f"Registros: {len(d_tratado)} | Variables fisicas: {len(VARIABLES)}\n"
        f"PC1: {var[0]*100:.3f}% | PC2: {var[1]*100:.3f}% | Acumulado PC1-PC2: {(var[0]+var[1])*100:.3f}%\n"
        f"Componentes para >=90%: {n90} | para >=95%: {n95}\n\n"
        "Interpretacion: los dos primeros componentes explican solo una fraccion limitada\n"
        "de la variabilidad total. El grafico PC1-PC2 sirve para exploracion, no para describir\n"
        "por completo el comportamiento del sistema. La figura 09 colorea cada punto segun el\n"
        "estado global derivado del mismo criterio robusto del codigo 05.\n"
        "La figura 07b muestra las cargas PC1-PC2 como flechas desde el origen; el angulo\n"
        "indica la orientacion de cada variable en el plano factorial.\n\n"
        "Variables con mayor carga absoluta en PC1:\n"
        + top_pc1.to_string() + "\n\n"
        "IMPORTANTE: el PCA no demuestra fallas ni reemplaza la interpretacion fisica.\n"
        "Se utilizaron 10 variables estandarizadas; los NaN del tratamiento 6sigma se imputaron\n"
        "por mediana solo para calcular el PCA. El estado de alerta se calculo sobre la base\n"
        "corregida sin eliminar esas observaciones.\n"
    )
    (salida / "10_interpretacion_y_limitaciones.txt").write_text(texto, encoding="utf-8")

    print(f"OK | Registros: {len(d_tratado)} | PC1: {var[0]*100:.2f}% | PC2: {var[1]*100:.2f}%")
    print(f"OK | Componentes para 90%: {n90}")
    print(f"Resultados guardados en: {salida}")

if __name__ == "__main__":
    principal()
