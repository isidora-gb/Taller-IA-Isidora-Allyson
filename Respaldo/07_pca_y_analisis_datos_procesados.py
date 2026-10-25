# -*- coding: utf-8 -*-
"""
TALLER IA 2 | 07 - PCA y analisis de datos procesados
Mantiene intactos los codigos 00-06 y los CSV de entrada.

Entrada preferida: Datos_tratados/datos_multisensor_tratados_6sigma.csv
Alternativa: Datos_corregidos/datos_multisensor_corregidos.csv;
             crea en memoria una copia con atipicos 6sigma marcados NaN.
Salidas: Resultados_07_PCA_y_procesados/
Dependencias: pandas, numpy, matplotlib, scikit-learn
"""
from pathlib import Path
import argparse
import warnings
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
COLOR = "#245F83"
ORANGE = "#CB7940"

def leer_csv(ruta):
    with open(ruta, "r", encoding="utf-8-sig") as f:
        cabecera = f.readline()
    sep = ";" if cabecera.count(";") > cabecera.count(",") else ","
    return pd.read_csv(ruta, sep=sep, decimal="," if sep == ";" else ".",
                       encoding="utf-8-sig", low_memory=False)

def guardar(df, ruta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df.to_csv(ruta, index=False, sep=";", decimal=",", encoding="utf-8-sig")

def configurar():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .16, "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })

def resumen_estadistico(df):
    filas = []
    for c in VARIABLES:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mu = x.mean()
        sd = x.std(ddof=1)
        filas.append(dict(variable=c, registros=len(x), validos=int(x.notna().sum()),
                          nulos=int(x.isna().sum()), media=mu, DS=sd,
                          minimo=x.min(), maximo=x.max(),
                          limite_inferior_6sigma=mu-6*sd,
                          limite_superior_6sigma=mu+6*sd))
    return pd.DataFrame(filas)

def graficar_matriz(corr, titulo, ruta):
    fig, ax = plt.subplots(figsize=(11.5, 9))
    mat = corr.to_numpy()
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(VARIABLES)), labels=VARIABLES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(VARIABLES)), labels=VARIABLES, fontsize=8)
    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            ax.text(j, i, f"{mat[i,j]:.2f}" if np.isfinite(mat[i,j]) else "—",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8, label="Coeficiente")
    ax.set_title(titulo, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)

def principal():
    parser = argparse.ArgumentParser(description="PCA y analisis conjunto despues de 6sigma")
    parser.add_argument("--carpeta", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--entrada", default=None, help="CSV multisensor tratado opcional")
    args = parser.parse_args()
    raiz = Path(args.carpeta).expanduser().resolve()
    carpeta = raiz / "Resultados_07_PCA_y_procesados"
    graficos = carpeta / "Graficos"
    graficos.mkdir(parents=True, exist_ok=True)
    configurar()
    tratados = raiz / "Datos_tratados" / "datos_multisensor_tratados_6sigma.csv"
    corregidos = raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv"
    fuente = Path(args.entrada).expanduser().resolve() if args.entrada else tratados
    origen = "CSV tratado previamente"
    if fuente.is_file():
        d = leer_csv(fuente)
    elif corregidos.is_file() and args.entrada is None:
        d = leer_csv(corregidos)
        origen = "Copia tratada en memoria desde CSV corregido (6sigma clasico)"
        for c in VARIABLES:
            x = pd.to_numeric(d[c], errors="coerce")
            mu, ds = x.mean(), x.std(ddof=1)
            if np.isfinite(ds) and ds > 0:
                d.loc[(x-mu).abs() > 6*ds, c] = np.nan
        # No se sobreescribe ni modifica ningun CSV previo.
    else:
        raise SystemExit(
            "No encontre la entrada. Ejecuta los codigos anteriores o usa "
            "--entrada RUTA_CSV. Busque: " + str(fuente))
    faltantes = [c for c in VARIABLES if c not in d.columns]
    if faltantes:
        raise SystemExit("Faltan columnas: " + ", ".join(faltantes))
    for c in VARIABLES:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(d) < 20:
        raise SystemExit("Se requieren al menos 20 registros.")
    if any(d[c].notna().sum() < 2 for c in VARIABLES):
        raise SystemExit("Una o mas variables tienen menos de 2 valores validos.")

    # Estadisticos y correlaciones de la copia tratada.
    estad = resumen_estadistico(d)
    guardar(estad, carpeta / "01_estadisticos_datos_tratados.csv")
    pearson = d[VARIABLES].corr(method="pearson")
    spearman = d[VARIABLES].corr(method="spearman")
    guardar(pearson.reset_index().rename(columns={"index": "variable"}),
            carpeta / "02_correlacion_pearson_tratados.csv")
    guardar(spearman.reset_index().rename(columns={"index": "variable"}),
            carpeta / "03_correlacion_spearman_tratados.csv")
    graficar_matriz(pearson, "Correlacion Pearson | Datos tratados 6sigma",
                   graficos / "01_matriz_pearson.png")
    graficar_matriz(spearman, "Correlacion Spearman | Datos tratados 6sigma",
                   graficos / "02_matriz_spearman.png")

    # Comparacion antes/despues: solo si la fuente original esta disponible.
    comparacion = False
    if corregidos.is_file():
        original = leer_csv(corregidos)
        if len(original) == len(d) and all(c in original.columns for c in VARIABLES):
            for c in VARIABLES:
                original[c] = pd.to_numeric(original[c], errors="coerce")
            antes = resumen_estadistico(original)
            comp = antes.merge(estad, on="variable", suffixes=("_antes", "_despues"))
            for nombre in ("media", "DS", "minimo", "maximo", "nulos"):
                comp[f"cambio_{nombre}"] = comp[f"{nombre}_despues"] - comp[f"{nombre}_antes"]
            guardar(comp, carpeta / "04_comparacion_antes_despues.csv")
            c0 = original[VARIABLES].corr()
            delta = pearson - c0
            guardar(delta.reset_index().rename(columns={"index": "variable"}),
                    carpeta / "05_cambio_correlacion_pearson.csv")
            graficar_matriz(delta, "Cambio de correlacion Pearson | Despues - Antes",
                           graficos / "03_cambio_correlaciones.png")
            fig, ax = plt.subplots(figsize=(11, 5))
            pos = np.arange(len(VARIABLES))
            ax.bar(pos-.19, antes["DS"], width=.38, label="Antes", color=COLOR)
            ax.bar(pos+.19, estad["DS"], width=.38, label="Despues", color=ORANGE)
            ax.set_xticks(pos, labels=VARIABLES, rotation=50, ha="right", fontsize=8)
            ax.set_ylabel("Desviacion estandar (unidades originales por variable)")
            ax.set_title("Comparacion de dispersion antes/despues 6sigma")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(graficos / "04_DS_antes_despues.png", bbox_inches="tight")
            plt.close(fig)
            comparacion = True

    # PCA sobre las 10 variables: imputacion temporal y estandarizacion.
    X = d[VARIABLES].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_std = StandardScaler().fit_transform(X_imp)
    modelo = PCA(n_components=len(VARIABLES), svd_solver="full")
    scores = modelo.fit_transform(X_std)
    var = modelo.explained_variance_ratio_
    acumulada = np.cumsum(var)
    tabla = pd.DataFrame({
        "componente": [f"PC{i}" for i in range(1, len(VARIABLES)+1)],
        "varianza_explicada": var, "porcentaje_varianza": var*100,
        "varianza_acumulada": acumulada,
        "porcentaje_acumulado": acumulada*100,
        "autovalor": modelo.explained_variance_,
    })
    guardar(tabla, carpeta / "06_varianza_explicada_PCA.csv")
    cargas = pd.DataFrame(modelo.components_.T, columns=[f"PC{i}" for i in range(1, 11)])
    cargas.insert(0, "variable", VARIABLES)
    guardar(cargas, carpeta / "07_cargas_PCA.csv")
    indice = (d["smp_desde_cero"] if "smp_desde_cero" in d.columns
              else pd.Series(np.arange(len(d)), name="indice_fila"))
    guardar(pd.DataFrame({"indice": indice, "PC1": scores[:,0], "PC2": scores[:,1],
                         "PC3": scores[:,2]}), carpeta / "08_scores_PCA.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pcs = np.arange(1, 11)
    ax.bar(pcs, var*100, color=COLOR, label="Individual")
    ax.set_xticks(pcs)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza individual (%)")
    ax2 = ax.twinx()
    ax2.plot(pcs, acumulada*100, marker="o", color=ORANGE, label="Acumulada")
    ax2.axhline(90, ls="--", lw=1, color=ORANGE, alpha=.8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Varianza acumulada (%)")
    ax.set_title("PCA | Varianza explicada y acumulada", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "05_scree_varianza_PCA.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    mat = modelo.components_.T[:, :5]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(10), labels=VARIABLES, fontsize=8)
    ax.set_xticks(range(5), labels=[f"PC{i}" for i in range(1, 6)])
    for i in range(10):
        for j in range(5):
            ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Carga (coeficiente)")
    ax.set_title("PCA | Cargas de las variables", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "06_cargas_PCA.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    paso = max(1, len(d)//9000)
    ix = np.arange(0, len(d), paso)
    sc = ax.scatter(scores[ix,0], scores[ix,1], c=ix, cmap="viridis",
                    alpha=.45, s=5, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Posicion de registro (no segundos)")
    ax.set_xlabel(f"PC1 ({var[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.2f}%)")
    ax.set_title("PCA | Proyeccion PC1-PC2", weight="bold")
    fig.tight_layout()
    fig.savefig(graficos / "07_proyeccion_PC1_PC2.png", bbox_inches="tight")
    plt.close(fig)

    n90 = int(np.searchsorted(acumulada, .90)+1)
    n95 = int(np.searchsorted(acumulada, .95)+1)
    top = cargas.set_index("variable")["PC1"].abs().sort_values(ascending=False).head(4)
    texto = (
        "TALLER IA 2 | ANALISIS DE DATOS PROCESADOS\n"
        f"Fuente: {fuente if fuente.exists() else corregidos}\n"
        f"Procedencia: {origen}\n"
        f"Registros: {len(d)}; variables: {len(VARIABLES)}\n"
        f"Valores faltantes en datos tratados (10 variables): {int(d[VARIABLES].isna().sum().sum())}\n"
        f"Comparacion con base corregida: {'SI' if comparacion else 'NO (base no disponible)'}\n"
        f"PC1: {var[0]*100:.3f}%; PC2: {var[1]*100:.3f}%; acumulado: {(var[0]+var[1])*100:.3f}%\n"
        f"Componentes para >=90%: {n90}; >=95%: {n95}\n"
        "Variables con mayores cargas absolutas en PC1:\n"
        + top.to_string() + "\n\n"
        "METODOLOGIA: PCA de 10 variables estandarizadas; los NaN del "
        "tratamiento 6sigma se imputan por mediana SOLO EN MEMORIA para PCA. "
        "Los estadisticos y correlaciones tratados conservan los NaN.\n"
        "IMPORTANTE: PCA explica variabilidad, no identifica causas fisicas, "
        "no demuestra fallas ni predice mantenimiento. SMP no representa segundos. "
        "La unidad del magnetometro requiere confirmacion instrumental.\n"
    )
    (carpeta / "09_interpretacion_y_limitaciones.txt").write_text(texto, encoding="utf-8")
    print(f"OK: {len(d)} registros | PC1 {var[0]*100:.2f}% | "
          f"PC2 {var[1]*100:.2f}% | PC90 {n90}")
    print("Resultados:", carpeta)

if __name__ == "__main__":
    principal()
