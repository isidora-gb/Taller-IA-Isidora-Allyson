
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SMP = "smp_continuo"
SENSORES = [
    "AcelX1_SisGenEol", "AcelY1_SisGenEol", "AcelZ1_SisGenEol",
    "GiroX_SisGenEol", "GiroY_SisGenEol", "GiroZ_SisGenEol",
    "MagneX_SisGenEol", "MagneY_SisGenEol", "MagneZ_SisGenEol",
    "Ter_SisGenEol",
]
MODELOS = ["Isolation Forest", "K-means", "DBSCAN (aprox.)"]
COLORES = ["#326C94", "#A44E79", "#638B6B"]
SEMILLA = 42


def resolver_entrada(raiz: Path, argumento: str | None) -> Path:
    if argumento:
        ruta = Path(argumento).expanduser()
        ruta = ruta if ruta.is_absolute() else raiz / ruta
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontro el CSV: {ruta}")
        return ruta
    for ruta in [
        raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv",
        raiz / "Datos_corregidos_desde_excel" / "datos_multisensor_corregidos.csv",
        raiz / "datos_multisensor_corregidos.csv",
    ]:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError("Falta Datos_corregidos/datos_multisensor_corregidos.csv")


def a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).astype(float).replace([np.inf, -np.inf], np.nan)


def cargar_datos(entrada: Path) -> tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_csv(entrada, sep=";", encoding="utf-8-sig", dtype="string")
    faltantes = [c for c in [SMP, *SENSORES] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}\nDisponibles: {list(df.columns)}")
    for columna in [SMP, *SENSORES]:
        df[columna] = a_numero(df[columna])
    if df[SMP].isna().any() or (df[SMP] % 1 != 0).any():
        raise ValueError("SMP continuo tiene valores nulos, invalidos o no enteros")
    df[SMP] = df[SMP].astype("int64")
    if df[SMP].duplicated().any():
        raise ValueError("Hay valores SMP continuo duplicados")
    df = df.sort_values(SMP, kind="mergesort").reset_index(drop=True)
    n = len(df)
    if n < 600:
        raise ValueError("Muy pocos registros para cinco cortes cronologicos: minimo 600")
    externo_entrenamiento = int(n * 0.70)
    smp = df[SMP].iloc[:externo_entrenamiento].to_numpy(dtype=np.int64)
    X = df[SENSORES].iloc[:externo_entrenamiento].to_numpy(dtype=float)
    if np.any(np.all(np.isnan(X), axis=0)):
        raise ValueError("Existe alguna variable totalmente vacia en entrenamiento externo")
    return smp, X, n


def contar_episodios(smp: np.ndarray, alertas: np.ndarray) -> int:
    """Continuidad solo con alertas en SMP consecutivos; no usa segundos."""
    if len(smp) != len(alertas):
        raise ValueError("SMP y alertas no tienen igual longitud")
    if not len(smp):
        return 0
    a = np.asarray(alertas, dtype=bool)
    cortes = np.r_[True, (np.diff(smp) != 1) | (~a[:-1])]
    return int(np.count_nonzero(a & cortes))


def distancias_centroides(modelo: KMeans, X: np.ndarray) -> np.ndarray:
    return np.min(modelo.transform(X), axis=1)


def escoger_k_en_subentrenamiento(X: np.ndarray, k_min: int, k_max: int,
                                   semilla: int, muestra_fit: int,
                                   muestra_eval: int) -> tuple[int, list[dict]]:
    """Muestra de ajuste y muestra de silhouette DISJUNTAS y ambas del pasado."""
    rng = np.random.default_rng(semilla)
    perm = rng.permutation(len(X))
    n_fit = min(muestra_fit, max(20, int(len(X) * .75)))
    n_eval = min(muestra_eval, len(X) - n_fit)
    if n_eval < 10:
        raise ValueError("No hay suficiente subentrenamiento para escoger K")
    XA, XB = X[perm[:n_fit]], X[perm[n_fit:n_fit + n_eval]]
    registros = []
    for k in range(k_min, min(k_max, n_fit - 1) + 1):
        mod = KMeans(n_clusters=k, n_init=10, random_state=semilla, max_iter=300)
        mod.fit(XA)
        etiquetas = mod.predict(XB)
        n_grupos = int(len(np.unique(etiquetas)))
        sil = (float(silhouette_score(XB, etiquetas))
               if 2 <= n_grupos <= len(XB) - 1 else np.nan)
        registros.append({"K": k, "silhouette_interno": sil,
                          "grupos_presentes": n_grupos,
                          "muestras_fit": len(XA), "muestras_eval": len(XB)})
    if not registros:
        raise ValueError("No se pudo evaluar K")
    validos = [r for r in registros if np.isfinite(r["silhouette_interno"])]
    if validos:
        elegido = sorted(validos, key=lambda r: (-r["silhouette_interno"], r["K"]))[0]["K"]
    else:
        elegido = k_min  # Fallback documentado si silhouette no existe.
    return int(elegido), registros


def ajustar_dbscan(X: np.ndarray, tamano_muestra: int, min_samples: int,
                    objetivo: float, semilla: int) -> tuple[NearestNeighbors, float, dict]:
    rng = np.random.default_rng(semilla)
    ids = rng.choice(len(X), min(len(X), tamano_muestra), replace=False)
    Xm = X[ids]
    if min_samples >= len(Xm):
        raise ValueError("min_samples debe ser menor que la muestra DBSCAN")
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=1)
    nn.fit(Xm)
    dist, _ = nn.kneighbors(Xm)
    kth = dist[:, -1]
    quantiles = (0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 0.995)
    epsilones = sorted(set(float(np.quantile(kth, q)) for q in quantiles
                            if float(np.quantile(kth, q)) > 0))
    if not epsilones:
        # Situacion excepcional con valores repetidos: radio positivo minimo.
        epsilones = [1e-10]
    opciones = []
    for eps in epsilones:
        m = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
        etiquetas = m.fit_predict(Xm)
        cores = len(m.core_sample_indices_)
        grupos = int(len(set(etiquetas) - {-1}))
        fraccion = float(np.mean(etiquetas == -1))
        if cores:
            opciones.append((abs(fraccion - objetivo), eps, m, grupos, fraccion))
    if not opciones:
        raise ValueError("DBSCAN no encontro nucleos con los radios explorados")
    _, eps, modelo, grupos, tasa_muestra = min(opciones, key=lambda r: (r[0], r[1]))
    cores = modelo.components_
    vecinos_core = NearestNeighbors(n_neighbors=1, n_jobs=1)
    vecinos_core.fit(cores)
    metadatos = {"eps": float(eps), "min_samples": int(min_samples),
                 "muestra_entrenamiento": len(Xm), "clusters_muestra": int(grupos),
                 "ruido_fraccion_muestra": float(tasa_muestra),
                 "nucleos_muestra": int(len(cores))}
    return vecinos_core, float(eps), metadatos


def evaluar_dbscan_aproximado(vecinos_core: NearestNeighbors, eps: float,
                               X: np.ndarray, lote: int = 2000) -> np.ndarray:
    """Ruido fuera de eps de su nucleo mas cercano; NO es un predict DBSCAN."""
    resultado = np.empty(len(X), dtype=bool)
    for inicio in range(0, len(X), lote):
        fin = min(len(X), inicio + lote)
        distancias, _ = vecinos_core.kneighbors(X[inicio:fin])
        resultado[inicio:fin] = distancias[:, 0] > eps
    return resultado


def jaccard(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter, (100. * inter / union if union else float("nan"))


def crear_graficos(resumen: pd.DataFrame, estabilidad: pd.DataFrame, salida: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for nombre, color in zip(MODELOS, COLORES):
        d = resumen[resumen["modelo"] == nombre]
        ax.plot(d["corte"], d["porcentaje_alertas_validacion"],
                marker="o", linewidth=1.8, color=color, label=nombre)
    ax.set(xlabel="Corte cronologico", ylabel="Alertas en bloque posterior (%)",
           title="Validacion cronologica: 5 cortes crecientes")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "01_alertas_por_corte.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(MODELOS))
    d = estabilidad.set_index("modelo").loc[MODELOS]
    ax.bar(x, d["media_porcentaje_alertas"].to_numpy(), width=.60,
           color=COLORES, alpha=.85)
    ax.errorbar(x, d["media_porcentaje_alertas"].to_numpy(),
                yerr=d["desviacion_estandar_porcentaje"].fillna(0).to_numpy(),
                fmt="none", ecolor="#333333", capsize=5)
    ax.set_xticks(x, MODELOS)
    ax.set(ylabel="% alertas de validacion (media ± DE entre cortes)",
           title="Variabilidad descriptiva de las alertas")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(salida / "02_estabilidad_modelos.png", dpi=300)
    plt.close(fig)


def ejecutar(entrada: Path, salida: Path, splits: int, fraccion_alerta: float,
            muestra_dbscan: int, min_samples: int, objetivo_ruido: float,
            muestra_k: int, muestra_silhouette: int, estimadores_if: int) -> None:
    if splits != 5:
        raise ValueError("Esta tarea usa exactamente cinco cortes: --cortes 5")
    if not 0 < fraccion_alerta < 1:
        raise ValueError("--fraccion-alerta debe estar entre 0 y 1")
    if not 0 < objetivo_ruido < 1:
        raise ValueError("--objetivo-ruido debe estar entre 0 y 1")
    if muestra_dbscan < 100 or min_samples < 3 or min_samples >= muestra_dbscan:
        raise ValueError("Aumentar --muestra-dbscan o reducir --min-samples")
    smp, X, n_total = cargar_datos(entrada)
    salida.mkdir(parents=True, exist_ok=True)
    cortes = TimeSeriesSplit(n_splits=splits)
    filas_resultados, filas_cortes, filas_pares, filas_k = [], [], [], []

    for numero, (idx_tr, idx_va) in enumerate(cortes.split(X), start=1):
        assert idx_tr.max() < idx_va.min(), "Se detecto filtracion temporal"
        smp_tr, smp_va = smp[idx_tr], smp[idx_va]
        imputador = SimpleImputer(strategy="median")
        escalador = StandardScaler()
        Xtr = escalador.fit_transform(imputador.fit_transform(X[idx_tr]))
        Xva = escalador.transform(imputador.transform(X[idx_va]))
        if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
            raise ValueError(f"Datos no finitos en corte {numero}")

        filas_cortes.append({
            "corte": numero, "train_registros": len(idx_tr),
            "train_smp_inicio": int(smp_tr[0]), "train_smp_fin": int(smp_tr[-1]),
            "validacion_registros": len(idx_va),
            "validacion_smp_inicio": int(smp_va[0]),
            "validacion_smp_fin": int(smp_va[-1]),
            "huecos_smp_entrenamiento": int(np.count_nonzero(np.diff(smp_tr) > 1)),
            "huecos_smp_validacion": int(np.count_nonzero(np.diff(smp_va) > 1)),
        })

        # 1) Isolation Forest. Ningun umbral usa el bloque posterior.
        forest = IsolationForest(n_estimators=estimadores_if, max_samples="auto",
                                 contamination="auto", random_state=SEMILLA + numero,
                                 n_jobs=-1)
        forest.fit(Xtr)
        score_tr = -forest.score_samples(Xtr)
        score_va = -forest.score_samples(Xva)
        umbral_if = float(np.quantile(score_tr, 1 - fraccion_alerta))
        ftr_if = score_tr >= umbral_if
        fva_if = score_va >= umbral_if

        # 2) K-means. K se escoge SOLO dentro de los datos anteriores al corte.
        elegido, tabla_ks = escoger_k_en_subentrenamiento(
            Xtr, 2, 6, SEMILLA + numero, muestra_k, muestra_silhouette)
        for item in tabla_ks:
            filas_k.append({"corte": numero, **item, "seleccionado": item["K"] == elegido})
        km = KMeans(n_clusters=elegido, random_state=SEMILLA + numero,
                    n_init=10, max_iter=300)
        km.fit(Xtr)
        dtr = distancias_centroides(km, Xtr)
        dva = distancias_centroides(km, Xva)
        umbral_km = float(np.quantile(dtr, 1 - fraccion_alerta))
        ftr_km = dtr >= umbral_km
        fva_km = dva >= umbral_km

        # 3) DBSCAN sobre una muestra pasada. La extension no es fit_predict nativo.
        core_nn, eps, meta_db = ajustar_dbscan(
            Xtr, muestra_dbscan, min_samples, objetivo_ruido, SEMILLA + numero)
        ftr_db = evaluar_dbscan_aproximado(core_nn, eps, Xtr)
        fva_db = evaluar_dbscan_aproximado(core_nn, eps, Xva)

        pred = {"Isolation Forest": (ftr_if, fva_if, umbral_if),
                "K-means": (ftr_km, fva_km, umbral_km),
                "DBSCAN (aprox.)": (ftr_db, fva_db, eps)}
        for nombre, (flags_tr, flags_va, umbral) in pred.items():
            filas_resultados.append({
                "corte": numero, "modelo": nombre,
                "train_registros": len(Xtr), "validacion_registros": len(Xva),
                "alertas_train": int(flags_tr.sum()),
                "porcentaje_alertas_train": 100 * float(flags_tr.mean()),
                "alertas_validacion": int(flags_va.sum()),
                "porcentaje_alertas_validacion": 100 * float(flags_va.mean()),
                "alertas_por_1000_validacion": 1000 * float(flags_va.mean()),
                "episodios_validacion": contar_episodios(smp_va, flags_va),
                "parametro_umbral_o_eps": float(umbral),
                "K": elegido if nombre == "K-means" else np.nan,
                "clusters_dbscan_muestra": (meta_db["clusters_muestra"]
                                           if nombre == "DBSCAN (aprox.)" else np.nan),
                "nucleos_dbscan_muestra": (meta_db["nucleos_muestra"]
                                         if nombre == "DBSCAN (aprox.)" else np.nan),
            })
        for ai, a in enumerate(MODELOS):
            for b in MODELOS[ai + 1:]:
                n, jac = jaccard(pred[a][1], pred[b][1])
                filas_pares.append({"corte": numero, "modelo_A": a, "modelo_B": b,
                                    "coincidencias_ambos": n, "jaccard_pct": jac})
        consensos = np.column_stack([pred[m][1] for m in MODELOS]).sum(axis=1)
        pd.DataFrame({
            SMP: smp_va,
            "alerta_isolation_forest": fva_if.astype(np.int8),
            "alerta_distancia_kmeans": fva_km.astype(np.int8),
            "alerta_ruido_dbscan_aprox": fva_db.astype(np.int8),
            "consenso_al_menos_2": (consensos >= 2).astype(np.int8),
            "consenso_los_3": (consensos == 3).astype(np.int8),
        }).to_csv(salida / f"02_predicciones_corte_{numero}.csv", sep=";", index=False,
                  encoding="utf-8-sig")
        print(f"Corte {numero}/5 | train={len(Xtr):,} | validacion={len(Xva):,} | "
              f"K={elegido} | eps={eps:.4f} | "
              f"alertas IF/KM/DBSCAN={sum(fva_if)}/{sum(fva_km)}/{sum(fva_db)}",
              flush=True)

    df_res = pd.DataFrame(filas_resultados)
    df_cortes = pd.DataFrame(filas_cortes)
    df_pares = pd.DataFrame(filas_pares)
    df_ks = pd.DataFrame(filas_k)
    estabilidad = (df_res.groupby("modelo", sort=False)["porcentaje_alertas_validacion"]
                   .agg(media_porcentaje_alertas="mean",
                        desviacion_estandar_porcentaje="std",
                        minimo_porcentaje="min", maximo_porcentaje="max")
                   .reset_index())
    for nombre, tabla in [
        ("00_resumen_cortes.csv", df_cortes),
        ("01_resultados_por_corte.csv", df_res),
        ("03_estabilidad_modelos.csv", estabilidad),
        ("04_coincidencias_jaccard.csv", df_pares),
        ("05_seleccion_k_interna.csv", df_ks),
    ]:
        tabla.to_csv(salida / nombre, sep=";", index=False, encoding="utf-8-sig",
                     float_format="%.8g")
    meta = {
        "fuente": str(entrada.resolve()),
        "filas_base_completa": n_total,
        "filas_entrenamiento_externo_utilizadas": len(X),
        "validacion_externa_15_pct_utilizada": False,
        "prueba_externa_15_pct_utilizada": False,
        "numero_cortes": 5,
        "criterio": "TimeSeriesSplit: origen movil sobre 70% inicial; sin mezcla",
        "gap": 0,
        "porque_gap_cero": "No se construyeron ventanas solapadas ni variables rezagadas; si se agregan, fijar gap >= longitud de ventana",
        "preprocesamiento": "imputador + StandardScaler ajustados en cada subentrenamiento",
        "umbral_IF_Kmeans": f"percentil {100*(1-fraccion_alerta):.4g} del subentrenamiento",
        "K": "seleccionado con silhouette en datos exclusivamente anteriores al corte; ajuste final en subentrenamiento",
        "DBSCAN": "ajustado sobre submuestra de subentrenamiento; eps calibrado para ruido heuristico en esa submuestra; extension a los bloques por punto nucleo mas proximo, aproximada",
        "muestra_dbscan_maxima": muestra_dbscan,
        "min_samples_dbscan": min_samples,
        "objetivo_ruido_dbscan": objetivo_ruido,
        "muestra_ajuste_K_maxima": muestra_k,
        "muestra_silhouette_maxima": muestra_silhouette,
        "estimadores_isolation_forest": estimadores_if,
        "semilla_base": SEMILLA,
        "interpretacion": "Sin etiquetas de fallas no es valido calcular exactitud, precision, recall, F1, ni AUC frente a fallas; tasas y Jaccard son descriptivos",
        "smp": "contador de adquisicion, no tiempo en segundos; episodios se interrumpen en huecos",
    }
    (salida / "00_metodologia.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    crear_graficos(df_res, estabilidad, salida)
    print("\nVALIDACION CRONOLOGICA DE 5 CORTES FINALIZADA")
    print(df_res[["corte", "modelo", "validacion_registros",
                  "alertas_validacion", "porcentaje_alertas_validacion",
                  "episodios_validacion"]].to_string(index=False, float_format=lambda f: f"{f:.3f}"))
    print(f"\nResultados en: {salida}")
    print("IMPORTANTE: este analisis no usa validacion ni prueba externas, no diagnostica averias y DBSCAN fuera de muestra es aproximado.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", type=str, help="CSV corregido: origen de los datos")
    p.add_argument("--salida", type=Path, help="Carpeta destino")
    p.add_argument("--cortes", type=int, default=5, help="Numero de cortes (5)")
    p.add_argument("--fraccion-alerta", type=float, default=.01,
                   help="Tasa de referencia para los umbrales IF y K-means (.01)")
    p.add_argument("--muestra-dbscan", type=int, default=2500,
                   help="Maximo de puntos del subentrenamiento para DBSCAN (2500)")
    p.add_argument("--min-samples", type=int, default=20,
                   help="min_samples para DBSCAN (20)")
    p.add_argument("--objetivo-ruido", type=float, default=.05,
                   help="Ruido orientativo en la muestra DBSCAN (.05)")
    p.add_argument("--muestra-k", type=int, default=2500,
                   help="Maximo de filas para buscar K en subentrenamiento (2500)")
    p.add_argument("--muestra-silhouette", type=int, default=700,
                   help="Maximo de filas para evaluar silhouette internamente (700)")
    p.add_argument("--estimadores-if", type=int, default=300,
                   help="Cantidad de arboles Isolation Forest (300)")
    args = p.parse_args()
    raiz = Path(__file__).resolve().parent
    entrada = resolver_entrada(raiz, args.entrada)
    salida = args.salida or raiz / "Resultados_Tarea3" / "06_validacion_5_cortes"
    ejecutar(entrada, salida, args.cortes, args.fraccion_alerta,
             args.muestra_dbscan, args.min_samples, args.objetivo_ruido,
             args.muestra_k, args.muestra_silhouette, args.estimadores_if)


if __name__ == "__main__":
    main()
