from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SMP = "smp_continuo"
CONJUNTOS = ("entrenamiento", "validacion", "prueba")
SEMILLA = 42


def leer_preparados(carpeta: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    partes: dict[str, pd.DataFrame] = {}
    for nombre in CONJUNTOS:
        ruta = carpeta / f"01_{nombre}_escalado.csv"
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró {ruta}. Ejecuta primero 08_preparar_datos_tarea3.py"
            )
        partes[nombre] = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    variables = [c for c in partes["entrenamiento"].columns if c.startswith("z_")]
    if len(variables) != 10:
        raise ValueError(f"Se esperaban diez variables estandarizadas; encontradas: {variables}")
    for nombre, datos in partes.items():
        if list(datos.columns) != [SMP, *variables]:
            raise ValueError(f"Columnas distintas en el conjunto {nombre}")
        if datos[SMP].isna().any() or not np.isfinite(datos[variables].to_numpy(float)).all():
            raise ValueError(f"El conjunto {nombre} contiene valores no válidos")
        if datos[SMP].duplicated().any() or not datos[SMP].is_monotonic_increasing:
            raise ValueError(f"Revisar duplicados u orden de SMP en {nombre}")
    if int(partes["entrenamiento"][SMP].iloc[-1]) >= int(partes["validacion"][SMP].iloc[0]):
        raise ValueError("La división entrenamiento/validación no es cronológica")
    if int(partes["validacion"][SMP].iloc[-1]) >= int(partes["prueba"][SMP].iloc[0]):
        raise ValueError("La división validación/prueba no es cronológica")
    return partes, variables


def seleccionar_k(X_train: np.ndarray, X_val: np.ndarray, ks: list[int],
                 tamano_entrenamiento: int, tamano_validacion: int) -> tuple[int, pd.DataFrame]:
    # Muestras deterministas. Ninguna fila de prueba interviene en la selección.
    rng = np.random.default_rng(SEMILLA)
    ids_train = np.sort(rng.choice(len(X_train), min(len(X_train), tamano_entrenamiento), replace=False))
    ids_val = np.sort(rng.choice(len(X_val), min(len(X_val), tamano_validacion), replace=False))
    A, B = X_train[ids_train], X_val[ids_val]
    filas: list[dict] = []
    for k in ks:
        if k >= min(len(A), len(B)):
            continue
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA, max_iter=300)
        modelo.fit(A)
        grupos_val = modelo.predict(B)
        n_observados = np.unique(grupos_val).size
        # Silhouette solo existe cuando hay al menos 2 grupos asignados.
        sil = (float(silhouette_score(B, grupos_val))
               if 2 <= n_observados <= len(B)-1 else np.nan)
        filas.append({
            "K": int(k),
            "inercia_muestra_entrenamiento": float(modelo.inertia_),
            "silhouette_muestra_validacion": sil,
            "grupos_presentes_validacion": int(n_observados),
            "muestras_entrenamiento_seleccion": len(A),
            "muestras_validacion_seleccion": len(B),
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        raise ValueError("No hay suficientes registros para evaluar los K seleccionados")
    validos = tabla.dropna(subset=["silhouette_muestra_validacion"])
    if validos.empty:
        raise ValueError("Silhouette no es definible para ningún K; revisar datos")
    # Mejor score; desempata por K menor. No se usa ninguna medición de prueba.
    ganador = validos.sort_values(["silhouette_muestra_validacion", "K"], ascending=[False, True]).iloc[0]
    return int(ganador["K"]), tabla


def episodios_por_smp(datos: pd.DataFrame) -> pd.DataFrame:
    smp = datos[SMP].to_numpy(dtype=np.int64)
    flags = datos["alerta_distancia_kmeans"].to_numpy(dtype=bool)
    comienzo = flags & np.r_[True, (~flags[:-1]) | (np.diff(smp) != 1)]
    inicios = np.flatnonzero(comienzo)
    filas = []
    for n_episodio, comienzo_idx in enumerate(inicios, start=1):
        fin = comienzo_idx
        while fin + 1 < len(datos) and flags[fin+1] and smp[fin+1] == smp[fin]+1:
            fin += 1
        filas.append({
            "episodio": n_episodio,
            "smp_inicial": int(smp[comienzo_idx]),
            "smp_final": int(smp[fin]),
            "longitud_en_muestras": int(fin - comienzo_idx + 1),
            "distancia_maxima": float(datos["distancia_al_centroide"].iloc[comienzo_idx:fin+1].max()),
        })
    return pd.DataFrame(filas, columns=[
        "episodio", "smp_inicial", "smp_final", "longitud_en_muestras", "distancia_maxima"
    ])


def guardar_graficos(tabla: pd.DataFrame, porcentajes: pd.DataFrame,
                     puntajes_prueba: pd.DataFrame, umbral: float, salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.plot(tabla["K"], tabla["silhouette_muestra_validacion"], marker="o", color="#914070", linewidth=1.7)
    ax.set(xlabel="Número de clusters (K)", ylabel="Silhouette (muestra de validación)",
           title="K-means | Selección de K")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(salida / "01_seleccion_k_silhouette.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.plot(tabla["K"], tabla["inercia_muestra_entrenamiento"], marker="o", color="#387E9F", linewidth=1.7)
    ax.set(xlabel="Número de clusters (K)", ylabel="Inercia (muestra de entrenamiento)",
           title="K-means | Inercia por número de grupos")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(salida / "02_inercia_k.png", dpi=300)
    plt.close(fig)

    tabla_pivote = porcentajes.pivot(index="conjunto", columns="cluster", values="porcentaje").fillna(0)
    tabla_pivote = tabla_pivote.reindex(CONJUNTOS)
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    tabla_pivote.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.65)
    ax.set(xlabel="Conjunto cronológico", ylabel="Porcentaje de registros (%)",
           title="K-means | Distribución de clusters")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster", bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "03_distribucion_clusters.png", dpi=300)
    plt.close(fig)

    prueba = puntajes_prueba
    alerta = prueba["alerta_distancia_kmeans"].astype(bool)
    fig, ax = plt.subplots(figsize=(11.3, 5.1))
    ax.plot(prueba[SMP], prueba["distancia_al_centroide"], color="#46637F", alpha=.7, linewidth=.8)
    ax.scatter(prueba.loc[alerta, SMP], prueba.loc[alerta, "distancia_al_centroide"],
               s=10, color="#C24354", label="Alertas exploratorias", zorder=3)
    ax.axhline(umbral, linestyle="--", color="#333333", label="Umbral fijo (entrenamiento)")
    ax.set(xlabel="SMP continuo (posición de adquisición, no segundos)",
           ylabel="Distancia al centroide más próximo (escala estandarizada)",
           title="K-means | Distancias en prueba")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "04_distancias_y_alertas_prueba.png", dpi=300)
    plt.close(fig)


def comparar_if(carpeta_if: Path, resultado: pd.DataFrame, salida: Path) -> str:
    """Comparación exploratoria opcional; solo prueba, sin cambiar modelos."""
    ruta = carpeta_if / "01_resultado_prueba.csv"
    if not ruta.exists():
        return "No se encontró salida de IF; comparación opcional omitida."
    otro = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    necesarias = {SMP, "alerta_isolation_forest"}
    if not necesarias.issubset(otro.columns):
        return "Salida IF sin columnas esperadas; comparación omitida."
    comparacion = resultado[[SMP, "alerta_distancia_kmeans"]].merge(
        otro[[SMP, "alerta_isolation_forest"]], on=SMP, how="inner", validate="one_to_one"
    )
    if len(comparacion) != len(resultado):
        return "SMP de IF y K-means no coinciden en prueba; comparación omitida."
    a = comparacion["alerta_distancia_kmeans"].astype(bool)
    b = comparacion["alerta_isolation_forest"].astype(bool)
    resumen = pd.DataFrame([{
        "registros_comparados": len(comparacion),
        "alertas_kmeans": int(a.sum()),
        "alertas_isolation_forest": int(b.sum()),
        "coincidencias_ambos": int((a & b).sum()),
        "solo_kmeans": int((a & ~b).sum()),
        "solo_isolation_forest": int((~a & b).sum()),
        "ninguno": int((~a & ~b).sum()),
        "nota": "Coincidencia entre detectores, NO exactitud diagnóstica",
    }])
    resumen.to_csv(salida / "05_comparacion_con_isolation_forest_prueba.csv", sep=";", index=False, encoding="utf-8-sig")
    return "Comparación con Isolation Forest (prueba) guardada."


def ejecutar(preparados: Path, salida: Path, carpeta_if: Path, fraccion_alerta: float,
            tamano_train: int, tamano_val: int) -> None:
    if not 0 < fraccion_alerta < 1:
        raise ValueError("La fracción de alertas debe estar entre 0 y 1")
    partes, variables = leer_preparados(preparados)
    salida.mkdir(parents=True, exist_ok=True)
    X_train = partes["entrenamiento"][variables].to_numpy(float)
    X_val = partes["validacion"][variables].to_numpy(float)
    k_elegido, tabla_k = seleccionar_k(X_train, X_val, list(range(2, 8)), tamano_train, tamano_val)
    tabla_k.to_csv(salida / "00_seleccion_k.csv", sep=";", index=False, encoding="utf-8-sig")

    modelo = KMeans(n_clusters=k_elegido, n_init=10, random_state=SEMILLA, max_iter=300)
    modelo.fit(X_train)  # Exclusivamente entrenamiento.
    dist_train = np.min(modelo.transform(X_train), axis=1)
    umbral = float(np.quantile(dist_train, 1.0 - fraccion_alerta))

    resultados = []
    resumen = []
    por_cluster = []
    for nombre in CONJUNTOS:
        bloque = partes[nombre]
        X = bloque[variables].to_numpy(float)
        grupos = modelo.predict(X)
        distancias = np.min(modelo.transform(X), axis=1)
        alertas = (distancias >= umbral).astype(int)
        resultado = pd.DataFrame({
            SMP: bloque[SMP].to_numpy(),
            "cluster": grupos,
            "distancia_al_centroide": distancias,
            "alerta_distancia_kmeans": alertas,
        })
        resultado.to_csv(salida / f"01_resultado_{nombre}.csv", sep=";", index=False, encoding="utf-8-sig")
        episodios = episodios_por_smp(resultado)
        episodios.to_csv(salida / f"02_episodios_{nombre}.csv", sep=";", index=False, encoding="utf-8-sig")
        resumen.append({
            "conjunto": nombre,
            "registros": len(bloque),
            "K": k_elegido,
            "grupos_presentes": int(np.unique(grupos).size),
            "alertas_distancia": int(alertas.sum()),
            "porcentaje_alertas": round(100.0 * alertas.mean(), 4),
            "alertas_por_1000_muestras": round(1000.0 * alertas.mean(), 3),
            "episodios": int(len(episodios)),
            "distancia_mediana": float(np.median(distancias)),
            "distancia_percentil_95": float(np.quantile(distancias, 0.95)),
        })
        for cluster, cantidad in zip(*np.unique(grupos, return_counts=True)):
            por_cluster.append({
                "conjunto": nombre, "cluster": int(cluster),
                "registros": int(cantidad), "porcentaje": 100 * int(cantidad) / len(bloque),
            })
        resultados.append(resultado)

    tabla_resumen = pd.DataFrame(resumen)
    tabla_cluster = pd.DataFrame(por_cluster)
    tabla_resumen.to_csv(salida / "00_resumen_kmeans.csv", sep=";", index=False, encoding="utf-8-sig")
    tabla_cluster.to_csv(salida / "00_distribucion_clusters.csv", sep=";", index=False, encoding="utf-8-sig")
    joblib.dump(modelo, salida / "00_modelo_kmeans.joblib")
    config = {
        "algoritmo": "KMeans", "K_elegido": k_elegido, "K_evaluados": list(range(2, 8)),
        "criterio": "mayor silhouette en muestra de validación",
        "muestra_entrenamiento_para_K": min(len(X_train), tamano_train),
        "muestra_validacion_para_K": min(len(X_val), tamano_val),
        "ajuste_modelo_final": "solo entrenamiento completo",
        "n_init": 10, "random_state": SEMILLA,
        "regla_alerta_adicional": "distancia al centroide >= percentil entrenamiento",
        "fraccion_alerta_referencia": fraccion_alerta,
        "umbral_distancia": umbral,
        "advertencia": "La distancia no es una probabilidad de falla; sin etiquetas no hay matriz de confusión",
    }
    (salida / "00_configuracion_kmeans.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    guardar_graficos(tabla_k, tabla_cluster, resultados[2], umbral, salida)
    mensaje = comparar_if(carpeta_if, resultados[2], salida)

    print("\nK-MEANS FINALIZADO")
    print(f"K elegido (silhouette de validación): {k_elegido}")
    print(f"Umbral de distancia fijo desde entrenamiento: {umbral:.6f}")
    print(tabla_resumen.to_string(index=False))
    print(f"\nResultados: {salida}")
    print(mensaje)
    print("Nota: las alertas por distancia son una regla exploratoria, NO fallas confirmadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparados", help="Carpeta de los datos del script 08")
    parser.add_argument("--salida", help="Carpeta de resultados de K-means")
    parser.add_argument("--carpeta-if", help="Carpeta de los resultados del script 09")
    parser.add_argument("--fraccion-alerta", type=float, default=0.01,
                        help="Fracción exploratoria del entrenamiento; por defecto 0.01")
    parser.add_argument("--muestra-entrenamiento", type=int, default=12000,
                        help="Muestra de entrenamiento para elegir K; por defecto 12000")
    parser.add_argument("--muestra-validacion", type=int, default=1500,
                        help="Muestra de validación para silhouette; por defecto 1500")
    args = parser.parse_args()
    if min(args.muestra_entrenamiento, args.muestra_validacion) < 30:
        parser.error("Las muestras deben contener al menos 30 observaciones")
    raiz = Path(__file__).resolve().parent
    ejecutar(
        Path(args.preparados).expanduser() if args.preparados else raiz / "Resultados_Tarea3" / "00_preparacion",
        Path(args.salida).expanduser() if args.salida else raiz / "Resultados_Tarea3" / "02_kmeans",
        Path(args.carpeta_if).expanduser() if args.carpeta_if else raiz / "Resultados_Tarea3" / "01_isolation_forest",
        args.fraccion_alerta, args.muestra_entrenamiento, args.muestra_validacion,
    )


if __name__ == "__main__":
    main()
