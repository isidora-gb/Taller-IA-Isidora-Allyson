
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

SMP = "smp_continuo"
CONJUNTOS = ("entrenamiento", "validacion", "prueba")
SEMILLA = 42


def leer_preparados(carpeta: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    partes = {}
    for conjunto in CONJUNTOS:
        ruta = carpeta / f"01_{conjunto}_escalado.csv"
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}. Ejecuta primero 08_preparar_datos_tarea3.py")
        partes[conjunto] = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    variables = [c for c in partes["entrenamiento"].columns if c.startswith("z_")]
    if len(variables) != 10:
        raise ValueError(f"Se requieren 10 variables estandarizadas, pero hay: {variables}")
    for nombre, datos in partes.items():
        if list(datos.columns) != [SMP, *variables]:
            raise ValueError(f"Columnas inesperadas en {nombre}")
        X = datos[variables].to_numpy(dtype=float)
        if not np.isfinite(X).all() or datos[SMP].isna().any():
            raise ValueError(f"Valores invalidos en {nombre}")
        if not datos[SMP].is_monotonic_increasing or datos[SMP].duplicated().any():
            raise ValueError(f"SMP no esta ordenado o tiene duplicados en {nombre}")
    if partes["entrenamiento"][SMP].iloc[-1] >= partes["validacion"][SMP].iloc[0]:
        raise ValueError("Entrenamiento y validacion no estan en orden cronologico")
    if partes["validacion"][SMP].iloc[-1] >= partes["prueba"][SMP].iloc[0]:
        raise ValueError("Validacion y prueba no estan en orden cronologico")
    return partes, variables


def elegir_parametros(X_muestra: np.ndarray, min_samples: int,
                      objetivo_ruido: float) -> tuple[DBSCAN, float, pd.DataFrame, np.ndarray]:
    """Explora eps de las distancias al k-esimo vecino; elige por heuristica.

    CRITERIO: acercarse a un 5% orientativo de ruido EN LA MUESTRA ENTRENADA,
    siempre con al menos un cluster. Esto no es una tasa de falla ni de error.
    """
    vecinos = NearestNeighbors(n_neighbors=min_samples, algorithm="auto", n_jobs=1)
    vecinos.fit(X_muestra)
    distancias, _ = vecinos.kneighbors(X_muestra)
    # Incluye al propio registro (distancia cero). k-esimo vecino, incluido el propio.
    k_dist = distancias[:, -1]
    cuantiles = (0.65, 0.75, 0.85, 0.90, 0.95, 0.98)
    candidatos = sorted({float(np.quantile(k_dist, q)) for q in cuantiles if np.quantile(k_dist, q) > 0})
    if not candidatos:
        raise ValueError("Las distancias al vecino son cero; DBSCAN no es informativo en esta muestra")
    opciones = []
    for eps in candidatos:
        modelo = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", algorithm="auto", n_jobs=1)
        etiquetas = modelo.fit_predict(X_muestra)
        ruido = int(np.count_nonzero(etiquetas == -1))
        clusters = int(len(set(etiquetas) - {-1}))
        cores = int(len(modelo.core_sample_indices_))
        fraccion = ruido / len(X_muestra)
        # Penalizar soluciones degeneradas; luego preferir ruido proximo al objetivo.
        penalizacion = 100 if clusters == 0 or cores == 0 else 0
        puntuacion = penalizacion + abs(fraccion - objetivo_ruido)
        opciones.append((puntuacion, eps, modelo, {
            "eps": eps, "min_samples": min_samples, "clusters_en_muestra": clusters,
            "ruido_en_muestra": ruido, "porcentaje_ruido_muestra": 100 * fraccion,
            "puntos_nucleo_muestra": cores, "criterio_heuristico": puntuacion,
        }))
    opciones.sort(key=lambda x: (x[0], x[1]))
    _, eps_elegido, modelo_elegido, _ = opciones[0]
    if len(modelo_elegido.core_sample_indices_) == 0:
        raise ValueError("Todos los candidatos resultaron solo ruido. Ajustar --min-samples o tamano de muestra")
    tabla = pd.DataFrame([item[3] for item in sorted(opciones, key=lambda x: x[1])])
    tabla["seleccionado"] = np.isclose(tabla["eps"], eps_elegido)
    return modelo_elegido, eps_elegido, tabla, np.sort(k_dist)


def proyectar_nuevos(puntos: np.ndarray, vecinos_core: NearestNeighbors,
                     etiquetas_core: np.ndarray, eps: float,
                     tamano_lote: int) -> tuple[np.ndarray, np.ndarray]:
    """Extension fija del DBSCAN entrenado: nucleo mas cercano dentro de eps."""
    etiquetas = np.empty(len(puntos), dtype=np.int32)
    distancias = np.empty(len(puntos), dtype=float)
    for inicio in range(0, len(puntos), tamano_lote):
        fin = min(inicio + tamano_lote, len(puntos))
        dist, ids = vecinos_core.kneighbors(puntos[inicio:fin], n_neighbors=1)
        d = dist[:, 0]
        etiqueta = etiquetas_core[ids[:, 0]].astype(np.int32, copy=True)
        etiqueta[d > eps] = -1
        distancias[inicio:fin] = d
        etiquetas[inicio:fin] = etiqueta
    return etiquetas, distancias


def construir_episodios(datos: pd.DataFrame) -> pd.DataFrame:
    """SMP consecutivos, sin conectar huecos ni contar duracion en segundos."""
    smp = datos[SMP].to_numpy(dtype=np.int64)
    bandera = datos["alerta_ruido_dbscan"].to_numpy(dtype=bool)
    cambios = np.diff(smp) != 1
    inicios = np.flatnonzero(bandera & np.r_[True, (~bandera[:-1] | cambios)])
    finales = np.flatnonzero(bandera & np.r_[(~bandera[1:] | cambios), True])
    filas = []
    for i, (inicio, fin) in enumerate(zip(inicios, finales), 1):
        filas.append({
            "episodio": i, "smp_inicial": int(smp[inicio]),
            "smp_final": int(smp[fin]),
            "longitud_en_muestras": int(fin - inicio + 1),
            "distancia_nucleo_maxima": float(datos["distancia_nucleo_mas_cercano"].iloc[inicio:fin+1].max()),
        })
    return pd.DataFrame(filas, columns=[
        "episodio", "smp_inicial", "smp_final", "longitud_en_muestras", "distancia_nucleo_maxima",
    ])


def comparar_modelos(resultado: pd.DataFrame, carpeta_if: Path,
                     carpeta_kmeans: Path, salida: Path) -> str:
    """Une las alertas de prueba por SMP. Coincidencia NO equivale a precision."""
    archivos = [
        (carpeta_if / "01_resultado_prueba.csv", "alerta_isolation_forest"),
        (carpeta_kmeans / "01_resultado_prueba.csv", "alerta_distancia_kmeans"),
    ]
    tabla = resultado[[SMP, "alerta_ruido_dbscan"]].copy()
    for ruta, columna in archivos:
        if not ruta.is_file():
            return f"Comparacion de tres modelos omitida: falta {ruta}"
        dato = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
        if not {SMP, columna}.issubset(dato.columns):
            return f"Comparacion omitida: falta columna {columna} en {ruta.name}"
        tabla = tabla.merge(dato[[SMP, columna]], on=SMP, how="left", validate="one_to_one")
    if len(tabla) != len(resultado) or tabla.isna().any().any():
        return "Comparacion omitida: SMP de tres modelos no coincide totalmente"
    columnas = ["alerta_isolation_forest", "alerta_distancia_kmeans", "alerta_ruido_dbscan"]
    for nombre in columnas:
        tabla[nombre] = tabla[nombre].astype(int)
    tabla["modelos_en_alerta"] = tabla[columnas].sum(axis=1)
    tabla.to_csv(salida / "05_comparacion_tres_modelos_prueba.csv", sep=";", index=False, encoding="utf-8-sig")
    combinaciones = (tabla.groupby(columnas, sort=True).size().reset_index(name="registros"))
    combinaciones.to_csv(salida / "05_coincidencias_combinaciones.csv", sep=";", index=False, encoding="utf-8-sig")
    informe = {
        "registros_comparados": int(len(tabla)),
        "alertas_isolation_forest": int(tabla[columnas[0]].sum()),
        "alertas_kmeans": int(tabla[columnas[1]].sum()),
        "alertas_dbscan": int(tabla[columnas[2]].sum()),
        "alertas_los_tres": int((tabla["modelos_en_alerta"] == 3).sum()),
        "alertas_al_menos_dos": int((tabla["modelos_en_alerta"] >= 2).sum()),
        "alertas_al_menos_uno": int((tabla["modelos_en_alerta"] >= 1).sum()),
        "ninguna_alerta": int((tabla["modelos_en_alerta"] == 0).sum()),
        "nota": "Coincidencias entre detectores, NO matriz de confusion ni exactitud diagnóstica",
    }
    pd.DataFrame([informe]).to_csv(salida / "05_resumen_comparacion_tres_modelos_prueba.csv",
                                 sep=";", index=False, encoding="utf-8-sig")
    return (f"Comparacion de tres algoritmos guardada. Coinciden los tres en "
            f"{informe['alertas_los_tres']} SMP de prueba.")


def guardar_graficos(tabla_parametros: pd.DataFrame, distancias_k: np.ndarray,
                     prueba: pd.DataFrame, eps: float, min_samples: int, salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(distancias_k) + 1), distancias_k, color="#46637F", lw=1.2)
    ax.axhline(eps, color="#9A325D", ls="--", label=f"eps seleccionado = {eps:.3f}")
    ax.set(xlabel="Posicion en la muestra ordenada", ylabel=f"Distancia al vecino {min_samples}",
           title="DBSCAN | Curva de distancias a vecinos (entrenamiento)")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "01_curva_distancias_vecinos.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tabla_parametros["eps"], tabla_parametros["porcentaje_ruido_muestra"],
            marker="o", color="#914070", label="Ruido de la muestra")
    ax.scatter([eps], [float(tabla_parametros.loc[tabla_parametros["seleccionado"],
                                                    "porcentaje_ruido_muestra"].iloc[0])],
               color="#BA253E", s=60, zorder=3, label="Seleccionado")
    ax.set(xlabel="eps (en variables estandarizadas)", ylabel="Registros como ruido (%)",
           title="DBSCAN | Sensibilidad a eps (entrenamiento)")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "02_sensibilidad_eps.png", dpi=300)
    plt.close(fig)

    bandera = prueba["alerta_ruido_dbscan"].astype(bool)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prueba[SMP], prueba["distancia_nucleo_mas_cercano"],
            color="#46637F", lw=0.7, alpha=0.8)
    ax.scatter(prueba.loc[bandera, SMP], prueba.loc[bandera, "distancia_nucleo_mas_cercano"],
               s=8, color="#BB3C54", label="Ruido (extension aproximada)")
    ax.axhline(eps, color="#2E3740", ls="--", label="eps fijo")
    ax.set(xlabel="SMP continuo (posicion, NO segundos)",
           ylabel="Distancia al punto nucleo mas cercano",
           title="DBSCAN | Registros de prueba fuera del radio eps")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "03_ruido_dbscan_prueba.png", dpi=300)
    plt.close(fig)


def ejecutar(preparados: Path, salida: Path, carpeta_if: Path, carpeta_kmeans: Path,
            muestra_train: int, min_samples: int, objetivo_ruido: float,
            tamano_lote: int) -> None:
    partes, variables = leer_preparados(preparados)
    if muestra_train < 200:
        raise ValueError("--muestra-entrenamiento debe ser >= 200")
    if min_samples < 3 or min_samples >= muestra_train:
        raise ValueError("--min-samples debe ser >= 3 y menor a la muestra")
    if not (0.0 < objetivo_ruido < 0.5):
        raise ValueError("--objetivo-ruido debe ser entre 0 y 0.5")
    if tamano_lote < 1:
        raise ValueError("--tamano-lote debe ser positivo")
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    X_train = partes["entrenamiento"][variables].to_numpy(dtype=float)
    indice_muestra = np.sort(rng.choice(len(X_train), min(len(X_train), muestra_train), replace=False))
    X_muestra = X_train[indice_muestra]
    modelo, eps, tabla_params, dist_k = elegir_parametros(X_muestra, min_samples, objetivo_ruido)
    tabla_params.to_csv(salida / "00_busqueda_eps.csv", sep=";", index=False, encoding="utf-8-sig")

    nucleos = modelo.components_
    etiquetas_nucleos = modelo.labels_[modelo.core_sample_indices_]
    vecinos_core = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1)
    vecinos_core.fit(nucleos)

    resultados = {}
    resumen = []
    distribucion = []
    for nombre in CONJUNTOS:
        bloque = partes[nombre]
        X = bloque[variables].to_numpy(dtype=float)
        grupos, distancias = proyectar_nuevos(X, vecinos_core, etiquetas_nucleos, eps, tamano_lote)
        alertas = (grupos == -1).astype(np.int8)
        tabla = pd.DataFrame({
            SMP: bloque[SMP].to_numpy(dtype=np.int64),
            "cluster_dbscan_aproximado": grupos,
            "distancia_nucleo_mas_cercano": distancias,
            "alerta_ruido_dbscan": alertas,
        })
        tabla.to_csv(salida / f"01_resultado_{nombre}.csv", sep=";", index=False, encoding="utf-8-sig")
        episodios = construir_episodios(tabla)
        episodios.to_csv(salida / f"02_episodios_{nombre}.csv", sep=";", index=False, encoding="utf-8-sig")
        n_clusters = len(set(grupos) - {-1})
        resumen.append({
            "conjunto": nombre, "registros": len(bloque),
            "eps": float(eps), "min_samples": int(min_samples),
            "clusters_presentes": n_clusters, "registros_ruido": int(alertas.sum()),
            "porcentaje_ruido": round(100 * alertas.mean(), 4),
            "ruido_por_1000_muestras": round(1000 * alertas.mean(), 3),
            "episodios_ruido": len(episodios),
            "distancia_nucleo_mediana": float(np.median(distancias)),
            "distancia_nucleo_p95": float(np.quantile(distancias, .95)),
            "metodo_fuera_muestra": "nucleo mas cercano en radio eps (aproximado)",
        })
        for grupo, cuenta in zip(*np.unique(grupos, return_counts=True)):
            distribucion.append({
                "conjunto": nombre, "cluster": int(grupo),
                "registros": int(cuenta), "porcentaje": 100 * int(cuenta) / len(grupos),
            })
        resultados[nombre] = tabla

    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(salida / "00_resumen_dbscan.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.DataFrame(distribucion).to_csv(salida / "00_distribucion_clusters_dbscan.csv", sep=";",
                                               index=False, encoding="utf-8-sig")
    joblib.dump({
        "modelo_dbscan_muestra": modelo, "vecinos_core": vecinos_core,
        "etiquetas_nucleos": etiquetas_nucleos, "eps": eps,
        "min_samples": min_samples, "variables": variables,
        "indice_muestra_entrenamiento": indice_muestra,
    }, salida / "00_modelo_dbscan_muestra.joblib")
    config = {
        "algoritmo": "DBSCAN", "random_state": SEMILLA,
        "ajuste": "muestra del bloque entrenamiento (sin validacion ni prueba)",
        "muestra_entrenamiento_efectiva": len(X_muestra),
        "min_samples": min_samples, "eps": eps,
        "fraccion_ruido_objetivo_exploratorio": objetivo_ruido,
        "criterio_eps": "distancia a min_samples-ésimo vecino; seleccionar tasa de ruido de muestra proxima a objetivo orientativo",
        "dbscan_muestra_clusters": int(len(set(modelo.labels_) - {-1})),
        "dbscan_muestra_ruido": int(np.count_nonzero(modelo.labels_ == -1)),
        "dbscan_muestra_nucleos": int(len(modelo.core_sample_indices_)),
        "extension_fuera_muestra": "nucleo mas cercano <= eps; sin reentrenar; asignacion aproximada",
        "advertencias": [
            "La clase -1 significa ruido de densidad, no averia confirmada",
            "DBSCAN sklearn no ofrece predict() nativo para registros nuevos",
            "Clasificacion de todos los SMP es una extension aproximada de la muestra entrenada",
            "SMP no representa segundos", "No hay etiquetas reales para calcular matriz de confusion",
            "Las tasas de ruido no son directamente equivalentes al 1% fijado en IF y K-means",
        ],
    }
    (salida / "00_configuracion_dbscan.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    guardar_graficos(tabla_params, dist_k, resultados["prueba"], eps, min_samples, salida)
    mensaje = comparar_modelos(resultados["prueba"], carpeta_if, carpeta_kmeans, salida)
    print("\nDBSCAN FINALIZADO")
    print(f"Muestra entrenamiento usada: {len(X_muestra)} registros")
    print(f"eps elegido: {eps:.6f} | min_samples: {min_samples}")
    print(f"Clusters en la muestra DBSCAN: {config['dbscan_muestra_clusters']} | "
          f"Ruido en la muestra: {config['dbscan_muestra_ruido']} | "
          f"Nucleos: {config['dbscan_muestra_nucleos']}")
    print(resumen_df.to_string(index=False))
    print(f"\nResultados: {salida}")
    print(mensaje)
    print("IMPORTANTE: sobre el conjunto completo se uso asignacion aproximada al nucleo mas cercano.")
    print("Las alertas por ruido no representan fallas verificadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparados", type=Path, help="Carpeta creada por el script 08")
    parser.add_argument("--salida", type=Path, help="Carpeta para los resultados DBSCAN")
    parser.add_argument("--carpeta-if", type=Path, help="Resultados del script 09")
    parser.add_argument("--carpeta-kmeans", type=Path, help="Resultados del script 10")
    parser.add_argument("--muestra-entrenamiento", type=int, default=4000,
                        help="Maximo de muestras usadas para entrenar DBSCAN (default 4000)")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Vecinos minimos requeridos en el radio eps (default 20)")
    parser.add_argument("--objetivo-ruido", type=float, default=0.05,
                        help="Heuristica exploratoria de ruido EN muestra: 0.05 (no umbral de falla)")
    parser.add_argument("--tamano-lote", type=int, default=2000,
                        help="Tamaño de los bloques para procesar el total (default 2000)")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    ejecutar(
        args.preparados or raiz / "Resultados_Tarea3" / "00_preparacion",
        args.salida or raiz / "Resultados_Tarea3" / "03_dbscan",
        args.carpeta_if or raiz / "Resultados_Tarea3" / "01_isolation_forest",
        args.carpeta_kmeans or raiz / "Resultados_Tarea3" / "02_kmeans",
        args.muestra_entrenamiento, args.min_samples, args.objetivo_ruido, args.tamano_lote,
    )


if __name__ == "__main__":
    main()
