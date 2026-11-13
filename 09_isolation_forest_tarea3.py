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
from sklearn.ensemble import IsolationForest

SMP = "smp_continuo"
CONJUNTOS = ("entrenamiento", "validacion", "prueba")


def leer_preparados(carpeta: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    partes = {}
    for conjunto in CONJUNTOS:
        ruta = carpeta / f"01_{conjunto}_escalado.csv"
        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta {ruta}. Primero ejecuta 08_preparar_datos_tarea3.py"
            )
        partes[conjunto] = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    variables = [c for c in partes["entrenamiento"].columns if c.startswith("z_")]
    if len(variables) != 10:
        raise ValueError(f"Se esperan 10 variables estandarizadas, encontradas: {variables}")
    for conjunto, df in partes.items():
        if list(df.columns) != [SMP, *variables]:
            raise ValueError(f"Las columnas no coinciden en {conjunto}")
        if df.isna().any().any() or not np.isfinite(df[variables].to_numpy()).all():
            raise ValueError(f"Hay valores no finitos en {conjunto}")
    return partes, variables


def episodios_por_smp(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrupa alertas solo si las posiciones SMP son realmente consecutivas."""
    smp = datos[SMP].to_numpy(dtype=np.int64)
    alertas = datos["alerta_isolation_forest"].to_numpy(dtype=bool)
    inicios = np.flatnonzero(alertas & (
        np.r_[True, ~alertas[:-1]] | np.r_[True, np.diff(smp) != 1]
    ))
    filas = []
    for numero, inicio in enumerate(inicios, 1):
        fin = inicio
        while fin + 1 < len(datos) and alertas[fin+1] and smp[fin+1]-smp[fin] == 1:
            fin += 1
        filas.append({
            "episodio": numero,
            "smp_inicial": int(smp[inicio]),
            "smp_final": int(smp[fin]),
            "longitud_en_muestras": int(fin - inicio + 1),
            "puntaje_maximo": float(datos["puntaje_anomalia"].iloc[inicio:fin+1].max()),
        })
    return pd.DataFrame(filas, columns=[
        "episodio", "smp_inicial", "smp_final", "longitud_en_muestras", "puntaje_maximo"
    ])


def graficar(distribuciones: dict[str, np.ndarray],
             resultado_prueba: pd.DataFrame, umbral: float,
             salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    colores = {"entrenamiento": "#347CA5", "validacion": "#D18B34", "prueba": "#9C3262"}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for nombre in CONJUNTOS:
        valores = distribuciones[nombre]
        ax.hist(valores, bins=65, density=True, alpha=0.42,
                color=colores[nombre], label=nombre.capitalize())
    ax.axvline(umbral, color="#292929", linestyle="--", lw=1.8,
               label="Umbral (solo entrenamiento)")
    ax.set(xlabel="Puntuación de anomalía (-score_samples)",
           ylabel="Densidad", title="Isolation Forest | Distribución de puntuaciones")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "01_distribucion_puntuaciones.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    prueba = resultado_prueba
    alertas = prueba["alerta_isolation_forest"].to_numpy(dtype=bool)
    ax.plot(prueba[SMP], prueba["puntaje_anomalia"], color="#46637F",
            alpha=0.62, lw=0.75, label="Puntuación")
    ax.scatter(prueba.loc[alertas, SMP],
               prueba.loc[alertas, "puntaje_anomalia"],
               s=12, color="#C24354", label="Alertas", zorder=3)
    ax.axhline(umbral, color="#333333", lw=1.4, linestyle="--",
               label="Umbral fijo")
    ax.set(xlabel="SMP continuo (posición de adquisición, no segundos)",
           ylabel="Puntuación de anomalía", title="Isolation Forest | Evaluación en prueba")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "02_alertas_prueba_por_smp.png", dpi=300)
    plt.close(fig)


def ejecutar(carpeta: Path, salida: Path, fraccion_alerta: float) -> None:
    if not 0.0 < fraccion_alerta < 1.0:
        raise ValueError("--fraccion-alerta debe ser mayor que cero y menor que uno")
    partes, variables = leer_preparados(carpeta)
    salida.mkdir(parents=True, exist_ok=True)
    modelo = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",   # NO se usa predict() ni su umbral automático
        random_state=42,
        n_jobs=-1,
    )
    X_train = partes["entrenamiento"][variables].to_numpy(dtype=float)
    modelo.fit(X_train)
    puntuaciones = {
        nombre: -modelo.score_samples(datos[variables].to_numpy(dtype=float))
        for nombre, datos in partes.items()
    }
    # Cuota fija inicial: aproximadamente 1% de entrenamiento con mayor puntuación.
    umbral = float(np.quantile(puntuaciones["entrenamiento"], 1.0 - fraccion_alerta))
    filas_resumen = []
    resultado_prueba = None
    for nombre in CONJUNTOS:
        datos = partes[nombre]
        puntajes = puntuaciones[nombre]
        flags = puntajes >= umbral
        resultado = pd.DataFrame({
            SMP: datos[SMP].to_numpy(),
            "puntaje_anomalia": puntajes,
            "alerta_isolation_forest": flags.astype(int),
        })
        resultado.to_csv(salida / f"01_resultado_{nombre}.csv", sep=";",
                         index=False, encoding="utf-8-sig")
        episodios = episodios_por_smp(resultado)
        episodios.to_csv(salida / f"02_episodios_{nombre}.csv", sep=";",
                         index=False, encoding="utf-8-sig")
        filas_resumen.append({
            "conjunto": nombre,
            "registros": len(datos),
            "alertas": int(flags.sum()),
            "porcentaje_alertas": round(100.0 * float(flags.mean()), 4),
            "alertas_por_1000_muestras": round(1000.0 * float(flags.mean()), 3),
            "episodios": len(episodios),
            "puntaje_mediano": float(np.median(puntajes)),
            "puntaje_percentil_95": float(np.quantile(puntajes, 0.95)),
        })
        if nombre == "prueba":
            resultado_prueba = resultado

    pd.DataFrame(filas_resumen).to_csv(
        salida / "00_resumen_isolation_forest.csv", sep=";", index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(modelo, salida / "00_modelo_isolation_forest.joblib")
    configuracion = {
        "algoritmo": "IsolationForest", "n_estimators": 300,
        "max_samples": "auto", "random_state": 42,
        "fraccion_alertas_referencia_entrenamiento": fraccion_alerta,
        "umbral_puntuacion": umbral,
        "convencion_puntuacion": "mayor => más anómalo (-score_samples)",
        "advertencia": "Sin etiquetas verificadas: NO se calculó exactitud ni matriz de confusión",
    }
    (salida / "00_configuracion_modelo.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert resultado_prueba is not None
    graficar(puntuaciones, resultado_prueba, umbral, salida)
    print("\nISOLATION FOREST FINALIZADO")
    print(pd.DataFrame(filas_resumen).to_string(index=False))
    print(f"\nUmbral fijo desde entrenamiento: {umbral:.6f}")
    print(f"Resultados: {salida}")
    print("Advertencia: las alertas no son averías verificadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparados", help="Carpeta generada por el script 08")
    parser.add_argument("--salida", help="Carpeta de resultados Isolation Forest")
    parser.add_argument("--fraccion-alerta", type=float, default=0.01,
                        help="Fracción de referencia en entrenamiento, por defecto 0.01")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    preparados = (Path(args.preparados).expanduser() if args.preparados
                  else raiz / "Resultados_Tarea3" / "00_preparacion")
    salida = (Path(args.salida).expanduser() if args.salida
              else raiz / "Resultados_Tarea3" / "01_isolation_forest")
    ejecutar(preparados, salida, args.fraccion_alerta)


if __name__ == "__main__":
    main()
