
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

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "Falta SHAP. Instala con el mismo Python con que corres el programa:\n"
        "python -m pip install shap"
    ) from exc

SMP = "smp_continuo"
SEMILLA = 42
SENSORES = [
    ("AcelX1_SisGenEol", "Ac X"), ("AcelY1_SisGenEol", "Ac Y"),
    ("AcelZ1_SisGenEol", "Ac Z"), ("GiroX_SisGenEol", "Gi X"),
    ("GiroY_SisGenEol", "Gi Y"), ("GiroZ_SisGenEol", "Gi Z"),
    ("MagneX_SisGenEol", "Ma X"), ("MagneY_SisGenEol", "Ma Y"),
    ("MagneZ_SisGenEol", "Ma Z"), ("Ter_SisGenEol", "Temp"),
]


def cargar(preparados: Path, carpeta_if: Path):
    archivos = [
        preparados / "01_entrenamiento_escalado.csv",
        preparados / "01_prueba_escalado.csv",
        carpeta_if / "01_resultado_prueba.csv",
        carpeta_if / "00_modelo_isolation_forest.joblib",
    ]
    for ruta in archivos:
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}. Revisa ejecucion de scripts 08 y 09.")
    train = pd.read_csv(archivos[0], sep=";", encoding="utf-8-sig")
    prueba = pd.read_csv(archivos[1], sep=";", encoding="utf-8-sig")
    resultado = pd.read_csv(archivos[2], sep=";", encoding="utf-8-sig")
    modelo = joblib.load(archivos[3])  # Cargar solo archivos joblib generados por el propio proyecto.
    vars_z = [f"z_{nombre}" for nombre, _ in SENSORES]
    for nombre, df in [("entrenamiento", train), ("prueba", prueba)]:
        if list(df.columns) != [SMP, *vars_z]:
            raise ValueError(f"Columnas no esperadas en {nombre}: {list(df.columns)}")
        if df.isna().any().any() or not np.isfinite(df[vars_z].to_numpy(float)).all():
            raise ValueError(f"Hay datos nulos o no finitos en {nombre}")
    necesarias = {SMP, "puntaje_anomalia", "alerta_isolation_forest"}
    if not necesarias.issubset(resultado.columns):
        raise ValueError("El resultado 09 no contiene todas las columnas requeridas")
    if len(prueba) != len(resultado) or not np.array_equal(
        prueba[SMP].to_numpy(), resultado[SMP].to_numpy()
    ):
        raise ValueError("Los SMP del script 08 y el resultado 09 de prueba no coinciden")
    # Verificar que los resultados guardados corresponden al modelo cargado.
    indices = np.linspace(0, len(prueba) - 1, min(15, len(prueba)), dtype=int)
    calculado = -modelo.score_samples(prueba.iloc[indices][vars_z].to_numpy(float))
    guardado = resultado.iloc[indices]["puntaje_anomalia"].to_numpy(float)
    if not np.allclose(calculado, guardado, atol=1e-6, rtol=1e-6):
        raise ValueError("El modelo guardado y los puntajes 09 no coinciden; ejecuta 09 de nuevo")
    return train, prueba, resultado, modelo, vars_z


def obtener_muestra(resultado: pd.DataFrame, n_alertas: int, n_normales: int) -> np.ndarray:
    rng = np.random.default_rng(SEMILLA)
    alerta = resultado["alerta_isolation_forest"].to_numpy(dtype=int) == 1
    indices_alerta = np.flatnonzero(alerta)
    indices_normales = np.flatnonzero(~alerta)
    if len(indices_alerta) == 0:
        raise ValueError("No se encontraron alertas de Isolation Forest en prueba")
    if len(indices_normales) == 0:
        raise ValueError("No hay registros normales en prueba para comparar")
    selec_a = rng.choice(indices_alerta, size=min(n_alertas, len(indices_alerta)), replace=False)
    selec_n = rng.choice(indices_normales, size=min(n_normales, len(indices_normales)), replace=False)
    return np.sort(np.concatenate([selec_a, selec_n]))


def graficar(importancias: pd.DataFrame, tabla: pd.DataFrame,
             valores: np.ndarray, base: np.ndarray, salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    vista = importancias.sort_values("shap_abs_media_total", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ypos = np.arange(len(vista))
    alto = 0.37
    ax.barh(ypos - alto / 2, vista["shap_abs_media_alertas"], height=alto,
            color="#A13D69", label="Alertas de prueba (muestra)")
    ax.barh(ypos + alto / 2, vista["shap_abs_media_normales"], height=alto,
            color="#497C9D", label="No alertas de prueba (muestra)")
    ax.set_yticks(ypos, vista["sensor"])
    ax.set(xlabel="Media de |SHAP| para -score_samples (unidades de puntuación)",
           title="SHAP | Influencia sobre la puntuación de Isolation Forest")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.13)
    fig.tight_layout()
    fig.savefig(salida / "01_importancia_shap_por_variable.png", dpi=300)
    plt.close(fig)

    # Seleccionamos el registro con mayor puntaje entre los registros explicados.
    i = int(tabla["puntaje_anomalia"].to_numpy().argmax())
    contribs = valores[i]
    nombres = [nombre for _, nombre in SENSORES]
    orden = np.argsort(np.abs(contribs))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(np.arange(len(orden)), contribs[orden],
            color=["#B64B64" if x >= 0 else "#467EAB" for x in contribs[orden]])
    ax.set_yticks(np.arange(len(orden)), np.array(nombres)[orden])
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set(xlabel="Contribución SHAP (positiva = aumenta anomalía)",
           title=f"Isolation Forest | Explicación de SMP {int(tabla[SMP].iloc[i])}")
    ax.grid(axis="x", alpha=0.13)
    fig.tight_layout()
    fig.savefig(salida / "02_explicacion_registro_alerta.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(tabla["puntaje_anomalia"], base + valores.sum(axis=1),
               c=tabla["alerta_isolation_forest"].map({0: "#497C9D", 1: "#B64B64"}),
               alpha=0.75)
    a = min(float(tabla["puntaje_anomalia"].min()), float((base + valores.sum(axis=1)).min()))
    b = max(float(tabla["puntaje_anomalia"].max()), float((base + valores.sum(axis=1)).max()))
    ax.plot([a, b], [a, b], ls="--", color="#333333")
    ax.set(xlabel="Puntuación real (-score_samples)",
           ylabel="Base SHAP + suma de contribuciones",
           title="Comprobación de aditividad SHAP")
    fig.tight_layout()
    fig.savefig(salida / "03_comprobacion_aditividad_shap.png", dpi=300)
    plt.close(fig)


def ejecutar(preparados: Path, carpeta_if: Path, salida: Path,
            n_alertas: int, n_normales: int, n_fondo: int) -> None:
    if min(n_alertas, n_normales, n_fondo) < 1:
        raise ValueError("--alertas, --normales y --fondo deben ser positivos")
    train, prueba, res, modelo, vars_z = cargar(preparados, carpeta_if)
    rng = np.random.default_rng(SEMILLA)
    idx_fondo = rng.choice(len(train), size=min(n_fondo, len(train)), replace=False)
    fondo = train.iloc[idx_fondo][vars_z].to_numpy(dtype=float)
    indices = obtener_muestra(res, n_alertas, n_normales)
    X = prueba.iloc[indices][vars_z].to_numpy(dtype=float)

    def puntuar(matriz: np.ndarray) -> np.ndarray:
        # MISMA salida del 09: mayor puntuación => más anómalo.
        return -modelo.score_samples(np.asarray(matriz, dtype=float))

    # Explanaciones agnósticas: a diferencia de TreeExplainer, aquí
    # explicamos directamente el puntaje -score_samples, no la longitud del camino.
    mascara = shap.maskers.Independent(fondo, max_samples=len(fondo))
    explainer = shap.Explainer(puntuar, mascara, algorithm="permutation",
                             feature_names=[nom for _, nom in SENSORES], seed=SEMILLA)
    explicaciones = explainer(X, max_evals=2 * len(vars_z) + 1,
                              batch_size=64, silent=True)
    valores = np.asarray(explicaciones.values, dtype=float)
    bases = np.broadcast_to(np.asarray(explicaciones.base_values, dtype=float), len(indices))
    puntajes = res.iloc[indices]["puntaje_anomalia"].to_numpy(dtype=float)
    error_max = float(np.max(np.abs(bases + valores.sum(axis=1) - puntajes)))

    tabla = res.iloc[indices][[SMP, "puntaje_anomalia", "alerta_isolation_forest"]].copy()
    tabla["shap_valor_base"] = bases
    tabla["shap_suma"] = valores.sum(axis=1)
    tabla["shap_puntaje_reconstruido"] = bases + valores.sum(axis=1)
    for j, (nombre, _) in enumerate(SENSORES):
        tabla[f"shap_{nombre}"] = valores[:, j]
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "01_explicaciones_registros_prueba.csv", sep=";", index=False,
                 encoding="utf-8-sig")

    mask_alertas = tabla["alerta_isolation_forest"].to_numpy(dtype=int) == 1
    importancia = pd.DataFrame({
        "sensor": [nom for _, nom in SENSORES],
        "columna_original": [col for col, _ in SENSORES],
        "shap_abs_media_total": np.abs(valores).mean(axis=0),
        "shap_abs_media_alertas": np.abs(valores[mask_alertas]).mean(axis=0),
        "shap_abs_media_normales": np.abs(valores[~mask_alertas]).mean(axis=0),
        "shap_media_con_signo_alertas": valores[mask_alertas].mean(axis=0),
    }).sort_values("shap_abs_media_total", ascending=False)
    importancia.to_csv(salida / "00_importancias_shap.csv", sep=";", index=False,
                       encoding="utf-8-sig")
    graficar(importancia, tabla, valores, bases, salida)
    metadata = {
        "modelo": "IsolationForest guardado por 09, no reentrenado",
        "salida_explicada": "-modelo.score_samples; valores grandes = más anómalo",
        "metodo_SHAP": "PermutationExplainer con fondo del entrenamiento",
        "registros_prueba_explicados": int(len(indices)),
        "alertas_explicadas": int(mask_alertas.sum()),
        "no_alertas_explicadas": int((~mask_alertas).sum()),
        "referencias_entrenamiento": int(len(fondo)),
        "error_maximo_aditividad": error_max,
        "advertencias": [
            "Las importancias son locales y dependen del fondo elegido.",
            "La muestra explicada no representa necesariamente toda la base.",
            "SHAP describe contribuciones a la salida del modelo, no causas ni fallas reales.",
        ],
    }
    (salida / "00_metadatos_shap.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSHAP PARA ISOLATION FOREST FINALIZADO")
    print(f"Registros de prueba explicados: {len(indices)} "
          f"({int(mask_alertas.sum())} alertas y {int((~mask_alertas).sum())} no alertas)")
    print(f"Referencias de entrenamiento: {len(fondo)}")
    print("\nVariables con mayor |SHAP| promedio EN LA MUESTRA:")
    print(importancia[["sensor", "shap_abs_media_total"]].head(10).to_string(index=False))
    print(f"Error máximo de reconstrucción SHAP: {error_max:.8g}")
    print(f"Resultados: {salida}")
    print("Nota: SHAP no diagnostica fallas ni establece causas físicas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparados", help="Carpeta del script 08")
    parser.add_argument("--isolation", help="Carpeta del script 09")
    parser.add_argument("--salida", help="Carpeta de salida SHAP")
    parser.add_argument("--alertas", type=int, default=20, help="Alertas de prueba a explicar")
    parser.add_argument("--normales", type=int, default=20, help="No alertas de prueba a explicar")
    parser.add_argument("--fondo", type=int, default=30, help="Registros de entrenamiento de referencia")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    ejecutar(
        Path(args.preparados).expanduser() if args.preparados else raiz / "Resultados_Tarea3" / "00_preparacion",
        Path(args.isolation).expanduser() if args.isolation else raiz / "Resultados_Tarea3" / "01_isolation_forest",
        Path(args.salida).expanduser() if args.salida else raiz / "Resultados_Tarea3" / "04_shap",
        args.alertas, args.normales, args.fondo,
    )


if __name__ == "__main__":
    main()
