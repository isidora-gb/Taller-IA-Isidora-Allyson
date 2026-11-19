
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

SMP = "smp_continuo"
SEMILLA = 42
NOMBRES_CORTOS = {
    "AcelX1_SisGenEol": "Ac X", "AcelY1_SisGenEol": "Ac Y",
    "AcelZ1_SisGenEol": "Ac Z", "GiroX_SisGenEol": "Gi X",
    "GiroY_SisGenEol": "Gi Y", "GiroZ_SisGenEol": "Gi Z",
    "MagneX_SisGenEol": "Ma X", "MagneY_SisGenEol": "Ma Y",
    "MagneZ_SisGenEol": "Ma Z", "Ter_SisGenEol": "Temp",
}


def exigir_archivo(ruta: Path) -> Path:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encuentra: {ruta}\nEjecuta primero los scripts 08 a 11.")
    return ruta


def leer_json(ruta: Path) -> dict:
    return json.loads(exigir_archivo(ruta).read_text(encoding="utf-8"))


def cargar_artefactos(raiz: Path) -> dict:
    resultados = raiz / "Resultados_Tarea3"
    preparacion = resultados / "00_preparacion"
    iforest = resultados / "01_isolation_forest"
    kmeans = resultados / "02_kmeans"
    dbscan = resultados / "03_dbscan"
    # Cargar solamente archivos joblib creados por este proyecto (nunca de fuentes desconocidas).
    prep = joblib.load(exigir_archivo(preparacion / "00_preprocesamiento.joblib"))
    modelo_if = joblib.load(exigir_archivo(iforest / "00_modelo_isolation_forest.joblib"))
    modelo_km = joblib.load(exigir_archivo(kmeans / "00_modelo_kmeans.joblib"))
    modelo_db = joblib.load(exigir_archivo(dbscan / "00_modelo_dbscan_muestra.joblib"))
    config_if = leer_json(iforest / "00_configuracion_modelo.json")
    config_km = leer_json(kmeans / "00_configuracion_kmeans.json")
    vars_originales = list(prep["variables"])
    vars_z = [f"z_{v}" for v in vars_originales]
    if len(vars_originales) != 10:
        raise ValueError("El preprocesamiento no contiene diez variables.")
    if vars_z != list(modelo_db["variables"]):
        raise ValueError("El orden de las variables no coincide con el DBSCAN entrenado.")
    if int(modelo_if.n_features_in_) != 10 or int(modelo_km.n_features_in_) != 10:
        raise ValueError("El numero de variables no coincide con los modelos guardados.")
    if not all(np.isfinite(float(v)) for v in (
        config_if["umbral_puntuacion"], config_km["umbral_distancia"], modelo_db["eps"]
    )):
        raise ValueError("Umbrales invalidos.")
    return {
        "prep": prep, "if": modelo_if, "km": modelo_km, "db": modelo_db,
        "umbral_if": float(config_if["umbral_puntuacion"]),
        "umbral_km": float(config_km["umbral_distancia"]),
        "variables": vars_originales, "vars_z": vars_z,
        "preparacion": preparacion,
    }


def a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    ).astype(float).replace([np.inf, -np.inf], np.nan)


def cargar_datos(raiz: Path, entrada: Path | None, modo: str, obj: dict) -> pd.DataFrame:
    if entrada is None:
        if modo == "nuevos":
            raise ValueError("Con --modo nuevos tienes que pasar --entrada nuevas_mediciones.csv")
        meta = leer_json(obj["preparacion"] / "00_metadatos.json")
        entrada = Path(meta["archivo_fuente"])
        if not entrada.is_file():
            # En caso de mover el proyecto a otra carpeta, probar ruta local conocida.
            entrada = raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv"
    elif not entrada.is_absolute():
        entrada = raiz / entrada
    exigir_archivo(entrada)
    df = pd.read_csv(entrada, sep=";", encoding="utf-8-sig", dtype="string")
    necesarias = [SMP, *obj["variables"]]
    faltan = [v for v in necesarias if v not in df.columns]
    if faltan:
        raise ValueError(f"Columnas faltantes en {entrada}: {faltan}")
    df = df[necesarias].copy()
    df[SMP] = a_numero(df[SMP])
    if df[SMP].isna().any() or (df[SMP] % 1 != 0).any():
        raise ValueError("SMP debe ser entero y estar completo.")
    df[SMP] = df[SMP].astype("int64")
    if df[SMP].duplicated().any():
        raise ValueError("Existen SMP duplicados; corrige los datos de entrada.")
    for sensor in obj["variables"]:
        df[sensor] = a_numero(df[sensor])
    if df.empty:
        raise ValueError("Archivo sin mediciones.")
    df = df.sort_values(SMP, kind="mergesort").reset_index(drop=True)
    if modo == "prueba":
        division = pd.read_csv(
            exigir_archivo(obj["preparacion"] / "00_division_cronologica.csv"),
            sep=";", encoding="utf-8-sig"
        )
        prueba = division.loc[division["conjunto"] == "prueba", SMP]
        smp_esperados = prueba.to_numpy(dtype=np.int64)
        subconjunto = df[df[SMP].isin(smp_esperados)].copy().reset_index(drop=True)
        if len(subconjunto) != len(smp_esperados) or not np.array_equal(
            subconjunto[SMP].to_numpy(dtype=np.int64), smp_esperados
        ):
            raise ValueError(
                "Para modo prueba, la entrada debe contener todos los SMP de la prueba "
                "original. Usa --modo nuevos si tienes un archivo diferente."
            )
        df = subconjunto
    else:
        # En modo nuevos hay que asegurar que no se pasan, por error, los datos históricos.
        limite_historico = int(pd.read_csv(
            exigir_archivo(obj["preparacion"] / "00_division_cronologica.csv"),
            sep=";", encoding="utf-8-sig", usecols=[SMP]
        )[SMP].max())
        if (df[SMP] <= limite_historico).any():
            raise ValueError(
                f"El modo nuevos requiere SMP posteriores a {limite_historico}, "
                "para no confundir registros historicos con nuevos."
            )
    return df


def calcular_episodios(tabla: pd.DataFrame, columna: str) -> pd.DataFrame:
    smp = tabla[SMP].to_numpy(np.int64)
    flags = tabla[columna].to_numpy(bool)
    inicios = np.flatnonzero(flags & (
        np.r_[True, ~flags[:-1]] | np.r_[True, np.diff(smp) != 1]
    ))
    fines = np.flatnonzero(flags & (
        np.r_[~flags[1:], True] | np.r_[np.diff(smp) != 1, True]
    ))
    filas = []
    for numero, (inicio, fin) in enumerate(zip(inicios, fines), 1):
        filas.append({
            "episodio": numero,
            "smp_inicial": int(smp[inicio]),
            "smp_final": int(smp[fin]),
            "longitud_en_muestras": int(fin - inicio + 1),
            "puntaje_maximo_if": float(tabla["puntaje_anomalia"].iloc[inicio:fin+1].max()),
        })
    return pd.DataFrame(filas, columns=[
        "episodio", "smp_inicial", "smp_final",
        "longitud_en_muestras", "puntaje_maximo_if",
    ])


def puntuar(df: pd.DataFrame, obj: dict) -> tuple[pd.DataFrame, np.ndarray]:
    vars_orig = obj["variables"]
    imputaciones = df[vars_orig].isna().sum(axis=1).to_numpy(int)
    x_imp = obj["prep"]["imputador"].transform(df[vars_orig].to_numpy(float))
    x = obj["prep"]["escalador"].transform(x_imp)
    if not np.isfinite(x).all():
        raise ValueError("El preprocesamiento genero datos no finitos")
    punt_if = -obj["if"].score_samples(x)
    aviso_if = punt_if >= obj["umbral_if"]
    cluster_km = obj["km"].predict(x)
    distancia_km = np.linalg.norm(x - obj["km"].cluster_centers_[cluster_km], axis=1)
    aviso_km = distancia_km >= obj["umbral_km"]
    # DBSCAN NO predice fuera de la muestra. Esta es solo una extension aproximada.
    vecinos = obj["db"]["vecinos_core"]
    distancia_db = np.empty(len(x), dtype=float)
    cluster_db = np.empty(len(x), dtype=int)
    for a in range(0, len(x), 4096):
        b = min(a + 4096, len(x))
        dist, idx = vecinos.kneighbors(x[a:b], n_neighbors=1)
        distancia_db[a:b] = dist[:, 0]
        cluster_db[a:b] = obj["db"]["etiquetas_nucleos"][idx[:, 0]]
    aviso_db = distancia_db > float(obj["db"]["eps"])
    cluster_db[aviso_db] = -1
    votos = aviso_if.astype(int) + aviso_km.astype(int) + aviso_db.astype(int)
    prioridad = np.select(
        [votos == 3, aviso_if & (votos == 2), aviso_if & (votos == 1),
         ~aviso_if & (votos == 2), ~aviso_if & (votos == 1)],
        ["prioridad_alta", "prioridad_media", "revision_if",
         "revision_complementaria", "solo_metodo_complementario"],
        default="sin_alerta"
    )
    tabla = pd.DataFrame({
        SMP: df[SMP].to_numpy(np.int64),
        "celdas_imputadas_para_modelo": imputaciones,
        "puntaje_anomalia": punt_if,
        "umbral_if_fijo": obj["umbral_if"],
        "alerta_isolation_forest": aviso_if.astype(int),
        "cluster_kmeans": cluster_km,
        "distancia_al_centroide": distancia_km,
        "alerta_distancia_kmeans": aviso_km.astype(int),
        "cluster_dbscan_aproximado": cluster_db,
        "distancia_nucleo_mas_cercano": distancia_db,
        "alerta_ruido_dbscan": aviso_db.astype(int),
        "numero_metodos_que_alertan": votos,
        "alerta_consenso_2_o_3": (votos >= 2).astype(int),
        "nivel_de_revision_propuesto": prioridad,
    })
    return tabla, x


def ejecutar_shap(tabla: pd.DataFrame, x: np.ndarray, obj: dict,
                  max_shap: int, fondo_n: int, salida: Path) -> tuple[int, float | None]:
    if max_shap == 0:
        return 0, None
    indices = np.flatnonzero(tabla["alerta_isolation_forest"].to_numpy(bool))
    if len(indices) == 0:
        pd.DataFrame(columns=[SMP, "sin_alertas_if_para_explicar"]).to_csv(
            salida / "04_shap_alertas_prioritarias.csv", sep=";", index=False,
            encoding="utf-8-sig"
        )
        return 0, None
    # Se explican primero las alertas IF con mayor puntaje; no todas las observaciones.
    orden = np.argsort(-tabla.iloc[indices]["puntaje_anomalia"].to_numpy())
    seleccion = indices[orden[:max_shap]]
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Falta SHAP. Instala: python -m pip install shap") from exc
    train = pd.read_csv(
        exigir_archivo(obj["preparacion"] / "01_entrenamiento_escalado.csv"),
        sep=";", encoding="utf-8-sig"
    )
    if list(train.columns) != [SMP, *obj["vars_z"]]:
        raise ValueError("Las variables del CSV de entrenamiento no coinciden")
    rng = np.random.default_rng(SEMILLA)
    inds_fondo = rng.choice(len(train), size=min(fondo_n, len(train)), replace=False)
    fondo = train.iloc[inds_fondo][obj["vars_z"]].to_numpy(float)

    def f(matriz: np.ndarray) -> np.ndarray:
        return -obj["if"].score_samples(np.asarray(matriz, float))

    mascara = shap.maskers.Independent(fondo, max_samples=len(fondo))
    explainer = shap.Explainer(
        f, mascara, algorithm="permutation",
        feature_names=[NOMBRES_CORTOS.get(v, v) for v in obj["variables"]],
        seed=SEMILLA,
    )
    explicacion = explainer(
        x[seleccion], max_evals=2 * len(obj["variables"]) + 1,
        batch_size=64, silent=True,
    )
    valores = np.asarray(explicacion.values, float)
    base = np.broadcast_to(np.asarray(explicacion.base_values, float), len(seleccion))
    reales = tabla.iloc[seleccion]["puntaje_anomalia"].to_numpy(float)
    errores = np.abs(base + valores.sum(axis=1) - reales)
    resultado = tabla.iloc[seleccion][[
        SMP, "puntaje_anomalia", "nivel_de_revision_propuesto",
        "numero_metodos_que_alertan"
    ]].copy().reset_index(drop=True)
    resultado["base_shap"] = base
    resultado["reconstruccion_shap"] = base + valores.sum(axis=1)
    resultado["error_reconstruccion"] = errores
    etiquetas = [NOMBRES_CORTOS.get(v, v) for v in obj["variables"]]
    for j, nombre in enumerate(etiquetas):
        resultado[f"shap_{nombre.replace(' ', '_')}"] = valores[:, j]
    resultado["tres_mayores_contribuciones_positivas"] = [
        ", ".join([etiquetas[j] for j in np.argsort(-fila) if fila[j] > 0][:3])
        for fila in valores
    ]
    resultado.to_csv(salida / "04_shap_alertas_prioritarias.csv", sep=";", index=False,
                     encoding="utf-8-sig", float_format="%.10g")
    importancia = pd.DataFrame({
        "sensor": etiquetas,
        "media_shap_absoluta_alertas_seleccionadas": np.abs(valores).mean(axis=0),
    }).sort_values("media_shap_absoluta_alertas_seleccionadas", ascending=False)
    importancia.to_csv(salida / "05_importancia_shap_alertas_seleccionadas.csv",
                       sep=";", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    vista = importancia.sort_values("media_shap_absoluta_alertas_seleccionadas")
    ax.barh(vista["sensor"], vista["media_shap_absoluta_alertas_seleccionadas"])
    ax.set(xlabel="Media de |SHAP| de la puntuacion IF",
           title="Contribuciones en alertas IF seleccionadas (no toda la base)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida / "02_shap_alertas_seleccionadas.png", dpi=250)
    plt.close(fig)
    return int(len(seleccion)), float(errores.max())


def grafico_monitoreo(tabla: pd.DataFrame, salida: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    alerta = tabla["alerta_isolation_forest"].to_numpy(bool)
    ax.plot(tabla[SMP], tabla["puntaje_anomalia"], linewidth=.8, alpha=.65,
            label="Puntuacion IF")
    ax.scatter(tabla.loc[alerta, SMP], tabla.loc[alerta, "puntaje_anomalia"],
               s=11, label="Alerta IF", color="#AD4161")
    ax.axhline(tabla["umbral_if_fijo"].iloc[0], ls="--", color="black",
               label="Umbral fijo de entrenamiento")
    ax.set(xlabel="SMP continuo (no equivale a segundos)",
           ylabel="Puntuacion -score_samples", title="Prototipo de monitoreo por lote")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(salida / "01_puntuacion_y_alertas.png", dpi=250)
    plt.close(fig)


def ejecutar(raiz: Path, entrada: Path | None, modo: str, max_shap: int,
             fondo_n: int, salida: Path) -> None:
    if max_shap < 0 or fondo_n < 1:
        raise ValueError("--max-shap debe ser >= 0 y --fondo >= 1")
    obj = cargar_artefactos(raiz)
    datos = cargar_datos(raiz, entrada, modo, obj)
    tabla, x = puntuar(datos, obj)
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "01_monitoreo_registros.csv", sep=";", index=False,
                 encoding="utf-8-sig", float_format="%.10g")
    episodios_if = calcular_episodios(tabla, "alerta_isolation_forest")
    episodios_consenso = calcular_episodios(tabla, "alerta_consenso_2_o_3")
    episodios_if.to_csv(salida / "02_episodios_isolation_forest.csv", sep=";",
                        index=False, encoding="utf-8-sig")
    episodios_consenso.to_csv(salida / "03_episodios_consenso.csv", sep=";",
                             index=False, encoding="utf-8-sig")
    grafico_monitoreo(tabla, salida)
    n_shap, error_shap = ejecutar_shap(tabla, x, obj, max_shap, fondo_n, salida)
    resumen = {
        "modo": modo,
        "registros": len(tabla),
        "alertas_if": int(tabla["alerta_isolation_forest"].sum()),
        "alertas_kmeans": int(tabla["alerta_distancia_kmeans"].sum()),
        "ruido_dbscan_aproximado": int(tabla["alerta_ruido_dbscan"].sum()),
        "consenso_2_o_3": int(tabla["alerta_consenso_2_o_3"].sum()),
        "coincidencia_los_3": int((tabla["numero_metodos_que_alertan"] == 3).sum()),
        "episodios_if": len(episodios_if),
        "episodios_consenso": len(episodios_consenso),
        "registros_con_alguna_imputacion": int((tabla["celdas_imputadas_para_modelo"] > 0).sum()),
        "alertas_if_explicadas_por_shap": n_shap,
        "error_maximo_reconstruccion_shap": error_shap,
    }
    pd.DataFrame([resumen]).to_csv(salida / "00_resumen_monitoreo.csv", sep=";",
                                   index=False, encoding="utf-8-sig")
    (salida / "00_limitaciones.json").write_text(json.dumps({
        "tipo": "prototipo retrospectivo por lotes; no sistema conectado en tiempo real",
        "preprocesamiento": "imputador y escalador entrenados en el script 08",
        "umbrales": "guardados por 09 y 10, no recalibrados en este programa",
        "dbscan": "extension aproximada por nucleo mas cercano; no predict nativo",
        "shap": "permutation SHAP de -score_samples, solo alertas IF seleccionadas; "
                "no participa en el consenso",
        "episodios": "continuidad solo por SMP consecutivo; longitud en muestras, no segundos",
        "prioridad": "regla exploratoria de apoyo a revision humana; no orden de mantenimiento",
        "evaluacion": "conjunto de prueba ya examinado; esto no es una nueva validacion independiente",
        "etiquetas": "no hay fallas confirmadas ni matriz de confusion real",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMONITOREO INTEGRADO CON SHAP FINALIZADO")
    print(pd.DataFrame([resumen]).to_string(index=False))
    print(f"\nResultados: {salida}")
    print("SHAP explica la puntuacion del IF; no diagnostica causas ni fallas.")
    print("Este es un prototipo por lotes, no un monitor en tiempo real.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", help="CSV corregido o nuevo CSV con diez senales y smp_continuo")
    parser.add_argument("--modo", choices=["prueba", "nuevos"], default="prueba",
                        help="Prueba retrospectiva o nuevos SMP posteriores al historial")
    parser.add_argument("--max-shap", type=int, default=20,
                        help="Maximo de alertas IF a explicar por SHAP (0 para omitir)")
    parser.add_argument("--fondo", type=int, default=30,
                        help="Numero de observaciones de entrenamiento para referencia SHAP")
    parser.add_argument("--salida", help="Carpeta de resultados")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    salida = Path(args.salida).expanduser() if args.salida else (
        raiz / "Resultados_Tarea3" / "07_monitoreo_integrado_shap"
    )
    if not salida.is_absolute():
        salida = raiz / salida
    ejecutar(raiz, Path(args.entrada).expanduser() if args.entrada else None,
             args.modo, args.max_shap, args.fondo, salida)


if __name__ == "__main__":
    main()
