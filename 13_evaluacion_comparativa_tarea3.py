
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMP = "smp_continuo"
PARTES = ("entrenamiento", "validacion", "prueba")
MODELOS = {
    "Isolation Forest": ("01_isolation_forest", "alerta_isolation_forest"),
    "K-means": ("02_kmeans", "alerta_distancia_kmeans"),
    "DBSCAN": ("03_dbscan", "alerta_ruido_dbscan"),
}
COLORES = {"Isolation Forest": "#356C91", "K-means": "#9A4872", "DBSCAN": "#BA862F"}
NOMBRES = {
    "AcelX1_SisGenEol": "Ac X", "AcelY1_SisGenEol": "Ac Y", "AcelZ1_SisGenEol": "Ac Z",
    "GiroX_SisGenEol": "Gi X", "GiroY_SisGenEol": "Gi Y", "GiroZ_SisGenEol": "Gi Z",
    "MagneX_SisGenEol": "Ma X", "MagneY_SisGenEol": "Ma Y", "MagneZ_SisGenEol": "Ma Z",
    "Ter_SisGenEol": "Temp",
}


def leer_csv(ruta: Path) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró {ruta}. Ejecuta los códigos anteriores primero.")
    return pd.read_csv(ruta, sep=";", encoding="utf-8-sig")


def unir_resultados(raiz_resultados: Path, parte: str) -> pd.DataFrame:
    tabla = None
    for nombre, (subcarpeta, columna) in MODELOS.items():
        archivo = raiz_resultados / subcarpeta / f"01_resultado_{parte}.csv"
        dato = leer_csv(archivo)
        if not {SMP, columna}.issubset(dato):
            raise ValueError(f"Falta {SMP} o {columna} en {archivo}")
        dato = dato[[SMP, columna]].copy()
        if dato[SMP].isna().any() or dato[SMP].duplicated().any():
            raise ValueError(f"SMP nulo o repetido en {archivo}")
        if not dato[columna].isin((0, 1, False, True)).all():
            raise ValueError(f"Las alertas de {nombre} no son binarias en {parte}")
        dato[columna] = dato[columna].astype(np.int8)
        if tabla is None:
            tabla = dato
        else:
            tabla = tabla.merge(dato, on=SMP, how="outer", validate="one_to_one")
    assert tabla is not None
    if tabla.isna().any().any():
        raise ValueError(f"Algunos SMP no coinciden entre los modelos en {parte}")
    tabla = tabla.sort_values(SMP).reset_index(drop=True)
    if not tabla[SMP].is_monotonic_increasing:
        raise ValueError(f"Hay problemas de orden en {parte}")
    tabla["cantidad_modelos_en_alerta"] = tabla[[x[1] for x in MODELOS.values()]].sum(axis=1)
    tabla["consenso_al_menos_2"] = (tabla["cantidad_modelos_en_alerta"] >= 2).astype(np.int8)
    tabla["consenso_los_3"] = (tabla["cantidad_modelos_en_alerta"] == 3).astype(np.int8)
    return tabla


def episodios_por_smp(smp: np.ndarray, alerta: np.ndarray, nombre: str) -> pd.DataFrame:
    smp = np.asarray(smp, dtype=np.int64)
    alerta = np.asarray(alerta, dtype=bool)
    if len(smp) != len(alerta):
        raise ValueError("La longitud de las alertas y de SMP no coincide")
    if len(smp) == 0:
        return pd.DataFrame(columns=["criterio", "smp_inicial", "smp_final", "longitud_muestras"])
    # Un episodio se corta ante salto de SMP o ausencia de alerta; SMP no expresa segundos.
    inicio = alerta & (np.r_[True, ~alerta[:-1]] | np.r_[True, np.diff(smp) != 1])
    fin = alerta & (np.r_[~alerta[1:], True] | np.r_[np.diff(smp) != 1, True])
    indices_inicio, indices_fin = np.flatnonzero(inicio), np.flatnonzero(fin)
    if len(indices_inicio) != len(indices_fin):
        raise AssertionError("La construcción de episodios falló")
    return pd.DataFrame({
        "criterio": nombre,
        "smp_inicial": smp[indices_inicio],
        "smp_final": smp[indices_fin],
        "longitud_muestras": indices_fin - indices_inicio + 1,
    })


def resumen_y_bloques(tablas: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    resumen, bloques = [], []
    for parte, tabla in tablas.items():
        smp = tabla[SMP].to_numpy()
        for nombre, (_, columna) in MODELOS.items():
            bandera = tabla[columna].to_numpy(dtype=bool)
            epi = episodios_por_smp(smp, bandera, nombre)
            resumen.append({
                "conjunto": parte, "modelo": nombre, "registros": len(tabla),
                "alertas": int(bandera.sum()),
                "porcentaje_alertas": 100 * bandera.mean(),
                "alertas_por_1000_muestras": 1000 * bandera.mean(),
                "episodios": len(epi),
                "longitud_media_episodio_muestras": float(epi["longitud_muestras"].mean()) if len(epi) else 0.0,
                "episodios_3_o_mas_muestras": int((epi["longitud_muestras"] >= 3).sum()),
            })
        for numero, indices in enumerate(np.array_split(np.arange(len(tabla)), 10), 1):
            if not len(indices):
                continue
            grupo = tabla.iloc[indices]
            for nombre, (_, columna) in MODELOS.items():
                bloques.append({
                    "conjunto": parte, "bloque": numero, "modelo": nombre,
                    "registros": len(grupo),
                    "smp_inicio": int(grupo[SMP].iloc[0]),
                    "smp_fin": int(grupo[SMP].iloc[-1]),
                    "alertas": int(grupo[columna].sum()),
                    "porcentaje_alertas": 100 * grupo[columna].mean(),
                })
    return pd.DataFrame(resumen), pd.DataFrame(bloques)


def coincidencias_pares(tabla: pd.DataFrame) -> pd.DataFrame:
    items = list(MODELOS.items())
    filas = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, col_a = items[i][0], items[i][1][1]
            b, col_b = items[j][0], items[j][1][1]
            fa, fb = tabla[col_a].to_numpy(dtype=bool), tabla[col_b].to_numpy(dtype=bool)
            interseccion = int((fa & fb).sum())
            union = int((fa | fb).sum())
            filas.append({
                "modelo_A": a, "modelo_B": b,
                "alertas_A": int(fa.sum()), "alertas_B": int(fb.sum()),
                "coincidencias_ambos": interseccion,
                "registros_marcados_por_alguno": union,
                "jaccard_pct": 100 * interseccion / union if union else np.nan,
                "advertencia": "Jaccard = coincidencia de alertas, no exactitud ni precision diagnostica",
            })
    return pd.DataFrame(filas)


def cambios_variables(raiz_resultados: Path) -> pd.DataFrame:
    preparados = raiz_resultados / "00_preparacion"
    datos = {parte: leer_csv(preparados / f"01_{parte}_escalado.csv") for parte in PARTES}
    columnas = [c for c in datos["entrenamiento"].columns if c.startswith("z_")]
    if len(columnas) != 10:
        raise ValueError("Se requieren las diez variables estandarizadas por el código 08")
    filas = []
    for col in columnas:
        valores = {p: datos[p][col].to_numpy(dtype=float) for p in PARTES}
        for parte in PARTES:
            if not np.isfinite(valores[parte]).all():
                raise ValueError(f"Variable no finita {col}, {parte}")
            filas.append({
                "variable": NOMBRES.get(col[2:], col), "columna_estandarizada": col,
                "conjunto": parte,
                "media_z": float(np.mean(valores[parte])),
                "mediana_z": float(np.median(valores[parte])),
                "p05_z": float(np.quantile(valores[parte], 0.05)),
                "p95_z": float(np.quantile(valores[parte], 0.95)),
                "diferencia_media_vs_entrenamiento_z": float(np.mean(valores[parte]) - np.mean(valores["entrenamiento"])),
                "diferencia_mediana_vs_entrenamiento_z": float(np.median(valores[parte]) - np.median(valores["entrenamiento"])),
            })
    return pd.DataFrame(filas)


def hacer_figuras(resumen: pd.DataFrame, bloques: pd.DataFrame, salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(3)
    ancho = 0.24
    for i, nombre in enumerate(MODELOS):
        filas = resumen[resumen["modelo"] == nombre].set_index("conjunto")
        tasa = filas.loc[list(PARTES), "porcentaje_alertas"].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * ancho, tasa, width=ancho, label=nombre, color=COLORES[nombre])
    ax.set_xticks(x, ["Entrenamiento", "Validación", "Prueba"])
    ax.set(ylabel="Registros señalados como inusuales (%)",
           title="Comparación descriptiva de tres métodos")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E3E3E3", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(salida / "01_comparacion_alertas.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharey=False, sharex=True)
    for ax, parte in zip(axes, PARTES):
        for nombre in MODELOS:
            df = bloques[(bloques["conjunto"] == parte) & (bloques["modelo"] == nombre)]
            ax.plot(df["bloque"], df["porcentaje_alertas"], marker="o", markersize=3.5,
                    linewidth=1.5, label=nombre, color=COLORES[nombre])
        ax.set_title(parte.capitalize())
        ax.set_ylabel("Alertas (%)")
        ax.grid(axis="y", color="#E3E3E3", alpha=0.7)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Bloque consecutivo de registros (1 a 10 por conjunto)")
    axes[-1].set_xticks(range(1, 11))
    fig.suptitle("Evolución de alertas por posición de adquisición (no por segundos)")
    fig.tight_layout()
    fig.savefig(salida / "02_evolucion_bloques.png", dpi=300)
    plt.close(fig)


def escribir_borrador(tablas: dict[str, pd.DataFrame], resumen: pd.DataFrame,
                      pares: pd.DataFrame, shap: pd.DataFrame | None, salida: Path) -> None:
    prueba = tablas["prueba"]
    n = len(prueba)
    tres = int(prueba["consenso_los_3"].sum())
    dos = int(prueba["consenso_al_menos_2"].sum())
    uno = int((prueba["cantidad_modelos_en_alerta"] >= 1).sum())
    encabezado = [
        "# Evaluación comparativa descriptiva – Tarea 3",
        "",
        "**Alcance:** monitoreo no supervisado de diez señales de sensores; sin etiquetas de fallas confirmadas.",
        "La prueba se revisa descriptivamente, no se usa para ajustar parámetros o elegir umbrales de nuevo.",
        "",
        f"En el conjunto de prueba se analizaron **{n:,} registros**.",
    ]
    for modelo in MODELOS:
        r = resumen[(resumen["modelo"] == modelo) & (resumen["conjunto"] == "prueba")].iloc[0]
        encabezado.append(
            f"- **{modelo}:** {int(r['alertas']):,} alertas ({r['porcentaje_alertas']:.2f} %) y "
            f"{int(r['episodios']):,} episodios de SMP consecutivos."
        )
    encabezado += [
        "",
        f"Los tres marcaron simultáneamente **{tres:,} registros** ",
        f"(al menos dos: **{dos:,}**; al menos uno: **{uno:,}**).",
        "La coincidencia puede servir para priorizar inspecciones, pero no mide precisión diagnóstica.",
        "",
        "## Coincidencia por pares (Jaccard)",
        "",
    ]
    for _, r in pares.iterrows():
        encabezado.append(
            f"- {r['modelo_A']} / {r['modelo_B']}: {int(r['coincidencias_ambos'])} coincidencias, "
            f"Jaccard = {r['jaccard_pct']:.1f} %."
        )
    if shap is not None and len(shap):
        top = shap.iloc[0]
        encabezado += [
            "", "## Explicabilidad SHAP", "",
            f"En la **muestra explicada** por el código 12, la mayor media de |SHAP| correspondió a "
            f"{top['sensor']} ({float(top['shap_abs_media_total']):.6f}).",
            "Se trata de una explicación del modelo, no de una causa física ni de la importancia global validada.",
        ]
    encabezado += [
        "", "## Advertencias metodológicas", "",
        "- Isolation Forest y K-means usan un umbral fijado al percentil 99 del entrenamiento (alrededor de 1 %).",
        "- DBSCAN seleccionó parámetros con una tasa orientativa de ruido en una muestra del entrenamiento; "
        "fuera de muestra emplea asignación aproximada al punto núcleo más cercano.",
        "- Comparar tasas no demuestra cuál modelo es mejor: las políticas de alerta no están igualadas.",
        "- La evaluación en la prueba ya ha sido inspeccionada. Cualquier ajuste posterior necesita nuevos datos "
        "retenidos para una comprobación independiente.",
        "- No hay verdad terreno para calcular exactitud, precisión, sensibilidad, F1, PR-AUC ni falsos positivos reales.",
        "- SMP es un contador de muestras: episodios y bloques no indican duración en segundos.",
        "- Cambios estadísticos en las señales no permiten inferir averías sin mediciones operativas e inspecciones.",
    ]
    (salida / "07_borrador_interpretacion.md").write_text("\n".join(encabezado) + "\n", encoding="utf-8")


def ejecutar(raiz_resultados: Path, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    tablas = {parte: unir_resultados(raiz_resultados, parte) for parte in PARTES}
    for antes, despues in zip(PARTES, PARTES[1:]):
        if int(tablas[antes][SMP].max()) >= int(tablas[despues][SMP].min()):
            raise ValueError("La división de los tres conjuntos no es cronológica")
    resumen, bloques = resumen_y_bloques(tablas)
    resumen.to_csv(salida / "00_resumen_modelos.csv", sep=";", index=False, encoding="utf-8-sig")
    bloques.to_csv(salida / "03_alertas_por_bloque.csv", sep=";", index=False, encoding="utf-8-sig")
    prueba = tablas["prueba"]
    pares = coincidencias_pares(prueba)
    pares.to_csv(salida / "01_coincidencias_pares_prueba.csv", sep=";", index=False, encoding="utf-8-sig")
    prueba.to_csv(salida / "02_alertas_y_consenso_prueba.csv", sep=";", index=False, encoding="utf-8-sig")
    epi = pd.concat([
        episodios_por_smp(prueba[SMP].to_numpy(), prueba[columna].to_numpy(), nombre)
        for columna, nombre in [("consenso_al_menos_2", "Al menos dos"), ("consenso_los_3", "Los tres")]
    ], ignore_index=True)
    epi.to_csv(salida / "04_episodios_consenso_prueba.csv", sep=";", index=False, encoding="utf-8-sig")
    cambios = cambios_variables(raiz_resultados)
    cambios.to_csv(salida / "05_cambios_variables_estandarizadas.csv", sep=";", index=False, encoding="utf-8-sig")
    ruta_shap = raiz_resultados / "04_shap" / "00_importancias_shap.csv"
    if ruta_shap.is_file():
        shap = leer_csv(ruta_shap)
        shap.to_csv(salida / "06_importancias_shap_muestra.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        shap = None
        print("Nota: no encontré SHAP; se omite su copia en la evaluación.")
    hacer_figuras(resumen, bloques, salida)
    escribir_borrador(tablas, resumen, pares, shap, salida)
    print("\nEVALUACIÓN COMPARATIVA FINALIZADA")
    print("\nResultados por conjunto:")
    print(resumen[["conjunto", "modelo", "registros", "alertas", "porcentaje_alertas", "episodios"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nCoincidencias en prueba:")
    print(pares[["modelo_A", "modelo_B", "coincidencias_ambos", "jaccard_pct"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nConsenso de al menos dos en prueba: {int(prueba['consenso_al_menos_2'].sum())}")
    print(f"Consenso de los tres en prueba: {int(prueba['consenso_los_3'].sum())}")
    print(f"\nArchivos: {salida}")
    print("No se calculan métricas supervisadas sin etiquetas verificadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados", help="Carpeta Resultados_Tarea3, por defecto al lado del script")
    parser.add_argument("--salida", help="Carpeta destino, por defecto Resultados_Tarea3/05_evaluacion_comparativa")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    resultados = Path(args.resultados).expanduser() if args.resultados else raiz / "Resultados_Tarea3"
    salida = Path(args.salida).expanduser() if args.salida else resultados / "05_evaluacion_comparativa"
    ejecutar(resultados, salida)


if __name__ == "__main__":
    main()
