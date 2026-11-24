
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

SMP = "smp_continuo"
SENSORES = [
    "AcelX1_SisGenEol", "AcelY1_SisGenEol", "AcelZ1_SisGenEol",
    "GiroX_SisGenEol", "GiroY_SisGenEol", "GiroZ_SisGenEol",
    "MagneX_SisGenEol", "MagneY_SisGenEol", "MagneZ_SisGenEol",
    "Ter_SisGenEol",
]
CORTOS = ["Ac X", "Ac Y", "Ac Z", "Gi X", "Gi Y", "Gi Z",
          "Ma X", "Ma Y", "Ma Z", "Temp"]
PARTES = ("entrenamiento", "validacion", "prueba")
COLORES = {"entrenamiento": "#2876A9", "validacion": "#D58B3F",
           "prueba": "#7D55A4"}


def ruta_entrada(raiz: Path, texto: str | None) -> Path:
    if texto:
        p = Path(texto).expanduser()
        p = p if p.is_absolute() else raiz / p
        if not p.is_file():
            raise FileNotFoundError(f"Archivo de entrada inexistente: {p}")
        return p.resolve()
    opciones = [
        raiz / "datos_multisensor_corregidos.csv",
        raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv",
        raiz / "Datos_corregidos_desde_excel" / "datos_multisensor_corregidos.csv",
    ]
    for p in opciones:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "No encontre datos_multisensor_corregidos.csv junto al codigo. "
        "Utiliza --entrada RUTA para indicar la ubicacion exacta."
    )


def numero(s: pd.Series) -> pd.Series:
    # El CSV de este proyecto usa punto y coma y coma decimal, incluso en E-3.
    textos = s.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(textos, errors="coerce").astype(float).replace(
        [np.inf, -np.inf], np.nan
    )


def indicador(s: pd.Series, nombre: str) -> pd.Series:
    valor = s.astype("string").str.strip().str.lower()
    positivos = {"true", "1", "si", "sí"}
    negativos = {"false", "0", "no", ""}
    sin_reconocer = ~valor.isin(positivos | negativos)
    if sin_reconocer.any():
        raise ValueError(f"Bandera {nombre}: valores no reconocidos: "
                         f"{valor[sin_reconocer].unique().tolist()[:5]}")
    return valor.isin(positivos)


def cargar_y_validar(origen: Path) -> tuple[pd.DataFrame, dict]:
    with origen.open("r", encoding="utf-8-sig") as f:
        cabecera = f.readline()
    separador = ";" if cabecera.count(";") > cabecera.count(",") else ","
    df = pd.read_csv(origen, sep=separador, encoding="utf-8-sig",
                     dtype="string", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Hay columnas duplicadas en la fuente")
    requeridas = [SMP, *SENSORES]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
    if len(df) < 30:
        raise ValueError("La base contiene menos de 30 registros")

    # No inferir valores ausentes del contador ni tratarlo como una variable fisica.
    df[SMP] = numero(df[SMP])
    if df[SMP].isna().any() or not (df[SMP] % 1 == 0).all():
        raise ValueError("smp_continuo debe ser entero, finito y sin valores ausentes")
    df[SMP] = df[SMP].astype("int64")
    if df[SMP].duplicated().any():
        raise ValueError("smp_continuo tiene identificadores duplicados")
    if not df[SMP].is_monotonic_increasing:
        raise ValueError("SMP no esta ordenado: revisar la secuencia antes de modelar")

    for c in SENSORES:
        # Registrar conversiones fallidas para no disfrazar errores de lectura.
        df[c] = numero(df[c])
    if df[SENSORES].isna().all(axis=1).any():
        raise ValueError("Hay filas completas sin mediciones fisicas")
    if (df[SENSORES].notna().sum(axis=0) == 0).any():
        raise ValueError("Existe una variable de sensor sin ninguna medicion")

    if "smp_original" in df.columns:
        df["smp_original"] = numero(df["smp_original"])
    if "smp_desde_cero" in df.columns:
        df["smp_desde_cero"] = numero(df["smp_desde_cero"])
        if not np.array_equal(df["smp_desde_cero"].to_numpy(),
                              (df[SMP] - df[SMP].iloc[0]).to_numpy()):
            raise ValueError("smp_desde_cero no concuerda con smp_continuo")
    for c in ("smp_estimado", "medicion_imputada"):
        if c in df.columns:
            df[c] = indicador(df[c], c)
        else:
            df[c] = False
    if "fila_excel" in df.columns:
        df["fila_excel"] = numero(df["fila_excel"]).astype("Int64")

    diferencias = np.diff(df[SMP].to_numpy(dtype="int64"))
    if (diferencias <= 0).any():
        raise ValueError("SMP no es estrictamente creciente")
    meta = {
        "registros": int(len(df)),
        "numero_variables_sensores": len(SENSORES),
        "smp_inicial": int(df[SMP].iloc[0]),
        "smp_final": int(df[SMP].iloc[-1]),
        "saltos_smp": int(np.count_nonzero(diferencias > 1)),
        "posiciones_smp_ausentes": int(np.maximum(diferencias - 1, 0).sum()),
        "filas_con_smp_estimado": int(df["smp_estimado"].sum()),
        "filas_con_medicion_imputada_en_origen": int(df["medicion_imputada"].sum()),
        "nulos_senales": int(df[SENSORES].isna().sum().sum()),
    }
    return df, meta


def csv_guardar(df: pd.DataFrame, p: Path) -> None:
    df.to_csv(p, sep=";", index=False, encoding="utf-8-sig",
              float_format="%.15g")


def partir(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    a, b = int(0.70 * n), int(0.85 * n)
    # Disjuntas y ordenadas, sin romper la base ni usar aleatoriedad.
    partes = {
        "entrenamiento": df.iloc[:a].copy(),
        "validacion": df.iloc[a:b].copy(),
        "prueba": df.iloc[b:].copy(),
    }
    if sum(len(p) for p in partes.values()) != n:
        raise AssertionError("La division perdio registros")
    if not (partes["entrenamiento"][SMP].max() < partes["validacion"][SMP].min()
            < partes["prueba"][SMP].min()):
        raise AssertionError("La division no respeta el orden SMP")
    return partes


def graficos(partes: dict[str, pd.DataFrame], escalados: dict,
             saltos: np.ndarray, salida: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    # 1) Segmentacion cronologica real del contador (no representa tiempo).
    fig, ax = plt.subplots(figsize=(11.5, 3.7))
    for i, nombre in enumerate(PARTES):
        d = partes[nombre]
        ax.hlines(2-i, int(d[SMP].iloc[0]), int(d[SMP].iloc[-1]),
                  color=COLORES[nombre], linewidth=10,
                  label=f"{nombre.capitalize()}: {len(d):,} registros".replace(",", "."))
    ax.set_yticks([2, 1, 0], ["Entrenamiento", "Validacion", "Prueba"])
    ax.set_xlabel("SMP continuo (contador; no segundos)")
    ax.set_title("Division cronologica de todos los registros")
    ax.grid(axis="x", alpha=.17)
    fig.tight_layout()
    fig.savefig(salida / "01_division_cronologica.png", dpi=250)
    plt.close(fig)

    # 2) Mediana e IQR por sensor y conjunto, calculados sobre TODAS las filas.
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.3), constrained_layout=True)
    for j, (col, corto) in enumerate(zip(SENSORES, CORTOS)):
        ax = axs.flat[j]
        for k, nombre in enumerate(PARTES):
            x = escalados["estandar"][nombre][:, j]
            q1, med, q3 = np.quantile(x, [.25, .5, .75])
            ax.plot([k, k], [q1, q3], color=COLORES[nombre], linewidth=5,
                    solid_capstyle="round")
            ax.scatter(k, med, color="black", s=20, zorder=3)
        ax.set_xticks([0, 1, 2], ["Ent.", "Val.", "Prueb."])
        ax.set_title(corto)
        ax.grid(axis="y", alpha=.15)
        if j in (0, 5):
            ax.set_ylabel("Escala estandar (ajustada en train)")
    fig.suptitle("Mediana (punto) y rango intercuartil (segmento) por conjunto",
                 fontsize=13, fontweight="bold")
    fig.savefig(salida / "02_distribucion_por_conjunto.png", dpi=230)
    plt.close(fig)

    # 3) Cuantificacion de nulos, mostrando explicitamente ceros.
    nulos = np.array([[int(partes[p][c].isna().sum()) for c in SENSORES]
                      for p in PARTES])
    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.imshow(nulos, cmap="Blues", aspect="auto", vmin=0,
              vmax=max(1, int(nulos.max())))
    ax.set_yticks(range(3), [p.capitalize() for p in PARTES])
    ax.set_xticks(range(10), CORTOS)
    for i in range(3):
        for j in range(10):
            ax.text(j, i, str(nulos[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("Valores ausentes por sensor y conjunto (antes de imputacion)")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(salida / "03_nulos_por_sensor.png", dpi=240)
    plt.close(fig)

    # 4) Donde se acumulan huecos del contador SMP; no son filas nulas.
    inicio = partes["entrenamiento"][SMP].iloc[0]
    todos = pd.concat([partes[p][[SMP]] for p in PARTES], ignore_index=True)[SMP].to_numpy()
    perdidos = np.r_[0, np.cumsum(np.maximum(np.diff(todos)-1, 0))]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.step(todos, perdidos, where="post", linewidth=1.4)
    for p in PARTES[1:]:
        ax.axvline(partes[p][SMP].iloc[0], linestyle="--", linewidth=1,
                   color=COLORES[p])
    ax.set_xlabel("SMP continuo (contador; no segundos)")
    ax.set_ylabel("Posiciones faltantes acumuladas")
    ax.set_title("Huecos en la secuencia SMP (no equivalen a mediciones vacias)")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(salida / "04_saltos_smp.png", dpi=240)
    plt.close(fig)


def ejecutar(origen: Path, salida: Path) -> dict:
    df, meta = cargar_y_validar(origen)
    partes = partir(df)
    train = partes["entrenamiento"]

    # Ajustar SOLO en entrenamiento, sin imputar ni sustituir valores extremos.
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(train[SENSORES].to_numpy(dtype=float))
    sc_std = StandardScaler().fit(X_train)
    sc_rob = RobustScaler(quantile_range=(25, 75)).fit(X_train)
    escaladores = {"estandar": sc_std, "robusto": sc_rob}

    salida.mkdir(parents=True, exist_ok=True)
    escalados = {"estandar": {}, "robusto": {}}
    calidad, resumen = [], []
    division = []
    columnas_z = ["z_" + c for c in SENSORES]
    for nombre in PARTES:
        bloque = partes[nombre]
        X = bloque[SENSORES].to_numpy(dtype=float)
        imputada = imp.transform(X)
        for tipo, esc in escaladores.items():
            Z = esc.transform(imputada)
            if Z.shape != (len(bloque), len(SENSORES)) or not np.isfinite(Z).all():
                raise ValueError(f"Escalamiento {tipo}/{nombre}: matriz no finita")
            escalados[tipo][nombre] = Z
            salida_z = pd.DataFrame(Z, columns=columnas_z)
            salida_z.insert(0, SMP, bloque[SMP].to_numpy(dtype="int64"))
            csv_guardar(salida_z, salida / f"01_{nombre}_{tipo}.csv")
        for c in SENSORES:
            x = bloque[c]
            calidad.append({
                "conjunto": nombre, "variable": c, "registros": len(bloque),
                "faltantes": int(x.isna().sum()), "distintos": int(x.nunique()),
                "media": float(x.mean()), "mediana": float(x.median()),
                "desviacion_estandar": float(x.std(ddof=1)),
                "minimo": float(x.min()), "maximo": float(x.max()),
            })
        resumen.append({
            "conjunto": nombre, "registros": len(bloque),
            "porcentaje": 100 * len(bloque) / len(df),
            "smp_inicial": int(bloque[SMP].iloc[0]),
            "smp_final": int(bloque[SMP].iloc[-1]),
            "celdas_senales_ausentes": int(bloque[SENSORES].isna().sum().sum()),
            "filas_preimputadas_origen": int(bloque["medicion_imputada"].sum()),
        })
        columnas_traza = [SMP, "smp_estimado", "medicion_imputada"]
        for opcional in ("smp_original", "smp_desde_cero", "fila_excel"):
            if opcional in bloque.columns:
                columnas_traza.append(opcional)
        d = bloque[columnas_traza].copy()
        d.insert(1, "conjunto", nombre)
        division.append(d)

    trazabilidad = pd.concat(division, ignore_index=True)
    assert len(trazabilidad) == len(df) and trazabilidad[SMP].is_unique
    csv_guardar(trazabilidad, salida / "00_division_cronologica.csv")
    csv_guardar(pd.DataFrame(resumen), salida / "00_resumen_division.csv")
    csv_guardar(pd.DataFrame(calidad), salida / "00_calidad_por_variable.csv")

    referencias = []
    for j, c in enumerate(SENSORES):
        original = train[c].dropna().to_numpy(dtype=float)
        referencias.append({
            "variable": c,
            "mediana_imputacion_entrenamiento": float(imp.statistics_[j]),
            "media_escalador_estandar": float(sc_std.mean_[j]),
            "desviacion_escalador_estandar": float(sc_std.scale_[j]),
            "centro_escalador_robusto": float(sc_rob.center_[j]),
            "escala_robusta_IQR": float(sc_rob.scale_[j]),
            "distintos_entrenamiento": int(pd.Series(original).nunique()),
        })
    csv_guardar(pd.DataFrame(referencias), salida / "00_parametros_entrenamiento.csv")
    joblib.dump({
        "variables": SENSORES,
        "columna_smp": SMP,
        "imputador": imp,
        "escalador_estandar": sc_std,
        "escalador_robusto": sc_rob,
        "ajuste_solo_entrenamiento": True,
    }, salida / "00_preprocesamiento.joblib")

    # El hash permite demostrar exactamente que archivo se utilizo.
    digest = hashlib.sha256()
    with origen.open("rb") as archivo:
        for bloque_bytes in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque_bytes)
    meta.update({
        "origen": str(origen), "sha256_origen": digest.hexdigest(),
        "regla_division": "70/15/15 en orden de SMP, sin mezclas",
        "conjuntos": resumen, "variables": SENSORES,
        "columnas_no_predictoras": [c for c in df if c not in SENSORES],
        "imputacion": "mediana calculada solo en entrenamiento",
        "escalado": ["StandardScaler", "RobustScaler (IQR 25-75)"],
        "tratamiento_outliers": "No se eliminan ni se sustituyen valores extremos",
        "alcance": "Preparacion para analisis no supervisado; no diagnostica fallas",
        "precaucion": "SMP ordena adquisiciones; sin frecuencia verificada no expresa tiempo",
    })
    (salida / "00_metadatos.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    graficos(partes, escalados, np.diff(df[SMP].to_numpy(dtype="int64")), salida)

    print("\nPREPARACION MULTISENSOR COMPLETADA")
    print("Entrada:", origen)
    print("Resultados:", salida.resolve())
    print("Total:", meta["registros"], "| Variables:", meta["numero_variables_sensores"])
    print(pd.DataFrame(resumen).to_string(index=False))
    print("Huecos SMP:", meta["posiciones_smp_ausentes"],
          "| Filas imputadas en origen:", meta["filas_con_medicion_imputada_en_origen"])
    print("Guardados: 6 CSV escalados, auditorias, preprocesamiento y 4 graficos.")
    print("IMPORTANTE: resultados antiguos de modelos no se mezclan con este proceso.")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="08 - Preparacion multisensor sin fuga temporal")
    parser.add_argument("--entrada", help="CSV multisensor corregido, si no esta junto al script")
    parser.add_argument("--salida", help="Carpeta de resultados (por defecto se usa la ya existente)")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    entrada = ruta_entrada(raiz, args.entrada)
    salida = Path(args.salida).expanduser() if args.salida else (
        raiz / "Resultados_Tarea3_Nuevos" / "00_preparacion")
    if not salida.is_absolute():
        salida = raiz / salida
    ejecutar(entrada, salida)


if __name__ == "__main__":
    main()
