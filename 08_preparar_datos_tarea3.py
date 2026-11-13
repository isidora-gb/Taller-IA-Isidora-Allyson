
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SENSORES = [
    "AcelX1_SisGenEol", "AcelY1_SisGenEol", "AcelZ1_SisGenEol",
    "GiroX_SisGenEol", "GiroY_SisGenEol", "GiroZ_SisGenEol",
    "MagneX_SisGenEol", "MagneY_SisGenEol", "MagneZ_SisGenEol",
    "Ter_SisGenEol",
]
SMP = "smp_continuo"   # Identificador/orden; NO es un tiempo en segundos.


def resolver_entrada(raiz: Path, argumento: str | None) -> Path:
    if argumento:
        ruta = Path(argumento).expanduser()
        if not ruta.is_absolute():
            ruta = raiz / ruta
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encuentra el CSV: {ruta}")
        return ruta

    candidatos = [
        raiz / "Datos_corregidos" / "datos_multisensor_corregidos.csv",
        raiz / "Datos_corregidos_desde_excel" / "datos_multisensor_corregidos.csv",
        raiz / "datos_multisensor_corregidos.csv",
    ]
    for ruta in candidatos:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "No encontré datos_multisensor_corregidos.csv. "
        "Ponlo dentro de Datos_corregidos/ o indícalo con --entrada."
    )


def a_numero(serie: pd.Series) -> pd.Series:
    # Los CSV históricos pueden usar coma o punto como separador decimal.
    return pd.to_numeric(
        serie.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).astype(float).replace([np.inf, -np.inf], np.nan)


def ejecutar(entrada: Path, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(entrada, sep=";", encoding="utf-8-sig", dtype="string")
    faltantes = [c for c in [SMP, *SENSORES] if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}. Columnas disponibles: {list(df.columns)}"
        )

    # Mantener las columnas de trazabilidad existentes, sin emplearlas como predictores.
    df[SMP] = a_numero(df[SMP])
    if df[SMP].isna().any():
        raise ValueError("El SMP continuo tiene valores ausentes o no numéricos.")
    if (df[SMP] % 1 != 0).any():
        raise ValueError("El SMP continuo contiene valores no enteros.")
    df[SMP] = df[SMP].astype("int64")
    if df[SMP].duplicated().any():
        raise ValueError("Hay SMP continuos duplicados; revisar el archivo de origen.")
    df = df.sort_values(SMP, kind="mergesort").reset_index(drop=True)

    for sensor in SENSORES:
        df[sensor] = a_numero(df[sensor])

    n = len(df)
    if n < 30:
        raise ValueError("Se requieren al menos 30 registros para esta división.")
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    limites = {
        "entrenamiento": (0, n_train),
        "validacion": (n_train, n_train + n_val),
        "prueba": (n_train + n_val, n),
    }
    train = df.iloc[:n_train]
    if train[SENSORES].isna().all(axis=0).any():
        vacios = train[SENSORES].columns[train[SENSORES].isna().all()].tolist()
        raise ValueError(f"No hay observaciones en entrenamiento para: {vacios}")

    imputador = SimpleImputer(strategy="median")
    escalador = StandardScaler()
    X_train_imputado = imputador.fit_transform(train[SENSORES])
    escalador.fit(X_train_imputado)

    resumen = []
    columnas_z = [f"z_{c}" for c in SENSORES]
    for nombre, (a, b) in limites.items():
        bloque = df.iloc[a:b]
        X = escalador.transform(imputador.transform(bloque[SENSORES]))
        normalizado = pd.DataFrame(X, columns=columnas_z)
        normalizado.insert(0, SMP, bloque[SMP].to_numpy())
        normalizado.to_csv(
            salida / f"01_{nombre}_escalado.csv",
            sep=";", index=False, encoding="utf-8-sig", float_format="%.10g",
        )
        resumen.append({
            "conjunto": nombre,
            "registros": int(b - a),
            "porcentaje": round(100 * (b-a)/n, 4),
            "smp_inicial": int(bloque[SMP].iloc[0]),
            "smp_final": int(bloque[SMP].iloc[-1]),
            "celdas_no_numericas_o_nulas": int(bloque[SENSORES].isna().sum().sum()),
        })

    # Guardar la referencia exacta de las particiones; NO modifica datos originales.
    division = df[[SMP]].copy()
    division["conjunto"] = (
        ["entrenamiento"] * n_train +
        ["validacion"] * n_val +
        ["prueba"] * (n - n_train - n_val)
    )
    division.to_csv(salida / "00_division_cronologica.csv", sep=";", index=False)
    pd.DataFrame(resumen).to_csv(
        salida / "00_resumen_division.csv", sep=";", index=False,
        encoding="utf-8-sig",
    )
    joblib.dump({
        "imputador": imputador,
        "escalador": escalador,
        "variables": SENSORES,
        "columna_smp": SMP,
    }, salida / "00_preprocesamiento.joblib")

    diferencias_smp = np.diff(df[SMP].to_numpy())
    metadata = {
        "archivo_fuente": str(entrada.resolve()),
        "filas_totales": n,
        "variables": SENSORES,
        "metodo": "separación por SMP, imputación mediana y StandardScaler",
        "ajuste_imputacion_escalado": "solo datos de entrenamiento",
        "numero_saltos_smp": int(np.count_nonzero(diferencias_smp > 1)),
        "posiciones_smp_faltantes": int(np.maximum(diferencias_smp - 1, 0).sum()),
        "division": resumen,
        "advertencia": "SMP no equivale a tiempo; sin etiquetas no se conocen fallas reales",
    }
    (salida / "00_metadatos.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPREPARACIÓN TAREA 3 FINALIZADA")
    print(f"Fuente: {entrada}\nSalida: {salida}")
    print(pd.DataFrame(resumen).to_string(index=False))
    print(f"Saltos de SMP: {metadata['numero_saltos_smp']}")
    print("Importante: ningún valor físico extremo fue eliminado.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", help="Ruta del CSV corregido de la Tarea 2")
    parser.add_argument("--salida", help="Carpeta de archivos preparados")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    entrada = resolver_entrada(raiz, args.entrada)
    salida = (Path(args.salida).expanduser() if args.salida
              else raiz / "Resultados_Tarea3" / "00_preparacion")
    ejecutar(entrada, salida)


if __name__ == "__main__":
    main()
