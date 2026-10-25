# -*- coding: utf-8 -*-
"""
TALLER IA 2 — Análisis exploratorio de anomalías por sensor.

USO NORMAL (NO REQUIERE ARGUMENTOS):
    python 05_rangos_anomalias_sensores.py

Lee automáticamente estos cuatro archivos, ubicados en Datos_corregidos, generado por el codigo 00:
    01_acelerometro_smp_ordenado.csv
    02_giroscopio_smp_ordenado.csv
    03_magnetometro_smp_ordenado.csv
    04_temperatura_smp_ordenado.csv

Produce Resultados_rangos_sensores/ con una carpeta para cada sensor, sin
alterar los CSV originales ni unir registros de sensores diferentes por SMP.

Estos rangos son SOLO estadísticos y experimentales, NO límites de seguridad,
falla física o mantenimiento establecidos por el fabricante.

Requisitos: pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unicodedata
import warnings

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # No abre ventanas que interrumpan el análisis.
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(
        f"Falta una biblioteca: {exc}. Ejecuta: python -m pip install pandas numpy matplotlib"
    )

# Criterios exploratorios
LIMITE_ALERTA = 3.5   # Iglewicz–Hoaglin; NIST para Z modificado.
LIMITE_EXTREMO = 6.0  # Elección del proyecto, NO valor de norma técnica.
MIN_OBSERVACIONES = 10
MAX_PUNTOS_LINEA = 22000
MAX_MARCADORES = 2500

ARCHIVOS = {
    "acelerometro": "01_acelerometro_smp_ordenado.csv",
    "giroscopio": "02_giroscopio_smp_ordenado.csv",
    "magnetometro": "03_magnetometro_smp_ordenado.csv",
    "temperatura": "04_temperatura_smp_ordenado.csv",
}
ETIQUETAS = ("NORMAL", "ALERTA", "CRITICO_ESTADISTICO")
COLORES = {
    "senal": "#25618D", "alerta": "#D48A24", "critico": "#B83844",
    "mediana": "#2E4051", "grid": "#DCE3EA",
}


def simplificar(texto: str) -> str:
    return "".join(
        letra for letra in unicodedata.normalize("NFKD", str(texto).lower())
        if not unicodedata.combining(letra)
    )


def ruta_usuario(texto: str, carpeta: Path) -> Path:
    ruta = Path(texto).expanduser()
    return ruta if ruta.is_absolute() else carpeta / ruta


def buscar_archivo(carpeta: Path, nombre: str, salida: Path) -> Path | None:
    # Se da prioridad al archivo en la misma carpeta que el script.
    directo = carpeta / nombre
    if directo.is_file():
        return directo
    encontrados = []
    for ruta in carpeta.rglob("*.csv"):
        if ruta.name.lower() != nombre.lower():
            continue
        if ruta == directo or salida == ruta or salida in ruta.parents:
            continue
        encontrados.append(ruta)
    if not encontrados:
        return None
    encontrados.sort(key=lambda p: (len(p.relative_to(carpeta).parts), str(p).lower()))
    if len(encontrados) > 1:
        print(f"[AVISO] Hay {len(encontrados)} copias de {nombre}. Se usará: {encontrados[0]}")
    return encontrados[0]


def leer_csv(ruta: Path) -> pd.DataFrame:
    # El archivo puede estar separado por comas o punto y coma.
    ultimo_error = None
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with ruta.open("r", encoding=codificacion) as archivo:
                encabezado = next((l for l in archivo if l.strip()), "")
            if not encabezado:
                raise ValueError("El CSV está vacío")
            separador = max((";", ",", "\t"), key=encabezado.count)
            if encabezado.count(separador) == 0:
                raise ValueError("No se reconoció el separador del CSV")
            tabla = pd.read_csv(ruta, sep=separador, encoding=codificacion,
                                low_memory=False, on_bad_lines="error")
            tabla.columns = [str(c).strip().lstrip("\ufeff") for c in tabla.columns]
            return tabla
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            ultimo_error = error
    raise ValueError(f"No pude leer {ruta.name}: {ultimo_error}")


def a_numero(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").astype(float).replace(
            [np.inf, -np.inf], np.nan)
    texto = serie.astype("string").str.strip().str.replace(" ", "", regex=False)
    normal = pd.to_numeric(texto, errors="coerce")
    con_coma = pd.to_numeric(texto.str.replace(",", ".", regex=False), errors="coerce")
    formato_europeo = pd.to_numeric(
        texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    opciones = (normal, con_coma, formato_europeo)
    mejor = max(opciones, key=lambda s: int(s.notna().sum()))
    return mejor.astype(float).replace([np.inf, -np.inf], np.nan)


def es_columna_sensor(nombre: str, sensor: str) -> bool:
    limpio = simplificar(nombre).replace("_", "").replace(" ", "")
    if any(x in limpio for x in ("smp", "timestamp", "fecha", "hora", "indice", "index", "unnamed")):
        return False
    prefijos = {
        "acelerometro": ("acel", "accel", "aceler", "accx", "accy", "accz"),
        "giroscopio": ("giro", "gyro", "giros", "girx", "giry", "girz", "gyr"),
        "magnetometro": ("mag", "magnet", "compass"),
        "temperatura": ("temp", "term", "thermal", "tersisgeneol"),
    }
    # "Ter_SisGenEol" es el nombre real de la columna de temperatura.
    return any(p in limpio for p in prefijos[sensor]) or (sensor == "temperatura" and limpio == "ter")


def clasificaciones(codigos: np.ndarray) -> np.ndarray:
    respuesta = np.full(codigos.size, "SIN_DATO", dtype=object)
    for codigo, etiqueta in enumerate(ETIQUETAS):
        respuesta[codigos == codigo] = etiqueta
    return respuesta


def calcular_columna(serie: pd.Series) -> dict:
    x = serie.to_numpy(dtype=float)
    validas = np.isfinite(x)
    n = int(validas.sum())
    resultado = {
        "n": n, "mediana": np.nan, "mad": np.nan, "escala": np.nan,
        "estado_referencia": "SIN_REFERENCIA", "observacion": "",
        "z": np.full(len(x), np.nan),
        "clases": np.full(len(x), -1, dtype=np.int8),
    }
    if n < MIN_OBSERVACIONES:
        resultado["observacion"] = "Menos de 10 datos numéricos válidos"
        return resultado
    mediana = float(np.median(x[validas]))
    mad = float(np.median(np.abs(x[validas] - mediana)))
    resultado.update(mediana=mediana, mad=mad)
    if not np.isfinite(mad) or mad <= 0:
        resultado["observacion"] = "MAD igual a cero; requiere evaluación especial"
        return resultado
    escala = mad / 0.6745
    z = np.full(len(x), np.nan)
    z[validas] = np.abs((x[validas] - mediana) / escala)
    clases = np.full(len(x), -1, dtype=np.int8)
    clases[validas] = 0
    clases[validas & (z > LIMITE_ALERTA)] = 1
    clases[validas & (z > LIMITE_EXTREMO)] = 2
    resultado.update(escala=escala, z=z, clases=clases,
                     estado_referencia="APTO_EXPLORATORIO")
    return resultado


def obtener_eje(tabla: pd.DataFrame) -> tuple[np.ndarray, str]:
    mapa = {simplificar(c): c for c in tabla.columns}
    for candidato in ("smp_continuo", "smp_desde_cero", "smp_original", "smp"):
        columna = mapa.get(candidato)
        if columna is None:
            continue
        x = a_numero(tabla[columna])
        if x.notna().all() and x.is_monotonic_increasing and x.nunique() > 1:
            return x.to_numpy(dtype=float), f"{columna} (índice, no tiempo)"
    return np.arange(1, len(tabla) + 1), "Número de fila (no tiempo)"


def muestra_indices(indices: np.ndarray, maximo: int) -> np.ndarray:
    if len(indices) <= maximo:
        return indices
    posiciones = np.linspace(0, len(indices) - 1, maximo).astype(int)
    return indices[posiciones]


def crear_grafico(sensor: str, columnas: list[str], tabla: pd.DataFrame,
                   calculos: dict, x: np.ndarray, nombre_eje: str, destino: Path) -> None:
    if not columnas:
        return
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.titlesize": 12,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(len(columnas), 1, figsize=(13, max(3.7, 3.0 * len(columnas))),
                             squeeze=False, constrained_layout=True)
    for ax, columna in zip(axes[:, 0], columnas):
        datos = calculos[columna]
        y = a_numero(tabla[columna]).to_numpy()
        indices = muestra_indices(np.arange(len(y)), MAX_PUNTOS_LINEA)
        ax.plot(x[indices], y[indices], color=COLORES["senal"], lw=0.75,
                alpha=0.9, rasterized=True, label="Señal (submuestreada para gráfico)")
        if datos["estado_referencia"] == "APTO_EXPLORATORIO":
            mediana, escala = datos["mediana"], datos["escala"]
            ax.axhline(mediana, color=COLORES["mediana"], lw=1.1,
                       alpha=0.7, label="Mediana")
            for limite, color, estilo, nombre in (
                (LIMITE_ALERTA, COLORES["alerta"], "--", "Límite alerta ±"),
                (LIMITE_EXTREMO, COLORES["critico"], ":", "Límite extremo ±"),
            ):
                ax.axhline(mediana + limite * escala, color=color, lw=1.15,
                           linestyle=estilo, label=nombre)
                ax.axhline(mediana - limite * escala, color=color, lw=1.15,
                           linestyle=estilo)
            for clase, color, nombre in ((1, COLORES["alerta"], "Alerta"),
                                         (2, COLORES["critico"], "Crítico estadístico")):
                marcadores = muestra_indices(np.flatnonzero(datos["clases"] == clase),
                                             MAX_MARCADORES)
                if len(marcadores):
                    ax.scatter(x[marcadores], y[marcadores], c=color, s=10,
                               marker="o", zorder=3, label=nombre, rasterized=True)
        else:
            ax.text(0.02, 0.94, "Sin umbrales fiables: " + datos["observacion"],
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.set_ylabel(columna)
        ax.grid(color=COLORES["grid"], linewidth=0.5, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=7, ncol=2)
    axes[0, 0].set_title(f"{sensor.capitalize()} | Rangos estadísticos exploratorios",
                         weight="bold", pad=14)
    axes[-1, 0].set_xlabel(nombre_eje)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)


def guardar_csv(tabla: pd.DataFrame, destino: Path) -> None:
    # Compatible con Excel en español (separador ; y decimal ,).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tabla.to_csv(destino, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def episodios(estados: np.ndarray, x: np.ndarray, sensor: str) -> pd.DataFrame:
    activa = np.isin(estados, ("ALERTA", "CRITICO_ESTADISTICO"))
    comienzos = np.flatnonzero(activa & ~np.r_[False, activa[:-1]])
    finales = np.flatnonzero(activa & ~np.r_[activa[1:], False])
    filas = []
    for i, f in zip(comienzos, finales):
        filas.append({
            "sensor": sensor, "fila_inicio": int(i + 1), "fila_fin": int(f + 1),
            "registros_seguidos": int(f - i + 1), "indice_inicio": x[i], "indice_fin": x[f],
            "nivel_maximo": ("CRITICO_ESTADISTICO" if
                             "CRITICO_ESTADISTICO" in estados[i:f+1] else "ALERTA"),
            "tres_o_mas_filas": bool(f - i + 1 >= 3),
        })
    return pd.DataFrame(filas, columns=["sensor", "fila_inicio", "fila_fin",
                                       "registros_seguidos", "indice_inicio", "indice_fin",
                                       "nivel_maximo", "tres_o_mas_filas"])


def procesar(sensor: str, archivo: Path, carpeta_salida: Path) -> dict:
    print(f"\n{'=' * 70}\nProcesando {sensor.upper()}: {archivo.name}")
    tabla = leer_csv(archivo)
    if tabla.empty:
        raise ValueError("El archivo no tiene filas")
    columnas = [c for c in tabla.columns if es_columna_sensor(c, sensor)
                and a_numero(tabla[c]).notna().sum() >= 1]
    if not columnas:
        raise ValueError("No se identifican columnas del sensor. Columnas presentes: "
                         + ", ".join(tabla.columns))
    print("Columnas detectadas:", ", ".join(columnas))
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    graficos = carpeta_salida / "Graficos"
    graficos.mkdir(exist_ok=True)
    calculos = {}
    resumen_variables = []
    umbrales = []
    salida_datos = tabla.copy()
    clases_general = np.full(len(tabla), -1, dtype=np.int8)
    for columna in columnas:
        s = a_numero(tabla[columna])
        obj = calcular_columna(s)
        calculos[columna] = obj
        clases_general = np.maximum(clases_general, obj["clases"])
        estado = clasificaciones(obj["clases"])
        if obj["estado_referencia"] != "APTO_EXPLORATORIO":
            estado[s.notna().to_numpy()] = "SIN_REFERENCIA"
        salida_datos[f"estado__{columna}"] = estado
        salida_datos[f"zmod_abs__{columna}"] = obj["z"]
        n_evaluable = int((obj["clases"] >= 0).sum())
        n_alerta = int((obj["clases"] == 1).sum())
        n_critico = int((obj["clases"] == 2).sum())
        resumen_variables.append({
            "sensor": sensor, "variable": columna, "registros": len(tabla),
            "evaluables": n_evaluable, "normal": int((obj["clases"] == 0).sum()),
            "alerta": n_alerta, "critico_estadistico": n_critico,
            "porcentaje_alerta_evaluables": round(100*n_alerta/n_evaluable, 3)
            if n_evaluable else np.nan,
            "porcentaje_critico_evaluables": round(100*n_critico/n_evaluable, 3)
            if n_evaluable else np.nan,
            "referencia": obj["estado_referencia"],
        })
        centro, escala = obj["mediana"], obj["escala"]
        umbrales.append({
            "sensor": sensor, "variable": columna, "n_validos": obj["n"],
            "mediana": centro, "MAD": obj["mad"],
            "normal_inferior": centro - LIMITE_ALERTA * escala,
            "normal_superior": centro + LIMITE_ALERTA * escala,
            "extremo_inferior": centro - LIMITE_EXTREMO * escala,
            "extremo_superior": centro + LIMITE_EXTREMO * escala,
            "referencia": obj["estado_referencia"],
            "observacion": obj["observacion"],
        })
    estados = clasificaciones(clases_general)
    referencia_incompleta = any(
        calculos[c]["estado_referencia"] != "APTO_EXPLORATORIO" for c in columnas
    )
    if referencia_incompleta:
        estados[clases_general < 0] = "SIN_REFERENCIA"
    salida_datos["estado_sensor"] = estados
    salida_datos["cobertura_parcial"] = referencia_incompleta
    x, eje = obtener_eje(tabla)
    crear_grafico(sensor, columnas, tabla, calculos, x, eje,
                  graficos / f"{sensor}_rangos_estadisticos.png")
    guardar_csv(salida_datos, carpeta_salida / "datos_clasificados.csv")
    guardar_csv(pd.DataFrame(umbrales), carpeta_salida / "umbrales_por_variable.csv")
    guardar_csv(pd.DataFrame(resumen_variables), carpeta_salida / "resumen_por_variable.csv")
    guardar_csv(episodios(estados, x, sensor), carpeta_salida / "episodios_anomalias.csv")
    evaluables = int((clases_general >= 0).sum())
    alerta = int((clases_general == 1).sum())
    critico = int((clases_general == 2).sum())
    resumen = {
        "sensor": sensor, "archivo": archivo.name, "filas": len(tabla),
        "variables_detectadas": len(columnas), "filas_evaluables": evaluables,
        "normal": int((clases_general == 0).sum()), "alerta": alerta,
        "critico_estadistico": critico, "sin_referencia_o_dato": len(tabla) - evaluables,
        "porcentaje_alerta_evaluables": round(100*alerta/evaluables, 3)
        if evaluables else np.nan,
        "porcentaje_critico_evaluables": round(100*critico/evaluables, 3)
        if evaluables else np.nan,
        "cobertura_parcial": referencia_incompleta,
    }
    guardar_csv(pd.DataFrame([resumen]), carpeta_salida / "resumen_sensor.csv")
    print("Filas:", len(tabla), "| Evaluables:", evaluables,
          "| Alerta:", alerta, "| Crítico estadístico:", critico)
    if referencia_incompleta:
        print("[AVISO] Una o más columnas carecen de referencia MAD válida.")
    return resumen


METODOLOGIA = """TALLER IA 2 — RANGOS DE ANOMALÍAS EXPLORATORIOS

ENTRADAS: cuatro CSV de acelerómetro, giroscopio, magnetómetro y temperatura,
corregidos estructuralmente mediante el código 00 y anteriores a cualquier eliminación de atípicos por 6 sigma.
Los archivos originales NO se editan y las filas NO se emparejan entre sensores.

ESTADÍSTICA (por columna, usando sus propios datos):
    Mediana = mediana(x)
    MAD = mediana(|x - mediana(x)|)
    Z modificado = 0.6745*(x - mediana)/MAD

    NORMAL:                  |Z modificado| <= 3.5
    ALERTA:                   3.5 < |Z modificado| <= 6.0
    CRITICO_ESTADISTICO:      |Z modificado| > 6.0
    SIN_REFERENCIA:          MAD=0 o menos de 10 valores válidos.
    SIN_DATO:                dato numérico inexistente o no interpretable.

La regla de |Z| > 3.5 se utiliza para DETECTAR POSIBLES VALORES ATÍPICOS,
no para declarar fallas. Fuente: NIST Engineering Statistics Handbook,
'Outlier Detection', https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
El umbral |Z| > 6 y la clasificación en tres niveles son PROPUESTAS
EXPLORATORIAS del proyecto, NO límites físicos ni criterios de fabricante.
Los límites numéricos por variable se obtienen como mediana ± (3.5 o 6)*MAD/0.6745.

INTERPRETACIÓN:
- 'Crítico estadístico' no equivale a daño, riesgo de seguridad ni necesidad
  confirmada de mantenimiento.
- Los límites calculados reflejan esta base, que NO está certificada como
  datos de operación sana. Se requiere validar con inspecciones y datos de falla.
- La temperatura se registra en su propia escala; nunca se supone que todas
  las magnitudes miden vibración.
- SMP es un ÍNDICE, no minutos, segundos ni fechas. Los episodios consecutivos
  se cuentan por filas en el orden almacenado, no por tiempo transcurrido.
- Las cuatro series no se sincronizan, combinan ni correlacionan por SMP.
- Porcentajes se expresan sobre las filas EVALUABLES de cada sensor; cautela
  al comparar sensores con distinto número de registros o muestreo.
- Si una o más variables tienen MAD cero o pocos datos, la evaluación del
  sensor será incompleta (cobertura_parcial=True).
- Los gráficos submuestrean la línea y limitan marcadores para visualizar
  muchos datos sin perder los conteos completos en los CSV.
- Los resultados serán más fiables al separar regímenes de operación,
  verificar las unidades y crear una referencia de funcionamiento saludable.
"""


def principal() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza automáticamente los cuatro CSV separados por sensor.")
    parser.add_argument("--carpeta", help="Carpeta donde buscar los CSV (por defecto, junto al script)")
    parser.add_argument("--salida", help="Carpeta de resultados (opcional)")
    for sensor in ARCHIVOS:
        parser.add_argument(f"--{sensor}", help=f"Ruta alternativa del CSV de {sensor}")
    args = parser.parse_args()
    carpeta = (Path(args.carpeta).expanduser().resolve() if args.carpeta
               else Path(__file__).resolve().parent / "Datos_corregidos")
    if not carpeta.is_dir():
        sys.exit(f"No existe la carpeta de trabajo: {carpeta}")
    raiz_proyecto = Path(__file__).resolve().parent
    salida = (ruta_usuario(args.salida, raiz_proyecto) if args.salida
              else raiz_proyecto / "Resultados_rangos_sensores").resolve()
    hallados = {}
    for sensor, nombre in ARCHIVOS.items():
        manual = getattr(args, sensor)
        ruta = ruta_usuario(manual, carpeta) if manual else carpeta / nombre
        if ruta is None or not ruta.is_file():
            print(f"[FALTA] {nombre}  (sensor: {sensor})")
        else:
            hallados[sensor] = ruta.resolve()
    if not hallados:
        sys.exit("\nNo encontré ningún CSV esperado. Ejecuta el codigo 00 o proporciona --carpeta, "
                 "o ejecuta con --carpeta RUTA_DE_LA_CARPETA.")
    salida.mkdir(parents=True, exist_ok=True)
    resultados = []
    errores = []
    for sensor, archivo in hallados.items():
        try:
            resultados.append(procesar(sensor, archivo, salida / sensor))
        except Exception as error:
            errores.append((sensor, str(error)))
            print(f"[ERROR EN {sensor.upper()}] {error}")
    if resultados:
        guardar_csv(pd.DataFrame(resultados), salida / "resumen_general.csv")
        (salida / "metodologia_y_limitaciones.txt").write_text(METODOLOGIA,
                                                               encoding="utf-8")
    print("\n" + "=" * 70)
    print("ANÁLISIS FINALIZADO — archivos originales intactos")
    print("Procesados:", len(resultados), "de", len(ARCHIVOS), "sensores")
    print("Resultados:", salida)
    if errores:
        print("Errores:")
        for sensor, error in errores:
            print(f" - {sensor}: {error}")
    if len(resultados) != len(ARCHIVOS):
        sys.exit("ATENCIÓN: faltó procesar alguno de los cuatro sensores.")
    print("Recuerda: 'CRÍTICO' significa anomalía estadística, NO falla confirmada.")


if __name__ == "__main__":
    principal()
