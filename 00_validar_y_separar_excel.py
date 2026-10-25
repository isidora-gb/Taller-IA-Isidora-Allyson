# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv
import math
import re
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
COLUMNAS = [
    'AcelX1_SisGenEol', 'AcelY1_SisGenEol', 'AcelZ1_SisGenEol',
    'GiroX_SisGenEol', 'GiroY_SisGenEol', 'GiroZ_SisGenEol',
    'MagneX_SisGenEol', 'MagneY_SisGenEol', 'MagneZ_SisGenEol',
    'Ter_SisGenEol', 'smp',
]
SENSORES = {
    '01_acelerometro_smp_ordenado.csv': COLUMNAS[:3],
    '02_giroscopio_smp_ordenado.csv': COLUMNAS[3:6],
    '03_magnetometro_smp_ordenado.csv': COLUMNAS[6:9],
    '04_temperatura_smp_ordenado.csv': COLUMNAS[9:10],
}
BASE = ['smp_original', 'smp_continuo', 'smp_desde_cero']


def indice_columna(referencia: str) -> int:
    m = re.fullmatch(r'([A-Z]+)\d+', referencia)
    if not m:
        raise ValueError(f'Celda sin referencia esperada: {referencia!r}')
    pos = 0
    for ch in m.group(1):
        pos = pos * 26 + ord(ch) - ord('A') + 1
    return pos


def texto_celda(celda: ET.Element, compartidas: list[str]) -> str:
    tipo = celda.get('t')
    valor = celda.findtext(NS + 'v')
    if tipo == 's' and valor is not None:
        return compartidas[int(valor)]
    if tipo == 'inlineStr':
        return ''.join(n.text or '' for n in celda.iter(NS + 't'))
    return '' if valor is None else valor


def filas_de_excel(ruta: Path):
    """Iteración en streaming sobre un XLSX de gran tamaño (sin carga total)."""
    with ZipFile(ruta) as z:
        compartidas = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as archivo:
                arbol = ET.parse(archivo).getroot()
            compartidas = [
                ''.join(t.text or '' for t in si.iter(NS + 't'))
                for si in arbol.findall(NS + 'si')
            ]
        # Archivo proporcionado: una sola hoja, localizada en sheet1.xml.
        hoja = 'xl/worksheets/sheet1.xml'
        if hoja not in z.namelist():
            raise ValueError('No encontré la hoja 1 dentro del XLSX.')
        with z.open(hoja) as f:
            contexto = ET.iterparse(f, events=('start', 'end'))
            _, raiz = next(contexto)
            for evento, elemento in contexto:
                if evento != 'end' or elemento.tag != NS + 'row':
                    continue
                numero = int(elemento.get('r'))
                celdas = {
                    indice_columna(c.get('r')): texto_celda(c, compartidas)
                    for c in elemento.findall(NS + 'c')
                }
                yield numero, celdas
                elemento.clear()
                # Evita acumular 143 mil objetos XML en memoria.
                raiz.clear()


def numero_validado(valor: str) -> bool:
    try:
        return math.isfinite(float(valor))
    except (TypeError, ValueError):
        return False


def numero_para_csv(valor: str) -> str:
    """CSV adaptado a Excel en español: ; separa, , es decimal."""
    return valor.replace('.', ',')


def formato_fila_excel(celdas: dict[int, str]) -> str:
    return ' | '.join(f'{i}:{v}' for i, v in sorted(celdas.items()))


def filas_con_marca_de_ultima(ruta: Path):
    """Devuelve (número fila, celdas, es_última) sin cargar todo el XLSX."""
    anterior = None
    for actual in filas_de_excel(ruta):
        if anterior is not None:
            yield anterior[0], anterior[1], False
        anterior = actual
    if anterior is not None:
        yield anterior[0], anterior[1], True


def promedio_dos_anteriores(registros: deque, pos: int) -> str:
    """Promedio verificable de dos valores observados previos (NO interpolación)."""
    if len(registros) != 2:
        raise ValueError('No hay dos registros completos previos para la imputación final.')
    a, b = Decimal(registros[0][pos]), Decimal(registros[1][pos])
    if not a.is_finite() or not b.is_finite():
        raise ValueError('Los dos vecinos anteriores no tienen valores finitos.')
    return str((a + b) / Decimal(2))


def procesar(entrada: Path, salida: Path) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    cifras = Counter()
    ubicaciones_anormales = []
    smp_anterior = None
    smp_inicial = None
    vueltas = 0
    anterior_continuo = None
    saltos = Counter()
    ids_vistos = set()  # pares (ciclo, SMP), valida no duplicidad
    registros_anteriores = deque(maxlen=2)
    archivos = {}
    try:
        maestro = salida / 'datos_multisensor_corregidos.csv'
        archivos['maestro'] = maestro.open('w', encoding='utf-8-sig', newline='')
        escritores = {}
        for nombre, columnas in SENSORES.items():
            archivos[nombre] = (salida / nombre).open('w', encoding='utf-8-sig', newline='')
            escritores[nombre] = csv.writer(archivos[nombre], delimiter=';')
            escritores[nombre].writerow(BASE + columnas + ['fila_excel', 'smp_estimado', 'medicion_imputada'])
        maestro_escritor = csv.writer(archivos['maestro'], delimiter=';')
        maestro_escritor.writerow(BASE + COLUMNAS[:-1] + ['fila_excel', 'smp_estimado', 'medicion_imputada'])
        archivos['auditoria'] = (salida / 'auditoria_filas.csv').open('w', encoding='utf-8-sig', newline='')
        auditoria = csv.writer(archivos['auditoria'], delimiter=';')
        auditoria.writerow(['fila_excel', 'diagnostico', 'columnas_presentes',
                            'smp_conservado', 'fragmentos_sin_smp', 'detalle'])
        archivos['huerfanos'] = (salida / 'fragmentos_no_recuperables.csv').open('w', encoding='utf-8-sig', newline='')
        huerfanos = csv.writer(archivos['huerfanos'], delimiter=';')
        huerfanos.writerow(['fila_excel', 'cantidad_celdas', 'razon', 'celdas_originales'])
        archivos['imputaciones'] = (salida / 'registro_imputaciones.csv').open('w', encoding='utf-8-sig', newline='')
        imputaciones = csv.writer(archivos['imputaciones'], delimiter=';')
        imputaciones.writerow(['fila_excel', 'campo', 'valor_asignado', 'metodo',
                              'filas_fuente', 'observacion'])
        for fila, celdas, es_ultima in filas_con_marca_de_ultima(entrada):
            if fila == 1:
                nombres = [celdas.get(k, '') for k in range(1, 12)]
                if nombres != COLUMNAS:
                    raise ValueError(f'Encabezado del Excel distinto al esperado: {nombres}')
                continue
            cifras['filas_de_datos_excel'] += 1
            columnas = sorted(celdas)
            n = len(columnas)
            # Un bloque válido debe terminar en un SMP entero y tener 10 señales.
            # Las 11 celdas finales DEBEN ser contiguas y numéricas.
            ultimas = columnas[-11:] if n >= 11 else []
            valido = (len(ultimas) == 11 and
                      ultimas == list(range(ultimas[0], ultimas[0] + 11)) and
                      all(numero_validado(celdas[k]) for k in ultimas))
            smp = None
            if valido:
                s_float = float(celdas[ultimas[-1]])
                valido = s_float.is_integer() and 0 <= s_float <= 65535
                if valido: smp = int(s_float)
            recuperada_terminal = False
            if (not valido and es_ultima and columnas == list(range(1, 9))
                    and all(numero_validado(celdas[k]) for k in range(1, 9))
                    and len(registros_anteriores) == 2 and smp_anterior is not None):
                # SOLO para la última fila de este Excel, sin vecinos posteriores.
                magnez = promedio_dos_anteriores(registros_anteriores, 8)
                temperatura = promedio_dos_anteriores(registros_anteriores, 9)
                registro = [celdas[k] for k in range(1, 9)] + [magnez, temperatura]
                smp = (smp_anterior + 1) % 65536  # Estimación de índice, NO valor original
                valido = True
                recuperada_terminal = True
                cifras['filas_finales_imputadas'] += 1
                for campo, valor, col in [('MagneZ_SisGenEol', magnez, 8),
                                          ('Ter_SisGenEol', temperatura, 9)]:
                    imputaciones.writerow([fila, campo, numero_para_csv(valor),
                        'Media de dos últimas muestras previas del mismo campo',
                        f'{fila - 2} y {fila - 1}',
                        'Estimado; no es una medición observada'])
                imputaciones.writerow([fila, 'smp_continuo', str(smp + vueltas * 65536),
                    'Avance unitario estimado desde SMP observado anterior',
                    f'{fila - 1}',
                    f'SMP original ausente; se estima el siguiente contador módulo 65536: {smp}'])
            if not valido:
                cifras['filas_incompletas_o_invalidas'] += 1
                huerfanos.writerow([fila, n, 'No hay registro completo de 11 valores con SMP',
                                    formato_fila_excel(celdas)])
                auditoria.writerow([fila, 'INCOMPLETA_EXCLUIDA', n, '', n,
                                    'Conservada en fragmentos_no_recuperables.csv'])
                ubicaciones_anormales.append(fila)
                continue
            if not recuperada_terminal:
                registro = [celdas[k] for k in ultimas[:-1]]
            if smp_anterior is not None and smp < smp_anterior:
                if smp_anterior >= 65000 and smp < 1000:
                    vueltas += 1
                    cifras['reinicios_smp'] += 1
                else:
                    cifras['decrementos_smp_inesperados'] += 1
            continuo = smp + vueltas * 65536
            if smp_inicial is None:
                smp_inicial = continuo
            if anterior_continuo is not None:
                salto = continuo - anterior_continuo
                saltos[salto] += 1
                if salto > 1:
                    cifras['muestras_smp_ausentes'] += salto - 1
                if salto <= 0:
                    cifras['smp_no_creciente'] += 1
            if continuo in ids_vistos:
                cifras['smp_duplicados'] += 1
            ids_vistos.add(continuo)
            anterior_continuo = continuo
            smp_anterior = smp
            base = ['' if recuperada_terminal else smp, continuo, continuo - smp_inicial]
            datos = [numero_para_csv(v) for v in registro]
            por_nombre = dict(zip(COLUMNAS[:10], datos))
            maestro_escritor.writerow(base + datos + [fila, recuperada_terminal, recuperada_terminal])
            for nombre, cols_sensor in SENSORES.items():
                hay_medicion_imputada = recuperada_terminal and nombre in (
                    '03_magnetometro_smp_ordenado.csv', '04_temperatura_smp_ordenado.csv')
                escritores[nombre].writerow(base + [por_nombre[c] for c in cols_sensor]
                                             + [fila, recuperada_terminal, hay_medicion_imputada])
            cifras['registros_completos_conservados'] += 1
            registros_anteriores.append(registro)
            sobrantes = columnas[:-11] if not recuperada_terminal else []
            if recuperada_terminal:
                auditoria.writerow([fila, 'ULTIMA_FILA_IMPUTADA', n, '', 3,
                    f'Conservadas A:H; MagneZ={magnez}; Temperatura={temperatura}; '
                    f'SMP reconstruido={smp}, smp_original vacío'])
                ubicaciones_anormales.append(fila)
            elif sobrantes:
                cifras['filas_con_fragmento'] += 1
                cifras['valores_huerfanos'] += len(sobrantes)
                detalles = {k: celdas[k] for k in sobrantes}
                huerfanos.writerow([fila, len(sobrantes),
                                    'Fragmento anterior a registro completo; sin SMP',
                                    formato_fila_excel(detalles)])
                auditoria.writerow([fila, 'REGISTRO_RECUPERADO_AL_FINAL', n, smp,
                                    len(sobrantes), 'Se conservó el bloque final de 11 celdas'])
                ubicaciones_anormales.append(fila)
            else:
                cifras['filas_11_columnas_correctas'] += 1
    finally:
        for file in archivos.values(): file.close()
    cifras['reinicios_smp'] = vueltas
    texto = f'''TALLER IA 2 - AUDITORIA DE ORIGEN Y SEPARACION DE VARIABLES

Origen: {entrada.name}
Filas de datos del Excel: {cifras['filas_de_datos_excel']}
Filas de 11 columnas correctas: {cifras['filas_11_columnas_correctas']}
Filas con fragmentos concatenados: {cifras['filas_con_fragmento']}
Filas incompletas definitivamente excluidas: {cifras['filas_incompletas_o_invalidas']}
Última fila conservada con imputación: {cifras['filas_finales_imputadas']}
Registros conservados para el análisis (incluye imputada): {cifras['registros_completos_conservados']}
Fragmentos sin identificador SMP: {cifras['filas_con_fragmento']} (y un SMP final no observado, estimado)
Reinicios de SMP detectados (contador 16 bits): {cifras['reinicios_smp']}
Muestras SMP no representadas por un registro completo: {cifras['muestras_smp_ausentes']}
Duplicados SMP continuo: {cifras['smp_duplicados']}
Orden SMP no creciente: {cifras['smp_no_creciente']}
Distribucion de los saltos del indice SMP: {dict(sorted(saltos.items()))}
Primer SMP continuo: {smp_inicial}
Ultimo SMP continuo: {anterior_continuo}

PROCEDIMIENTO:
- En las filas con mas de 11 valores, se recupera SOLO el bloque completo
  de 10 variables + SMP ubicado al FINAL de la fila de Excel.
- El fragmento adicional a su izquierda carece de SMP; no se le asigna uno.
- Los fragmentos descartados de la tabla principal NO se destruyen: quedan
  documentados en fragmentos_no_recuperables.csv.
- La última fila de 8 celdas queda DENTRO de los CSV analíticos.
- Sus 8 señales reales no se modifican. Solo MagneZ y temperatura se estiman
  con la MEDIA DE LAS DOS MEDICIONES ANTERIORES de cada variable (sin vecino futuro).
- smp_original en esa fila queda vacío, porque no fue observado; se estima
  smp_continuo dando el siguiente paso (+1) sobre el contador anterior.
- `registro_imputaciones.csv` guarda valores/método; `smp_estimado` y
  `medicion_imputada` marcan el último registro en los CSV por sensor.
- El indice SMP se conserva y se desenrolla tras sus reinicios 65535 -> 0.
- fila_excel permite rastrear cada registro de salida a la fuente original.
- Solo se imputan 2 celdas de medición y 1 contador de la última fila, con auditoría.
- No se eliminan valores por criterios estadísticos.
- El eje SMP NO es tiempo real: sin frecuencia de muestreo no se estiman horas.
- No se supone ninguna unidad fisica no documentada para los sensores.

SIGUIENTE PASO:
Ejecutar los analisis 01, 02, 03, 04 y 05 exclusivamente sobre los CSV de ESTA carpeta.
Los resultados previos basados en filas desplazadas quedan invalidados.
'''
    (salida / 'informe_validacion.txt').write_text(texto, encoding='utf-8')
    print(texto)
    print('Carpeta:', salida.resolve())
    return dict(cifras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, default=Path(__file__).resolve().parent / 'CF-2023-I-204-1230_DAT.xlsx')
    parser.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'Datos_corregidos')
    args = parser.parse_args()
    if not args.entrada.is_file():
        parser.error(f'Archivo no encontrado: {args.entrada}')
    procesar(args.entrada, args.salida)


if __name__ == '__main__':
    main()
