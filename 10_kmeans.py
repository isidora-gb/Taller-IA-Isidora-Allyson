
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_samples, silhouette_score

RAIZ = Path(__file__).resolve().parent
SMP = "smp_continuo"
PARTES = ("entrenamiento", "validacion", "prueba")
FAMILIAS = {"todas": (list(range(10)), "Las 10 variables juntas")}
CORTOS = ["Ac X", "Ac Y", "Ac Z", "Gi X", "Gi Y", "Gi Z",
          "Ma X", "Ma Y", "Ma Z", "Temp"]
COLORES_PARTES = {"entrenamiento": "#3278a7", "validacion": "#d18a44",
                  "prueba": "#835bb1"}
SEED = 42


def guardar_csv(tabla, ruta):
    pd.DataFrame(tabla).to_csv(ruta, index=False, sep=";", encoding="utf-8-sig",
                               float_format="%.10g")


def guardar_figura(fig, ruta):
    fig.savefig(ruta, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def leer_08(preparados: Path, escalados: list[str]):
    """Comprueba exactamente la salida del 08 actual, no la del 08 antiguo."""
    meta = preparados / "00_metadatos.json"
    if not meta.is_file():
        raise FileNotFoundError(f"Falta {meta}. Ejecuta: python 08_preparacion_datos.py")
    info = json.loads(meta.read_text(encoding="utf-8"))
    nombres = list(info.get("variables", []))
    if len(nombres) != 10:
        raise ValueError("Los metadatos del 08 deben incluir exactamente 10 sensores.")
    particion = pd.read_csv(preparados / "00_division_cronologica.csv", sep=";",
                            usecols=[SMP, "conjunto"])
    if particion[SMP].duplicated().any() or not particion[SMP].is_monotonic_increasing:
        raise ValueError("La división 08 tiene SMP repetidos o desordenados.")
    conjuntos = {}
    for escalado in escalados:
        conjuntos[escalado] = {}
        for parte in PARTES:
            ruta = preparados / f"01_{parte}_{escalado}.csv"
            if not ruta.is_file():
                raise FileNotFoundError(f"Falta {ruta}; ejecuta el 08 ACTUAL.")
            t = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
            cols = ["z_" + x for x in nombres]
            if list(t.columns) != [SMP, *cols]:
                raise ValueError(f"El 08 dejó columnas incompatibles en {ruta.name}")
            ref = particion.loc[particion.conjunto == parte, SMP].to_numpy(dtype=np.int64)
            if not np.array_equal(ref, t[SMP].to_numpy(dtype=np.int64)):
                raise ValueError(f"El SMP no coincide entre división y {ruta.name}")
            x = t[cols].to_numpy(dtype=np.float64, copy=True)
            if not np.isfinite(x).all():
                raise ValueError(f"Hay valores no finitos en {ruta.name}")
            conjuntos[escalado][parte] = (ref, x)
    if len(particion) != info.get("registros") or len(particion) != 143561:
        # Si cambia la fuente real, detenerse en vez de reciclar resultados.
        raise ValueError("La base no coincide con la fuente de 143.561 registros acordada.")
    return info, nombres, conjuntos


def opciones_modelo(k, n_init, seed, max_iter=300):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                  max_iter=max_iter, random_state=seed, algorithm="lloyd")


def dispersión_modelo(modelo, x):
    """Suma de cuadrados a centroides asignados (no es tasa de error)."""
    etiquetas = modelo.predict(x)
    d = x - modelo.cluster_centers_[etiquetas]
    return float(np.einsum("ij,ij->", d, d, optimize=True)), etiquetas


def colores_clusters(k):
    """Colores contrastados: G0 azul, G1 naranja (evita dos azules claros)."""
    if k <= 10:
        paleta = colormaps["tab10"]
        return [paleta(i) for i in range(k)]
    if k <= 20:
        paleta = colormaps["tab20"]
        orden = list(range(0, 20, 2)) + list(range(1, 20, 2))
        return [paleta(orden[i]) for i in range(k)]
    paleta = colormaps["gist_ncar"]
    return [paleta(i / max(1, k - 1)) for i in range(k)]


def crear_curvas(tabla, prefijo, destino, titulo):
    t = tabla.sort_values("K")
    elegido = t.loc[t["seleccionado"]].iloc[0]
    k_best = int(elegido.K)
    specs = [
        ("01_error_porcentual_vs_k", "Error de agrupamiento (%)",
         "K-Means | Error (%) frente a K", "dispersion_residual_pct", "#b67540"),
        ("02_codo_inercia", "Inercia SSE (entrenamiento)",
         "K-Means | Método del codo", "inercia_entrenamiento", "#2b739f"),
        ("03_silueta_vs_k", "Coeficiente de silueta (muestra de validación)",
         "K-Means | Coeficiente de silueta frente a K", "silueta_validacion_muestra", "#875ba3"),
    ]
    for nombre, ylabel, subt, columna, color in specs:
        fig, ax = plt.subplots(figsize=(9.0, 5.1))
        ax.plot(t.K, t[columna], marker="o", ms=3.1, color=color, lw=1.65)
        ax.axvline(k_best, color="#ad4552", ls="--", lw=1.15,
                   label=f"K seleccionado = {k_best}")
        ax.set(xlabel="K (número de grupos)", ylabel=ylabel,
               title=f"{subt}\n{titulo}")
        ax.set_xlim(1, max(2, int(t.K.max())))
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="best", frameon=False, fontsize=8)
        if columna == "dispersion_residual_pct":
            ax.set_ylim(bottom=0, top=max(102, t[columna].max()*1.05))
            ax.text(.5, -.20,
                    "Error (%) = 100 × inercia(K) / inercia(K=1). "
                    "No es porcentaje de fallas ni de clasificaciones incorrectas.",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8,
                    color="#555")
            fig.subplots_adjust(bottom=.26)
        if columna == "silueta_validacion_muestra":
            ax.axhline(0, color="gray", lw=.7)
        guardar_figura(fig, destino / f"{nombre}_{prefijo}.png")

def siluetas_por_grupo(X_sample, labels_sample, elegido, prefijo, destino, titulo):
    grupos = np.unique(labels_sample)
    if len(grupos) < 2 or len(grupos) >= len(X_sample):
        return
    sil = silhouette_samples(X_sample, labels_sample)
    promedio = float(np.mean(sil))
    colores = colores_clusters(int(np.max(labels_sample)) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    actual = 5
    for g in grupos:
        vals = np.sort(sil[labels_sample == g]); n = len(vals)
        ax.fill_betweenx(np.arange(actual, actual+n), 0, vals,
                         color=colores[int(g)], alpha=.77)
        ax.text(-.08, actual+n/2, f"G{g} (n={n})", fontsize=8, va="center")
        actual += n + 7
    ax.axvline(promedio, color="#bf4255", lw=1.4, ls="--",
               label=f"Silueta media = {promedio:.3f}")
    ax.axvline(0, lw=.7, color="#444")
    ax.set(xlim=(-.5, 1), xlabel="Silueta por observación",
           ylabel="Registros de validación incluidos en la muestra",
           title=f"K-Means | Siluetas individuales (K={elegido})\n{titulo}")
    ax.set_yticks([]); ax.legend(frameon=False, fontsize=8)
    guardar_figura(fig, destino / f"04_silueta_por_grupo_{prefijo}.png")


def limites_plot(puntos_dict):
    todos = np.concatenate(list(puntos_dict.values()))
    lo = np.min(todos, axis=0); hi = np.max(todos, axis=0)
    centro = np.median(todos, axis=0)
    q0, q1 = np.quantile(todos, [0.005, 0.995], axis=0)
    def extender(a, b):
        r = np.maximum(b-a, 1e-5)
        return np.column_stack([a-.055*r, b+.055*r])
    completo = extender(lo,hi)
    zoom = extender(q0,q1)
    return completo, zoom, centro


def pca_y_visualizacion(tablas: dict, indices: list[int], prefijo: str,
                        titulo: str, destino: Path, labels_por_parte: dict,
                        k: int, max_legenda: int=12):
    Xtr = tablas["entrenamiento"][1][:,indices]
    Xv = {p: tablas[p][1][:,indices] for p in PARTES}
    componentes = len(indices)
    if componentes >= 2:
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(Xtr)  # jamás sobre validación o prueba
        planos = {p: pca.transform(Xv[p]) for p in PARTES}
        xlabel = f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f} % var.)"
        ylabel = f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f} % var.)"
        nota = "PCA ajustado solo en entrenamiento; K-Means agrupó en variables originales escaladas."
        var_2d = float(pca.explained_variance_ratio_.sum())
    else:
        # Graficar un eje físico, con jitter ÚNICAMENTE visual.
        planos = {}
        for p in PARTES:
            x = Xv[p][:,0]
            seed = SEED + PARTES.index(p)
            jitter = np.random.default_rng(seed).uniform(-.28,.28,len(x))
            planos[p] = np.column_stack([x, jitter])
        xlabel = "Temperatura escalada (una variable)"
        ylabel = "Dispersión vertical artificial (solo visual)"
        nota = "El desplazamiento vertical es artificial y no se usó para agrupar."
        var_2d = 1.0
    completos, zoom, centro = limites_plot(planos)
    colores = colores_clusters(k)
    for p in PARTES:
        z = planos[p]
        grupos = labels_por_parte[p]
        for encuadre, limites in (("completo",completos),("zoom",zoom)):
            fig,ax = plt.subplots(figsize=(10.1, 6.0))
            for g in range(k):
                mask = grupos == g
                if not np.any(mask):
                    continue
                ax.scatter(z[mask,0], z[mask,1], s=1.2 if p=="entrenamiento" else 3,
                           color=colores[g], alpha=.38 if p=="entrenamiento" else .53,
                           linewidths=0, rasterized=True,
                           label=f"G{g}: {int(mask.sum()):,}" if g<max_legenda else None)
            ax.set(xlabel=xlabel, ylabel=ylabel,
                   title=f"K-Means | {titulo}\n{p.capitalize()}: {len(grupos):,} registros | K={k} | {encuadre.upper()}")
            ax.set_xlim(limites[0,0],limites[0,1]);ax.set_ylim(limites[1,0],limites[1,1])
            ax.grid(alpha=.12)
            leg = ax.legend(fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01,1),
                            borderaxespad=0, frameon=False, markerscale=3)
            if k > max_legenda:
                ax.text(1.01,.08,f"+ {k-max_legenda} grupos\n(no listados)",
                        transform=ax.transAxes, va="bottom", fontsize=8)
            ax.text(0, -.13, f"{nota} " + (
                "Zoom: algunos puntos quedan fuera del encuadre, ninguno se elimina del cálculo."
                if encuadre=="zoom" else "Todos los puntos representados."),
                transform=ax.transAxes, fontsize=7.4, color="#555")
            guardar_figura(fig,destino/f"05_grupos_{p}_{encuadre}_{prefijo}.png")
    return var_2d


def graficos_adicionales(tablas:dict, indices, labels:dict, modelo, prefijo:str,
                         titulo:str, destino:Path):
    k = modelo.n_clusters
    cuenta=[]
    for p in PARTES:
        counts=np.bincount(labels[p],minlength=k)
        for g in range(k):
            cuenta.append({"conjunto":p,"grupo":g,"registros":int(counts[g]),
                          "porcentaje":100*counts[g]/len(labels[p])})
    counts_df = pd.DataFrame(cuenta)
    guardar_csv(counts_df, destino/f"02_tamanos_grupos_{prefijo}.csv")
    fig,ax=plt.subplots(figsize=(9.5,5.0))
    x=np.arange(3); fondo=np.zeros(3); cols=colores_clusters(k)
    for g in range(k):
        d=counts_df[counts_df.grupo==g].set_index("conjunto")
        valores=d.loc[list(PARTES),"porcentaje"].to_numpy()
        ax.bar(x,valores,bottom=fondo,color=cols[g],label=f"G{g}" if g<12 else None)
        fondo+=valores
    ax.set(xticks=x,xticklabels=[p.capitalize() for p in PARTES],
           ylabel="Distribución de asignaciones (%)",ylim=(0,100),
           title=f"K-Means | Participación de cada grupo\n{titulo}")
    ax.legend(frameon=False,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    guardar_figura(fig,destino/f"06_distribucion_grupos_{prefijo}.png")

    # Evolución a lo largo de SMP; todos los puntos, no segundos.
    fig,axs=plt.subplots(3,1,figsize=(13,7.4),sharey=True)
    for ax,p in zip(axs,PARTES):
        smp=tablas[p][0]
        ax.scatter(smp,labels[p],s=.35 if p=="entrenamiento" else 1.0,
                   c=np.array(cols)[labels[p]],alpha=.42,rasterized=True,lw=0)
        ax.set_ylabel(p.capitalize()+"\nGrupo")
        ax.grid(axis="y",alpha=.15)
    axs[-1].set_xlabel("SMP continuo (índice de adquisición; NO segundos)")
    axs[0].set_title(f"K-Means | Asignaciones por SMP\n{titulo}")
    fig.tight_layout()
    guardar_figura(fig,destino/f"07_evolucion_clusters_smp_{prefijo}.png")

    # Perfil de centroides en escalamiento de entrada (no diagnósticos físicos).
    fig,ax=plt.subplots(figsize=(max(7.8,.52*k+3), max(3.8,.4*len(indices)+2)))
    centros=modelo.cluster_centers_.T
    vmax=max(1.,float(np.quantile(np.abs(centros),.99)))
    im=ax.imshow(centros,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set(xticks=np.arange(k),xticklabels=[f"G{i}" for i in range(k)],
           yticks=np.arange(len(indices)),yticklabels=[CORTOS[i] for i in indices],
           xlabel="Grupo",title=f"K-Means | Perfiles de centroides (escala transformada)\n{titulo}")
    if k>16: ax.tick_params(axis="x",labelrotation=90,labelsize=7)
    fig.colorbar(im,ax=ax,label="Valor del centroide en la escala utilizada")
    fig.tight_layout()
    guardar_figura(fig,destino/f"08_perfil_centroides_{prefijo}.png")


def analizar_escenario(tablas:dict, nombre:str,indices:list[int],escala:str,
                       argumentos,destino:Path):
    prefijo=f"{nombre}_{escala}"; titulo=f"{FAMILIAS[nombre][1]} | {escala.capitalize()}"
    Xtr=tablas["entrenamiento"][1][:,indices]
    Xval=tablas["validacion"][1][:,indices]
    # Evita K superiores a la cantidad de vectores distintos (temperatura discreta).
    distintos=len(np.unique(Xtr, axis=0))
    Kmax=min(argumentos.max_k, distintos)
    if Kmax<2:
        print(f"[OMITIDO] {titulo}: solo {distintos} patrones distintos",flush=True)
        return {"escenario":prefijo,"estado":"sin K valido","patrones_distintos":distintos}
    # Misma muestra de VALIDACIÓN para todos los K, reproducible y sin tocar PRUEBA.
    rng=np.random.default_rng(SEED)
    ids=np.sort(rng.choice(len(Xval),size=min(argumentos.muestra_silueta,len(Xval)),replace=False))
    print(f"\n[{prefijo}] train={len(Xtr):,} | validación={len(Xval):,} | "
          f"K=1..{Kmax}, muestra silueta={len(ids):,}",flush=True)
    baseline=opciones_modelo(1,argumentos.n_init_barrido,SEED,
                           argumentos.max_iter_barrido).fit(Xtr)
    sse0=float(baseline.inertia_)
    sse0_val, _ = dispersión_modelo(baseline, Xval)
    modelos = {}
    filas = [{"K": 1, "inercia_entrenamiento": sse0,
              "dispersion_residual_pct": 100.0, "reduccion_dispersion_pct": 0.0,
              "SSE_validacion": sse0_val, "SSE_validacion_por_registro": sse0_val / len(Xval),
              "silueta_validacion_muestra": np.nan,
              "grupos_en_validacion": 1, "grupo_menor_validacion": len(Xval),
              "grupo_mayor_validacion": len(Xval), "grupos_en_muestra_silueta": 1,
              "entrenamiento_total": len(Xtr), "validacion_total": len(Xval),
              "filas_silueta": len(ids), "aviso_convergencia": False,
              "escenario": nombre, "escalado": escala}]
    for k in range(2,Kmax+1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",ConvergenceWarning)
            model=opciones_modelo(k,argumentos.n_init_barrido,SEED,
                                 argumentos.max_iter_barrido).fit(Xtr)
        sse_val,etq_val=dispersión_modelo(model,Xval)
        etq_sample=etq_val[ids]
        grupos_sample=len(np.unique(etq_sample))
        sil=(float(silhouette_score(Xval[ids],etq_sample))
             if 2<=grupos_sample<len(ids) else float("nan"))
        valores=(np.bincount(etq_val,minlength=k))
        filas.append({"K":k,"inercia_entrenamiento":float(model.inertia_),
                      "dispersion_residual_pct":100*model.inertia_/sse0 if sse0>0 else np.nan,
                      "reduccion_dispersion_pct":100*(1-model.inertia_/sse0) if sse0>0 else np.nan,
                      "SSE_validacion":sse_val,"SSE_validacion_por_registro":sse_val/len(Xval),
                      "silueta_validacion_muestra":sil,
                      "grupos_en_validacion":int(np.count_nonzero(valores)),
                      "grupo_menor_validacion":int(valores.min()),
                      "grupo_mayor_validacion":int(valores.max()),
                      "grupos_en_muestra_silueta":grupos_sample,
                      "entrenamiento_total":len(Xtr),"validacion_total":len(Xval),
                      "filas_silueta":len(ids),"aviso_convergencia":bool(caught),
                      "escenario":nombre,"escalado":escala})
        modelos[k]=model
        if (k==2 or k%5==0 or k==Kmax):
            print(f"  K={k:2d}: silueta={sil:.4f}, dispersión residual={100*model.inertia_/sse0:.2f}%",flush=True)
    tabla=pd.DataFrame(filas)
    seleccion=tabla.dropna(subset=["silueta_validacion_muestra"])
    if seleccion.empty:
        raise ValueError(f"No hay siluetas válidas para {prefijo}; revisar datos")
    ordenado=seleccion.sort_values(["silueta_validacion_muestra","K"],ascending=[False,True])
    # El barrido usa un número pequeño de inicializaciones para economizar
    # cálculo SIN recortar observaciones. Reentrenar los primeros candidatos
    # con MÁS inicializaciones y 300 iteraciones, luego revalidar silueta.
    finalistas=[]
    for fila in ordenado.head(argumentos.finalistas).itertuples():
        k_c=int(fila.K)
        mod=opciones_modelo(k_c,argumentos.n_init_final,SEED,max_iter=300).fit(Xtr)
        labs=mod.predict(Xval[ids])
        s=(float(silhouette_score(Xval[ids],labs))
           if 2<=len(np.unique(labs))<len(labs) else float("nan"))
        finalistas.append({"K":k_c,"silueta_preliminar":float(fila.silueta_validacion_muestra),
                           "silueta_refinada":s,"inercia_train_refinada":float(mod.inertia_),
                           "modelo":mod})
    elegibles=[r for r in finalistas if np.isfinite(r["silueta_refinada"])]
    if not elegibles:
        raise ValueError(f"Ningún K candidato fue válido tras reajuste ({prefijo})")
    seleccionado=sorted(elegibles,key=lambda d:(-d["silueta_refinada"],d["K"]))[0]
    mejor_k=int(seleccionado["K"])
    final=seleccionado["modelo"]
    nueva_silueta=float(seleccionado["silueta_refinada"])
    guardar_csv([{k:v for k,v in f.items() if k!="modelo"}
                 for f in finalistas],destino/f"00_finalistas_refinados_{prefijo}.csv")
    etiquetas={p:final.predict(tablas[p][1][:,indices]) for p in PARTES}
    idsval=etiquetas["validacion"][ids]
    tabla["seleccionado"]=(tabla.K==mejor_k)
    guardar_csv(tabla,destino/f"00_barrido_k_{prefijo}.csv")
    crear_curvas(tabla,prefijo,destino,titulo)
    siluetas_por_grupo(Xval[ids],idsval,mejor_k,prefijo,destino,titulo)
    var_2d=pca_y_visualizacion(tablas,indices,prefijo,titulo,destino,etiquetas,mejor_k)
    graficos_adicionales(tablas,indices,etiquetas,final,prefijo,titulo,destino)
    unidas=[]
    for p in PARTES:
        smp=tablas[p][0]; z=tablas[p][1][:,indices]
        d=z-final.cluster_centers_[etiquetas[p]]
        dist=np.sqrt(np.einsum("ij,ij->i",d,d,optimize=True))
        unidas.append(pd.DataFrame({SMP:smp,"conjunto":p,"cluster":etiquetas[p],
                                    "distancia_centroide":dist}))
    guardar_csv(pd.concat(unidas,ignore_index=True),destino/f"01_grupos_todos_registros_{prefijo}.csv")
    joblib.dump({"modelo":final,"indices":indices,"variables":[CORTOS[i] for i in indices],
                 "escalado":escala,"mejor_k":mejor_k},destino/f"00_modelo_{prefijo}.joblib")
    resumen={"escenario":prefijo,"familia":nombre,"escalado":escala,
            "patrones_distintos_train":distintos,"mejor_k":mejor_k,
            "silueta_barrido_k_elegido":float(tabla.loc[tabla.K==mejor_k,"silueta_validacion_muestra"].iloc[0]),
            "silueta_final_validacion_muestra":nueva_silueta,
            "varianza_pca_2d_pct":100*var_2d,
            "n_entrenamiento":len(Xtr),"n_validacion":len(Xval),
            "n_prueba":len(tablas["prueba"][0]),"n_muestra_silueta":len(ids),
            "inercia_final":float(final.inertia_),"Kmax_evaluado":Kmax,
            "regla":"criba por silueta muestral (barrido) y seleccion tras refinar finalistas en validación",
            "advertencia":"silueta de agrupamiento NO es exactitud diagnóstica"}
    print(f"  >> K elegido={mejor_k}; silueta final muestral={nueva_silueta:.4f}; "
          f"PCA2D={100*var_2d:.1f}% var.",flush=True)
    return resumen


def resumen_grafico(resumen,destino):
    tab=resumen.dropna(subset=["silueta_final_validacion_muestra"])
    if tab.empty:return
    tab=tab.sort_values("silueta_final_validacion_muestra")
    fig,ax=plt.subplots(figsize=(10,max(4.5,.42*len(tab)+1.4)))
    nombres=[f"{r.familia} · {r.escalado} (K={r.mejor_k})" for r in tab.itertuples()]
    barras=ax.barh(nombres,tab.silueta_final_validacion_muestra,color="#5b8caa",alpha=.86)
    ax.axvline(0,color="gray",lw=.8)
    ax.set(xlabel="Silueta media en una muestra de validación",
           title="K-Means | Comparación exploratoria de escenarios\n"
                 "No comparable como exactitud de fallas")
    ax.grid(axis="x",alpha=.13)
    guardar_figura(fig,destino/"00_comparacion_silueta_escenarios.png")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preparacion",default=str(RAIZ/"Resultados_Tarea3_Nuevos"/"00_preparacion"))
    ap.add_argument("--salida",default=str(RAIZ/"Resultados_Tarea3_Nuevos"/"02_kmeans"))
    ap.add_argument("--max-k",type=int,default=40)
    ap.add_argument("--muestra-silueta",type=int,default=1800)
    ap.add_argument("--n-init-barrido",type=int,default=1,
                    help="Inicializaciones por K en exploración (sobre entrenamiento completo)")
    ap.add_argument("--max-iter-barrido",type=int,default=100,
                    help="Máximo de iteraciones de la criba inicial")
    ap.add_argument("--n-init-final",type=int,default=10)
    ap.add_argument("--finalistas",type=int,default=3,
                    help="K más prometedores que se vuelven a ajustar rigurosamente")
    ap.add_argument("--escalados",nargs="+",choices=["estandar","robusto"],
                    default=["estandar"])
    args=ap.parse_args()
    if min(args.max_k,args.muestra_silueta) < 2 or \
            min(args.n_init_barrido,args.n_init_final,args.max_iter_barrido,args.finalistas) < 1:
        raise ValueError("K >= 2, muestra >= 2 y recuentos de iteraciones/inicializaciones >= 1")
    fuente=Path(args.preparacion).expanduser()
    destino=Path(args.salida).expanduser()
    datos,nombres,escenarios=leer_08(fuente,args.escalados)
    destino.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9.4,
                         "axes.spines.top":False,"axes.spines.right":False})
    # Conservar solo resultados del modelo de 10 variables, nunca los resultados
    # antiguos por familias. Se admite comparar escalados, no separar sensores.
    archivo_resumen=destino/"00_resumen_escenarios.csv"
    acumulado={}
    if archivo_resumen.exists():
        anteriores=pd.read_csv(archivo_resumen,sep=";",encoding="utf-8-sig")
        if "escenario" in anteriores.columns:
            for registro in anteriores.to_dict(orient="records"):
                if str(registro["escenario"]).startswith("todas_"):
                    acumulado[str(registro["escenario"])]=registro
    recientes=[]
    for escalado in args.escalados:
        r = analizar_escenario(escenarios[escalado], "todas", list(range(10)),
                              escalado, args, destino)
        acumulado[r["escenario"]] = r
        recientes.append(r["escenario"])
        guardar_csv(pd.DataFrame(list(acumulado.values())), archivo_resumen)
    tabla=pd.DataFrame(list(acumulado.values()))
    resumen_grafico(tabla,destino)
    nota={"fuente_08":str(fuente.resolve()),"registros":datos["registros"],"escenarios_recalculados":recientes,
          "escenarios_en_resumen":list(acumulado), "alcance":"10 variables juntas; sin modelos por sensor",
          "variables":nombres,"particion":"cronológica 70/15/15",
          "parametros":{"max_k":args.max_k,
                        "muestra_silueta":args.muestra_silueta,
                        "n_init_barrido":args.n_init_barrido,"max_iter_barrido":args.max_iter_barrido,
                        "n_init_final":args.n_init_final,"finalistas":args.finalistas,
                        "escalados":args.escalados,"variables_combinadas":nombres},
          "explicaciones":[
              "El entrenamiento utiliza TODAS las 100.492 filas para cada K.",
              "La selección usa una muestra fija de validación para la métrica silueta por coste O(n²).",
              "Las asignaciones y diagramas PCA muestran todas las filas de cada conjunto.",
              "Las curvas son de exploración (K-Means con menos inicializaciones); los 3 K finalistas se reentrenan y reevalúan en validación.",
              "Error de agrupamiento (%)=100*inercia(K)/inercia(K=1), NO porcentaje de clasificaciones incorrectas.",
              "No se entrenan modelos independientes por sensor ni por familia: se usan simultáneamente las 10 señales.",
              "El PCA no se empleó para entrenar K-Means; solo representa 2D.",
              "El porcentaje de error es una transformación de la inercia, no un error de clasificación.",
              "Los resultados son agrupaciones exploratorias, no diagnósticos de fallas."]}
    (destino/"00_metodologia.json").write_text(json.dumps(nota,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nK-MEANS FINALIZADO |",len(recientes),"escenarios recalculados | salida:",destino.resolve())


if __name__=="__main__":
    main()
