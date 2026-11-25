"""
SISTEMA INTELIGENTE DE ANÁLISIS DE VENTAS Y RECOMENDACIONES COMERCIALES
Lenguajes de Programación - Grupo 4 - UTP

Paradigmas:
  * Unidad 1 - Funcional : funciones puras, map / filter / reduce
  * Unidad 2 - Lógica    : motor de reglas (hechos -> reglas -> conclusiones)
  * Unidad 3 - Multiparadigma: clase SistemaAnalisisVentas (POO) con menú interactivo

Uso:
    python sistema_ventas.py                      -> menú interactivo
    python sistema_ventas.py --csv ruta.csv       -> indicar el CSV a mano
    python sistema_ventas.py --auto               -> ejecuta todo sin menú (para capturas)

Dataset: Supermarket Sales (Kaggle), archivo supermarket_sales.csv junto a este script.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # guarda PNG sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # evita errores de tildes en consolas Windows antiguas
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ANCHO = 78
UMBRAL_RATING = 7.0
UMBRAL_PREMIUM = 500.0
COLUMNAS_OBLIGATORIAS = [
    "Branch", "Customer type", "Product line", "Unit price", "Quantity", "Payment",
]


def titulo(texto: str) -> None:
    print("\n" + "=" * ANCHO)
    print(texto)
    print("=" * ANCHO)


def dinero(x: float) -> str:
    return f"${x:,.2f}"


# ==============================================================================
# CARGA Y LIMPIEZA (pandas + numpy)
# ==============================================================================
def generar_datos_demo(n: int = 300, semilla: int = 42) -> pd.DataFrame:
    """Datos SINTÉTICOS solo para probar el programa si falta el CSV real."""
    rng = np.random.default_rng(semilla)
    lineas = ["Health and beauty", "Electronic accessories", "Home and lifestyle",
              "Sports and travel", "Food and beverages", "Fashion accessories"]
    sucursal_ciudad = {"A": "Yangon", "B": "Mandalay", "C": "Naypyitaw"}
    suc = rng.choice(list(sucursal_ciudad), n)
    precio = np.round(rng.uniform(10, 100, n), 2)
    cant = rng.integers(1, 11, n)
    sub = precio * cant
    fechas = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 89, n), unit="D")
    return pd.DataFrame({
        "Invoice ID": [f"DEMO-{i:04d}" for i in range(n)],
        "Branch": suc,
        "City": [sucursal_ciudad[s] for s in suc],
        "Customer type": rng.choice(["Member", "Normal"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Product line": rng.choice(lineas, n),
        "Unit price": precio,
        "Quantity": cant,
        "Tax 5%": np.round(sub * 0.05, 4),
        "Total": np.round(sub * 1.05, 4),
        "Date": fechas.strftime("%m/%d/%Y"),
        "Time": [f"{h}:{m:02d}" for h, m in zip(rng.integers(10, 21, n), rng.integers(0, 60, n))],
        "Payment": rng.choice(["Ewallet", "Cash", "Credit card"], n),
        "gross income": np.round(sub * 0.05, 4),
        "Rating": np.round(rng.uniform(4, 10, n), 1),
    })


def localizar_csv(ruta: Optional[str]) -> Optional[str]:
    aqui = os.path.dirname(os.path.abspath(__file__))
    nombres = ["supermarket_sales.csv", "SuperMarket Analysis.csv", "supermarket_sales - Sheet1.csv"]
    candidatos = [ruta] if ruta else []
    for base in (os.getcwd(), aqui):
        candidatos += [os.path.join(base, n) for n in nombres]
    return next((c for c in candidatos if c and os.path.isfile(c)), None)


def cargar_y_limpiar(ruta: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Devuelve (DataFrame limpio, bitácora de limpieza)."""
    archivo = localizar_csv(ruta)
    if archivo:
        df = pd.read_csv(archivo)
        origen = archivo
        es_demo = False
    else:
        df = generar_datos_demo()
        origen = "DATOS DE DEMOSTRACIÓN (sintéticos)"
        es_demo = True

    log: Dict[str, Any] = {"origen": origen, "es_demo": es_demo, "filas_originales": len(df)}
    df.columns = [c.strip() for c in df.columns]

    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas obligatorias: {faltan}")

    # 1) duplicados (por Invoice ID si existe)
    antes = len(df)
    df = df.drop_duplicates(subset="Invoice ID") if "Invoice ID" in df.columns else df.drop_duplicates()
    log["duplicados"] = antes - len(df)

    # 2) estandarización de texto
    for col in ["City", "Customer type", "Gender", "Product line", "Payment"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                       .str.capitalize().replace({"Nan": np.nan, "None": np.nan, "": np.nan}))
    df["Branch"] = df["Branch"].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "": np.nan})
    antes = len(df)
    df = df.dropna(subset=["Branch", "Product line", "Customer type", "Payment"])
    log["sin_campos_clave"] = antes - len(df)

    # 3) numéricos y casos de borde
    for col in ["Unit price", "Quantity", "Tax 5%", "Total", "gross income", "Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    antes = len(df)
    df = df[(df["Unit price"] > 0) & (df["Quantity"] > 0)].copy()
    log["precio_o_cantidad_invalidos"] = antes - len(df)

    subtotal = df["Unit price"] * df["Quantity"]
    df["Tax 5%"] = df["Tax 5%"].fillna(np.round(subtotal * 0.05, 4))
    total_esperado = np.round(subtotal * 1.05, 4)
    inconsistentes = (df["Total"] - total_esperado).abs() > 0.05
    log["totales_recalculados"] = int(inconsistentes.sum() + df["Total"].isna().sum())
    df["Total"] = np.where(inconsistentes | df["Total"].isna(), total_esperado, df["Total"])
    df["gross income"] = df["gross income"].fillna(df["Tax 5%"])
    log["ratings_nulos"] = int(df["Rating"].isna().sum())
    df["Rating"] = df["Rating"].fillna(df["Rating"].median()).clip(1.0, 10.0)

    # 4) fecha y franja horaria (np.select)
    if "Date" in df.columns:
        fecha = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        fecha = fecha.fillna(pd.to_datetime(df["Date"], errors="coerce"))
        df["Mes"] = fecha.dt.month_name().fillna("Desconocido")
    else:
        df["Mes"] = "Desconocido"
    if "Time" in df.columns:
        hora = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
        df["Hora"] = hora.fillna(-1).astype(int)
    else:
        df["Hora"] = -1
    df["Franja"] = np.select(
        [(df["Hora"] >= 10) & (df["Hora"] < 14),
         (df["Hora"] >= 14) & (df["Hora"] < 18),
         (df["Hora"] >= 18) & (df["Hora"] <= 22)],
        ["Mañana (10h-13h)", "Tarde (14h-17h)", "Noche (18h-22h)"],
        default="Sin hora",
    )
    log["filas_validas"] = len(df)
    return df.reset_index(drop=True), log


# ==============================================================================
# UNIDAD 1 - PROGRAMACIÓN FUNCIONAL
# ==============================================================================
def enriquecer_transaccion(registro: Dict[str, Any]) -> Dict[str, Any]:
    """Función pura: devuelve un registro nuevo, no modifica el original."""
    cantidad = int(registro["Quantity"])
    return {
        **registro,
        "Ticket_Neto": round(float(registro["Total"]) / 1.05, 2),
        "Es_Compra_Volumen": cantidad >= 7,
        "Rentabilidad_Unitaria": round(float(registro["gross income"]) / cantidad, 2),
    }


def es_premium(umbral: float) -> Callable[[Dict[str, Any]], bool]:
    """Función de orden superior: construye el predicado de venta premium."""
    return lambda r: r["Total"] >= umbral and r["Rating"] >= UMBRAL_RATING


def procesar_funcional(registros: List[Dict[str, Any]], umbral: float = UMBRAL_PREMIUM) -> Dict[str, Any]:
    enriquecidos = list(map(enriquecer_transaccion, registros))
    premium = list(filter(es_premium(umbral), enriquecidos))
    miembros = list(filter(lambda r: r["Customer type"] == "Member", enriquecidos))
    volumen = list(filter(lambda r: r["Es_Compra_Volumen"], enriquecidos))
    suma = lambda campo: (lambda lista: reduce(lambda acc, r: acc + r[campo], lista, 0.0))
    return {
        "total_registros": len(enriquecidos),
        "ingreso_total": round(suma("Total")(enriquecidos), 2),
        "ganancia_total": round(suma("gross income")(enriquecidos), 2),
        "umbral_premium": umbral,
        "cantidad_premium": len(premium),
        "ingreso_premium": round(suma("Total")(premium), 2),
        "cantidad_miembros": len(miembros),
        "cantidad_volumen": len(volumen),
        "ticket_promedio": round(suma("Total")(enriquecidos) / max(len(enriquecidos), 1), 2),
    }


# ==============================================================================
# UNIDAD 2 - PROGRAMACIÓN LÓGICA (motor de reglas: hechos -> reglas -> conclusiones)
# ==============================================================================
@dataclass(frozen=True)
class Regla:
    id: str
    nombre: str
    condicion: Callable[[Dict[str, Any]], bool]
    conclusion: Callable[[Dict[str, Any]], str]
    tipo: str = "COMPLEMENTARIA"  # CLASIFICACION (excluyente) o COMPLEMENTARIA
    alerta: str = ""
    etiqueta: str = ""


@dataclass
class Inferencia:
    categoria: str
    clasificacion: str = ""
    alerta: str = ""
    recomendaciones: List[str] = field(default_factory=list)
    reglas_disparadas: List[str] = field(default_factory=list)


def _base_reglas() -> List[Regla]:
    return [
        # --- clasificación: se dispara solo la primera que cumple ---
        Regla("R1", "Estrella",
              lambda h: h["ingreso_alto"] and h["rating_alto"],
              lambda h: "Mantener stock prioritario, exhibición preferencial y venta cruzada.",
              "CLASIFICACION", "INFO", "ESTRELLA"),
        Regla("R2", "Vaca lechera en riesgo",
              lambda h: h["ingreso_alto"] and not h["rating_alto"],
              lambda h: "Auditar calidad del producto, proveedores y atención en sala.",
              "CLASIFICACION", "WARNING", "VACA LECHERA EN RIESGO"),
        Regla("R3", "Nicho de crecimiento",
              lambda h: (not h["ingreso_alto"]) and h["rating_alto"],
              lambda h: "Aumentar visibilidad y promociones: el cliente la valora, falta volumen.",
              "CLASIFICACION", "SUCCESS", "NICHO DE CRECIMIENTO"),
        Regla("R4", "Baja rotación",
              lambda h: (not h["ingreso_alto"]) and not h["rating_alto"],
              lambda h: "Revisar precios, evaluar descontinuar o liquidar excedentes.",
              "CLASIFICACION", "CRITICAL", "BAJA ROTACIÓN"),
        # --- complementarias: pueden dispararse varias a la vez ---
        Regla("R5", "Fidelización de miembros",
              lambda h: h["pct_miembros"] >= h["pct_miembros_global"],
              lambda h: f"Miembros = {h['pct_miembros']:.1f}% (global {h['pct_miembros_global']:.1f}%): "
                        "lanzar puntos dobles para fidelizar."),
        Regla("R6", "Captación de miembros",
              lambda h: h["pct_miembros"] < h["pct_miembros_global"],
              lambda h: f"Miembros = {h['pct_miembros']:.1f}% (global {h['pct_miembros_global']:.1f}%): "
                        "ofrecer membresía en caja al comprar esta línea."),
        Regla("R7", "Medio de pago dominante",
              lambda h: h["pago_dominante_pct"] >= 40.0,
              lambda h: f"{h['pago_dominante']} concentra {h['pago_dominante_pct']:.1f}% de las compras: "
                        "negociar alianza o descuento con ese medio de pago."),
        Regla("R8", "Calidad crítica",
              lambda h: h["rating"] < 6.0,
              lambda h: f"Rating {h['rating']:.2f} < 6.0: intervención inmediata en calidad."),
        Regla("R9", "Sucursal rezagada",
              lambda h: h["suc_rezagada"] != h["suc_lider"],
              lambda h: f"Impulsar la línea en la sucursal {h['suc_rezagada']} (menor ingreso); "
                        f"la sucursal {h['suc_lider']} lidera."),
        Regla("R10", "Franja horaria pico",
              lambda h: h["franja_pico"] != "Sin hora",
              lambda h: f"Concentrar reposición y personal en la franja {h['franja_pico']}."),
    ]


class MotorReglasComerciales:
    """Motor de inferencia: aplica las reglas sobre los hechos de cada línea de producto."""

    def __init__(self, reglas: Optional[List[Regla]] = None):
        self.reglas = reglas or _base_reglas()

    def inferir(self, hechos: Dict[str, Any]) -> Inferencia:
        res = Inferencia(categoria=hechos["categoria"])
        for regla in self.reglas:
            if not regla.condicion(hechos):
                continue
            if regla.tipo == "CLASIFICACION":
                if res.clasificacion:      # ya hay una clasificación: es excluyente
                    continue
                res.clasificacion, res.alerta = regla.etiqueta, regla.alerta
            res.recomendaciones.append(regla.conclusion(hechos))
            res.reglas_disparadas.append(regla.id)
        return res


# ==============================================================================
# UNIDAD 3 - MULTIPARADIGMA (POO + funcional + lógico + imperativo)
# ==============================================================================
class SistemaAnalisisVentas:
    def __init__(self, ruta_csv: Optional[str] = None):
        self.ruta_csv = ruta_csv
        self.df = pd.DataFrame()
        self.log: Dict[str, Any] = {}
        self.motor = MotorReglasComerciales()
        self.hechos: List[Dict[str, Any]] = []
        self.inferencias: List[Inferencia] = []
        self.resumen: Dict[str, Any] = {}
        self.mediana_ingresos = 0.0

    # ---------- preparación ----------
    def cargar(self) -> None:
        self.df, self.log = cargar_y_limpiar(self.ruta_csv)
        if self.df.empty:
            raise ValueError("No quedaron transacciones válidas tras la limpieza.")
        self._analizar()

    def _analizar(self, umbral: float = UMBRAL_PREMIUM) -> None:
        self.resumen = procesar_funcional(self.df.to_dict(orient="records"), umbral)
        self.hechos = self._construir_hechos()
        self.inferencias = [self.motor.inferir(h) for h in self.hechos]

    def _construir_hechos(self) -> List[Dict[str, Any]]:
        df = self.df
        por_linea = df.groupby("Product line")
        ingresos = por_linea["Total"].sum()
        self.mediana_ingresos = float(np.median(ingresos))
        pct_global = float((df["Customer type"] == "Member").mean() * 100)
        hechos = []
        for linea in ingresos.sort_values(ascending=False).index:
            sub = df[df["Product line"] == linea]
            pagos = sub["Payment"].value_counts(normalize=True) * 100
            por_suc = sub.groupby("Branch")["Total"].sum()
            franjas = sub.groupby("Franja")["Total"].sum()
            hechos.append({
                "categoria": linea,
                "ingresos": float(ingresos[linea]),
                "unidades": int(sub["Quantity"].sum()),
                "transacciones": len(sub),
                "rating": float(sub["Rating"].mean()),
                "pct_miembros": float((sub["Customer type"] == "Member").mean() * 100),
                "pct_miembros_global": pct_global,
                "participacion": float(ingresos[linea] / ingresos.sum() * 100),
                "ingreso_alto": bool(ingresos[linea] >= self.mediana_ingresos),
                "rating_alto": bool(sub["Rating"].mean() >= UMBRAL_RATING),
                "pago_dominante": pagos.idxmax(),
                "pago_dominante_pct": float(pagos.max()),
                "suc_lider": por_suc.idxmax(),
                "suc_rezagada": por_suc.idxmin(),
                "franja_pico": franjas.idxmax(),
            })
        return hechos

    # ---------- reportes (cada opción del menú) ----------
    def reporte_limpieza(self) -> None:
        titulo("CARGA Y LIMPIEZA DE DATOS")
        l = self.log
        print(f"Origen de datos            : {l['origen']}")
        if l["es_demo"]:
            print("!! ATENCIÓN: no se encontró supermarket_sales.csv. Se usan datos SINTÉTICOS de prueba.")
            print("!! Coloca el CSV de Kaggle junto al script para obtener resultados reales.")
        print(f"Filas originales           : {l['filas_originales']}")
        print(f"Duplicados eliminados      : {l['duplicados']}")
        print(f"Sin campos clave           : {l['sin_campos_clave']}")
        print(f"Precio/cantidad inválidos  : {l['precio_o_cantidad_invalidos']}")
        print(f"Totales recalculados       : {l['totales_recalculados']}")
        print(f"Ratings nulos (imputados)  : {l['ratings_nulos']}")
        print(f"Transacciones válidas      : {l['filas_validas']}")

    def reporte_resumen(self, umbral: Optional[float] = None) -> None:
        if umbral is not None:
            self._analizar(umbral)
        r = self.resumen
        titulo("RESUMEN GENERAL (CAPA FUNCIONAL)")
        print(f"Transacciones procesadas   : {r['total_registros']:,}")
        print(f"Ingreso total acumulado    : {dinero(r['ingreso_total'])}")
        print(f"Ganancia bruta acumulada   : {dinero(r['ganancia_total'])}")
        print(f"Ticket promedio            : {dinero(r['ticket_promedio'])}")
        pct_p = r['cantidad_premium'] / r['total_registros'] * 100
        print(f"Ventas premium (>= {dinero(r['umbral_premium'])} y rating >= {UMBRAL_RATING}): "
              f"{r['cantidad_premium']:,} ({pct_p:.1f}%) por {dinero(r['ingreso_premium'])}")
        pct_m = r['cantidad_miembros'] / r['total_registros'] * 100
        print(f"Compras de miembros        : {r['cantidad_miembros']:,} ({pct_m:.1f}%)")
        print(f"Compras de volumen (>=7 u.): {r['cantidad_volumen']:,}")

    def reporte_clasificacion(self) -> None:
        titulo("CLASIFICACIÓN Y RECOMENDACIONES POR LÍNEA DE PRODUCTO (CAPA LÓGICA)")
        print(f"Mediana de ingresos por línea = {dinero(self.mediana_ingresos)} | "
              f"Umbral de rating = {UMBRAL_RATING}\n")
        filas = [{
            "Línea de producto": h["categoria"],
            "Ingresos": dinero(h["ingresos"]),
            "Rating": f"{h['rating']:.2f}",
            "% Miembros": f"{h['pct_miembros']:.1f}",
            "Clasificación": i.clasificacion,
            "Alerta": i.alerta,
        } for h, i in zip(self.hechos, self.inferencias)]
        print(pd.DataFrame(filas).to_string(index=False))
        print("\nUse la opción 3 para ver las recomendaciones detalladas de una línea.")

    def reporte_detalle(self, indice: int) -> None:
        h, inf = self.hechos[indice], self.inferencias[indice]
        titulo(f"DETALLE: {h['categoria'].upper()}")
        print("HECHOS observados:")
        print(f"  Ingresos {dinero(h['ingresos'])} ({h['participacion']:.1f}% del total) | "
              f"Unidades {h['unidades']:,} | Transacciones {h['transacciones']:,}")
        print(f"  Rating promedio {h['rating']:.2f} | Miembros {h['pct_miembros']:.1f}% | "
              f"Pago dominante {h['pago_dominante']} ({h['pago_dominante_pct']:.1f}%)")
        print(f"  Sucursal líder {h['suc_lider']} | rezagada {h['suc_rezagada']} | pico {h['franja_pico']}")
        print(f"\nCONCLUSIÓN: {inf.clasificacion}  [Alerta: {inf.alerta}]")
        print(f"Reglas disparadas: {', '.join(inf.reglas_disparadas)}")
        print("\nRECOMENDACIONES:")
        for n, rec in enumerate(inf.recomendaciones, 1):
            print(f"  {n}. {rec}")

    def reporte_sucursales(self) -> None:
        titulo("ANÁLISIS POR SUCURSAL")
        g = self.df.groupby("Branch")
        ing = self.df.groupby(["Branch", "Product line"])["Total"].sum()
        mejor_linea = ing.groupby(level=0).idxmax().map(lambda ix: ix[1])
        t = pd.DataFrame({
            "Ciudad": g["City"].first() if "City" in self.df.columns else "-",
            "Transacciones": g.size(),
            "Ingresos": g["Total"].sum(),
            "Ticket prom.": g["Total"].mean(),
            "Rating prom.": g["Rating"].mean(),
            "Mejor línea": mejor_linea,
        })
        for col in ["Ingresos", "Ticket prom."]:
            t[col] = t[col].map(dinero)
        t["Rating prom."] = t["Rating prom."].map("{:.2f}".format)
        print(t.to_string())
        mejor = self.df.groupby("Branch")["Total"].sum().idxmax()
        print(f"\nSucursal con mayor facturación: {mejor}")

    def reporte_pagos_horarios(self) -> None:
        titulo("MÉTODOS DE PAGO Y FRANJAS HORARIAS")
        pagos = self.df.groupby("Payment")["Total"].agg(["count", "sum"])
        pagos["% ingresos"] = pagos["sum"] / pagos["sum"].sum() * 100
        pagos = pagos.rename(columns={"count": "Transacciones", "sum": "Ingresos"})
        pagos["Ingresos"] = pagos["Ingresos"].map(dinero)
        pagos["% ingresos"] = pagos["% ingresos"].map("{:.1f}%".format)
        print(pagos.to_string())
        print()
        fr = self.df.groupby("Franja")["Total"].agg(["count", "sum", "mean"])
        fr = fr.rename(columns={"count": "Transacciones", "sum": "Ingresos", "mean": "Ticket prom."})
        fr["Ingresos"] = fr["Ingresos"].map(dinero)
        fr["Ticket prom."] = fr["Ticket prom."].map(dinero)
        print(fr.to_string())

    # ---------- salidas ----------
    def generar_graficos(self, salida: str = "reporte_ventas_grupo4.png") -> str:
        colores = {"INFO": "#2e7d32", "SUCCESS": "#1565c0", "WARNING": "#ef6c00", "CRITICAL": "#c62828"}
        datos = sorted(zip(self.hechos, self.inferencias), key=lambda p: p[0]["ingresos"])
        fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5))
        ejes[0].barh([h["categoria"] for h, _ in datos], [h["ingresos"] for h, _ in datos],
                     color=[colores[i.alerta] for _, i in datos])
        ejes[0].axvline(self.mediana_ingresos, color="black", linestyle="--", label="Mediana")
        ejes[0].set_title("Ingresos por línea de producto (color = nivel de alerta)")
        ejes[0].set_xlabel("Ingresos (USD)")
        ejes[0].legend()
        tabla = self.df.pivot_table(index="Branch", columns="Payment", values="Total",
                                    aggfunc="sum", fill_value=0)
        tabla.plot(kind="bar", ax=ejes[1], colormap="viridis")
        ejes[1].set_title("Ingresos por sucursal y método de pago")
        ejes[1].set_xlabel("Sucursal")
        ejes[1].set_ylabel("Ingresos (USD)")
        ejes[1].tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(salida, dpi=150)
        plt.close(fig)
        return os.path.abspath(salida)

    def exportar_csv(self, salida: str = "reporte_recomendaciones.csv") -> str:
        filas = [{
            "Linea_Producto": h["categoria"], "Ingresos_USD": round(h["ingresos"], 2),
            "Participacion_%": round(h["participacion"], 1), "Rating_Prom": round(h["rating"], 2),
            "Pct_Miembros": round(h["pct_miembros"], 1), "Clasificacion": i.clasificacion,
            "Alerta": i.alerta, "Reglas": " ".join(i.reglas_disparadas),
            "Recomendaciones": " | ".join(i.recomendaciones),
        } for h, i in zip(self.hechos, self.inferencias)]
        pd.DataFrame(filas).to_csv(salida, index=False, encoding="utf-8-sig")
        return os.path.abspath(salida)

    # ---------- interfaz de consola (imperativo) ----------
    def ejecutar_auto(self) -> None:
        self.reporte_limpieza()
        self.reporte_resumen()
        self.reporte_clasificacion()
        for i in range(len(self.hechos)):
            self.reporte_detalle(i)
        self.reporte_sucursales()
        self.reporte_pagos_horarios()
        print(f"\n[OK] Gráfico guardado en: {self.generar_graficos()}")
        print(f"[OK] Reporte CSV guardado en: {self.exportar_csv()}")

    def menu(self) -> None:
        opciones = (
            "\n" + "-" * ANCHO + "\n"
            " 1. Resumen general (ingresos, ganancia, ventas premium)\n"
            " 2. Clasificación y alertas por línea de producto\n"
            " 3. Recomendaciones detalladas de UNA línea de producto\n"
            " 4. Análisis por sucursal\n"
            " 5. Métodos de pago y franjas horarias\n"
            " 6. Cambiar umbral de venta premium\n"
            " 7. Generar gráficos (PNG)\n"
            " 8. Exportar reporte de recomendaciones (CSV)\n"
            " 9. Ver informe de limpieza de datos\n"
            " 0. Salir\n" + "-" * ANCHO
        )
        titulo("SISTEMA INTELIGENTE DE ANÁLISIS DE VENTAS Y RECOMENDACIONES - GRUPO 4")
        self.reporte_limpieza()
        while True:
            print(opciones)
            op = pedir("Elija una opción: ")
            if op == "1":
                self.reporte_resumen()
            elif op == "2":
                self.reporte_clasificacion()
            elif op == "3":
                for n, h in enumerate(self.hechos, 1):
                    print(f"  {n}. {h['categoria']}")
                sel = pedir("Número de la línea: ")
                if sel.isdigit() and 1 <= int(sel) <= len(self.hechos):
                    self.reporte_detalle(int(sel) - 1)
                else:
                    print("Opción no válida.")
            elif op == "4":
                self.reporte_sucursales()
            elif op == "5":
                self.reporte_pagos_horarios()
            elif op == "6":
                val = pedir(f"Nuevo umbral en USD (actual {self.resumen['umbral_premium']:.0f}): ")
                try:
                    nuevo = float(val.replace(",", "."))
                    if nuevo <= 0:
                        raise ValueError
                    self.reporte_resumen(nuevo)
                except ValueError:
                    print("Valor no válido: ingrese un número mayor que 0.")
            elif op == "7":
                print(f"[OK] Gráfico guardado en: {self.generar_graficos()}")
            elif op == "8":
                print(f"[OK] Reporte guardado en: {self.exportar_csv()}")
            elif op == "9":
                self.reporte_limpieza()
            elif op == "0":
                print("Hasta luego.")
                break
            else:
                print("Opción no válida.")


def pedir(texto: str) -> str:
    try:
        return input(texto).strip()
    except EOFError:
        return "0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sistema inteligente de análisis de ventas - Grupo 4")
    ap.add_argument("--csv", help="ruta del CSV (por defecto busca supermarket_sales.csv)")
    ap.add_argument("--auto", action="store_true", help="ejecuta todos los reportes sin menú")
    args = ap.parse_args()
    sistema = SistemaAnalisisVentas(args.csv)
    try:
        sistema.cargar()
    except (ValueError, pd.errors.ParserError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    sistema.ejecutar_auto() if args.auto else sistema.menu()


if __name__ == "__main__":
    main()
