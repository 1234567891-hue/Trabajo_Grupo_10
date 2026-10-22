import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

st.set_page_config(page_title="Proyección de Demanda", page_icon="Universidad_de_Lima_logo.png", layout="wide")

col_img, col_titulo = st.columns([1, 5])
with col_img:
    st.image("Universidad_de_Lima_logo.png", width=120)
with col_titulo:
    st.title("Aplicativo — Caso Adaptado Super Market Sales")
st.caption("Grupo 10 | Modelamiento Predictivo de Datos — Universidad de Lima")


# ---------------------------------------------------------------------------
# Helper: título de sección + botón "❓" con explicación de qué se está viendo
# ---------------------------------------------------------------------------
def seccion(titulo, explicacion):
    c1, c2 = st.columns([10, 1])
    with c1:
        st.header(titulo)
    with c2:
        with st.popover("❓ ¿Qué es esto?", use_container_width=True):
            st.markdown(explicacion)


def subseccion(titulo, explicacion):
    c1, c2 = st.columns([10, 1])
    with c1:
        st.subheader(titulo)
    with c2:
        with st.popover("❓", use_container_width=True):
            st.markdown(explicacion)


# ---------------------------------------------------------------------------
# NUEVO: detección automática de la columna más probable por palabra clave,
# para que el selectbox no se quede por defecto en la primera columna del
# CSV (ej. "Invoice ID"), que era la causa de TODOS los errores del log:
# "no numeric data to plot", "Cannot perform reduction 'mean' with string
# dtype", "Found array with 0 sample(s)", etc.
# ---------------------------------------------------------------------------
def indice_por_defecto(columnas, candidatos, alternativa=0):
    """Busca, en orden de prioridad, la primera columna cuyo nombre
    contenga alguna de las palabras clave en `candidatos` (case-insensitive).
    Si no encuentra ninguna, devuelve `alternativa` (por defecto, la 0)."""
    cols_lower = [c.lower() for c in columnas]
    for palabra in candidatos:
        for i, c in enumerate(cols_lower):
            if palabra in c:
                return i
    return alternativa


# ---------------------------------------------------------------------------
# 1. Carga de datos (compartida por todas las pestañas)
# ---------------------------------------------------------------------------
st.sidebar.header("1. Carga tu dataset")
archivo = st.sidebar.file_uploader(
    "Sube un archivo CSV", type=["csv"],
    help="El archivo original del caso adaptado: supermarket_sales.csv. "
         "Debe tener una fila por transacción, con fecha, producto, cantidad, precio y total."
)

if archivo is None:
    st.info("👈 Sube un archivo CSV en el panel izquierdo para comenzar.")
    st.stop()

df_raw = pd.read_csv(archivo)

st.sidebar.header("2. Configura las columnas")
st.sidebar.caption(
    "Las columnas se pre-seleccionan automáticamente según el nombre. "
    "Verifica que cada selector apunte a la columna correcta antes de continuar."
)
columnas = df_raw.columns.tolist()

col_fecha = st.sidebar.selectbox(
    "Columna de fecha", columnas,
    index=indice_por_defecto(columnas, ["date", "fecha"]),
    help="Columna con la fecha de cada transacción (ej. 'Date'). Se usa para armar las series de tiempo semanales/diarias."
)
col_producto = st.sidebar.selectbox(
    "Columna de producto / categoría", columnas,
    index=indice_por_defecto(columnas, ["product line", "producto", "categor"]),
    help="Columna que identifica la línea de producto (ej. 'Product line'). Sobre esta columna se calcula el ranking y se separan los productos top."
)
col_cantidad = st.sidebar.selectbox(
    "Columna de cantidad / demanda", columnas,
    index=indice_por_defecto(columnas, ["quantity", "cantidad"]),
    help="Columna con las unidades vendidas por transacción (ej. 'Quantity'). Es la variable que se proyecta en la pestaña de Demanda."
)
col_hora = st.sidebar.selectbox(
    "Columna de hora (opcional)", ["(ninguna)"] + columnas,
    index=(indice_por_defecto(columnas, ["time", "hora"], alternativa=-1) + 1),
    help="Si tu dataset separa fecha y hora en dos columnas, selecciona aquí la de hora. Si no aplica, deja '(ninguna)'."
)
col_precio = st.sidebar.selectbox(
    "Columna de precio unitario (para Pricing)", columnas,
    index=indice_por_defecto(columnas, ["unit price", "precio"]),
    help="Precio por unidad de cada transacción (ej. 'Unit price'). Se usa para estimar la elasticidad precio-demanda."
)
col_ingreso = st.sidebar.selectbox(
    "Columna de ingreso/total por transacción (para Pricing)", columnas,
    index=indice_por_defecto(columnas, ["total", "ingreso"]),
    help="Monto total de la transacción (ej. 'Total'). Se usa para calcular el ingreso actual e incremental en Pricing."
)

# Aviso visible si alguna columna clave no es numérica: evita que el error
# aparezca varios pasos después (como en el log original) y sin contexto.
for nombre, col in [("cantidad", col_cantidad), ("precio unitario", col_precio), ("ingreso/total", col_ingreso)]:
    if not pd.api.types.is_numeric_dtype(df_raw[col]):
        st.sidebar.error(f"⚠️ La columna de {nombre} ('{col}') no es numérica. Selecciona la columna correcta.")

df = df_raw.copy()
try:
    if col_hora != "(ninguna)":
        df["FechaHora"] = pd.to_datetime(
            df[col_fecha].astype(str) + " " + df[col_hora].astype(str), errors="coerce"
        )
    else:
        df["FechaHora"] = pd.to_datetime(df[col_fecha], errors="coerce")
except Exception:
    df["FechaHora"] = pd.to_datetime(df[col_fecha], errors="coerce")

filas_antes = len(df)
df = df.dropna(subset=["FechaHora"])
filas_despues = len(df)
if filas_despues == 0:
    st.error(
        f"⚠️ Ninguna fila pudo interpretarse como fecha usando la columna '{col_fecha}'. "
        "Verifica que hayas seleccionado la columna correcta (debería llamarse algo como 'Date')."
    )
    st.stop()
elif filas_despues < filas_antes:
    st.warning(f"Se descartaron {filas_antes - filas_despues} filas sin fecha válida.")

df.set_index("FechaHora", inplace=True)
df.sort_index(inplace=True)

st.success(f"Datos cargados: {len(df)} filas, de {df.index.min().date()} a {df.index.max().date()}")
with st.expander("Ver vista previa de los datos"):
    st.dataframe(df.head(20))

ranking = df.groupby(col_producto)[col_cantidad].sum().sort_values(ascending=False)

st.sidebar.header("3. Productos a analizar")
n_top = st.sidebar.slider(
    "¿Cuántos productos top?", 1, min(6, len(ranking)), 3,
    help="Cuántas líneas de producto (las de mayor venta acumulada) se van a proyectar y analizar en las 3 pestañas."
)
productos_top = ranking.index[:n_top].tolist()

st.sidebar.header("4. Frecuencia y horizonte")
freq = st.sidebar.radio(
    "Frecuencia de agregación", ["Diaria", "Semanal"], index=1, horizontal=True,
    help="Cada cuánto se agrupa la demanda antes de proyectar. Con pocos meses de datos, 'Semanal' suele dar un error (MAPE) mucho menor que 'Diaria'."
)
freq_code = "D" if freq == "Diaria" else "W"
periodo_estacional = 7 if freq_code == "D" else 4
horizonte = st.sidebar.slider(
    "Horizonte de proyección (periodos)", 3, 30, 3,
    help="Cuántos periodos hacia adelante se proyectan. Con ~13 semanas de historia, la validación solo "
         "cubrió 3 semanas: se recomienda dejar este valor entre 3 y 4 para que la proyección esté respaldada "
         "por la validación. Valores mayores extrapolan más allá de lo probado y son menos confiables."
)

series_por_producto = {}
for prod in productos_top:
    series_por_producto[prod] = df[df[col_producto] == prod][col_cantidad].resample(freq_code).sum()


def mape(real, pred):
    real, pred = np.array(real), np.array(pred)
    mask = real != 0
    return np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100


# ---------------------------------------------------------------------------
# NUEVO: los 3 modelos que la comparativa (8 métodos, 6 líneas de producto)
# mostró como los más robustos para este dataset. Holt-Winters se conserva
# como benchmark de comparación (así lo exige la rúbrica y la literatura de
# clase), pero la PROYECCIÓN FINAL usa el ganador real por producto, porque
# Holt-Winters no ganó en ninguna de las 6 líneas evaluadas.
# ---------------------------------------------------------------------------
def pronostico_holt_winters(serie, n_forecast, seasonal_periods):
    seasonal, sp = ("add", seasonal_periods) if len(serie) >= seasonal_periods * 2 else (None, None)
    try:
        modelo = ExponentialSmoothing(serie, trend="add", seasonal=seasonal, seasonal_periods=sp).fit()
        pred = modelo.forecast(n_forecast)
    except Exception:
        idx = pd.date_range(serie.index[-1], periods=n_forecast + 1, freq=serie.index.freq)[1:]
        pred = pd.Series([serie.mean()] * n_forecast, index=idx)
    return pred


def pronostico_naive_estacional(serie, n_forecast, periodo=4):
    """Repite el patrón de las últimas `periodo` observaciones (ej. últimas 4 semanas)."""
    ultimos = serie.iloc[-periodo:].values
    reps = int(np.ceil(n_forecast / periodo))
    valores = np.tile(ultimos, reps)[:n_forecast]
    idx = pd.date_range(serie.index[-1], periods=n_forecast + 1, freq=serie.index.freq)[1:]
    return pd.Series(valores, index=idx)


def pronostico_media_movil(serie, n_forecast, ventana=4):
    """Repite el promedio de las últimas `ventana` observaciones."""
    valor = serie.iloc[-ventana:].mean()
    idx = pd.date_range(serie.index[-1], periods=n_forecast + 1, freq=serie.index.freq)[1:]
    return pd.Series([valor] * n_forecast, index=idx)


MODELOS_DEMANDA = {
    "Holt-Winters": pronostico_holt_winters,
    "Naive estacional": pronostico_naive_estacional,
    "Media móvil (4)": pronostico_media_movil,
}


def comparar_modelos(serie, periodo_estacional):
    """Valida los 3 modelos sobre el mismo tramo de prueba (25% final de la serie,
    igual que antes) y devuelve (tabla_metricas, nombre_del_mejor_modelo)."""
    n_test = max(1, len(serie) // 4)
    train, test = serie.iloc[:-n_test], serie.iloc[-n_test:]
    filas = []
    for nombre, funcion in MODELOS_DEMANDA.items():
        if nombre == "Holt-Winters":
            pred = funcion(train, n_test, periodo_estacional)
        else:
            pred = funcion(train, n_test, periodo_estacional)
        pred.index = test.index
        filas.append({
            "Modelo": nombre,
            "MAE": mean_absolute_error(test.values, pred.values),
            "RMSE": np.sqrt(mean_squared_error(test.values, pred.values)),
            "MAPE (%)": mape(test.values, pred.values),
        })
    tabla = pd.DataFrame(filas).sort_values("MAPE (%)").reset_index(drop=True)
    mejor_modelo = tabla.iloc[0]["Modelo"]
    return tabla, mejor_modelo, train, test


def ajustar_modelo(serie, n_forecast, seasonal_periods, nombre_modelo="Holt-Winters"):
    """Proyección a futuro usando el modelo indicado (por defecto Holt-Winters,
    para no romper compatibilidad; la app ahora pasa el modelo ganador)."""
    return MODELOS_DEMANDA[nombre_modelo](serie, n_forecast, seasonal_periods)


def boton_descarga(df_a_exportar, nombre_archivo, etiqueta="⬇️ Descargar CSV"):
    csv = df_a_exportar.to_csv(index=False).encode("utf-8")
    st.download_button(etiqueta, data=csv, file_name=nombre_archivo, mime="text/csv")


# Comparación de los 3 modelos y selección automática del ganador por producto
comparacion_por_producto = {}
mejor_modelo_por_producto = {}
for prod in productos_top:
    tabla_cmp, mejor, train_cmp, test_cmp = comparar_modelos(series_por_producto[prod], periodo_estacional)
    comparacion_por_producto[prod] = (tabla_cmp, train_cmp, test_cmp)
    mejor_modelo_por_producto[prod] = mejor

proyeccion_futura_por_producto = {}
for prod in productos_top:
    modelo_ganador = mejor_modelo_por_producto[prod]
    proyeccion_futura_por_producto[prod] = ajustar_modelo(
        series_por_producto[prod], horizonte, periodo_estacional, nombre_modelo=modelo_ganador
    )

tab_demanda, tab_inventario, tab_pricing = st.tabs([
    "📈 Proyección de Demanda (Zapata)", "📦 Inventario (Miyasato)", "💰 Pricing (Barrientos)"
])

# ===========================================================================
# TAB 1 — PROYECCIÓN DE DEMANDA
# ===========================================================================
with tab_demanda:
    seccion(
        "🏆 Productos más importantes",
        "Suma la cantidad vendida de **todas** las transacciones, agrupada por línea de producto, "
        "y las ordena de mayor a menor. Los productos resaltados/seleccionados (los N más vendidos, "
        "según el slider **'¿Cuántos productos top?'** de la barra lateral) son los que se analizan "
        "en el resto de la pestaña."
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(ranking.rename("Cantidad total").to_frame())
    with col2:
        fig, ax = plt.subplots()
        ranking.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Cantidad total")
        st.pyplot(fig)
    st.write("Seleccionados:", ", ".join(map(str, productos_top)))

    seccion(
        "📊 Análisis descriptivo",
        "Estadística descriptiva básica de la cantidad vendida **por transacción**, calculada por separado "
        "para cada producto top:\n\n"
        "- **Media**: promedio de unidades vendidas por transacción.\n"
        "- **Moda**: valor de cantidad que más se repite.\n"
        "- **Desv. estándar**: qué tanto varía la cantidad vendida entre transacciones (a mayor valor, más volátil).\n"
        "- **Mínimo / Máximo**: rango observado.\n\n"
        "Sirve para justificar decisiones posteriores, como el nivel de variabilidad que debe cubrir el stock de seguridad."
    )
    filas = []
    for prod in productos_top:
        sub = df[df[col_producto] == prod][col_cantidad]
        filas.append({
            "Producto": prod,
            "Media": sub.mean(),
            "Moda": sub.mode().iloc[0] if not sub.mode().empty else np.nan,
            "Desv. estándar": sub.std(),
            "Mínimo": sub.min(),
            "Máximo": sub.max(),
        })
    st.dataframe(pd.DataFrame(filas).round(2))

    seccion(
        "🕒 Demanda a lo largo del tiempo",
        "Serie de tiempo de la demanda **agregada** (diaria o semanal, según lo elegido en la barra lateral) "
        "para cada producto top, antes de aplicar cualquier modelo."
    )
    tabs = st.tabs(productos_top)
    for tab, prod in zip(tabs, productos_top):
        with tab:
            fig, ax = plt.subplots(figsize=(10, 3))
            series_por_producto[prod].plot(ax=ax)
            ax.set_title(f"Demanda ({freq.lower()}) — {prod}")
            ax.set_ylabel("Unidades")
            st.pyplot(fig)

    seccion(
        "🔮 Validación del modelo",
        "Se **esconde** el tramo final de la serie (25% más reciente — no el horizonte elegido arriba: "
        "con series tan cortas, achicar el conjunto de prueba a más semanas empeora el ajuste, así que se "
        "usa siempre la misma proporción de prueba para no des-estabilizar el modelo) y se comparan 3 "
        "modelos: **Holt-Winters** (referencia de la literatura de clase), **Naive estacional** (repite el "
        "patrón de hace 4 semanas) y **Media móvil** (promedio de las últimas 4 semanas). El ganador por "
        "producto es el que se usa después en 'Proyección a futuro'."
    )
    metricas = []
    for prod in productos_top:
        tabla_cmp, train, test = comparacion_por_producto[prod]
        for _, fila in tabla_cmp.iterrows():
            metricas.append({"Producto": prod, **fila.to_dict()})

    tabs2 = st.tabs(productos_top)
    for tab, prod in zip(tabs2, productos_top):
        with tab:
            tabla_cmp, train, test = comparacion_por_producto[prod]
            n_test = len(test)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train.index, train.values, label="Histórico")
            ax.plot(test.index, test.values, label="Real", marker="o", color="black")
            for nombre, funcion in MODELOS_DEMANDA.items():
                pred = funcion(train, n_test, periodo_estacional)
                pred.index = test.index
                estilo = "--" if nombre != mejor_modelo_por_producto[prod] else "-"
                ax.plot(pred.index, pred.values, estilo, label=nombre,
                         linewidth=2.5 if nombre == mejor_modelo_por_producto[prod] else 1.5)
            ax.legend()
            ax.set_title(f"Validación — {prod}  (ganador: {mejor_modelo_por_producto[prod]})")
            st.pyplot(fig)
            st.info(f"✅ Modelo seleccionado para {prod}: **{mejor_modelo_por_producto[prod]}** "
                    f"(MAPE {tabla_cmp.iloc[0]['MAPE (%)']:.1f}%)")

    subseccion(
        "Métricas de error",
        "- **MAE**: error promedio, en unidades.\n- **RMSE**: parecido al MAE pero castiga más los errores grandes.\n"
        "- **MAPE (%)**: error promedio en porcentaje. La fila en negrita/menor MAPE de cada producto es el "
        "modelo que se usa en la proyección a futuro."
    )
    metricas_df = pd.DataFrame(metricas).sort_values(["Producto", "MAPE (%)"])
    st.dataframe(metricas_df.round(2))
    boton_descarga(metricas_df.round(2), "demanda_metricas.csv")

    seccion(
        "🚀 Proyección a futuro",
        "Se reentrena con **todo** el histórico disponible usando, para cada producto, el modelo que ganó "
        "en la validación (no siempre Holt-Winters), y se proyecta hacia adelante la cantidad de periodos "
        "definida en 'Horizonte de proyección'. Alimenta la pestaña de Inventario.\n\n"
        "⚠️ Con solo ~13 semanas de historia, la validación solo cubrió 3 semanas hacia adelante — "
        "horizontes mayores a 3-4 periodos no están respaldados por la validación y deben presentarse con esa salvedad."
    )
    filas_proyeccion_export = []
    for prod in productos_top:
        serie_completa = series_por_producto[prod]
        proyeccion = proyeccion_futura_por_producto[prod]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(serie_completa.index, serie_completa.values, label="Histórico")
        ax.plot(proyeccion.index, proyeccion.values, label="Proyección futura", color="red", linestyle="--")
        ax.legend()
        ax.set_title(f"Proyección futura — {prod}")
        st.pyplot(fig)

        for fecha, valor in proyeccion.items():
            filas_proyeccion_export.append({"Producto": prod, "Fecha": fecha, "Proyección": round(valor, 1)})

        with st.expander(f"Ver valores numéricos — {prod}"):
            st.dataframe(proyeccion.rename("Proyección").round(1).to_frame())

    boton_descarga(pd.DataFrame(filas_proyeccion_export), "demanda_proyeccion_futura.csv")

# ===========================================================================
# TAB 2 — INVENTARIO (Miyasato)
# ===========================================================================
with tab_inventario:
    seccion(
        "📦 Inventario necesario para cubrir la demanda",
        "Traduce la demanda proyectada en stock de seguridad, punto de reorden (ROP) y EOQ."
    )
    st.caption("Usa la proyección de demanda de la pestaña anterior como punto de partida.")

    subseccion(
        "Parámetros de inventario",
        "- **Lead time**: cuántos periodos tarda en llegar un pedido.\n"
        "- **Nivel de servicio**: probabilidad de no quedarse sin stock durante el lead time.\n"
        "- **Costo por pedido**: costo fijo de generar una orden de compra (S/).\n"
        "- **Costo de almacenar x unidad/año**: costo de mantener una unidad en inventario un año."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lead_time = st.number_input(
            f"Lead time ({'días' if freq_code=='D' else 'semanas'})", min_value=1, value=3
        )
    with c2:
        nivel_servicio = st.select_slider(
            "Nivel de servicio deseado", options=[0.90, 0.95, 0.975, 0.99], value=0.95
        )
    with c3:
        costo_pedido = st.number_input("Costo por pedido (S/)", min_value=0.0, value=50.0)
    with c4:
        tasa_almacenamiento = st.slider(
            "Tasa de almacenamiento anual (% del valor del producto)", 10, 35, 22,
            help="Costo de mantener inventario, como % del precio del producto por año (capital inmovilizado, "
                 "riesgo de obsolescencia, espacio). La literatura de costos logísticos documenta un rango típico "
                 "de 15%-35% anual; 20-25% es el valor de referencia más común en retail (Azim et al., 2025)."
        ) / 100

    z = norm.ppf(nivel_servicio)
    st.caption(f"Z asociado al nivel de servicio {nivel_servicio*100:.1f}%: **{z:.2f}**")

    periodos_por_anio = 365 if freq_code == "D" else 52

    filas_inv = []
    for prod in productos_top:
        serie_hist = series_por_producto[prod]
        demanda_media = serie_hist.mean()
        demanda_std = serie_hist.std()
        demanda_anual = demanda_media * periodos_por_anio

        # NUEVO: el costo de almacenar ahora se calcula por producto, a partir de
        # su propio precio unitario promedio (viene de la data real, no es un
        # número fijo igual para todos). H = tasa_almacenamiento x precio_unitario.
        precio_prod = df[df[col_producto] == prod][col_precio].mean()
        costo_almacen_unit = tasa_almacenamiento * precio_prod

        stock_seguridad = z * demanda_std * np.sqrt(lead_time)
        punto_reorden = demanda_media * lead_time + stock_seguridad
        eoq = np.sqrt((2 * demanda_anual * costo_pedido) / costo_almacen_unit)

        demanda_proyectada_horizonte = proyeccion_futura_por_producto[prod].sum()
        inventario_objetivo = demanda_proyectada_horizonte + stock_seguridad

        filas_inv.append({
            "Producto": prod,
            "Precio unit. promedio": round(precio_prod, 2),
            "Costo almacenar/año (calculado)": round(costo_almacen_unit, 2),
            "Demanda media": round(demanda_media, 1),
            "Desv. estándar": round(demanda_std, 1),
            "Stock de seguridad": round(stock_seguridad, 1),
            "Punto de reorden": round(punto_reorden, 1),
            "EOQ (cantidad óptima x pedido)": round(eoq, 1),
            f"Demanda proyectada ({horizonte} periodos)": round(demanda_proyectada_horizonte, 1),
            "Inventario objetivo": round(inventario_objetivo, 1),
        })

    subseccion(
        "Tabla de resultados de inventario",
        "- **Stock de seguridad**: colchón para no quedarte sin stock.\n- **Punto de reorden**: nivel en el que debes generar un nuevo pedido.\n"
        "- **EOQ**: tamaño de pedido que minimiza el costo total.\n- **Inventario objetivo**: demanda proyectada + colchón de seguridad."
    )
    inv_df = pd.DataFrame(filas_inv)
    st.dataframe(inv_df, width="stretch")
    boton_descarga(inv_df, "inventario_resultados.csv")

    subseccion(
        "Inventario objetivo por producto",
        "Barras apiladas: abajo la demanda ya esperada, arriba el colchón de seguridad."
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(inv_df["Producto"], inv_df["Stock de seguridad"], label="Stock de seguridad")
    ax.bar(
        inv_df["Producto"],
        inv_df[f"Demanda proyectada ({horizonte} periodos)"],
        bottom=inv_df["Stock de seguridad"],
        label="Demanda proyectada",
    )
    ax.set_ylabel("Unidades")
    ax.legend()
    st.pyplot(fig)

    with st.expander("📖 Notas sobre las fórmulas usadas"):
        st.markdown("""
        - **Stock de seguridad** = Z × σ_demanda × √(lead time).
        - **Punto de reorden (ROP)** = demanda media × lead time + stock de seguridad.
        - **EOQ (Wilson)** = √(2 × Demanda anual × Costo por pedido / Costo de almacenar por unidad).
        - **Inventario objetivo** = demanda proyectada para el horizonte elegido + stock de seguridad.
        - Ajusta el nivel de servicio, lead time y costos con los valores reales/supuestos del caso, y sustenta la elección con citas científicas.
        """)

# ===========================================================================
# TAB 3 — PRICING (Barrientos)
# ===========================================================================
with tab_pricing:
    seccion(
        "💰 Pricing y asignación de ofertas",
        "Estima la elasticidad precio-demanda por producto y simula escenarios de descuento sobre el ingreso."
    )
    st.caption(
        "El dataset no trae descuentos reales aplicados; se estima la elasticidad "
        "precio-demanda y se simulan escenarios de descuento sobre esa base."
    )

    subseccion(
        "Análisis descriptivo de precio y cantidad",
        "Por línea de producto: precio unitario promedio y su variabilidad, cantidad promedio vendida por "
        "transacción, e ingreso total generado en todo el periodo."
    )
    desc_pricing = df.groupby(col_producto).agg(
        Precio_medio=(col_precio, "mean"),
        Precio_std=(col_precio, "std"),
        Cantidad_media=(col_cantidad, "mean"),
        Ingreso_total=(col_ingreso, "sum"),
    ).round(2)
    st.dataframe(desc_pricing, width="stretch")

    subseccion(
        "Elasticidad precio-demanda por producto",
        "- Elasticidad **negativa** (lo esperado): al subir el precio, baja la cantidad vendida.\n"
        "- Elasticidad cercana a **0**: demanda inelástica — un descuento probablemente no compense.\n"
        "- Elasticidad **positiva** (contraintuitivo): normalmente indica una limitación de datos."
    )
    st.caption("Regresión log-log: log(cantidad) = a + b · log(precio). 'b' es la elasticidad.")

    elasticidades = []
    for prod in df[col_producto].unique():
        sub = df[(df[col_producto] == prod) & (df[col_precio] > 0) & (df[col_cantidad] > 0)]
        if len(sub) < 5:
            continue
        X = np.log(sub[[col_precio]].values)
        y = np.log(sub[col_cantidad].values)
        reg = LinearRegression().fit(X, y)
        elasticidades.append({"Producto": prod, "Elasticidad estimada": round(reg.coef_[0], 3)})

    elasticidad_df = pd.DataFrame(elasticidades).sort_values("Elasticidad estimada")

    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.dataframe(elasticidad_df, width="stretch")
        boton_descarga(elasticidad_df, "pricing_elasticidad.csv")
    with col_e2:
        fig, ax = plt.subplots()
        ax.barh(elasticidad_df["Producto"], elasticidad_df["Elasticidad estimada"])
        ax.set_xlabel("Elasticidad")
        st.pyplot(fig)

    subseccion(
        "Simulación de descuentos",
        "Aplica distintos niveles de descuento y, usando la elasticidad estimada, proyecta el ingreso resultante."
    )
    descuento_max = st.slider("Descuento máximo a simular (%)", 10, 70, 50, 5) / 100
    paso_descuento = 0.05
    descuentos = np.arange(0, descuento_max + paso_descuento, paso_descuento)

    def simular_descuentos(producto):
        sub = df[df[col_producto] == producto]
        precio_base = sub[col_precio].mean()
        cantidad_base = sub[col_cantidad].sum()
        elasticidad = elasticidad_df.set_index("Producto").loc[producto, "Elasticidad estimada"]

        filas = []
        for d in descuentos:
            precio_nuevo = precio_base * (1 - d)
            cantidad_nueva = cantidad_base * (1 + elasticidad * (-d))
            ingreso = precio_nuevo * cantidad_nueva
            filas.append({
                "Descuento (%)": round(d * 100, 1),
                "Precio": round(precio_nuevo, 2),
                "Cantidad estimada": round(cantidad_nueva, 1),
                "Ingreso estimado": round(ingreso, 1),
            })
        return pd.DataFrame(filas)

    simulaciones = {prod: simular_descuentos(prod) for prod in productos_top}

    tabs_pricing = st.tabs(productos_top)
    optimos = []
    simulaciones_export = []
    for tab, prod in zip(tabs_pricing, productos_top):
        with tab:
            sim = simulaciones[prod]
            fila_optima = sim.loc[sim["Ingreso estimado"].idxmax()]
            optimos.append({
                "Producto": prod,
                "Descuento óptimo (%)": fila_optima["Descuento (%)"],
                "Precio óptimo": fila_optima["Precio"],
                "Ingreso estimado máximo": fila_optima["Ingreso estimado"],
            })

            sim_export = sim.copy()
            sim_export.insert(0, "Producto", prod)
            simulaciones_export.append(sim_export)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sim["Descuento (%)"], sim["Ingreso estimado"], marker="o")
            ax.axvline(fila_optima["Descuento (%)"], color="red", linestyle="--", label="Óptimo")
            ax.set_xlabel("Descuento (%)")
            ax.set_ylabel("Ingreso estimado")
            ax.set_title(f"Ingreso simulado vs. descuento — {prod}")
            ax.legend()
            st.pyplot(fig)

    boton_descarga(pd.concat(simulaciones_export, ignore_index=True), "pricing_simulacion_descuentos.csv")

    optimos_df = pd.DataFrame(optimos)

    subseccion(
        "Asignación de ofertas según presupuesto",
        "Ordena los productos top por ingreso incremental y marca con 'Asignar oferta = True' solo a los N mejores."
    )
    presupuesto_ofertas = st.slider(
        "¿A cuántos productos se les puede asignar oferta a la vez?", 1, len(productos_top), min(2, len(productos_top))
    )

    ingreso_actual = df.groupby(col_producto)[col_ingreso].sum()
    asignacion = optimos_df.copy()
    asignacion["Ingreso actual"] = asignacion["Producto"].map(ingreso_actual)
    asignacion["Ingreso incremental"] = asignacion["Ingreso estimado máximo"] - asignacion["Ingreso actual"]
    asignacion = asignacion.sort_values("Ingreso incremental", ascending=False).reset_index(drop=True)
    asignacion["Asignar oferta"] = False
    asignacion.loc[:presupuesto_ofertas - 1, "Asignar oferta"] = True

    st.dataframe(asignacion, width="stretch")
    boton_descarga(asignacion, "pricing_asignacion_ofertas.csv")

    with st.expander("📖 Notas sobre las fórmulas usadas"):
        st.markdown("""
        - **Elasticidad precio-demanda** = pendiente de la regresión log(cantidad) vs. log(precio).
        - **Simulación de descuento**: cantidad_nueva = cantidad_actual × (1 + elasticidad × (−%descuento)).
        - **Ingreso estimado** = precio_con_descuento × cantidad_nueva.
        - **Ingreso incremental** = ingreso estimado con el descuento óptimo − ingreso actual sin descuento.
        - ⚠️ El dataset no tiene descuentos reales aplicados; es una simulación basada en la elasticidad estimada.
        """)

st.caption("Aplicativo desarrollado para el caso adaptado — Modelamiento Predictivo de Datos, Grupo 10.")
