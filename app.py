import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="App NumPy y Pandas", layout="centered")

# ===================== FUNCIONES DE EJERCICIOS (SIN CAMBIOS) =====================

def ejercicio_1():
    st.header("Ejercicio 1: Estadísticas con NumPy y Pandas")
    
    Array1 = np.arange(1, 101)

    media = np.mean(Array1)
    mediana = np.median(Array1)
    varianza = np.var(Array1)
    percentil_90 = np.percentile(Array1, 90)

    # Mostrar resultados individuales
    st.subheader("Resultados individuales")
    st.write(f"Media: {media:.2f}")
    st.write(f"Mediana: {mediana}")
    st.write(f"Varianza: {varianza:.2f}")
    st.write(f"Percentil 90: {percentil_90}")

    # Crear un DataFrame con los resultados
    resultados = pd.DataFrame({
        "Estadística": ["Media", "Mediana", "Varianza", "Percentil 90"],
        "Valor": [media, mediana, varianza, percentil_90]
    })

    # Mostrar tabla con pandas
    st.subheader("Tabla de resultados")
    st.dataframe(resultados)

    # Gráfica simple del array
    st.subheader("Visualización del Array (1 al 100)")
    plt.figure(figsize=(8, 4))
    plt.plot(Array1, color="skyblue", marker="o", markersize=3)
    plt.title("Array del 1 al 100")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True)
    st.pyplot(plt)


def ejercicio_2():
    st.header("Ejercicio 2: Matriz Aleatoria 5x5 y Cálculos Numéricos")

    matriz = np.random.randn(5, 5)

    # Calcular determinante y traza
    determinante = np.linalg.det(matriz)
    traza = np.trace(matriz)

    # Mostrar matriz como tabla con pandas
    st.subheader("Matriz 5x5 generada:")
    st.dataframe(pd.DataFrame(matriz, columns=[f"C{i+1}" for i in range(5)]))

    # Mostrar resultados
    st.subheader("Resultados:")
    st.write(f" Determinante: {determinante:.4f}")
    st.write(f" Traza: {traza:.4f}")


def ejercicio_3():
    st.header("Ejercicio 3: Distribución de frecuencias (0 al 10)")

    # Generar 1000 enteros aleatorios entre 0 y 10
    valores = np.random.randint(0, 11, 1000)

    # Calcular distribución de frecuencias con pandas
    conteo = pd.Series(valores).value_counts().sort_index()

    # Crear un DataFrame con los resultados
    df_frecuencias = pd.DataFrame({
        "Número": conteo.index,
        "Frecuencia": conteo.values
    })

    # Mostrar la tabla
    st.subheader("Tabla de distribución de frecuencias")
    st.dataframe(df_frecuencias)

    # Mostrar gráfico de barras
    st.subheader("Gráfico de barras de la distribución")
    plt.figure(figsize=(8, 4))
    plt.bar(df_frecuencias["Número"], df_frecuencias["Frecuencia"], color="skyblue", edgecolor="black")
    plt.xlabel("Número")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de frecuencias de 1000 enteros (0 al 10)")
    st.pyplot(plt)


def ejercicio_4():
    st.header("Ejercicio 4: Normalización de un Vector")

    # Opción del usuario
    opcion = st.radio(
        "Selecciona cómo deseas obtener el vector:",
        ("Ingresar manualmente", "Generar aleatoriamente")
    )

    if opcion == "Ingresar manualmente":
        entrada = st.text_input(" Ingresa los valores separados por comas (ejemplo: 2, 4, 6, 8):")
        if entrada:
            try:
                v = np.array([float(x.strip()) for x in entrada.split(",")])
            except ValueError:
                st.error("Por favor ingresa solo números separados por comas.")
                return
        else:
            st.warning("Por favor ingresa un vector para continuar.")
            return
    else:
        tamaño = st.slider("Selecciona el tamaño del vector aleatorio:", 5, 20, 10)
        v = np.random.randint(0, 100, tamaño)
        st.write("🔹 Vector generado:", v)

    # Calcular normalización
    media = np.mean(v)
    desviacion = np.std(v)

    if desviacion == 0:
        st.error(" La desviación estándar es 0, no se puede normalizar.")
        return

    v_normalizado = (v - media) / desviacion

    # Mostrar resultados
    st.subheader(" Resultados:")
    st.write(f"Media: {media:.2f}")
    st.write(f"Desviación estándar: {desviacion:.2f}")

    # Crear tabla con pandas
    df = pd.DataFrame({
        "Valor Original": v,
        "Normalizado": v_normalizado
    })

    st.dataframe(df)

    # Mostrar gráfico comparativo
    st.subheader(" Gráfico comparativo")
    fig, ax = plt.subplots()
    ax.plot(v, label="Original", marker="o")
    ax.plot(v_normalizado, label="Normalizado", marker="s")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def ejercicio_estudiantes():
    st.title(" Gestión de Jóvenes del Ciclo")
    st.divider()

    # ======== INICIALIZAR DATOS ========
    if "df_estudiantes" not in st.session_state:
        data_inicial = [
            ("MARIUXI ANDREA ", "CALLE DUMAGUALA", 20, 8.5, "Matemáticas"),
            ("MAURA MILETH", " CALLE LEON", 21, 7.8, "Programación"),
            ("STEVEN ALEXANDER", " CARPIO CHILLOGALLO", 19, 9.0, "Base de Datos"),
            ("ERICK FERNANDO ", "CHACON AVILA", 22, 6.7, "Matemáticas"),
            ("EDWIN ALEXANDER", " CHOEZ DOMINGUEZ", 20, 7.9, "Base de Datos"),
            ("ADRIANA VALENTINA ", "CORNEJO ULLOA", 21, 9.1, "Matemáticas"),
            ("DAVID ALFONSO", " ESPINOZA CHÉVEZ", 22, 8.0, "Programación"),
            ("ANTHONY MAURICIO ", "FAJARDO VASQUEZ", 20, 7.5, "Base de Datos"),
            ("FREDDY ISMAEL", " GOMEZ ORDOÑEZ", 23, 8.8, "Matemáticas"),
            ("WENDY NICOLE ", "LLIVICHUZHCA MAYANCELA", 19, 9.3, "Programación"),
            ("ALEXANDER ISMAEL ", "LOJA LLIVICHUZHCA", 21, 9.0, "Base de Datos"),
            ("DAVID ALEXANDER ", "LOPEZ SALTOS", 22, 8.4, "Matemáticas"),
            ("VICTOR JONNATHAN ", "MENDEZ VILLA", 20, 7.7, "Programación"),
            ("JOHN SEBASTIAN", " MONTENEGRO CALLE", 21, 8.9, "Base de Datos"),
            ("CARMEN ELIZABETH ", "NEIRA INGA", 22, 8.1, "Matemáticas"),
            ("JOEL STALYN ", "PESANTEZ BERREZUETA", 23, 7.6, "Programación"),
            ("GILSON STALYN ", "TENEMEA AGUILAR", 20, 9.2, "Base de Datos"),
            ("KENNY ALEXANDER", " VALDIVIESO CORONEL", 21, 8.5, "Matemáticas"),
        ]
        st.session_state.df_estudiantes = pd.DataFrame(
            data_inicial,
            columns=["Nombres", "Apellidos", "Edad", "Notas", "Materias"]
        )

    df = st.session_state.df_estudiantes

    # ======== BOTONES DE ACCIÓN ========
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(" Actualizar Lista", use_container_width=True):
            st.rerun()
    with col2:
        if st.button(" Cargar CSV", use_container_width=True):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Archivo CSV", csv, "estudiantes.csv", "text/csv")

    st.divider()

    # ======== FORMULARIO PARA AGREGAR ========
    st.subheader(" Agregar Nuevo Joven")

    with st.form("form_agregar", clear_on_submit=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            nombre = st.text_input("Nombre")
        with col2:
            apellido = st.text_input("Apellido")
        with col3:
            edad = st.number_input("Edad", min_value=15, max_value=35, step=1)
        with col4:
            nota = st.number_input("Nota", min_value=0.0, max_value=10.0, step=0.1)
        with col5:
            materia = st.text_input("Materia")

        agregar = st.form_submit_button(" Agregar Joven")

        if agregar:
            if nombre and apellido and materia:
                nuevo = pd.DataFrame([[nombre, apellido, edad, nota, materia]],
                                     columns=["Nombres", "Apellidos", "Edad", "Notas", "Materias"])
                st.session_state.df_estudiantes = pd.concat([df, nuevo], ignore_index=True)
                st.success(f"Joven {nombre} agregado correctamente ")
                st.rerun()
            else:
                st.warning(" Debes llenar al menos los campos: Nombre, Apellido y Materia.")

    st.divider()

    # ======== ELIMINAR JOVEN ========
    st.subheader(" Eliminar Joven")
    if not df.empty:
        col1, col2 = st.columns([3, 3])
        with col1:
            seleccion = st.selectbox("Selecciona el joven a eliminar:", df["Nombres"] + " " + df["Apellidos"])
        with col2:
            if st.button(" Eliminar ", use_container_width=True):
                idx = df.index[(df["Nombres"] + " " + df["Apellidos"]) == seleccion][0]
                st.session_state.df_estudiantes = df.drop(index=idx).reset_index(drop=True)
                st.success(f"Joven {seleccion} eliminado correctamente 🧹")
                st.rerun()
    else:
        st.info("No hay jóvenes registrados actualmente.")

    st.divider()

    # ======== MOSTRAR TABLA ========
    st.subheader(" Lista de Jóvenes")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.info(f"Total de jóvenes: {len(df)}")


# =====================================================
# 🎯 CREAR DATASET BASE
# =====================================================
def crear_dataset():
    """Crea el DataFrame base para todos los ejercicios"""
    datos = {
        "fecha": pd.date_range("2024-01-01", periods=12, freq="ME"),
        "categoria": ["A", "B", "C"] * 4,
        "producto": ["P1", "P2", "P3"] * 4,
        "precio": [10, 12, 9, 11, 10, 8, 12, 14, 9, 13, 12, 10],
        "cantidad": [5, 3, 6, 2, 8, 1, 4, 5, 7, 3, 2, 6],
    }
    df = pd.DataFrame(datos)
    # Calcular ventas (precio * cantidad)
    df['ventas'] = df['precio'] * df['cantidad']
    return df


# =====================================================
# 1️⃣ Cargar CSV y mostrar las primeras 10 filas
# =====================================================
def ejercicio_1_panda():
    st.subheader("📊 Ejercicio 1: Dataset de Ventas")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💡 Usando dataset predefinido de ventas 2024")
    
    with col2:
        if st.button("🔄 Cargar Dataset"):
            df = crear_dataset()
            st.session_state['df'] = df
            st.success("✅ Dataset cargado")
    
    # Si existe el dataset, mostrarlo
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.write("**📋 Primeras 10 filas del DataFrame:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write("**📊 Información del Dataset:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", len(df))
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            st.metric("Productos", df['producto'].nunique())
        with col4:
            st.metric("Ventas Total", f"${df['ventas'].sum()}")


# =====================================================
# 2️⃣ Calcular venta total por producto (ordenada)
# =====================================================
def ejercicio_2_panda(df):
    st.subheader("📈 Ejercicio 2: Venta Total por Producto")
    
    if "producto" in df.columns and "ventas" in df.columns:
        # Calcular total por producto
        total_por_producto = df.groupby("producto")["ventas"].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**💰 Ventas por Producto:**")
            resultado_df = pd.DataFrame({
                'Producto': total_por_producto.index,
                'Ventas Totales': total_por_producto.values
            })
            st.dataframe(resultado_df, use_container_width=True)
        
        with col2:
            st.write("**📊 Gráfico de Barras:**")
            fig, ax = plt.subplots(figsize=(8, 5))
            total_por_producto.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title("Ventas Totales por Producto", fontsize=14, fontweight='bold')
            ax.set_ylabel("Ventas ($)", fontsize=12)
            ax.set_xlabel("Producto", fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()
    else:
        st.error("❌ El DataFrame debe tener las columnas 'producto' y 'ventas'.")


# =====================================================
# 3️⃣ Identificar valores faltantes y aplicar imputación
# =====================================================
def ejercicio_3_panda(df):
    st.subheader("🔧 Ejercicio 3: Imputación de Valores Faltantes")
    
    # Crear una copia con valores faltantes para demostración
    df_con_nulos = df.copy()
    
    # Agregar valores nulos aleatorios
    if st.button("🎲 Generar valores faltantes aleatorios"):
        np.random.seed(42)
        for col in ['precio', 'cantidad', 'ventas']:
            indices = np.random.choice(df_con_nulos.index, size=2, replace=False)
            df_con_nulos.loc[indices, col] = np.nan
        st.session_state['df_con_nulos'] = df_con_nulos
        st.success("✅ Valores nulos generados en precio, cantidad y ventas")
    
    # Usar el DataFrame con nulos si existe
    df_trabajo = st.session_state.get('df_con_nulos', df_con_nulos)
    
    # Mostrar valores faltantes
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔍 Valores faltantes por columna:**")
        nulos_df = pd.DataFrame({
            'Columna': df_trabajo.columns,
            'Valores Nulos': df_trabajo.isnull().sum().values,
            'Porcentaje': (df_trabajo.isnull().sum().values / len(df_trabajo) * 100).round(2)
        })
        st.dataframe(nulos_df, use_container_width=True)
    
    with col2:
        st.write("**⚙️ Configuración de Imputación:**")
        estrategia = st.selectbox(
            "Selecciona la estrategia:",
            ["media", "mediana", "moda"],
            help="Método para reemplazar valores faltantes"
        )
        
        if st.button("✨ Aplicar Imputación"):
            df_imputado = df_trabajo.copy()
            
            for col in df_imputado.select_dtypes(include=[np.number]).columns:
                if df_imputado[col].isnull().sum() > 0:
                    if estrategia == "media":
                        valor = df_imputado[col].mean()
                    elif estrategia == "mediana":
                        valor = df_imputado[col].median()
                    else:
                        valor = df_imputado[col].mode()[0]
                    
                    df_imputado[col] = df_imputado[col].fillna(valor)
            
            st.session_state['df'] = df_imputado
            st.success(f"✅ Valores imputados usando la {estrategia}")
            
            st.write("**📊 DataFrame después de la imputación:**")
            st.dataframe(df_imputado, use_container_width=True)
            
            return df_imputado


# =====================================================
# 4️⃣ Crear tabla dinámica (ventas por mes y producto)
# =====================================================
def ejercicio_4_panda(df):
    st.subheader("📅 Ejercicio 4: Tabla Dinámica - Ventas por Mes y Producto")

    if "fecha" in df.columns and "producto" in df.columns and "ventas" in df.columns:
        # Preparar datos
        df_copia = df.copy()
        df_copia["fecha"] = pd.to_datetime(df_copia["fecha"], errors="coerce")
        df_copia["mes"] = df_copia["fecha"].dt.month_name()
        
        # Crear tabla dinámica
        tabla = pd.pivot_table(
            df_copia, 
            values="ventas", 
            index="mes", 
            columns="producto", 
            aggfunc="sum", 
            fill_value=0
        )
        
        # Reordenar meses cronológicamente
        orden_meses = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        tabla = tabla.reindex([m for m in orden_meses if m in tabla.index])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📊 Tabla Dinámica:**")
            st.dataframe(tabla.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
            
            # Estadísticas adicionales
            st.write("**📈 Totales por Producto:**")
            totales = tabla.sum().sort_values(ascending=False)
            st.dataframe(totales, use_container_width=True)
        
        with col2:
            st.write("**📉 Gráfico de Ventas:**")
            fig, ax = plt.subplots(figsize=(8, 6))
            tabla.plot(kind="bar", ax=ax, width=0.8)
            ax.set_title("Ventas Mensuales por Producto", fontsize=14, fontweight='bold')
            ax.set_xlabel("Mes", fontsize=12)
            ax.set_ylabel("Ventas ($)", fontsize=12)
            ax.legend(title='Producto', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.error("❌ El DataFrame debe tener las columnas 'fecha', 'producto' y 'ventas'.")


# =====================================================
# 5️⃣ MERGE entre DataFrames
# =====================================================
def ejercicio_5_panda():
    st.subheader("🔗 Ejercicio 5: Merge entre DataFrames")
    
    st.info("💡 Vamos a combinar información de productos con sus ventas")
    
    # Crear DataFrames de ejemplo
    productos_df = pd.DataFrame({
        'producto': ['P1', 'P2', 'P3'],
        'nombre_completo': ['Laptop', 'Mouse', 'Teclado'],
        'stock': [50, 200, 150]
    })
    
    # Usar datos del dataset principal si existe
    if 'df' in st.session_state:
        ventas_df = st.session_state['df'][['producto', 'ventas', 'cantidad']].copy()
    else:
        ventas_df = crear_dataset()[['producto', 'ventas', 'cantidad']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📦 Tabla Productos:**")
        st.dataframe(productos_df, use_container_width=True)
    
    with col2:
        st.write("**💰 Tabla Ventas:**")
        st.dataframe(ventas_df.head(6), use_container_width=True)
    
    st.write("---")
    
    # Configurar merge
    tipo_merge = st.selectbox(
        "Selecciona el tipo de merge:",
        ['left', 'right', 'inner', 'outer'],
        help="left: mantiene todos de la izquierda | inner: solo coincidencias"
    )
    
    if st.button("🔀 Realizar Merge"):
        # Merge
        resultado = pd.merge(ventas_df, productos_df, on='producto', how=tipo_merge)
        
        st.success(f"✅ Merge realizado con éxito usando '{tipo_merge}' join")
        
        st.write("**📊 Resultado del Merge:**")
        st.dataframe(resultado, use_container_width=True)
        
        # Validación
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Registros resultado", len(resultado))
        with col2:
            st.metric("📦 Registros productos", len(productos_df))
        with col3:
            st.metric("💰 Registros ventas", len(ventas_df))


# ===================== MENÚ LATERAL POR CATEGORÍAS =====================

with st.sidebar:
    st.title("Menú de Navegación")

    
    # Seleccionar categoría principal
    categoria = st.radio(
        "Selecciona una categoría:",
        [" Ejercicios NumPy", " Gestión de Datos", " Ejercicios Pandas"]
    )
    
    st.write("---")
    
    # Mostrar opciones según la categoría seleccionada
    if categoria == " Ejercicios NumPy":
        opcion = st.radio(
            "Selecciona un ejercicio:",
            [
                "Ejercicio 1",
                "Ejercicio 2",
                "Ejercicio 3",
                "Ejercicio 4"
            ]
        )

    elif categoria == " Gestión de Datos":  # Gestión de Datos
        opcion = st.radio(
            "Selecciona una opción:",
            [
                "Jóvenes del ciclo"
            ]
        )
    
        
    elif categoria == " Ejercicios Pandas":  # Gestión de Datos
        opcion = st.radio(
            "Selecciona una opción:",
            [
                "Ejercicio 1_panda",
                "Ejercicio 2_panda",
                "Ejercicio 3_panda",
                "Ejercicio 4_panda",
                "Ejercicio 5_panda"
            ]
        )
    
    
    st.write("---")
    st.write("**Autor:** Alexander Loja")
    st.write("**Curso:** M6A")

# ===================== CONTENIDO PRINCIPAL =====================


# Mostrar la opción seleccionada
if opcion == "Ejercicio 1":
    ejercicio_1()
elif opcion == "Ejercicio 2":
    ejercicio_2()
elif opcion == "Ejercicio 3":
    ejercicio_3()
elif opcion == "Ejercicio 4":
    ejercicio_4()
elif opcion == "Jóvenes del ciclo":
    ejercicio_estudiantes()

# ===================== EJERCICIOS PANDAS =====================
elif opcion == "Ejercicio 1_panda": 
    ejercicio_1_panda()

elif opcion == "Ejercicio 2_panda":
    if 'df' in st.session_state:
        ejercicio_2_panda(st.session_state['df'])  # ← Pasar el DataFrame
    else:
        st.warning("⚠️ Primero carga el dataset en el Ejercicio 1")
        st.info("👉 Ve al Ejercicio 1 y presiona 'Cargar Dataset'")

elif opcion == "Ejercicio 3_panda":
    if 'df' in st.session_state:
        ejercicio_3_panda(st.session_state['df'])  # ← Pasar el DataFrame
    else:
        st.warning("⚠️ Primero carga el dataset en el Ejercicio 1")
        st.info("👉 Ve al Ejercicio 1 y presiona 'Cargar Dataset'")

elif opcion == "Ejercicio 4_panda":
    if 'df' in st.session_state:
        ejercicio_4_panda(st.session_state['df'])  # ← Pasar el DataFrame
    else:
        st.warning("⚠️ Primero carga el dataset en el Ejercicio 1")
        st.info("👉 Ve al Ejercicio 1 y presiona 'Cargar Dataset'")

elif opcion == "Ejercicio 5_panda": 
    ejercicio_5_panda()