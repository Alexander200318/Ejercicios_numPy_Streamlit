import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ejercicio_1():
    st.header("Ejercicio 1: Estadísticas con NumPy y Pandas")
    st.subheader("Autor: Alexander Loja    M6A")

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






# ===================== NUEVA SECCIÓN: DATAFRAME EDITABLE =====================

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

# ===================== INTERFAZ PRINCIPAL =====================

st.set_page_config(page_title="App NumPy y Pandas", layout="centered")

st.title(" Aplicación con Ejercicios y Gestión de Datos")

# Menú superior
st.markdown("""
    <style>
    div[data-baseweb="select"] {
        width: 300px !important;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    opcion = st.selectbox(
        "Selecciona una sección:",
        ["Ejercicio 1", "Ejercicio 2", "Ejercicio 3", "Ejercicio 4", "Jóvenes del ciclo"]
    )

st.divider()

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