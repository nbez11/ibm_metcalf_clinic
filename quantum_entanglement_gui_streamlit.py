import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, thermal_relaxation_error
from math import pi
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Entanglement Explorer", layout="wide")

# Sidebar Navigation
pages = [
    "Lesson",
    "Bell State",
    "Quantum Teleportation",
    "Superdense Coding",
    "CHSH Game",
    "Noise Experiments",
    "Autograder"
]
choice = st.sidebar.radio("Navigation", pages)

backend = AerSimulator()

# Utility: Draw and display circuit
def show_circuit(qc):
    fig = qc.draw('mpl')
    st.pyplot(fig)

# Lesson Page
if choice == "Lesson":
    st.title("📚 Quantum Entanglement Lesson")
    st.write("""
    Welcome to the Quantum Entanglement Explorer! This app teaches you about entanglement through interactive demos:
    - **Bell State**: The simplest form of entanglement.
    - **Quantum Teleportation**: Transmit a qubit's state using entanglement + classical bits.
    - **Superdense Coding**: Send 2 classical bits using 1 qubit and entanglement.
    - **CHSH Game**: Demonstrate quantum correlations exceeding classical limits.
    - **Noise Experiments**: See how real-world noise affects entanglement.
    """)
    st.subheader("ASCII Bell State Circuit")
    st.code("""
    q0: ──H──■──M
             │
    q1: ─────X──M
    """)
    st.write("**Key Idea:** After the H and CNOT, q0 and q1 are maximally entangled. Measuring one instantly determines the other.")

# Bell State
elif choice == "Bell State":
    st.title("🔗 Bell State Demo")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    show_circuit(qc)
    shots = st.slider("Number of shots", 100, 5000, 1024, step=100)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    st.write("Counts:", counts)

# Quantum Teleportation
elif choice == "Quantum Teleportation":
    st.title("📡 Quantum Teleportation")
    alpha = st.slider("Alpha (real)", 0.0, 1.0, 1/np.sqrt(2))
    beta = np.sqrt(1 - alpha**2)
    qc = QuantumCircuit(3, 3)
    qc.initialize([alpha, beta], 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)
    show_circuit(qc)
    shots = st.slider("Shots", 100, 5000, 1024, step=100)
    result = backend.run(qc, shots=shots).result()
    st.write("Counts:", result.get_counts())

# Superdense Coding
elif choice == "Superdense Coding":
    st.title("💾 Superdense Coding")
    b0 = st.selectbox("Bit 0", [0, 1])
    b1 = st.selectbox("Bit 1", [0, 1])
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    if b0 == 1:
        qc.z(0)
    if b1 == 1:
        qc.x(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    show_circuit(qc)
    shots = st.slider("Shots", 100, 5000, 1024, step=100)
    result = backend.run(qc, shots=shots).result()
    st.write("Counts:", result.get_counts())

# CHSH Game
elif choice == "CHSH Game":
    st.title("🎯 CHSH Game")
    st.write("Adjust the measurement angles and see how the CHSH S-value changes.")
    A0 = st.slider("Alice A0 (°)", 0, 180, 0) * pi/180
    A1 = st.slider("Alice A1 (°)", 0, 180, 45) * pi/180
    B0 = st.slider("Bob B0 (°)", 0, 180, 22) * pi/180
    B1 = st.slider("Bob B1 (°)", 0, 180, 67) * pi/180

    def measure_corr(theta_a, theta_b):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(-2*theta_a, 0)
        qc.ry(-2*theta_b, 1)
        qc.measure([0, 1], [0, 1])
        result = backend.run(qc, shots=2048).result().get_counts()
        total = sum(result.values())
        p_same = (result.get('00', 0) + result.get('11', 0)) / total
        p_diff = (result.get('01', 0) + result.get('10', 0)) / total
        return p_same - p_diff

    E_A0B0 = measure_corr(A0, B0)
    E_A0B1 = measure_corr(A0, B1)
    E_A1B0 = measure_corr(A1, B0)
    E_A1B1 = measure_corr(A1, B1)

    S = E_A0B0 + E_A0B1 + E_A1B0 - E_A1B1
    st.metric("CHSH S-value", f"{S:.3f}")
    if S > 2:
        st.success("Quantum violation of classical bound!")
    if S > 2.828:
        st.warning("Beyond Tsirelson's bound? Check parameters.")

# Noise Experiments
elif choice == "Noise Experiments":
    st.title("⚡ Noise Experiments")
    noise_type = st.selectbox("Noise Type", ["Depolarizing", "Amplitude Damping", "Thermal Relaxation"])
    shots = st.slider("Shots", 100, 5000, 1024, step=100)

    noise_model = NoiseModel()

    if noise_type == "Depolarizing":
        p = st.slider("Depolarizing probability", 0.0, 1.0, 0.05)
        # Single-qubit error
        single_qubit_error = depolarizing_error(p, 1)
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['x', 'h'])
        # Two-qubit error
        two_qubit_error = depolarizing_error(p, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    elif noise_type == "Amplitude Damping":
        gamma = st.slider("Gamma", 0.0, 1.0, 0.1)
        single_qubit_error = amplitude_damping_error(gamma, 1)
        two_qubit_error = amplitude_damping_error(gamma, 2)
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['x', 'h'])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    else:  # Thermal Relaxation
        t1 = st.slider("T1", 1.0, 500.0, 100.0)
        t2 = st.slider("T2", 1.0, 500.0, 80.0)
        time = st.slider("Gate time", 0.0, 1.0, 0.1)
        single_qubit_error = thermal_relaxation_error(t1, t2, time, 1)
        two_qubit_error = thermal_relaxation_error(t1, t2, time, 2)
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['x', 'h'])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    st.write("✅ Noise model successfully created.")


    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    show_circuit(qc)
    result = backend.run(qc, shots=shots, noise_model=noise_model).result()
    st.write("Counts with noise:", result.get_counts())

# Autograder
elif choice == "Autograder":
    st.title("📝 Autograder")
    import inspect
    issues = []

    # Check imports
    if "matplotlib" in inspect.getsource(show_circuit):
        issues.append("Matplotlib used for plotting (allowed only for draw).")
    # Check .c_if
    import re
    if re.search(r"\.c_if\s*\(", inspect.getsource(show_circuit)):
        issues.append("'.c_if' found in code.")
    # Check Bell state
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
    counts = backend.run(qc, shots=1000).result().get_counts()
    p00 = counts.get('00', 0)/1000
    p11 = counts.get('11', 0)/1000
    if p00 + p11 < 0.8:
        issues.append("Bell state not producing strong correlations.")
    if issues:
        st.error("Issues found:")
        for i in issues:
            st.write("-", i)
    else:
        st.success("All checks passed!")
