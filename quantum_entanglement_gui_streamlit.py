import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, thermal_relaxation_error
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate, QFT
from math import pi, gcd, floor, log
import matplotlib.pyplot as plt
from fractions import Fraction

st.set_page_config(page_title="Quantum Entanglement Explorer", layout="wide")

# Sidebar Navigation
pages = [
    "Lesson",
    "Bell State",
    "Quantum Teleportation",
    "Superdense Coding",
    "CHSH Game",
    "SKQD Algorithm",
    "Grover's Algorithm",
    "Shor's Algorithm",
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

#SKQD Algorithm
elif choice == "SKQD Algorithm":
    st.title("Sample-based Krylov Quantum Diagonalization (SKQD)")
    st.link_button("Check out the IBM article this section is based on", "https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/skqd")
    st.write("""
        The Sample-based Krylov Quantum Diagonalization (SKQD) algorithm is a
        hybrid quantum-classical method used to estimate the energy levels
        (eigenvalues) of a quantum system's Hamiltonian (a Hamiltonian describes
        all energies, interactions, and rules of a system).
    """)
    
    st.subheader("How it works")
    st.write("""
    1. Build a Krylov Subspace  
        - Start with a simple state |v⟩.  
        - Apply the Hamiltonian H repeatedly to generate states:  
            |v⟩, H|v⟩, H²|v⟩, …  
        - These states form a small (example)subspace that captures the important
            physics.

    2. Estimate Overlaps on a Quantum Computer  
        - We need to find out how similar the states are to each other.  
        - Using quantum circuits (like swap tests), we estimate the inner products:  
        - Sᵢⱼ = ⟨vᵢ | vⱼ⟩  
        - Hᵢⱼ = ⟨vᵢ | H | vⱼ⟩  
        - Because our quantum computers can only give probabilistic results, we 
            repeat experiments many times (this is the sample-based part).

    3. Classical Diagonalization  
        - The measured overlaps form a small matrix.  
        - A classical computer diagonalizes this small matrix to approximate the
            true energy levels (eigenvalues) of the big Hamiltonian.
    """)

    st.subheader("Role of Entanglement")
    st.write("""
    - Entanglement is used in SKQD during the overlap estimation step. 
    - To compare two quantum states (for example |vᵢ⟩ and |vⱼ⟩), we need 
        circuits like the swap test. These circuits rely on creating entanglement 
        between an ancilla qubit (a helper qubit) and the system qubits. The 
        expectation value of the ancilla quibit determines the value overlap
        of the quantum states.  
    - This entanglement lets us extract information about how similar two states 
        are, without fully measuring or destroying them. 
    - In other words, entanglement links multiple states together so that we can
        measure how similar they are to eachother.
    - Without entanglement, we would not be able to build the Krylov subspace
        to efficiently estimates the overlaps of the states.
    """)

    st.subheader("Why it is important")
    st.write("""
    - Avoids heavy quantum algorithms: Unlike full quantum phase estimation,
        SKQD works on near-term (NISQ) devices. In essence instead of using the 
        extremely deep circuits that quantum phase estimation uses to find energy 
        levels, SKQD can use much shallower circtuis that use more practical
        repeated sample.
    - Efficient: A few Krylov states often give very accurate energy estimates.
    - Applications: 
        -Quantum chemistry: it allows for the ground state energy of 
            molecules of complex sysetms to be calculated.
        - Materials: science: as it may allow us to better understand the flow of
            electrons by knowing their quantum properties which may lead to 
            efficientcy gains.
    """)
    
    st.subheader("In short:")  
    st.write("""
    SKQD lets a quantum computer provide just enough information (via sampling) so
    that a classical computer can do the hard math of diagonalizing the system's
    Hamiltonian. To give an analogy that may be a bit easier to understand, we
    begin with a single state before repeatedly applying a chain of the system's
    rules. This in turn generates a chain of related states which in turn allows
    us to use difference combinations of them in order to created the krylov subspace.
    From this subspace we can then analyze it in order to understand the physics
    of the larger system. Once this is achieved we can then switch back to classical
    computing in order to diagonalize the states in order to find the energy of
    our total system. 
    """)
    
# Grover
elif choice == "Grover's Algorithm":
    st.title("Grover's Algorithm")
    st.link_button("Check out the IBM article this section is based on", "https://quantum.cloud.ibm.com/docs/en/tutorials/grovers-algorithm")
    st.write("""
    Grover’s algorithm is a quantum search method that provides a quadratic speedup 
    for finding marked states in an unsorted database.

    Key Ideas:
    - Oracle: Marks the solution states by flipping their phase (meaning a state
             with a phase of -1).
    - Applification circuit/Diffusion operator: Amplifies the marked states’ amplitudes.
    - Iteration: Repeated oracle + applification circuit increases the probability of measuring a solution.
    - After about √N iterations (where N = 2^n), the marked states are aplified so much that it has the
             highest probability of being measured
    """)

    st.subheader("Role of Entanglement")
    st.write("""
    - Grover’s algorithm uses entanglement because the oracle and diffusion operators 
      require multi-qubit operations (such as multi-controlled gates).
    - The gates entangle qubits, making their amplitudes to become combined and interfered with.
    - This interference is what allows Grover’s algorithm to amplify certain
        marked states and suppress the non-marked states.
    """)

    # User input
    num_qubits = st.slider("Number of qubits", 2, 4, 3)
    marked_states_input = st.text_input(
        "Enter marked states (comma-separated, e.g. 011,100)", "011,100"
    )
    marked_states = [s.strip() for s in marked_states_input.split(",") if s.strip()]

    def grover_oracle(marked_states, n_qubits):
        qc = QuantumCircuit(n_qubits)
        for state in marked_states:
            rev_state = state[::-1]
            zero_inds = [i for i, bit in enumerate(rev_state) if bit == "0"]
            if zero_inds:
                qc.x(zero_inds)
            if n_qubits > 1:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            else:
                qc.z(0)
            if zero_inds:
                qc.x(zero_inds)
        return qc

    # Build Diffusion Operator
    def diffusion_operator(n_qubits):
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        if n_qubits > 1:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        else:
            qc.z(0)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    optimal_iter = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(len(marked_states) / 2**num_qubits)))))
    st.write(f"Optimal number of Grover iterations: **{optimal_iter}**")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = grover_oracle(marked_states, num_qubits)
    diffusion = diffusion_operator(num_qubits)

    for _ in range(optimal_iter):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure(range(num_qubits), range(num_qubits))

    show_circuit(qc)
    # Run on AerSimulator
    shots = st.slider("Shots", 100, 5000, 1024, step=100)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    st.write("Counts:", counts)

    st.write("""
    Notice that the marked states appear with higher probability 
    than the unmarked states due to amplitude amplification.
    """)

# Shor's Algo
elif choice == "Shor's Algorithm":
    st.title("Shor's Algorithm")
    st.link_button("Check out the IBM article this section is based on", "https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm")
    st.write("""
    This tab reproduces the IBM tutorial’s approach:
    1) Build order-finding as phase estimation for the modular-multiply unitary $\(M_a\)$ with $\(N=15, a=2\)$.
    2) Run on a simulator to get the counting-register distribution (peaks near multiples of $\(k/r\)$).
    3) Use continued fractions to recover the order $\(r\)$, then compute non-trivial factors via
       $\\(\\gcd(a^{r/2} \\pm 1, N)\\)$.
    """)

    N = 15
    a = 2  # follow the IBM tutorial exactly

    # Minimal swap-based implementations for M2 and M4 (mod 15)
    def M2mod15():
        """Permutation for b=2 (mod 15) implemented with swaps (as in IBM tutorial)."""
        U = QuantumCircuit(4)
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)
        U = U.to_gate()
        U.name = "M_2"
        return U

    def M4mod15():
        """Permutation for b=4 (mod 15) implemented with swaps (as in IBM tutorial)."""
        U = QuantumCircuit(4)
        U.swap(1, 3)
        U.swap(0, 2)
        U = U.to_gate()
        U.name = "M_4"
        return U

    def a2kmodN(a, k, N):
        """Compute a^(2^k) mod N by repeated squaring."""
        val = a
        for _ in range(k):
            val = (val * val) % N
        return val

    # Qubit counts: 4 target qubits for N=15; 8 control qubits gives good precision
    num_target = floor(log(N - 1, 2)) + 1  # = 4
    num_control = st.slider("Counting (control) qubits m", 5, 10, 8)
    shots = st.slider("Shots", 100, 5000, 1024, step=100)

    # Build list of needed Mb unitaries for b = a^(2^k) mod N, k=0..m-1
    k_list = range(num_control)
    b_list = [a2kmodN(a, k, N) for k in k_list]

    st.caption(f"b values (a^(2^k) mod {N}) for a={a}: {b_list}")

    # Construct the phase estimation (order finding) circuit
    control = QuantumRegister(num_control, name="C")
    target = QuantumRegister(num_target, name="T")
    creg = ClassicalRegister(num_control, name="out")
    circuit = QuantumCircuit(control, target, creg)

    # Prepare |1> on the target register (|0001⟩ in little-endian is X on T0)
    circuit.x(num_control)  # note: T qubits follow C in allocation; this targets T[0]

    # Hadamards on control; controlled-M_b per control qubit (skip identities b=1)
    for k, q in enumerate(control):
        circuit.h(q)
        b = b_list[k]
        if b == 2:
            circuit.compose(M2mod15().control(), qubits=[q] + list(target), inplace=True)
        elif b == 4:
            circuit.compose(M4mod15().control(), qubits=[q] + list(target), inplace=True)
        else:
            # b == 1 acts as identity; nothing to add
            pass

    # Inverse QFT on the control register, then measure it
    circuit.compose(QFT(num_control, inverse=True), qubits=control, inplace=True)
    circuit.measure(control, creg)

    st.subheader("Circuit")
    show_circuit(circuit)

    #Run on your AerSimulator backend
    result = backend.run(circuit, shots=shots).result()
    counts = result.get_counts()

    st.subheader("Counts (control register)")
    st.write(counts)

    # Post-processing: continued fractions to estimate r
    # Convert each measured bitstring to decimal and phase
    rows = []
    phases = []
    for bitstr, cnt in counts.items():
        dec = int(bitstr, 2)  # control register shown MSB->LSB; IBM tutorial uses int(bitstr,2)
        phase = dec / (2**num_control)
        phases.append((bitstr, dec, phase, cnt))
        rows.append([bitstr, dec, f"{dec}/{2**num_control}", f"{phase:.4f}", cnt])

    st.subheader("Measured phases")
    st.table(rows)

    # Guess r values from phases via continued fractions (limit denominator by N)
    guesses = []
    for bitstr, dec, phase, cnt in phases:
        frac = Fraction(phase).limit_denominator(N)
        r_guess = frac.denominator
        guesses.append((bitstr, phase, f"{frac.numerator}/{frac.denominator}", r_guess, cnt))

    st.subheader("Continued-fractions estimates for r")
    st.table([[b, f"{ph:.4f}", fr, r, c] for (b, ph, fr, r, c) in guesses])

    # Try to recover factors using any even r guess (and phase != 0)
    def try_factor_from_r(a, N, r):
        if r <= 0 or r % 2 == 1:
            return None
        x = pow(a, r // 2, N)
        f1, f2 = gcd(x - 1, N), gcd(x + 1, N)
        nontrivial = sorted({f for f in (f1, f2) if f not in (1, N)})
        return nontrivial or None

    found = set()
    for _, phase, _, r_guess, _ in guesses:
        if phase == 0.0:
            continue
        facs = try_factor_from_r(a, N, r_guess)
        if facs:
            for f in facs:
                found.add(f)
    if found:
        st.success(f"Non-trivial factor(s) recovered: {sorted(found)}  (Product {np.prod(sorted(found))})")
    else:
        st.info("No non-trivial factors from this run. Increase shots or try again — some outcomes give r=1 or an odd divisor of r.")


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
