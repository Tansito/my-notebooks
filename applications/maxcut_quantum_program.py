# General imports
import math
import numpy as np

# Dynamical decoupling imports
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

# Qiskit ansatz circuits
from qiskit.circuit.library import RealAmplitudes
from qiskit.compiler import transpile

# Qiskit primitives
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_aer.primitives import Sampler as AerSampler

# Qiskit runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler, Session

# quadratic_program
from quadratic_program import QuadraticProgram

# Docplex - classical description of optimization problems
from docplex.mp.model import Model

# translations
from translators import docplex_mp_to_qp
from translators import qubo_to_sparse_pauli_op

from permute_sparse_pauli_op import permute_sparse_pauli_op

# workflows 
from workflows import QuadraticProgramPostprocess, QuadraticProgramConverter

# SPSA
from spsa import minimize_spsa

# rustworkx graph library
import rustworkx as rs
from rustworkx.visualization import mpl_draw

# Middleware
import os
from quantum_serverless import save_result

# Step 1

service = QiskitRuntimeService(
    channel="ibm_quantum", 
    token="<YOUR_IBM_TOKEN_HERE>"
)

backend = service.get_backend('ibm_kyiv')

def random_adjacency_graph(N, density=0.5):
    """Build random adjacency graph of a given density

    Parameters:
        N (int): Matrx dimension
        density (float): Density of non-zero elements, default=0.5

    Returns:
        ndarray: Adjacency matrix as NumPy array
    """
    off_elems = N*(N-1)//2
    num_elems = math.ceil(off_elems * density)
    inds = np.sort(np.random.choice(off_elems, size=num_elems, replace=False))

    M = np.zeros((N, N), dtype=float)
    for k in inds:
        i = N - 2 - int(math.sqrt(-8*k + 4*N*(N-1)-7)/2 - 0.5)
        j = (k + i + 1 - N*(N-1)//2 + (N-i)*((N-i)-1)//2)
        M[i,j] = 1

    M = M + M.T
    return M

def build_model(G):
    mdl = Model(name="Max-cut")
    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(G.num_nodes())}
    objective = mdl.sum(w * x[i] * (1 - x[j]) + w * x[j] * (1 - x[i]) for i, j, w in G.weighted_edge_list())
    mdl.maximize(objective)
    return mdl

def build_quadratic_program(mdl):
    qp = docplex_mp_to_qp(mdl)
    return qp

def build_quadratic_transformer(qp):
    quadratic_transformer = QuadraticProgramConverter()
    qubo = quadratic_transformer.run(qp)
    hamiltonian, offset = qubo_to_sparse_pauli_op(qubo)
    return quadratic_transformer, qubo, hamiltonian

def quantum_solution_setup(hamiltonian):
    ansatz = RealAmplitudes(hamiltonian.num_qubits, entanglement = 'linear', reps=2)
    return ansatz

# Step 2

def optimize_circuits(ansatz):
    ansatz_ibm = transpile(ansatz, backend=backend, optimization_level=3, scheduling_method='alap')
    return ansatz_ibm


def apply_backend_constraints(ansatz_ibm):
    durations = InstructionDurations.from_backend(backend)
    constraints = backend.configuration().timing_constraints
    
    # CPMG sequence
    dd_sequence = [XGate(), XGate()]
    # Fraction of duration to place in-between DD sequence gates
    spacing = [1/4, 1/2, 1/4]
    
    pm = PassManager([ALAPScheduleAnalysis(durations),
                  PadDynamicalDecoupling(durations, dd_sequence, spacing=spacing,
                                         pulse_alignment=constraints['pulse_alignment'])])
    dd_circs = pm.run(ansatz_ibm)
    return dd_circs

def optimize_operators(ansatz, ansatz_ibm, hamiltonian):
    layout = ansatz_ibm.layout.initial_layout
    hamiltonian_ibm = permute_sparse_pauli_op(hamiltonian,layout, ansatz.qubits)
    return hamiltonian_ibm

# Step 3

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return cost

def setup_estimator_sampler():
    #session = Session(backend=backend)
    #estimator = Estimator(session=session, options={"shots": int(1e4)})
    #sampler = Sampler(session=session, options={"shots": int(1e4)})
    #estimator = QiskitEstimator(options={"shots": int(1e4)})
    #sampler = QiskitSampler(options={"shots": int(1e4)})

    estimator = AerEstimator()
    sampler = AerSampler()
    return estimator, sampler

def minimize(dd_circs, hamiltonian_ibm, estimator):
    x0 = 2*np.pi*np.random.random(size=dd_circs.num_parameters)
    res = minimize_spsa(cost_func, x0, args=(dd_circs, hamiltonian_ibm, estimator), maxiter=5)
    return res

def compute_distribution(dd_circs, sampler, res):
    # Assign solution parameters to ansatz
    qc = dd_circs.assign_parameters(res.x)
    qc.measure_all()
    samp_dist = sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]
    # Close the session since we are now done with it
    #session.close()
    return samp_dist

# Step 4

def post_processing(qubo, quadratic_transformer, samp_dist):
    solution = QuadraticProgramPostprocess(qubo, quadratic_transformer).run(samp_dist)
    return solution  

# Step 1

N = 10
density = 0.6
M = random_adjacency_graph(N, density)
G = rs.PyGraph.from_adjacency_matrix(M)
mpl_draw(G, with_labels=True, node_color='cyan')

mdl = build_model(G)
print(mdl.export_as_lp_string())

qp = build_quadratic_program(mdl)
print(qp.prettyprint())

quadratic_transformer, qubo, hamiltonian = build_quadratic_transformer(qp)
print(qubo.prettyprint())
print(hamiltonian)

ansatz = quantum_solution_setup(hamiltonian)
print(ansatz.decompose())

# Step 2

ansatz_ibm = optimize_circuits(ansatz)
print(ansatz_ibm.depth())

dd_circs = apply_backend_constraints(ansatz_ibm)
dd_circs.draw(idle_wires=False)

hamiltonian_ibm = optimize_operators(ansatz, ansatz_ibm, hamiltonian)

# Step 3

estimator, sampler = setup_estimator_sampler()
res = minimize(dd_circs, hamiltonian_ibm, estimator)
print(res)

samp_dist = compute_distribution(dd_circs, sampler, res)

# Step 4

solution = post_processing(qubo, quadratic_transformer, samp_dist)

save_result({
    "matrix": M.tolist(),
    "solution": solution.tolist()
})
