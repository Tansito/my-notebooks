# General imports
import numpy as np

# Qiskit ansatz circuits
from qiskit.circuit.library import RealAmplitudes
from qiskit.compiler import transpile

# Qiskit primitives
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit.primitives import Sampler as QiskitSampler

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

# permute pauli operator
from permute_sparse_pauli_op import permute_sparse_pauli_op

# workflows 
from workflows import QuadraticProgramPostprocess, QuadraticProgramConverter

# SPSA
from spsa import minimize_spsa

# Middleware
import os
from quantum_serverless import save_result

# Step 1

def build_model():
    mdl = Model("docplex model")
    x = mdl.binary_var("x")
    y = mdl.binary_var("y")
    z = mdl.binary_var("z")
    mdl.minimize(x*y - x*z + 2*y*z + x - 2*y - 3*z)
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
    ansatz_ibm = transpile(ansatz, basis_gates = ['cz', 'sx', 'rz'],  coupling_map =[[0, 1], [1, 2]], optimization_level=3)
    return ansatz_ibm

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
    estimator = QiskitEstimator(options={"shots": int(1e4)})
    sampler = QiskitSampler(options={"shots": int(1e4)})
    return estimator, sampler

def minimize(ansatz, ansatz_ibm, hamiltonian_ibm, estimator):
    x0 = 2*np.pi*np.random.random(size=ansatz.num_parameters)
    res = minimize_spsa(cost_func, x0, args=(ansatz_ibm, hamiltonian_ibm, estimator), maxiter=100)
    return res

def compute_distribution(ansatz_ibm, sampler, res):
    # Assign solution parameters to ansatz
    qc = ansatz_ibm.assign_parameters(res.x)
    qc.measure_all()
    samp_dist = sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]
    return samp_dist

# Step 4

def post_processing(qubo, quadratic_transformer, samp_dist):
    solution = QuadraticProgramPostprocess(qubo, quadratic_transformer).run(samp_dist)
    return solution

# Step 1

mdl = build_model()
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
print(ansatz_ibm)

hamiltonian_ibm = optimize_operators(ansatz, ansatz_ibm, hamiltonian)
print(hamiltonian_ibm)

# Step 3

estimator, sampler = setup_estimator_sampler()
res = minimize(ansatz, ansatz_ibm, hamiltonian_ibm, estimator)
print(res)

samp_dist = compute_distribution(ansatz_ibm, sampler, res)

# Step 4

solution = post_processing(qubo, quadratic_transformer, samp_dist)

save_result(solution.tolist())
