# General imports
import numpy as np

# Qiskit ansatz circuits
from qiskit.circuit.library import TwoLocal

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

# workflows 
from workflows import QuadraticProgramPostprocess, QuadraticProgramConverter

# SPSA
from spsa import minimize_spsa

# Middleware
import os
from quantum_serverless import save_result
    
value_of_items = [4, 3, 5, 6, 7]
weight_of_items = [2, 2, 4, 5, 6]
max_weight = 12

# Step 1

def build_model():
    mdl = Model(name="0-1 Knapsack problem")
    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(value_of_items))}
    mdl.maximize(mdl.sum(value_of_items[i] * x[i] for i in x))
    mdl.add_constraint(mdl.sum(weight_of_items[i] * x[i] for i in x) <= max_weight);
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
    ansatz = TwoLocal(hamiltonian.num_qubits, 'ry', 'cx', 'linear', reps=1)
    return ansatz

# Step 2

# EMPTY

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
    estimator = QiskitEstimator(options={"shots": int(1e4)})
    sampler = QiskitSampler(options={"shots": int(1e4)})
    return estimator, sampler

def minimize(ansatz, hamiltonian, estimator):
    x0 = 2*np.pi*np.random.random(size=ansatz.num_parameters)
    res = minimize_spsa(cost_func, x0,
                        args=(ansatz, hamiltonian, estimator),
                        maxiter=100)
    return res

def compute_distribution(ansatz, sampler, res):
    # Assign solution parameters to ansatz
    qc = ansatz.assign_parameters(res.x)
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

# Step 3 (there is no step 2)

estimator, sampler = setup_estimator_sampler()

res = minimize(ansatz, hamiltonian, estimator)
print(res)

samp_dist = compute_distribution(ansatz, sampler, res)

# Step 4

solution = post_processing(qubo, quadratic_transformer, samp_dist)
print(solution)

save_result({
    "value": sum(np.array(value_of_items)*solution),
    "weight": sum(np.array(weight_of_items)*solution)
})
