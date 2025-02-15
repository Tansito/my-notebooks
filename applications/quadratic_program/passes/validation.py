# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
QUBO workflow
"""
from functools import wraps


def validate_output_type(func):     
    @wraps(func)
    def wrapper(*args, **kwargs):
        output_types = args[0].output_types
        out = func(*args, **kwargs)
        if not isinstance(out, output_types):
            raise TypeError(f'Output type not in {output_types}')
        return out
    return wrapper  
