"""Shared synthetic customer/account population for the seven v1.3 channel
generators (guide section 7). Deterministic for a given (seed, n,
reference_date), the same contract src/generator/customers.py uses,
simplified to what these generators need: no online-feature-store profile
fields -- those belong to the existing v1.1/v1.2 shared feature store,
untouched by this guide.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class SyntheticCustomer:
    customer_id: str
    account_id: str


def generate_customers(n: int, seed: int, reference_date: date) -> list[SyntheticCustomer]:
    """Deterministic for a given (n, seed, reference_date). `reference_date`
    is accepted for contract-parity with src/generator/customers.py and
    reserved for future tenure-dependent logic; this function does not
    currently read it."""
    del reference_date
    rng = np.random.default_rng(seed)
    customers = []
    for i in range(n):
        customer_id = f"FIC{1000 + i}"
        account_id = f"FIA{100000 + int(rng.integers(0, 900000))}"
        customers.append(SyntheticCustomer(customer_id=customer_id, account_id=account_id))
    return customers
