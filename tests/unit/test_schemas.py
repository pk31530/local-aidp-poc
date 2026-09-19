import pytest
from pydantic import ValidationError

from src.common.schemas import SUPPORTED_SCHEMA_VERSIONS, Transaction

VALID_KWARGS = dict(
    transaction_id="TX1",
    customer_id="C1",
    transaction_timestamp="2026-06-01T10:00:00+05:30",
    amount=100.50,
    merchant="Grocery",
    country="India",
    device_id="DEV1",
    payment_method="CARD",
)


def test_default_schema_version_is_supported():
    tx = Transaction(**VALID_KWARGS)
    assert tx.schema_version == 1
    assert tx.schema_version in SUPPORTED_SCHEMA_VERSIONS


def test_supported_schema_version_is_accepted():
    tx = Transaction(**VALID_KWARGS, schema_version=1)
    assert tx.schema_version == 1


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        Transaction(**VALID_KWARGS, schema_version=99)


@pytest.mark.parametrize("schema_version", [0, -1])
def test_non_positive_schema_version_is_rejected(schema_version):
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        Transaction(**VALID_KWARGS, schema_version=schema_version)
