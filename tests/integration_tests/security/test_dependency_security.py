"""Exercise security-sensitive behavior in the installed dependency set."""

import ssl

import certifi
import pytest
import requests


@pytest.mark.integration
def test_certifi_distrusts_compromised_emsign_root() -> None:
    context = ssl.create_default_context(cafile=certifi.where())
    common_names = {
        value
        for certificate in context.get_ca_certs()
        for relative_name in certificate["subject"]
        for key, value in relative_name
        if key == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in common_names


@pytest.mark.unit
def test_requests_preserves_json_decode_error_contract() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"not-json"
    with pytest.raises(requests.exceptions.JSONDecodeError):
        response.json()
