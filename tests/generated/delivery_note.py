from __future__ import annotations

import re
import sys
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)


metamodel_version = "1.12.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias=True,
        validate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=False,
    )


class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key: str):
        return getattr(self.root, key)

    def __getitem__(self, key: str):
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.root[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta(
    {
        "default_prefix": "dn",
        "default_range": "string",
        "description": "Schema for extracting structured data from delivery notes in English, French, or German.",
        "id": "https://example.org/tests/delivery_note",
        "imports": ["linkml:types"],
        "name": "delivery_note",
        "prefixes": {
            "dn": {"prefix_prefix": "dn", "prefix_reference": "https://example.org/tests/delivery_note/"},
            "linkml": {"prefix_prefix": "linkml", "prefix_reference": "https://w3id.org/linkml/"},
        },
        "source_file": "/home/tcl/prj/genai-tk/tests/linkml_src/delivery_note.yaml",
    }
)


class DeliveryNoteLine(ConfiguredBaseModel):
    """
    A product line item on a delivery note.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({"from_schema": "https://example.org/tests/delivery_note"})

    product_number: str = Field(
        default=...,
        description="""product or item reference number (EN: product number / part number, FR: numéro de produit / référence article, DE: Artikelnummer / Produktnummer / HTZ)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine"]}},
    )
    quantity: float = Field(
        default=...,
        description="""quantity delivered (EN: quantity, FR: quantité, DE: Menge)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine"]}},
    )
    lot_number: Optional[str] = Field(
        default=None,
        description="""manufacturing lot or batch number, if present (EN: lot/batch number, FR: numéro de lot, DE: Chargennummer)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine"]}},
    )
    serial_number: Optional[str] = Field(
        default=None,
        description="""individual unit serial number, if present (EN: serial number, FR: numéro de série, DE: Seriennummer)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine"]}},
    )
    order_number: Optional[str] = Field(
        default=None,
        description="""customer or supplier order reference, if present (EN: order number / PO number, FR: numéro de commande, DE: Bestellnummer)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine", "DeliveryNote"]}},
    )


class DeliveryNote(ConfiguredBaseModel):
    """
    Delivery note document representation.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://example.org/tests/delivery_note", "tree_root": True}
    )

    delivery_note_number: str = Field(
        default=...,
        description="""unique identifier of the delivery note (EN: delivery note number, FR: numéro de BL / bon de livraison, DE: Lieferscheinnummer, lieferung)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNote"]}},
    )
    order_number: Optional[str] = Field(
        default=None,
        description="""customer or supplier order reference, if present (EN: order number / PO number, FR: numéro de commande, DE: Bestellnummer)""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNoteLine", "DeliveryNote"]}},
    )
    lines: list[DeliveryNoteLine] = Field(
        default=...,
        description="""list of product lines on the delivery note""",
        json_schema_extra={"linkml_meta": {"domain_of": ["DeliveryNote"]}},
    )


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
DeliveryNoteLine.model_rebuild()
DeliveryNote.model_rebuild()
