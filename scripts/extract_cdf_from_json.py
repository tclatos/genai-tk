#!/usr/bin/env python3
"""
Extract classifications from JSON data and export to CSV format.

Reads a JSON file with nested classification structures and extracts
the classification values to create a CDF (Comma-Delimited Format) file.
"""

import csv
import json
from pathlib import Path
from typing import Any


def extract_classification(obj: Any, default: str = "NA") -> str:
    """Extract classification value from a nested object."""
    if isinstance(obj, dict):
        return obj.get("classification", default)
    return default


def extract_country(obj: Any, default: str = "NA") -> str:
    """Extract country from a nested normalized_location object."""
    if isinstance(obj, dict):
        normalized_location = obj.get("normalized_location", {})
        if isinstance(normalized_location, dict):
            return normalized_location.get("country", default)
    return default


def normalize_service_window(classification: str) -> str:
    """Convert SW_8X5 format to 8x5 format."""
    if classification.startswith("SW_"):
        # Remove SW_ prefix and lowercase x
        return classification.replace("SW_", "").lower()
    return classification


def normalize_resource_allocation(classification: str) -> str:
    """Normalize resource allocation values."""
    # Convert DEDICATED to Dedicated, SHARED to Shared
    if classification == "DEDICATED":
        return "Dedicated"
    elif classification == "SHARED":
        return "Shared"
    return classification


def normalize_itsm(classification: str) -> str:
    """Normalize ITSM classification values."""
    # INCL_SMC_ATF, INCL_SMC, EXCLUDED
    mapping = {
        "INCL_SMC_ATF": "Included (SMC+ATF)",
        "INCL_SMC": "Included (SMC)",
        "EXCLUDED": "Excluded",
    }
    return mapping.get(classification, classification)


def normalize_tooling(classification: str) -> str:
    """Normalize tooling classification values."""
    mapping = {
        "ATOS_TOOLING": "Atos Tooling",
        "CUSTOMER_TOOLING": "Customer Tooling",
        "NA": "NA",
        "EXCLUDED": "Excluded",
    }
    return mapping.get(classification, classification)


def extract_cdf_from_json(json_file: str, output_file: str, include_metadata: bool = False) -> None:
    """
    Extract classifications from JSON and output as CSV.

    Args:
        json_file: Path to input JSON file
        output_file: Path to output CSV file
        include_metadata: If True, include empty columns for Unit_of_Measure,
                         L3_Service_Line, Base_volume_Y1-Y5

    Example:
        extract_cdf_from_json('MERGED.json', 'output.csv')
    """

    # Read JSON file
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("Warning: No results found in JSON")
        return

    # Define CSV headers
    if include_metadata:
        csv_headers = [
            "RU_Name",
            "Unit_of_Measure",
            "L3_Service_Line",
            "Base_volume_Y1",
            "Base_volume_Y2",
            "Base_volume_Y3",
            "Base_volume_Y4",
            "Base_volume_Y5",
            "SLA",
            "ServiceWindow",
            "Shared_Dedicated",
            "Tooling",
            "DC_Location",
            "Governance",
            "ITSM",
            "Delivery_Model",
            "Offshore_Location",
            "Software",
            "Hardware",
        ]
    else:
        csv_headers = [
            "RU_Name",
            "SLA",
            "ServiceWindow",
            "Shared_Dedicated",
            "Tooling",
            "DC_Location",
            "Governance",
            "ITSM",
            "Delivery_Model",
            "Offshore_Location",
            "Software",
            "Hardware",
        ]

    rows = []

    # Extract data from each result
    for result in results:
        row = {}
        row["RU_Name"] = result.get("ru_name", "").strip()

        # Extract classification fields
        row["SLA"] = extract_classification(result.get("sla"))
        row["ServiceWindow"] = normalize_service_window(extract_classification(result.get("service_window")))
        row["Shared_Dedicated"] = normalize_resource_allocation(
            extract_classification(result.get("resource_allocation"))
        )
        row["Tooling"] = normalize_tooling(extract_classification(result.get("tooling")))
        row["DC_Location"] = extract_classification(result.get("datacenter_ownership"))
        row["Governance"] = extract_classification(result.get("governance"))
        row["ITSM"] = normalize_itsm(extract_classification(result.get("itsm")))
        row["Delivery_Model"] = extract_classification(result.get("delivery_model"))
        row["Offshore_Location"] = extract_country(result.get("offshore_location"))
        row["Software"] = extract_classification(result.get("software"))
        row["Hardware"] = extract_classification(result.get("hardware"))

        # Add metadata columns if requested (empty for now)
        if include_metadata:
            row["Unit_of_Measure"] = ""
            row["L3_Service_Line"] = ""
            row["Base_volume_Y1"] = ""
            row["Base_volume_Y2"] = ""
            row["Base_volume_Y3"] = ""
            row["Base_volume_Y4"] = ""
            row["Base_volume_Y5"] = ""

        rows.append(row)

    # Write to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Successfully extracted {len(rows)} rows to {output_file}")
    print(f"  CSV file: {output_path.resolve()}")


if __name__ == "__main__":
    import sys

    # Default paths
    DEFAULT_JSON = "MERGED.json"
    DEFAULT_OUTPUT = "output.csv"

    # Accept command-line arguments
    json_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    csv_output = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    include_meta = "--with-metadata" in sys.argv

    try:
        extract_cdf_from_json(json_input, csv_output, include_metadata=include_meta)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
