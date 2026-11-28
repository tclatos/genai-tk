# Phase 0 Baseline Capture & Pilot Contract Specifications

This document records the Phase 0 baseline behavior, schemas, output layouts, and verified test fixtures for migrating structured extraction from BAML to LinkML.

---

## 1. Baseline Pilot Contracts

### 1.1 Contract 1: `Resume` (Flat model with lists and required fields)

- **Source BAML**: `tests/baml_src/resume.baml`
- **Fields**:
  - `name`: string (description: "name of the candidate") [REQUIRED]
  - `email`: string [REQUIRED]
  - `experience`: list of strings [REQUIRED]
  - `skills`: list of strings [REQUIRED]
- **Output Convention**:
  - Saved to `<output_dir>/Resume/<stem>.json`
  - Recorded in `<output_dir>/manifest.json` with source content hash and output relative path.
- **Representative Input**:
  ```text
  John Smith
  john.smith@example.com

  Experience:
  - Senior Software Engineer at TechCorp (2020-2024)
  - Backend Developer at CloudSolutions (2017-2020)

  Skills:
  - Python
  - Rust
  - Docker
  - PostgreSQL
  ```
- **Expected Baseline JSON**:
  ```json
  {
    "name": "John Smith",
    "email": "john.smith@example.com",
    "experience": [
      "Senior Software Engineer at TechCorp (2020-2024)",
      "Backend Developer at CloudSolutions (2017-2020)"
    ],
    "skills": [
      "Python",
      "Rust",
      "Docker",
      "PostgreSQL"
    ]
  }
  ```

---

### 1.2 Contract 2: `DeliveryNote` (Nested objects, optional scalar values, numbers, multilingual)

- **Source BAML**: `tests/baml_src/delivery_note.baml`
- **Root Class**: `DeliveryNote`
  - `delivery_note_number`: string (description: "unique identifier of the delivery note") [REQUIRED]
  - `order_number`: optional string (description: "customer or supplier order reference") [OPTIONAL]
  - `lines`: list of `DeliveryNoteLine` (description: "list of product lines on the delivery note") [REQUIRED]
- **Child Class**: `DeliveryNoteLine`
  - `product_number`: string (description: "product or item reference number") [REQUIRED]
  - `quantity`: float (description: "quantity delivered") [REQUIRED]
  - `lot_number`: optional string (description: "manufacturing lot or batch number") [OPTIONAL]
  - `serial_number`: optional string (description: "individual unit serial number") [OPTIONAL]
  - `order_number`: optional string (description: "order reference for line") [OPTIONAL]
- **Output Convention**:
  - Saved to `<output_dir>/DeliveryNote/<stem>.json`
- **Representative Input (French sample)**:
  ```text
  BON DE LIVRAISON
  N° de BL : BL-2024-00123
  N° de commande : CMD-9876

  Lignes :
  | Réf. produit | Désignation          | N° de lot  | N° de série | Quantité |
  |--------------|----------------------|------------|-------------|----------|
  | ART-001      | Capteur de pression  | LOT-2024A  |             | 10       |
  | ART-002      | Valve de régulation  | LOT-2024B  | SN-00045    | 2        |
  | ART-003      | Filtre hydraulique   |            |             | 5        |
  ```
- **Expected Baseline JSON**:
  ```json
  {
    "delivery_note_number": "BL-2024-00123",
    "order_number": "CMD-9876",
    "lines": [
      {
        "product_number": "ART-001",
        "quantity": 10.0,
        "lot_number": "LOT-2024A",
        "serial_number": null,
        "order_number": null
      },
      {
        "product_number": "ART-002",
        "quantity": 2.0,
        "lot_number": "LOT-2024B",
        "serial_number": "SN-00045",
        "order_number": null
      },
      {
        "product_number": "ART-003",
        "quantity": 5.0,
        "lot_number": null,
        "serial_number": null,
        "order_number": null
      }
    ]
  }
  ```

---

### 1.3 Contract 3: Enum & Permissible-Value Description Preservation

- **Goal**: Verify that enum descriptions AND individual permissible-value descriptions are preserved through `SchemaView` and prompt glossaries, overcoming JSON Schema limitations.
- **Fixture Spec**:
  ```yaml
  enums:
    ProcessingStatus:
      description: Lifecycle processing status of an item.
      permissible_values:
        PENDING:
          description: Awaiting initial review or ingestion.
        IN_PROGRESS:
          description: Active processing is ongoing.
        COMPLETED:
          description: Successfully analyzed and validated.
        FAILED:
          description: Extraction or validation encountered an unrecoverable error.
  ```

---

## 2. Graph Pilot Baselines (SCNF & EKG-Atos)

### 2.1 SCNF Pilot: PCD and W8/POSE

- **PCD Extraction**: `examples/_out/PCD_DREUX_Protection.json`
  - Total nodes: 55
    - Track: 4
    - Turnout: 19
    - LogicalSection: 9
    - RailwayDevice: 15
    - RailwaySite: 1
    - OperationalArea: 1
    - ExternalConnection: 2
    - UnresolvedItem: 4
  - Total relationships: 92
    - CONNECTS_TRACK
    - COVERS_TRACK
    - LOCATED_ON_TRACK
    - CONTINUES_TRACK
    - Structural containment: HAS_SITE, HAS_AREA, HAS_TRACK, HAS_TURNOUT, etc.
- **W8/POSE Extraction**: `examples/_out/W8_P1_1_01.3_U_00_POSE.json`
  - Contains railway protections (`ProtectionFerroviaire`), zone elementary protections (ZEP), track equipment, and cross-references.
- **Text-to-Cypher Baseline**:
  - Relies on markdown generated from `GraphSchema` documentation.
  - Queries like "Find all turnouts connected to track V1" translate to:
    ```cypher
    MATCH (t:Turnout)-[:CONNECTS_TRACK]->(tr:Track {code: 'V1'}) RETURN t.name, t.type
    ```

### 2.2 EKG-Atos Rainbow Relationship Properties

Verified target-node `p_*_` property convention:
- `Person.p_role_`:
  - `Customer -[HAS_CONTACT]-> Person`: role=Sponsor
  - `ReviewedOpportunity -[HAS_TEAM_MEMBER]-> Person`: role=Lead
- `Partner.p_role_`:
  - `ReviewedOpportunity -[HAS_PARTNER]-> Partner`: role=Subcontractor
- `RiskAnalysis.p_mitigation_strategy_`, `RiskAnalysis.p_risk_description_`:
  - `ReviewedOpportunity -[HAS_RISK]-> RiskAnalysis`: mitigation_strategy=Reserve, risk_description=Budget
- `Competitor.p_name_`, `Competitor.p_comment_`:
  - `ReviewedOpportunity -[HAS_COMPETITOR]-> Competitor`: name=Capgemini, comment=Incumbent
- `TechnicalComponent.p_purpose_`:
  - `SWArchitectureDocument -[USED_TECHNOLOGY]-> TechnicalComponent`: purpose=Data processing pipeline
- `Solution.p_purpose_`:
  - `SWArchitectureDocument -[USED_SOLUTION]-> Solution`: purpose=API Gateway

---

## 3. Workflow & Cache Invalidation Baseline

- `ManifestCache` format:
  - Key: source path or identifier
  - Fingerprint: content hash (xxh3_64)
  - Code version: prompt / schema fingerprint
  - Outputs: dictionary mapping output keys (`output_path`)
  - Skips processing when fingerprint and code version match, unless `--force` is provided.
- Cache invalidation triggers:
  1. Source file modified (content hash changes)
  2. Schema or prompt modified (code version / prompt fingerprint changes)
  3. Model override changes
  4. Force flag (`--force`)
