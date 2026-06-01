// contract_ru_extraction.baml
// Unified BAML schema + prompt for extracting contract classifications per Resource Unit (RU).
//

// ---------- ENUMS ----------

enum MatchBasis {
  @@description("Explains how a field value was determined")

  DIRECT @description("Explicit mention tied to the RU.")
  INFERRED @description("Derived via normalization or rule.")
  NOT_FOUND @description("No evidence; fallback rule applied.")
}

enum ITSMClassification {
  @@alias("ITSM Classification")
  @@description(#"
    ITSM / SMC / ATF classification:
    - Use 'Incl SMC/ATF' when both SMC and ATF are mentioned
    - Use 'Incl SMC' when only SMC is mentioned
    - Use 'Incl ATF' when only ATF is mentioned
    - Use 'Excluded' otherwise
  "#)
  INCL_SMC_ATF @alias("Incl SMC/ATF")
  INCL_SMC @alias("Incl SMC")
  INCL_ATF @alias("Incl ATF")
  EXCLUDED @alias("Excluded")
}

enum ResourceAllocationModel {
  @@alias("Resource Allocation Model")

  SHARED @alias("Shared")  @description("pooled, shared, multi-tenant, mixed, or uncertain cases.")
  DEDICATED @alias("Dedicated") @description("dedicated team or service desk, exclusive or single-client setup.")
}

enum DeliveryModel {
  @@alias("Delivery Model")
  @@description("Normalized delivery model for the RU.")

  ONSITE @alias("Onsite")
  ONSHORE @alias("Onshore")
  NEARSHORE @alias("Nearshore")
  OFFSHORE @alias("Offshore")
  ONSHORE_OFFSHORE @alias("Onshore/Offshore")
  ONSHORE_NEARSHORE @alias("Onshore/Nearshore")
  OFFSHORE_NEARSHORE @alias("Offshore/Nearshore")
  ONSHORE_NEARSHORE_OFFSHORE @alias("Onshore/Nearshore/Offshore")
}

enum DatacenterOwnership {
  @@alias("Datacenter Ownership")
  ATOS_DATACENTER @alias("Atos Datacenter")  @description( "Atos or supplier facilities")
  CUSTOMER_DATACENTER @alias("Customer Datacenter") @description( "Customer-owned or on-premise facilities")
  NA @alias("NA") @description( "Not applicable or not evidenced")
}

enum GovernanceCoverage {
  @@alias("Governance Coverage")
  @@description(#"
    Governance coverage classification:
    - 'Included': governance, service management, reporting, reviews,
      SLA tracking, or escalation management are part of the RU
    - 'Excluded': otherwise
  "#)
  INCLUDED @alias("Included") 
  EXCLUDED @alias("Excluded")
}

enum ServiceWindow {
  @@alias("Service Window")

  SW_8X5 @alias("8x5") @description("standard business hours or fallback")
  SW_12X5 @alias("12x5") @description("about 12 weekday hours")
  SW_16X5 @alias("16x5") @description("extended weekday plus possible Saturday coverage")
  SW_24X7 @alias("24x7") @description("continuous coverage")
}

enum SLAType {
  @@alias("SLA Type")
  @@description(#"
    SLA strictness classification:
    - 'Stringent': high, critical, production, minimal downtime, 24x7
    - 'Standard': normal, managed service, helpdesk, service desk
    - 'Lenient': low-priority or best-effort
  "#)
  LENIENT @alias("Lenient")
  STANDARD @alias("Standard")
  STRINGENT @alias("Stringent")
}

enum ToolingOwnership {
  @@alias("Tooling Ownership")
  @@description(#"
    Tooling ownership classification:
    - 'Atos Tooling': required tools are supplier-provided
    - 'Customer Tooling': customer-provided
    - 'NA': tooling not applicable, missing, or unclear
  "#)
  ATOS_TOOLING @alias("Atos Tooling")
  CUSTOMER_TOOLING @alias("Customer Tooling")
  NA @alias("NA")
}

enum InclusionStatus {
  @@alias("Inclusion Status")
  @@description(#"
    Generic inclusion/exclusion classification for software and hardware scope:
    - 'Included': when the RU explicitly includes that scope
    - 'Excluded': otherwise
  "#)
  INCLUDED @alias("Included")
  EXCLUDED @alias("Excluded")
}

enum Country {
  @@description("Normalized country vocabulary used in delivery model and location fields.")
  FRANCE @alias("France")
  GERMANY @alias("Germany")
  UNITED_KINGDOM @alias("United Kingdom")
  UNITED_STATES @alias("United States")
  ESTONIA @alias("Estonia")
  NETHERLANDS @alias("Netherlands")
  ARGENTINA @alias("Argentina")
  AUSTRALIA @alias("Australia")
  BELGIUM @alias("Belgium")
  BRAZIL @alias("Brazil")
  CHINA @alias("China")
  DENMARK @alias("Denmark")
  FINLAND @alias("Finland")
  HONG_KONG @alias("Hong Kong")
  SINGAPORE @alias("Singapore")
  SPAIN @alias("Spain")
  SWITZERLAND @alias("Switzerland")
  INDIA @alias("India")
  MEXICO @alias("Mexico")
  EGYPT @alias("Egypt")
  MALAYSIA @alias("Malaysia")
  MOROCCO @alias("Morocco")
  PHILIPPINES @alias("Philippines")
  POLAND @alias("Poland")
  ROMANIA @alias("Romania")
  AUSTRIA @alias("Austria")
  NA @alias("NA")
}

enum OnshoreCountry {
  @@description("Countries treated as onshore when normalizing delivery model and delivery location.")
  FRANCE @alias("France")
  GERMANY @alias("Germany")
  UNITED_KINGDOM @alias("United Kingdom")
  UNITED_STATES @alias("United States")
  ESTONIA @alias("Estonia")
  NETHERLANDS @alias("Netherlands")
  ARGENTINA @alias("Argentina")
  AUSTRALIA @alias("Australia")
  BELGIUM @alias("Belgium")
  BRAZIL @alias("Brazil")
  CHINA @alias("China")
  DENMARK @alias("Denmark")
  FINLAND @alias("Finland")
  HONG_KONG @alias("Hong Kong")
  SINGAPORE @alias("Singapore")
  SPAIN @alias("Spain")
  SWITZERLAND @alias("Switzerland")
  AUSTRIA @alias("Austria")
}

enum OffshoreCountry {
  @@description(#"
    Countries treated as offshore when normalizing delivery model and location.
    Default offshore fallback: India (except Philippines for offshore Service Desk
    when no country is specified).
  "#)
  INDIA @alias("India")
  MEXICO @alias("Mexico")
  BRAZIL @alias("Brazil")
  CHINA @alias("China")
  EGYPT @alias("Egypt")
  MALAYSIA @alias("Malaysia")
  MOROCCO @alias("Morocco")
  PHILIPPINES @alias("Philippines")
  SPAIN @alias("Spain")
}

enum NearshoreCountry {
  @@description(#"
    Countries treated as nearshore when normalizing delivery model and location.
    Poland is the default nearshore country in supported Onshore/Nearshore combinations.
  "#)
  POLAND @alias("Poland")
  ROMANIA @alias("Romania")
}

// ---------- HELPER CLASSES ----------

class Source {
   @@description("Source used to extract the evidence")
  file string @description("filename of the document (whose extension can be PDF, DOCX, HTML, ....) in the provided merged_contract_files_markdown") 
  section string? @description("specific section, clause, page number, sheet number or heading where the evidence was found.")
}

class EvidenceNote {
  @@description("Evidence explaining the classification.")
  basis MatchBasis @description("DIRECT / INFERRED / NOT_FOUND")
  evidence string @description("Short supporting statement.")
  applied_rule string? @description("Rule used when inferred or fallback.")
  source Source
}

class SingleCountryLocation {
  @@description("Single-country location.")
  country Country
}

class ModeOnlyLocation {
  @@description("Only delivery model known, no country.")
  mode DeliveryModel
}

class NoLocation {
  @@description("No reliable location could be extracted.")
  reason string
}

// ---------- CATEGORY ASSESSMENTS ----------

class ITSMAssessment {
  @@description("ITSM/SMC/ATF assessment.")
  classification ITSMClassification
  note EvidenceNote
}

class ResourceAllocationAssessment {
  @@description("Resource allocation assessment.")
  classification ResourceAllocationModel
  note EvidenceNote
}

class DeliveryModelAssessment {
  @@description("Delivery model and location assessment.")
  classification DeliveryModel
  normalized_location SingleCountryLocation | ModeOnlyLocation | NoLocation
  note EvidenceNote
}

class DatacenterOwnershipAssessment {
  @@description("Datacenter ownership assessment.")
  classification DatacenterOwnership
  note EvidenceNote
}

class GovernanceAssessment {
  @@description("Governance coverage assessment.")
  classification GovernanceCoverage
  note EvidenceNote
}

class OffshoreLocationAssessment {
  @@description("Offshore location assessment.")
  normalized_location SingleCountryLocation | NoLocation
  note EvidenceNote
}

class ServiceWindowAssessment {
  @@description("Service window assessment.")
  classification ServiceWindow
  note EvidenceNote
}

class SLAAssessment {
  @@description("SLA assessment.")
  classification SLAType
  note EvidenceNote
}

class ToolingAssessment {
  @@description("Tooling ownership assessment.")
  classification ToolingOwnership
  note EvidenceNote
}

class SoftwareAssessment {
  @@description("Software scope assessment.")
  classification InclusionStatus
  note EvidenceNote
}

class HardwareAssessment {
  @@description("Hardware scope assessment.")
  classification InclusionStatus
  note EvidenceNote
}

// ---------- MAIN OUTPUT ----------

class RUExtraction {
  @@description("All classifications for one RU.")
  ru_name string
  itsm ITSMAssessment
  resource_allocation ResourceAllocationAssessment
  delivery_model DeliveryModelAssessment
  datacenter_ownership DatacenterOwnershipAssessment
  governance GovernanceAssessment
  offshore_location OffshoreLocationAssessment
  service_window ServiceWindowAssessment
  sla SLAAssessment
  tooling ToolingAssessment
  software SoftwareAssessment
  hardware HardwareAssessment
}

class RUExtractionBatch {
  @@description("List of RU results.")
  results RUExtraction[]
}

// ---------- FUNCTION ----------
function ExtractRUFacts(merged_contract_files_markdown: string) -> RUExtractionBatch {
  client "openai/gpt-4o-mini"
 
  prompt #"
You are a precise contract-analysis extraction engine.

Task
- Read contract documentation files compiled as Markdown in `merged_contract_files_markdown`.
- For each RU in the RU LIST, return exactly one `RUExtraction`.
- Use only information grounded in `merged_contract_files_markdown`.
- Provide best evidence for each classification, citing the specific section or clause where it was found.

Global extraction rules
- Use only the provided contract markdown.
- Do not invent, assume, generalize, or merge facts across unrelated RUs.
- Treat each RU name as a business label, not as a generic keyword.
- Match RU names robustly across formatting differences such as line breaks, repeated spaces, punctuation differences, table formatting, or layout artifacts.
- Preserve the RU order exactly as listed.
- Fill every field for every RU.
- Keep `EvidenceNote.evidence` concise: one or two sentences maximum.
- Do not include citations, links, filenames from outside the input, or extra metadata in the extracted text itself.
- Because only one source input is provided, always set `EvidenceNote.source.file = "merged_contract_files_markdown"`.
- Set `EvidenceNote.source.section` to the best available locator such as heading, clause, appendix, table name, page marker, or sheet name; otherwise null.

EvidenceNote rules
- Use `basis = DIRECT` when the classification is supported by an explicit mention tied to the RU.
  - Populate `evidence`
  - Omit `applied_rule`
- Use `basis = INFERRED` when evidence exists but the final value requires a normalization rule or default mapping.
  - Populate both `evidence` and `applied_rule`
- Use `basis = NOT_FOUND` when no relevant evidence exists for that field.
  - Set `evidence = "No relevant mention found."`
  - Populate `applied_rule` with the fallback rule that determined the value

Classification rules by field

1) ITSM Classification
Objective
- Find information related to ITSM, SMC, and ATF for the RU, especially governance, roles, responsibilities, and operational structures.

Rules
- If the RU context mentions both SMC and ATF -> `Incl SMC/ATF`
- If it mentions SMC but not ATF -> `Incl SMC`
- If it mentions ATF but not SMC -> `Incl ATF`
- If neither SMC nor ATF is mentioned, or if uncertain -> `Excluded`

2) Resource Allocation Model
Objective
- Determine whether resources or support teams are shared or dedicated for the RU.

Rules
- If the RU mentions shared resources across multiple clients or services, pooled delivery, or multi-tenant setup -> `Shared`
- If the RU mentions dedicated team, exclusive allocation, dedicated infrastructure, dedicated service desk, or single-client setup -> `Dedicated`
- If both shared and dedicated elements are mentioned -> `Shared`
- If the RU mentions SPOC or dedicated service desk -> `Dedicated`
- If uncertain -> `Shared`

3) Delivery Model
Objective
- Determine how the RU is geographically delivered or supported.

Normalization rules
- Onshore countries: France, Germany, United Kingdom, United States, Estonia, Netherlands, Argentina, Australia, Belgium, Brazil, China, Denmark, Finland, Hong Kong, Singapore, Spain, Switzerland
- Offshore countries: India, Mexico, Egypt, Malaysia, Morocco, Philippines
- Nearshore countries: Poland, Romania

Classification rules
- If the text explicitly says `Onsite` -> `Onsite`
- If the text explicitly says `Offshore/Nearshore` -> `Offshore/Nearshore`
- If the text explicitly says `Onshore/Nearshore/Offshore` -> `Onshore/Nearshore/Offshore`
- If the RU references one onshore country only -> `Onshore`
- If the RU references one offshore country only -> `Offshore`
- If the RU references one nearshore country only -> `Nearshore`
- If the RU references United States/India -> `Onshore/Offshore`
- If the RU references one of these pairs:
  Austria/Poland, Brazil/Poland, Finland/Poland, France/Poland, Germany/Poland, Netherlands/Poland, Singapore/Poland, Switzerland/Poland, United Kingdom/Poland, United States/Poland
  -> `Onshore/Nearshore`
- If information is missing or unclear -> `Offshore`

`delivery_model.normalized_location` rules
- Use `SingleCountryLocation` for one normalized country
- Use `OnshoreOffshoreLocation` for an Onshore/Offshore pair
- Use `OnshoreNearshoreLocation` for an Onshore/Nearshore pair
- Use `ModeOnlyLocation` when the delivery mode is clear but no country can be normalized
- Use `NoLocation` when no reliable normalized location can be produced

4) Datacenter Ownership
Objective
- Find datacenter ownership information for the RU.

Scope constraints
- Do not treat legal entity addresses, postal addresses, or cloud regions as datacenter ownership evidence.

Rules
- If documentation mentions Atos or supplier facilities, supplier datacenter, or service running in supplier facilities -> `Atos Datacenter`
- If documentation mentions client-owned, customer-owned, or on-premise infrastructure -> `Customer Datacenter`
- If the RU is a people-based service such as support, service desk, integration, management, or consulting -> `NA`
- If the RU mentions monitoring and managing servers in supplier facilities -> `Atos Datacenter`
- If uncertain -> `NA`

5) Governance Coverage
Objective
- Determine whether governance activities are included in the RU.

Rules
- If the RU includes governance, service management, reporting, reviews, SLA tracking, or escalation management -> `Included`
- If the RU says governance is not included or handled separately, or there is a separate governance fee -> `Excluded`
- If the RU itself is about governance or service management -> `Included`
- If the RU is about operational tasks such as installations, changes, or deployments -> `Excluded`
- If the text describes governance as providing transparency, accountability, and compliance -> `Included`
- If the text describes lack of transparency, accountability, or compliance in governance -> `Excluded`
- If uncertain -> `Excluded`

6) Offshore Location (Normalized Location)
Objective
- Extract the normalized delivery or support location for the RU, based on where services, teams, or resources are delivered from.

Rules
- If the RU indicates Nearshore, identify the country from: Poland, Romania, Mexico
  - If the country is not defined -> use `NoLocation`
- If the RU indicates Onshore, identify the country from the onshore list
  - If the country is not defined -> use `NoLocation`
- If the RU indicates Offshore, identify the country from: India, Mexico, Brazil, China, Egypt, Malaysia, Morocco, Philippines, Spain
  - If the offshore country is not defined -> use India
  - If Offshore is indicated and Atos is requested to deliver Service Desk and the country is not defined -> use Philippines
- If the RU indicates `Onshore/Offshore` -> use `OnshoreOffshoreLocation` with United States / India
- If the RU indicates `Onshore/Nearshore`, identify the documented pair among:
  Austria/Poland, Brazil/Poland, Finland/Poland, France/Poland, Germany/Poland, Netherlands/Poland, Singapore/Poland, Switzerland/Poland, United Kingdom/Poland, United States/Poland
  - If the pair is not defined -> use `NoLocation`

`offshore_location.normalized_location` type rules
- Use `SingleOffshoreLocation` for one offshore or nearshore country
- Use `OnshoreOffshoreLocation` for an Onshore/Offshore pair
- Use `OnshoreNearshoreLocation` for an Onshore/Nearshore pair
- Use `NoLocation` when no reliable normalized location can be produced

7) Service Window
Objective
- Determine the service hours / support coverage model for the RU.

Rules
- If the text mentions `24/7`, `round the clock`, `continuous`, or the RU name itself clearly indicates 24/7 -> `24x7`
- If the text mentions `extended hours`, around `07:00-21:00`, or weekdays plus Saturday coverage -> `16x5`
- If the text mentions `12 hours`, around `07:00-19:00`, or equivalent -> `12x5`
- If the text mentions `business hours`, `BD 08:00-17:00`, `08:00-17:00`, or equivalent -> `8x5`
- If explicit hours are provided, infer the closest valid category from the hours
- If uncertain -> `8x5`

8) SLA Type
Objective
- Determine the SLA strictness for the RU from measurable SLA terms.

What to look for
- SLA tables
- Severity matrices
- Response / resolution times
- Availability targets
- Escalation protocols
- Service credits
- Reporting obligations
- Tower-specific KPIs
- Measurable exclusions such as planned maintenance or third-party failures

Do not use as SLA evidence
- Internal staffing
- Licensing or hardware scope
- Qualitative claims without measurable targets
- Internal targets or KPIs without target values

Rules
- If the text mentions 24/7, critical, high priority, minimal downtime, production, or explicit high SLA level -> `Stringent`
- If the text mentions extended hours, enhanced, monitoring, managed service, helpdesk, service desk, or explicit normal SLA level -> `Standard`
- If the text mentions best effort, low priority, planning, governance, license management, or explicit low SLA level -> `Lenient`
- If the RU name contains `SLA level High` -> `Stringent`
- If the RU name contains `SLA level Normal` -> `Standard`
- If the RU name contains `SLA level Low` -> `Lenient`
- If no clear SLA information is found -> `Standard`

9) Tooling Ownership
Objective
- Determine whether operational tooling is Atos-provided, customer-provided, or not applicable.

What to include
- Tooling directly linked to the RU
- ITSM, CMDB, monitoring, automation, observability, orchestration, DevOps platforms
- ServiceNow, Dynatrace, UiPath, Azure DevOps, CMDB systems
- Integration requirements and tooling ownership responsibilities
- Monitoring / observability platforms such as Dynatrace, Azure Monitor, log analytics
- RPA, CI/CD, orchestration platforms
- Diagrams or tables listing tooling platforms and operational responsibilities
- Focus especially on whether the tooling is Atos-managed or Customer-managed

What to exclude
- Implementation services
- Labor and hardware costs
- OS, middleware, productivity suites, SaaS platforms, business applications
- Licensing, subscriptions, support models
- Infrastructure software, end-user software, business/application software
- Middleware and utility software management
- Application support services
- Antivirus, encryption, compliance software

Rules
- If the tooling is provided, developed, owned, or primarily used by Atos as part of the service -> `Atos Tooling`
- If the tooling is customer-provided or customer-owned -> `Customer Tooling`
- If tooling is not part of scope, missing, or cannot be classified -> `NA`
- If uncertain -> `NA`

10) Software Inclusion Status
Objective
- Determine whether software scope is included for the RU.

What to include
- Software types directly linked to the RU, including OS, middleware, productivity suites, SaaS, business applications
- Licensing tables, subscription terms, software usage rights
- Middleware and utility software responsibilities
- Application support and maintenance
- Certificate lifecycle management, antivirus, encryption, compliance software responsibilities
- KPIs related to software performance, ticket resolution, change management, customer satisfaction
- AIX OS instances and middleware
- AIX LPAR management and capacity planning software
- Apache HTTP Server
- Backup software and archiving applications
- Tape backup management and scheduling software
- Batch job software and automation scripts
- Virtualization and server management software
- Network monitoring and infrastructure management software
- IBM DB2, MS SQL, MySQL, Oracle, PostgreSQL
- Storage management software
- IBM i OS and related management software
- HA software for IBM i
- Linux OS licenses
- SharePoint
- Software updates and patch management
- Virtual machine OS and applications
- VMware and Hyperware virtualization software
- Virtual CPU / memory allocation and management software

What to exclude
- Physical hardware components
- Hardware infrastructure or maintenance
- Physical servers, rack servers, backup media, or storage devices
- Non-software services such as physical installation, upgrades, transport
- Hardware clusters, appliances, physical infrastructure
- Hardware-only responsibilities or physical asset management
- Tooling platforms already covered under `tooling`
- ITSM tools, monitoring tools, automation / DevOps tools, security tooling
- Public or private cloud safety measures
- Implementation services, labor, hardware costs

Rules
- If the RU involves software licensing, subscriptions, software management, software maintenance, patching, or updates -> `Included`
- If the RU is about hardware, physical infrastructure, or people-based services without software scope -> `Excluded`
- If uncertain -> `Excluded`

11) Hardware Inclusion Status
Objective
- Determine whether hardware scope is included for the RU.

What to include
- Hardware directly linked to the RU, including servers, storage, network devices, end-user devices, and specialized equipment
- Hosting environments, storage systems, network appliances, datacenter hardware configurations
- On-site responsibilities related to device installation, maintenance, transfers, and readiness
- Specialized devices such as POS systems, routers, cash registers
- Hardware asset management practices linked to the RU
- Warranty, support, maintenance terms, spare parts, inspections
- Dedicated hardware for AIX LPARs
- Hardware processing resources allocated to AIX pools
- Blade and rack servers running AIX LPARs
- Physical LTO7 tapes and backup storage space
- Physical blade servers, physical network infrastructure
- High-performance storage hardware
- IBM i instances and physical IBM i servers
- CPW for IBM
- IBM i memory and storage hardware
- Tape drive hardware
- IBM i HA solution hardware
- Mid-range physical storage hardware
- NetApp physical storage hardware
- Physical infrastructure in provider premises
- Virtual hardware resources such as vCPU and memory when explicitly treated as hardware scope
- VMware / Hyperware virtual hardware instances
- Virtual RAM and CPU allocations

What to exclude
- Software, middleware, backup services
- Databases
- Non-AIX servers, other OS, applications, generic services
- Public cloud, private cloud, cloud infrastructure, cloud servers, cloud databases
- Cloud instances included only as datacenter capacity
- Generic virtual servers or VMware platform references without explicit hardware scope
- SQL/database development, migration, management
- Platform installation, instance creation/deletion
- Software licenses and management services
- Physical installation or upgrades not tied to AIX or IBM i hardware
- Vendor hardware/software and third-party services not tied to the RU scope

Rules
- If the RU involves physical hardware, end-user devices, hardware maintenance, hardware refresh, or lifecycle management -> `Included`
- If the RU is a people-based service such as service desk, consulting, or ITSM integration -> `Excluded`
- If the RU is about software, licensing, or monitoring without hardware -> `Excluded`
- If uncertain -> `Excluded`

Output constraints
- Use only the declared schema.
- Ensure each classification value matches the corresponding enum alias exactly.
- Keep all extracted evidence concise and grounded only in `merged_contract_files_markdown`.

RU LIST:
- Managed Transition project Workplace
- ITSM Integration
- SPOC SERVICE DESK Large package, monthly base fee
  Level 1. and level 2. support
  Remote support
  Dedicated team
  Responsibilities mentioned in RFP
  Normal working hours
- SD Workstations (if not included in row 9)
- SD POS (if not included in row 9)
- SD Mobile devices (if not included in row 9)
- On site support tasks
- On-site support - (SLA low service) price indicator
- On-site support - (SLA normal service)price indicator
- On-site support - (SLA high service)price indicator
- Extended SD service hours weekdays 07:00-21:30 and Saturdays 08:00-18:30, Additional price
- SD specialist 2. level
- Workstation management SCCM
- POS Management SCCM
- Mobile device management service Intune
- Application packaging, distribution and updates
- Hardware driver updates
- Security updates
- Equipment buffering, pre-installation and delivery
- Asset management

merged_contract_files_markdown:
{{ merged_contract_files_markdown }}

Return data in the exact schema below.
{{ ctx.output_format }}
"#

}
