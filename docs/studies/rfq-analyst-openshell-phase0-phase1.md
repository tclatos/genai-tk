# RFQ Analyst on OpenShell: Phase 0 and Phase 1 execution report

**Status:** Phase 0 passed; Phase 1 whole-worker packaging passed on the local pilot host.

**Evidence date:** 25 November 2026

**Target workload:** `rfq-analyst`, LangChain Deep Agent profile `rfq-analyst`

**OpenShell:** stable `v0.1.10`

**Scope:** local Docker-backed whole-worker execution

## 1. Executive summary

OpenShell was successfully used to run the real `rfq-analyst` Deep Agent as a
canonical sandboxed process. The OpenRouter API key remained outside the
workload environment, outbound access was mediated and limited to OpenRouter,
unapproved egress was denied, the workload ran as UID/GID 1000 without a Docker
socket, and NeMo Relay agent spans reached a collector through OpenShell's
private trace relay.

The strongest measured result was the credential boundary:

- the workload received an `OPENROUTER_API_KEY` value, but not the real key;
- OpenShell accepted the corresponding placeholder only on
  `openrouter.ai:443`;
- a request to `example.com` failed;
- the supervisor emitted an OCSF `NET:OPEN [MED] DENIED` record;
- OpenRouter HTTP L7 traffic emitted an `HTTP:POST [INFO] ALLOWED` record;
- the workload never received the OTLP collector credentials or public
  collector location;
- two relayed NeMo Relay spans contained the exact OpenShell sandbox ID in
  `openshell.sandbox.id`.

Phase 0 therefore passed and Phase 1 was executed. The Phase 1 result is a
repeatable RFQ whole-worker package:

- a 658,613,957-byte non-root image;
- UID/GID 1000;
- explicit canonical command;
- hard-required Landlock policy;
- only `/app/data`, `/home/appuser`, and `/tmp` writable;
- OpenRouter provider profile with L7 enforcement;
- baked model metadata to avoid runtime metadata downloads;
- generated BAML client included;
- no `agent-sandbox`/OpenSandbox runtime package;
- worker-only dependency lock generated and consumed with `uv sync --locked`;
- local NeMo Relay OTLP export enabled through OpenShell's injected endpoint.

### Promotion decision

**Proceed with OpenShell as an optional production containment layer for RFQ
Analyst, subject to a native supported-Linux or explicitly qualified WSL
deployment.** Do not treat this local result as production qualification.

The present host is WSL 2, which OpenShell v0.1.10 classifies as experimental.
Phase 2 must reproduce the security and functional gates on a supported native
Linux host or Kubernetes cluster. The Phase 0/1 work proves compatibility and
strong control-plane value, but not production platform qualification.

## 2. What was tested

The test exercised the RFQ-specific stack rather than a generic sample:

- RFQ project package and CLI;
- `rfq-analyst` Deep Agent profile;
- 17 loaded tools;
- two configured sub-agents:
  - `pricing-specialist`;
  - `risk-auditor`;
- 11 RFQ skills;
- NeMo Relay ATOF subscriber;
- OpenRouter model `deepseek/deepseek-v4-flash`;
- generated BAML types used by RFQ pricing modules;
- OpenShell provider attachment and policy composition;
- OpenShell L4/L7 network enforcement;
- OpenShell agent OTLP trace relay.

The tested prompt deliberately avoided RFQ tools and external data retrieval.
This isolated OpenShell compatibility, model connectivity, credential handling,
policy enforcement, and telemetry. It did not validate a complete Alko or
customer-document analysis against pricing databases or private graph stores.

## 3. Environment and version baseline

| Item | Result |
|---|---|
| Host | WSL 2, Linux kernel 6.18.58.1 |
| OpenShell | `0.1.10` stable |
| OpenShell CLI | verified release archive, SHA-256 checked |
| OpenShell gateway | `quay.io/ai-openshell/gateway:0.1.10` |
| OpenShell supervisor | release-matched sandbox runtime `0.1.10` |
| Compute driver | Docker |
| Docker | 29.6.0 client/server |
| Python in worker | 3.12 |
| OpenRouter endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| Gateway authentication | mTLS |
| Gateway publication | loopback only |
| Anonymous OpenShell telemetry | disabled |
| Agent workload image | `rfq-analyst-openshell:phase1` |
| Final image digest | `sha256:434b2d26133a5fb4447ca2a4cd7f611f8456b6b3abbb9e5e553299c9affd273b` |
| Image size | 658,613,957 bytes |
| Workload identity | `1000:1000` |

The package-managed installer was attempted but could not complete because the
shared host did not provide non-interactive `sudo`. The documented containerized
gateway deployment was used instead. It was bound to loopback and mTLS was kept
enabled; the installer documentation's unauthenticated TLS-disabled shortcut was
not used.

## 4. Architecture exercised

The pilot used the whole-worker model recommended in the original integration
study:

```text
RFQ host launcher
  -> OpenShell Gateway (mTLS)
  -> Docker compute driver
  -> OpenShell Supervisor
  -> rfq-analyst Deep Agent worker
  -> OpenRouter (credential injected only by Supervisor proxy)
```

NeMo Relay used a separate observation path:

```text
rfq-analyst / NeMo Relay
  -> OpenShell-injected private OTLP relay endpoint
  -> OpenShell Supervisor
  -> local pilot collector
```

The host launcher and OpenShell Supervisor are not the same component. The host
launcher is application-owned and manages the workload request. The Supervisor
is OpenShell-owned, created per sandbox, and enforces the policy and credential
mediation path inside the sandbox boundary.

No OpenSandbox container, nested Docker daemon, or Docker socket was required by
the final RFQ worker.

## 5. Phase 0 execution

### 5.1 Preflight

The host passed the Docker prerequisites and the RFQ project ran successfully
outside OpenShell:

- host RFQ offline-agent baseline: **7.70 s**, exit 0;
- NeMo Relay ATOF subscriber activated;
- RFQ CLI exposed the `rfq-analyst` profile and expected agent commands;
- Docker 29.6.0 met the documented OpenShell Docker minimum;
- OpenRouter credentials were available outside the repository.

The WSL 2 support classification was recorded as a qualification limitation.

### 5.2 OpenShell control-plane gate

The CLI, gateway, and supervisor runtime were pinned to `v0.1.10`. The release
archive and sandbox binary were verified before use.

A callback-port conflict was encountered because another local service occupied
port 8080. The gateway was moved to 18080, with the same internal and callback
port explicitly configured. This demonstrated that a containerized Docker
gateway must align its published port with the callback endpoint handed to the
Supervisor.

The final gateway status was:

- connected;
- authenticated with mTLS;
- version 0.1.10.

### 5.3 Small sandbox isolation smoke test

A Python-only retained sandbox proved the runtime path independently from RFQ:

| Check | Result |
|---|---|
| Effective UID | 1000 |
| Effective GID | 1000 |
| Docker socket | absent |
| `/tmp` write | allowed |
| `https://example.com` | denied |
| OCSF denial | present |

The observed denial was:

```text
NET:OPEN [MED] DENIED /usr/local/bin/python3.12(65)
-> example.com:443 [reason:transparent_tcp_policy_denied]
```

This confirmed actual default-deny behavior rather than relying on policy
inspection alone.

### 5.4 RFQ process and policy gates

The RFQ worker was launched under an explicit policy. The policy:

- required Landlock rather than accepting best-effort fallback;
- set UID/GID 1000;
- made `/app`, `/opt/venv`, `/usr`, `/lib`, and `/etc` read-only;
- made `/app/data`, `/tmp`, and `/home/appuser` writable;
- allowed network destinations only through attached provider contributions.

The resulting RFQ checks passed:

- OpenRouter credential environment value present as a placeholder;
- placeholder did not have the normal real-key shape;
- OTLP endpoint present in the sandbox;
- no OTLP header secret visible;
- Docker socket absent;
- approved `/app/data` write succeeded;
- unapproved egress denied;
- RFQ agent profile list succeeded in **7.61 s**.

### 5.5 Real RFQ model call

The real `rfq-analyst` Deep Agent loaded:

- 17 tools;
- 11 skills;
- `pricing-specialist`;
- `risk-auditor`;
- the Deep Agents planning stack.

It called:

```text
POST https://openrouter.ai/api/v1/chat/completions
```

and returned:

```text
PHASE0_RFQ_AGENT_OK
```

The measured cold OpenShell create-and-run duration was **33.64 s**. The OpenShell
provisioning portion was sub-second after the image was present; most time was
RFQ Python/package startup, model initialization, and inference.

OpenShell logged:

```text
NET:OPEN [INFO] ALLOWED ... openrouter.ai:443
  [policy:_provider_phase0_openrouter engine:opa]

HTTP:POST [INFO] ALLOWED
POST https://openrouter.ai/api/v1/chat/completions
  [policy:_provider_phase0_openrouter engine:l7]
```

### 5.6 NeMo Relay ATOF gate

The workload wrote a real NeMo Relay trajectory:

- run ID: `01a21432-64a1-7f60-bf75-e5fb82042fd2`;
- profile: `RFQ Analyst`;
- model scope: `deepseek/deepseek-v4-flash`;
- 11 `skill.load` marks;
- start and end agent scopes;
- downloadable `events.jsonl` and `meta.json`.

The ATOF trajectory was exported before deleting the sandbox. This confirmed an
important lifecycle requirement: ephemeral sandbox deletion would otherwise lose
trajectory files stored under `/app/data`.

### 5.7 NeMo Relay OTLP gate

The first OTLP attempt exposed a real integration defect. NeMo Relay rejected
OpenShell's private HTTP relay endpoint because it was not loopback or HTTPS:

```text
OTLP trace endpoints require HTTPS; http to http://192.0.0.8:4318 is allowed
only for localhost or loopback IP addresses unless ... allow_insecure is enabled
```

Phase 1 fixed this in genai-tk by setting NeMo Relay's
`OpenTelemetryConfig.allow_insecure` only when the configured endpoint
explicitly uses `http://`. OpenShell remained responsible for the mediation and
did not expose the collector address or secret to the workload.

The corrected run emitted two spans:

- `deepseek/deepseek-v4-flash`;
- `RFQ Analyst`.

The captured resource attributes included:

```json
{
  "openshell.sandbox.id": "<sandbox UUID>",
  "openshell.telemetry.source": "agent",
  "service.name": "genai-tk-GenAI_demo",
  "telemetry.sdk.language": "rust"
}
```

The captured sandbox UUID exactly matched the OpenShell sandbox UUID. The
span-tagging gate passed.

## 6. Phase 0 findings

### 6.1 Strong benefits

1. **Credential containment**

   The model credential was absent as a real value from the workload. Static
   credential endpoint binding prevented a placeholder from being used on an
   unrelated allowed destination.

2. **Deny-by-default egress**

   A destination not in the effective provider policy was denied. The decision
   produced an OCSF record usable by a security platform.

3. **Process and filesystem isolation**

   OpenShell did not rely on the RFQ image's own trust boundary. UID 1000 and the
   filesystem policy were enforced by the sandbox runtime.

4. **No nested Docker**

   Whole-worker containment avoided the Docker-in-sandbox pattern. The RFQ image
   could omit OpenSandbox packages.

5. **Framework neutrality**

   The worker was still a normal genai-tk/LangChain Deep Agent. OpenShell did not
   require RFQ-specific agent logic to know about the sandbox.

6. **Security telemetry**

   OpenShell supplied OCSF process/network/HTTP decisions independently of RFQ
   application tracing.

7. **Agent telemetry correlation**

   OpenShell added the sandbox identity to NeMo Relay spans, allowing agent
   traces and security decisions to be correlated.

### 6.2 Compatibility and packaging findings

1. **Deep Agent `backend: none` imported OpenSandbox**

   genai-tk unconditionally imported `AioSandboxBackend` in two Deep Agent
   paths. A `none` backend therefore failed if the AIO optional extra was not
   installed.

   Phase 1 changed backend capability detection to use `add_volume` rather than
   concrete `AioSandboxBackend` imports. The final package proves that the RFQ
   Deep Agent works with `agent_sandbox` absent.

2. **The RFQ wheel did not contain generated BAML modules**

   `rfq_analyst.baml_client` existed in the checkout but was not present in the
   installed wheel. The Phase 1 image explicitly copies the generated BAML
   package into the installed package directory.

3. **The full RFQ dependency set was unsuitable for an inference worker**

   The standard RFQ metadata includes document conversion, Streamlit, and
   obsolete `harnessing` extra selection. The full resolution also reached
   `laya -> torch -> CUDA`. A single initial build started downloads including
   hundreds of MiB of NVIDIA libraries.

   The Phase 1 worker derives a dedicated inference-only manifest from the
   current project metadata and excludes:

   - RFQ UI dependencies;
   - `docling` and the obsolete `harnessing` extra;
   - `laya`, which pulled Torch/CUDA into this worker.

   A dedicated dependency extra should replace this temporary manifest
   transformation in a subsequent cleanup.

4. **Runtime model metadata was fetched from `models.dev`**

   The first worker lacked `data/models_dev.json`, so RFQ attempted a network
   fetch that policy correctly denied, then continued with reduced metadata.
   Phase 1 bakes the existing metadata snapshot. This removed the policy denial
   in normal operation.

5. **NeMo Relay rejected OpenShell's private HTTP relay address**

   The required `allow_insecure` setting was added for explicitly configured
   HTTP endpoints.

6. **OTLP service naming did not honor the desired RFQ project label**

   The captured resource name was `genai-tk-GenAI_demo`, not
   `rfq-analyst-openshell`. This does not break correlation, because the
   sandbox attributes were correct, but the monitoring configuration merge
   should be cleaned up before production.

## 7. Phase 1 implementation

Phase 1 package files are under
[`/home/tcl/prj/rfq-analyst/deploy/openshell`](../../../rfq-analyst/deploy/openshell):

| Artifact | Purpose |
|---|---|
| `Dockerfile` | Non-root immutable RFQ whole-worker runtime |
| `build.sh` | Builds a filtered context and a worker-specific locked dependency set |
| `policy.yaml` | Hard-required filesystem/process policy |
| `openrouter-profile.yaml` | Provider and L7 endpoint profile |
| `monitoring.yaml` | RFQ NeMo Relay/OTLP worker configuration |
| `README.md` | Build, provider, policy, and run procedure |

The following genai-tk changes were also implemented:

- removed two unconditional OpenSandbox imports from Deep Agent construction;
- enabled NeMo Relay for explicit HTTP OTLP relay endpoints;
- added a focused NeMo Relay configuration test.

The files are:

- `/home/tcl/prj/genai-tk/genai_tk/agents/langchain/factory.py`;
- `/home/tcl/prj/genai-tk/genai_tk/extra/monitoring/nemo_relay_setup.py`;
- `/home/tcl/prj/genai-tk/tests/unit_tests/extra/monitoring/test_nemo_relay_setup.py`.

### 7.1 Build properties

The final Phase 1 build:

- used a **42.27 MB** transferred context;
- generated a worker-specific lock from the filtered manifest;
- used `uv sync --locked --no-dev --no-editable`;
- installed 308 packages;
- excluded OpenSandbox/AIO sandbox packages;
- excluded Torch/CUDA from the worker path;
- produced a **658,613,957-byte** image;
- produced digest
  `sha256:434b2d26133a5fb4447ca2a4cd7f611f8456b6b3abbb9e5e553299c9affd273b`;
- ran as UID/GID 1000;
- exposed `python -m genai_tk.main.cli` as the image entry point.

### 7.2 Packaged-image validation

The final locked image was run with the documented OpenShell command. It:

- loaded NeMo Relay OTLP export;
- did not contact `models.dev`;
- loaded 17 tools and 11 skills;
- called OpenRouter successfully;
- returned:

  ```text
  PHASE1_LOCKED_OK
  ```

- emitted OpenRouter allow events;
- produced sandbox-tagged OTLP spans;
- contained no installed `agent_sandbox` package;
- imported the packaged BAML client successfully.

The measured 32.95-second full create-and-run result from the preceding packaged
run establishes a representative cold-process upper baseline for the simple
model-call scenario. A normal local offline response took 7.70 seconds, so the
small model smoke test adds setup/inference cost rather than proving that all
RFQ analyses are 25 seconds slower. A statistically useful latency SLO requires
multiple warm and cold samples on the target platform.

## 8. Security assessment

### 8.1 Controls proven

| Control | Evidence |
|---|---|
| Workload is non-root | UID/GID 1000 |
| Docker socket isolation | socket absent |
| Filesystem policy | hard Landlock requirement accepted |
| Default network deny | `example.com` denied |
| Provider-scoped network allow | OpenRouter L4 and L7 allow records |
| Credential opacity | workload did not observe a real-key shape |
| Collector credential separation | no OTLP headers exposed in workload |
| Security events | OCSF deny and allow records |
| Trace correlation | exact sandbox UUID added to agent spans |
| Secret-free image build | `.env` and uploaded RFQ data excluded |
| Generated BAML packaging | import succeeded |
| No nested sandbox runtime | `agent_sandbox` absent |

### 8.2 Controls not proven or not supplied

OpenShell and this pilot do not:

- prevent prompt injection;
- determine whether an allowed RFQ or pricing operation is semantically
  appropriate;
- redact prompts, system instructions, or tool arguments from NeMo Relay OTLP
  payloads;
- protect data once it is sent to an intentionally allowed destination;
- replace user authentication, tenant isolation, authorization, or approval
  workflows;
- secure a private pricing database unless a separate provider/profile or
  policy is authored and tested;
- prove CNI enforcement, because no Kubernetes cluster was used;
- prove OpenShell HA, PostgreSQL failover, rolling restart, or gateway scaling;
- prove native Linux production kernel qualification, because the host was
  WSL 2;
- scan the final image for vulnerabilities;
- sign or publish the image to an immutable registry;
- establish an image rollback procedure;
- qualify all RFQ tools, especially private pricing and graph dependencies.

The ATOF and OTLP span payloads contain agent inputs and outputs. Those traces
must be treated as sensitive data, even when OpenShell protects the transport
and collector credentials.

## 9. Tradeoffs and limitations

### 9.1 Whole-worker containment

**Benefit:** maximum boundary coverage with low RFQ coupling. Model calls,
Python tools, files, and sub-agents share the outer policy.

**Cost:** the entire RFQ dependency graph becomes part of the worker image. The
Phase 1 dependency filtering is necessary to avoid a multi-gigabyte UI/OCR/CUDA
image for a pure model-turn worker.

**Operational consequence:** different RFQ profiles need distinct images or
capability profiles. Document ingestion, interactive Streamlit, pricing
database use, and simple model chat should not automatically share one image
and one broad policy.

### 9.2 Credential mediation

**Benefit:** the workload does not need the real provider secret.

**Cost:** every compatible model host requires a reviewed provider profile, and
binary authorization is digest/path sensitive. Updating interpreter paths can
invalidate policy unless the profile is updated and tested.

### 9.3 OpenTelemetry relay

**Benefit:** the collector endpoint and collector credentials stay outside the
workload, and each span gains a sandbox UUID.

**Cost:** the pilot needed an explicit NeMo Relay change to permit OpenShell's
private non-loopback HTTP endpoint. The collector still receives trace
payloads, so data classification and retention controls remain necessary.

### 9.4 Static policy

**Benefit:** model and filesystem access are explicit and reviewable.

**Cost:** new agent dependencies, endpoints, or filesystem changes require a
policy update and, for startup fields, sandbox recreation. An RFQ tool that
discovers endpoints dynamically will need an explicit extension design rather
than a broad wildcard.

### 9.5 WSL 2

**Benefit:** this host was sufficient to prove the Docker driver, policy engine,
credential proxy, RFQ process, and trace relay.

**Cost:** OpenShell classifies WSL 2 as experimental. The security claims in
this report must not be used as production platform qualification.

## 10. Residual Phase 1 work

Phase 1 functionality passed, but the following cleanup should happen before
Phase 2:

1. Replace the Phase 1 `sed`-derived worker manifest with an explicit
   `worker`/`agent` optional dependency split in RFQ and genai-tk.
2. Decide whether `laya` belongs in genai-tk core or a decision-model extra.
   It currently materially changes image size through Torch/CUDA.
3. Add the generated BAML client to RFQ wheel contents or produce it in the
   image build from pinned BAML source.
4. Add a CI test that starts a Deep Agent with `backend: none` while AIO
   sandbox packages are absent.
5. Fix monitoring service-name selection so OpenShell RFQ spans identify the
   intended service instead of `genai-tk-GenAI_demo`.
6. Define trace redaction and prompt-data retention. At minimum, make prompt
   payload export an explicit profile choice.
7. Pin and verify `genai-tk`, `genai-graph`, and `prefect-yaml` by immutable
   commit or registry artifact instead of copying sibling checkouts.
8. Add image SBOM generation, vulnerability scanning, signing, digest capture,
   and immutable registry publication.
9. Add a policy test suite for:
   - private graph access;
   - pricing database access;
   - model access;
   - denied metadata addresses;
   - denied arbitrary DNS;
   - no Docker socket;
   - approved and denied write paths.
10. Add graceful ATOF export to the host orchestrator before ephemeral sandbox
    deletion.

## 11. Phase 2 entry criteria and plan

Phase 2 should run on a supported native Linux or Kubernetes deployment. Entry
criteria:

- Phase 1 package is rebuilt from immutable commits;
- OpenShell release, gateway image, supervisor image, and worker image are
  digest-pinned;
- the CNI is known to enforce ingress and egress NetworkPolicy where
  Kubernetes is selected;
- RFQ provider credentials are held in the selected OpenShell credential
  backend;
- trace retention and redaction policy is approved;
- image scanning and signing are available;
- a rollback image and previous policy are retained.

Phase 2 work:

1. Deploy OpenShell Helm chart or native Linux gateway with OIDC.
2. Publish the Phase 1 worker to a controlled registry.
3. Add one RFQ workspace, model provider, and environment-specific policy.
4. Verify direct workload egress is impossible at the CNI level.
5. Exercise RFQ graph and pricing endpoints under dedicated policies.
6. Test sandbox quotas, concurrency, cancellation, timeout, and cleanup.
7. Export OCSF JSON/OTLP and agent OTLP to the designated collectors.
8. Validate gateway restart, Supervisor loss, and interrupted command recovery.
9. Measure cold/warm latency and peak memory over repeated runs.
10. Test ATOF retrieval and retention across sandbox teardown.
11. Run a real RFQ acceptance sample against controlled synthetic or authorized
    data.
12. Document rollback and emergency policy-disable procedures.

## 12. Test evidence summary

| Test | Result |
|---|---|
| OpenShell CLI release checksum | Pass |
| Gateway mTLS connection | Pass |
| OpenShell version | 0.1.10 |
| Default unapproved network request | Denied |
| OCSF denial emitted | Pass |
| Real OpenRouter key visible in workload | No |
| Placeholder/opaque credential visible | Yes |
| OpenRouter connection | Allowed |
| OpenRouter POST L7 policy | Allowed |
| Docker socket available to RFQ worker | No |
| Workload identity | UID/GID 1000 |
| Hard Landlock policy loaded | Pass |
| RFQ agent startup without AIO sandbox package | Pass |
| RFQ agent model response | Pass |
| Generated BAML client in worker | Pass |
| NeMo Relay ATOF file export | Pass |
| NeMo Relay OTLP spans | Pass |
| Sandbox UUID in OTLP resources | Pass |
| Runtime `models.dev` fetch in final run | Absent |
| Phase 1 image build from filtered locked manifest | Pass |
| Final Phase 1 smoke command | Pass |
| Native supported-Linux qualification | Not run |
| Kubernetes/CNI qualification | Not run |
| Full RFQ customer workflow | Not run |

## 13. Validation performed

The following checks were run:

```bash
cd /home/tcl/prj/genai-tk
uv run pytest \
  tests/unit_tests/extra/monitoring/test_nemo_relay_setup.py -q
uv run ruff check \
  genai_tk/agents/langchain/factory.py \
  genai_tk/extra/monitoring/nemo_relay_setup.py \
  tests/unit_tests/extra/monitoring/test_nemo_relay_setup.py
```

The RFQ package was additionally validated with:

```bash
cd /home/tcl/prj/rfq-analyst
deploy/openshell/build.sh
openshell policy lint deploy/openshell/policy.yaml
openshell profile lint -f deploy/openshell/openrouter-profile.yaml
openshell sandbox create ... \
  -- python -m genai_tk.main.cli agents run rfq-analyst ...
```

The final locked image returned `PHASE1_LOCKED_OK`, emitted the expected
OpenRouter allow records, and sent spans containing the exact sandbox UUID.

## 14. Important operational note

The IDE selection contained a LangSmith credential. The value was not copied
into source, images, reports, or provider configuration. It has nevertheless
been exposed in conversation context and should be rotated in the credential
store. LangSmith tracing was explicitly disabled for the RFQ pilot.

## 15. Conclusion

OpenShell provided a concrete, measurable security improvement for RFQ Analyst:
opaque endpoint-bound model credentials, enforced outbound policy, non-root
filesystem/process isolation, OCSF decisions, and sandbox-tagged agent
telemetry. The RFQ agent remained a standard genai-tk Deep Agent and did not
require an OpenShell-specific model or orchestration rewrite.

The whole-worker route is ready for a controlled Phase 2 pilot. The current
result is not a production readiness claim because WSL 2 is experimental and the
private RFQ databases, full acceptance workload, Kubernetes CNI, HA, image
signing, vulnerability scanning, and trace-governance controls remain
unqualified.
