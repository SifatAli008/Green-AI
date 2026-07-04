# System Architecture Diagrams (Mermaid)

This README contains Mermaid diagrams describing the proposed retrieval‑augmented small language model (SLM) system for medical AI. You can copy these snippets into any Mermaid‑compatible viewer (e.g., VS Code extension, Obsidian, `mermaid.live`) to render them.

## 1. High‑Level Software Architecture

```mermaid
flowchart TB
    %% ===== Layers =====
    subgraph L0[Users & External Systems]
        U1[Clinicians\n(Web UI)]
        U2[EHR / HIS\n(CDS Hooks, Plugins)]
        U3[Research Tools\nAnalytics, BI]
    end

    subgraph L1[Application Layer]
        A1[Web Frontend\nDashboard & QA UI]
        A2[EHR Integration Service\nSMART on FHIR / CDS Hooks]
        A3[API Gateway\nREST/JSON, Auth, Rate Limiting]
    end

    subgraph L2[Service Layer\n(RAG Services)]
        S1[Query Orchestrator\nRouting & Logging]
        S2[Preprocessing Service\nNormalize, De‑ID, Intent Detection]
        S3[Retrieval Service\nHybrid Search Orchestrator]
        S4[Generation Service\nSLM Inference]
        S5[Safety & Verification Service\nHallucination / Consistency Check]
        S6[Feedback Service\nCollect Ratings, Edits]
    end

    subgraph L3[Data & Knowledge Layer]
        D1[Clinical Data Ingestion\nHL7 FHIR, DICOM, CSV]
        D2[Preprocessed Text Store\nSectioned Clinical Notes]
        D3[Relational DB\nMetadata, Feedback, Logs]
        D4[Vector DB\nFAISS / ChromaDB]
        D5[Lexical Index\nBM25 / Inverted Index]
        D6[Structured KBs\nGuidelines, Ontologies, Drug DBs]
    end

    subgraph L4[Model & Algorithm Layer]
        M1[Embedding Model\nBioBERT / ClinicalBERT]
        M2[Small Language Model\n2–7B SLM (vLLM Engine)]
        M3[Reranker / Scorer\nRelevance & Diversity]
        M4[Trainer / Fine‑tuner\nOffline Jobs]
    end

    subgraph L5[Infrastructure Layer]
        I1[GPU Nodes\nSLM Inference, Training]
        I2[CPU Nodes\nETL, Indexing, Services]
        I3[Object Storage\nDocuments, Checkpoints]
        I4[Container Orchestrator\nKubernetes / On‑Prem Scheduler]
        I5[Monitoring Stack\nPrometheus / Grafana]
        I6[CI/CD Pipeline\nBuild & Deploy]
    end

    subgraph L6[Security & Compliance Layer]
        C1[Identity & Access Mgmt\nRBAC, SSO, MFA]
        C2[Encryption\nTLS, Disk Encryption]
        C3[Audit & Logging\nPHI Access Trails]
        C4[Privacy & DP Controls\nDe‑ID, DP‑SLMs]
        C5[Regulatory Governance\nHIPAA, GDPR, EU AI Act]
    end

    %% ===== Flows from Users to Application Layer =====
    U1 -->|HTTP(S)| A1
    U2 -->|FHIR/CDS Hooks| A2
    U3 -->|APIs| A3

    A1 -->|REST/JSON| A3
    A2 -->|REST/JSON| A3

    %% ===== Application to Service Layer =====
    A3 -->|Authenticated Request| S1
    S1 --> S2

    %% ===== Service Layer: Retrieval + Generation =====
    %% Preprocessing to Retrieval
    S2 -->|Preprocessed Query| S3

    %% Retrieval fan-out
    S3 -->|Lexical Query| D5
    S3 -->|Embedding Request| M1
    M1 -->|Embeddings| D4

    D5 -->|Candidate IDs| S3
    D4 -->|Nearest Neighbours| S3

    S3 -->|Top‑k Evidence IDs| M3
    M3 -->|Ranked Evidence| S3

    S3 -->|Context Chunks + Metadata| S4

    %% Generation + Safety
    S4 -->|Draft Answer + Evidence| S5
    S5 -->|Verified Answer + Flags| S1
    S1 -->|Response| A3
    A3 --> U1
    A3 --> U2

    %% Feedback Loop
    U1 -->|Ratings, Edits| A1
    A1 -->|Feedback| S6
    S6 --> D3
    S6 --> M4
    M4 -->|Updated Weights / Encoders| M1
    M4 -->|Updated SLM| M2

    %% ===== Data Layer Connections =====
    D1 -->|ETL Jobs| D2
    D2 -->|Chunking + Embedding| M1
    M1 -->|Initial Index Build| D4
    D2 -->|Index Build| D5
    D6 -->|Structured Facts| S3
    D6 -->|Grounding Info| S4

    %% ===== Infrastructure Bindings =====
    S1 & S2 & S3 & S4 & S5 & S6 --> I2
    M1 & M2 & M3 & M4 --> I1
    D2 & D3 & D4 & D5 & D6 --> I3
    I1 & I2 & D2 & D3 & D4 & D5 & D6 --> I4
    I1 & I2 & S1 & S2 & S3 & S4 & S5 & S6 --> I5
    I6 --> I4

    %% ===== Security & Compliance Overlays =====
    C1 --> A1
    C1 --> A2
    C1 --> A3
    C1 --> S1
    C2 --> I1
    C2 --> I2
    C2 --> I3
    C3 --> D3
    C3 --> S1
    C4 --> S2
    C4 --> D1
    C4 --> M4
    C5 --> C1
    C5 --> C4
```

## 2. Layered System Architecture (Paper Layers)

The following Mermaid diagram matches the five layers described in the paper:
Infrastructure Layer, Data Layer, Algorithm Layer, Application Layer, and Security \& Compliance Layer.

```mermaid
flowchart TB
    %% ========== APPLICATION LAYER ==========
    subgraph APP[Application Layer]
        APP1[Clinician Web UI\nQA, Documentation Support]
        APP2[EHR / HIS Integration\nSMART on FHIR / CDS Hooks]
        APP3[API Gateway\nREST/JSON, Auth, Rate Limiting]
    end

    %% ========== ALGORITHM LAYER ==========
    subgraph ALG[Algorithm Layer]
        ALG1[Preprocessing & Orchestration\nNormalize, De‑ID, Intent Detection]
        ALG2[Embedding Model\nBioBERT / ClinicalBERT]
        ALG3[Vector DB Client\nFAISS / ChromaDB]
        ALG4[Hybrid Retrieval\nSparse + Dense + Reranker]
        ALG5[Small Language Model\n2–7B SLM (vLLM Engine)]
        ALG6[Verifier / Safety Module\nHallucination & Consistency Check]
        ALG7[Trainer / Fine‑tuner\nOffline Updates]
        ALG8[External Knowledge Bases\nGuidelines, Ontologies, Drug DBs]
    end

    %% ========== DATA LAYER ==========
    subgraph DATA[Data Layer]
        D1[Clinical Data Ingestion\nHL7 FHIR, DICOM, CSV]
        D2[Preprocessing Pipeline\nDe‑ID, Sectioning, NLP]
        D3[Clinical Stores\nStructured DBs + Text Repository]
        D4[Vector Index\nFAISS / ChromaDB]
        D5[Lexical Index\nBM25 / Inverted Index]
        D6[Feedback & Logs Store]
    end

    %% ========== INFRASTRUCTURE LAYER ==========
    subgraph INF[Infrastructure Layer]
        I1[GPU Nodes\nSLM Inference & Training]
        I2[CPU Nodes\nETL, Indexing, Services]
        I3[Shared Storage\nDatabases, Object Store]
        I4[Cluster Orchestrator\nKubernetes / On‑prem Scheduler]
        I5[Monitoring Stack\nMetrics, Dashboards, Alerts]
    end

    %% ========== SECURITY & COMPLIANCE LAYER ==========
    subgraph SEC[Security & Compliance Layer]
        S1[Identity & Access Mgmt\nRBAC, SSO, MFA]
        S2[Encryption\nTLS, Disk Encryption]
        S3[Audit & Traceability\nQuery & PHI Access Logs]
        S4[Privacy Controls\nDe‑ID, DP‑SLMs, Data Residency]
        S5[Regulatory Governance\nHIPAA, GDPR, EU AI Act]
    end

    %% ===== FLOWS: APPLICATION -> ALGORITHM =====
    APP1 -->|Clinician queries / feedback| APP3
    APP2 -->|Context + queries| APP3
    APP3 -->|Authenticated request| ALG1

    %% ===== ALGORITHM <-> DATA =====
    ALG1 -->|Normalized query| ALG4
    ALG4 -->|Lexical filter| D5
    ALG4 -->|Embed request| ALG2
    ALG2 -->|Embeddings| D4
    D5 -->|Candidate IDs| ALG4
    D4 -->|Nearest neighbours| ALG4
    ALG4 -->|Top‑k evidence| ALG5
    ALG8 -->|Structured facts| ALG4
    ALG8 -->|Grounding context| ALG5

    ALG5 -->|Draft answer + evidence| ALG6
    ALG6 -->|Verified answer + uncertainty| APP3
    APP3 --> APP1
    APP3 --> APP2

    %% Feedback loop
    APP1 -->|Ratings, edits| D6
    D6 -->|Supervision| ALG7
    ALG7 -->|Updated encoders / SLM| ALG2
    ALG7 --> ALG5

    %% ===== DATA FLOW PIPELINES =====
    D1 -->|ETL jobs| D2
    D2 -->|Cleaned & de‑identified data| D3
    D3 -->|Chunking + embedding| ALG2
    D3 -->|Index build| D5
    ALG2 -->|Initial embeddings| D4

    %% ===== INFRASTRUCTURE BINDINGS =====
    ALG2 & ALG5 & ALG7 --> I1
    ALG1 & ALG3 & ALG4 & ALG6 --> I2
    D3 & D4 & D5 & D6 --> I3
    I1 & I2 & I3 --> I4
    I1 & I2 & ALG1 & ALG4 & ALG5 & ALG6 --> I5

    %% ===== SECURITY OVERLAY =====
    S1 --> APP1 & APP2 & APP3
    S2 --> I1 & I2 & I3
    S3 --> D6 & APP3
    S4 --> D1 & D2 & ALG7
    S5 --> S1 & S4
```

## 3. RAG Pipeline Detail

```mermaid
flowchart LR
    Q[Clinician Query] --> P1[Preprocess Query\nNormalize, Intent Detection]

    P1 --> R1[Hybrid Retrieval Orchestrator]

    subgraph RETRIEVAL[Retrieval Stage]
        R2[Lexical Filter\nBM25 / Keyword Rules]
        R3[Dense Search\nVector DB (FAISS / ChromaDB)]
        R4[Reranker\nTop‑k Evidence Selection]
    end

    R1 --> R2 --> R3 --> R4

    R4 --> C1[Context Builder\nAssemble Evidence + Metadata]
    C1 --> P2[Prompt Orchestrator\nInstructions + Evidence + Query]

    P2 --> M1[SLM Inference\n2–7B Small Language Model]
    M1 --> V1[Verifier / Safety Check\nHallucinations, Consistency]

    V1 --> A1[Final Answer\nCitations + Uncertainty Labels]
    A1 --> F1[Feedback Capture\nClinician Ratings, Edits]
    F1 --> U1[Continuous Learning\nUpdate Retrieval, Prompts, Models]
```

You can further simplify or split these diagrams (e.g., by layer or by subsystem) if you need smaller figures for documentation or slide decks.

## 4. Implementation and Reproducibility (Text Summary)

### 4.1 Software Architecture
The software architecture for the proposed retrieval‑augmented small language model (SLM) in medical AI follows a modular, multi‑layered design that separates infrastructure, data management, algorithms, applications, and compliance concerns. This separation of concerns supports scalability, maintainability, and regulatory conformance in hospital and public‑health deployments.

Concretely, the architecture is organized into five layers:

- **Infrastructure Layer**: GPU and CPU nodes, shared storage, container orchestration, and monitoring that provide resilient compute, storage, and networking for all services.
- **Data Layer**: Ingestion pipelines for HL7 FHIR and DICOM, de‑identification and NLP preprocessing, and structured/textual stores including vector and lexical indices.
- **Algorithm Layer**: Embedding encoders, hybrid retrieval over FAISS/ChromaDB and BM25, the 2–7B SLM inference engine, verifier/safety modules, and offline training/fine‑tuning jobs.
- **Application Layer**: Clinician‑facing web UI, EHR/HIS integrations, and REST APIs that embed the system into clinical workflows and expose decision‑support functionality.
- **Security and Compliance Layer**: Identity and access management, encryption, audit logging, privacy‑preserving mechanisms (de‑identification, DP‑aware SLMs), and governance processes aligned with HIPAA, GDPR, and the EU AI Act.

#### Infrastructure Layer
The infrastructure layer provisions compute, storage, and networking resources for end‑to‑end clinical workloads. GPU‑enabled servers host the SLM inference engine, while CPU nodes handle data preprocessing, indexing, and logging. Containerization (e.g., Docker) and orchestration (e.g., Kubernetes or equivalent on‑premise schedulers) are used to scale retrieval and inference replicas independently, maintain high availability, and isolate workloads across institutions or departments.

#### Data Layer
The data layer manages ingestion, preprocessing, and storage of medical data. Structured clinical data (e.g., demographics, laboratory values, diagnoses) are exposed through standards such as HL7 FHIR, while unstructured data (discharge summaries, pathology reports) are stored as normalized clinical text. Where imaging metadata are needed, the system integrates with DICOM‑compliant PACS. A preprocessing pipeline performs de‑identification, sentence segmentation, section detection, and controlled vocabulary mapping before persisting data into relational stores and document collections used by the retrieval layer.

#### Algorithm Layer
The algorithm layer contains the small language model, the retrieval components, and connections to external knowledge bases. A distilled or instruction‑tuned SLM (e.g., 2–7B parameters) runs behind a vLLM‑style inference engine, fronted by a hybrid retrieval stack that combines sparse keyword search with dense vector search over FAISS or ChromaDB indices. Biomedical ontologies and curated guideline collections are integrated as structured knowledge bases to ground responses, and this layer is responsible for training, fine‑tuning, inference, and continuous learning from feedback.

#### Application Layer
The application layer provides clinician‑facing tools and integrations with existing hospital systems. This includes web dashboards for question answering and documentation support, EHR plug‑ins that surface suggestions within clinician workflows, and RESTful APIs for integration with clinical decision support, order entry, and triage systems. All user interfaces are designed to surface retrieved evidence alongside generated answers, expose uncertainty indicators, and support clinician feedback on correctness and usefulness.

#### Security and Compliance Layer
The security and compliance layer implements privacy‑preserving mechanisms and governance features required for HIPAA, GDPR, and EU AI Act alignment. Role‑based access control is enforced across all components, with audit logging of queries, retrieved documents, and model outputs. Data at rest are encrypted using disk‑level encryption, while data in transit are protected using TLS. Differential privacy, anonymization of identifiers, and strict data residency controls ensure that PHI remains within approved jurisdictions and that cross‑border model evaluation uses synthetic or de‑identified data.

### 4.2 Retrieval‑Augmented Generation Pipeline
The retrieval‑augmented generation (RAG) pipeline operationalizes the interaction between retrieval components and the SLM. It is implemented as a sequence of services that can be independently scaled and monitored.

#### Embedding Model and Chunking
Clinical documents and guideline texts are segmented into semantically coherent chunks (e.g., sections for indications, contraindications, dosing, and adverse effects). Each chunk is embedded using a biomedical encoder such as BioBERT or ClinicalBERT, optionally fine‑tuned for clinical similarity tasks. Chunk sizes and overlap are empirically selected to balance recall (capturing sufficient context) and latency.

#### Vector Database and Hybrid Retrieval
Embeddings are stored in a vector database such as FAISS or ChromaDB, configured for approximate nearest‑neighbour search and sharded across nodes when datasets exceed single‑node memory. At query time, the system first applies lexical filtering (BM25 or keyword rules) to narrow the candidate set, then performs dense retrieval to obtain the top‑k evidence chunks. This hybrid strategy improves both recall and precision while keeping latency within clinical expectations.

#### Prompt Orchestration and SLM Inference
Retrieved evidence is assembled into structured prompts that include the user query, provenance metadata (source, date, tier of authority), and concise instructions emphasizing citation of sources, honest acknowledgment of uncertainty, and avoidance of unsupported recommendations. The SLM generates candidate answers, which are optionally passed through a verifier module to flag unsupported or low‑confidence statements. Temperature and maximum‑token settings are tuned to favour concise, guideline‑aligned outputs rather than speculative reasoning.

### 4.3 Integration and Deployment

#### API Design
Core RAG functions—document ingestion, embedding, retrieval, and SLM inference—are exposed via RESTful APIs with JSON payloads. These APIs are consumed by EHR plug‑ins, hospital portals, and research dashboards. API gateways enforce authentication, rate limiting, and request validation, and separate internal administrative endpoints (e.g., re‑indexing, model updates) from clinician‑facing endpoints.

#### Continuous Learning and Feedback
Clinician feedback on each response (correct, partially correct, unsafe, missing information) is logged and stored in a feedback database. Periodic offline training jobs use this feedback to update retrieval ranking models, refine prompts, and, when governance permits, fine‑tune the SLM on de‑identified corrections. New external evidence (e.g., updated guidelines, new clinical trials) is ingested on a scheduled basis and re‑indexed, ensuring that retrieval and generation reflect the latest medical knowledge.

#### Security Measures
All components are deployed within a zero‑trust perimeter, with network segmentation between application, data, and infrastructure layers. Secrets (database credentials, API keys, encryption keys) are stored in secure vault services and rotated regularly. Access to logs containing PHI is tightly controlled, and synthetic or redacted logs are used for most engineering and monitoring tasks. Penetration tests and privacy impact assessments are conducted prior to production deployment and after major updates.

### 4.4 Evaluation, Monitoring, and Reproducibility

#### Operational Monitoring
Centralized logging and metrics collection track query volume, latency distributions for retrieval and generation, GPU and memory utilization, and failure rates. Specialized dashboards report clinical metrics such as retrieval coverage for guideline citations, hallucination‑flag rates, and distribution of uncertainty labels. Alerts are configured for latency spikes, error bursts, or anomalous access patterns.

#### Regulatory and Privacy Audits
Regular internal audits verify that PHI handling, access controls, and logging conform to institutional policies and external regulations (HIPAA, GDPR, EU AI Act). Data‑processing agreements and records of processing activities are maintained, and privacy threat models are updated as system capabilities evolve. When models are retrained or new data sources are added, change‑impact assessments are documented and reviewed by clinical governance boards.

#### Software Stack and Artefact Availability
To support reproducibility, the full software stack (model versions, embedding encoders, retrieval configuration, and evaluation scripts) is defined in version‑controlled configuration files and container images. Where licensing permits, configuration templates, synthetic datasets, and evaluation code are made publicly available, enabling other institutions to replicate the pipeline on their own data while preserving local data sovereignty.


