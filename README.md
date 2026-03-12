# Resume Analyzer Agent

A data‑science powered pipeline that ingests job descriptions and resumes and produces a tailored resume
highlighting gaps, human‑focused bullet points, and ATS‑compliant formatting. The architecture is built
as a sequence of specialized agents orchestrated by a SequentialAgent.

## Features

* **Extraction Agent** – parses skills and timelines from resume and JD.
* **Gap Analysis Agent** – identifies mismatches between candidate background and job requirements.
* **Generation Agent** – crafts impactful, human‑readable bullet points.
* **Validation Agent** – performs ATS checks, consistency verifications and date validations.
* **Integration Agent** – assembles the final document.

## Quick start

1. Clone repository and create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the main script with your inputs:

```bash
python main.py --resume resume.pdf --jd jd.txt
```

(Adjust flags to your deployment; `main.py` exposes a CLI.)

## Project Structure

```
agents/           # individual agent implementations
llm/              # LLM client and schemas
pipeline/         # orchestration code
tools/            # helpers for parsing, analysis, etc.
sample_data/      # example job description and resume fragments
state/            # session state management
```

## Flow Chart


```mermaid

flowchart TD
    Start((User Input)):::input --> Root[Root: SequentialAgent]:::root

    subgraph Pipeline["Linear Orchestration"]
        direction TB
        Root --> S1[Extraction Agent]:::extract
        S1 --> S2[Gap Analysis Agent]:::gap
        S2 --> S3[Generation Agent]:::generate
        S3 --> S4[Validation Agent]:::validate
        S4 --> S5[Integration Agent]:::integrate
    end

    %% Step outputs / artifacts
    S1 -.-> T1(["Skills & Timeline"]):::artifact
    S2 -.-> T2(["JD vs Resume Gap"]):::artifact
    S3 -.-> T3(["Human-Impact Bullets"]):::artifact
    S4 -.-> T4(["ATS + Consistency + Timeframe Checks"]):::artifact

    %% Shared state
    State[(Session State)]:::state
    S1 -.-> State
    S2 -.-> State
    S3 -.-> State
    S4 -.-> State
    S5 -.-> State

    S5 --> End((Tailored Resume)):::output

    %% Color Classes
    classDef input fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1,stroke-width:2px;
    classDef root fill:#E8F5E9,stroke:#43A047,color:#1B5E20,stroke-width:2px;

    classDef extract fill:#FFF3E0,stroke:#FB8C00,color:#E65100,stroke-width:2px;
    classDef gap fill:#F3E5F5,stroke:#8E24AA,color:#4A148C,stroke-width:2px;
    classDef generate fill:#E0F7FA,stroke:#00ACC1,color:#006064,stroke-width:2px;
    classDef validate fill:#FFFDE7,stroke:#FDD835,color:#F57F17,stroke-width:2px;
    classDef integrate fill:#EDE7F6,stroke:#5E35B1,color:#311B92,stroke-width:2px;

    classDef artifact fill:#ECEFF1,stroke:#546E7A,color:#263238,stroke-dasharray: 5 5;
    classDef state fill:#F1F8E9,stroke:#7CB342,color:#33691E,stroke-width:2px;

    classDef output fill:#E8EAF6,stroke:#3949AB,color:#1A237E,stroke-width:2px;
  ```
