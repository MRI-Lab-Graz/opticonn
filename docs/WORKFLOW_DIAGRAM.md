# Braingraph Pipeline Workflow

```mermaid
graph TD
    %% Setup and Installation
    A[🛠️ Setup Environment] --> B[00_install_new.sh]
    B --> C[validate_setup.py]
    C --> D{Setup Valid?}
    D -->|No| E[Fix Issues]
    E --> C
    D -->|Yes| F[🚀 Ready to Start]
    
    %% Pipeline Entry Points
    F --> G[Choose Workflow]
    G --> H[JSON Test Configuration]
    G --> I[Manual Step-by-Step]
    G --> J[Legacy Shell Scripts]
    
    %% JSON Test Configuration Path (RECOMMENDED)
    H --> K[bootstrap_qa_validator.py create<br/>test_full_pipeline.json<br/>test_all_subjects.json<br/>test_extraction_only.json]
    K --> L[json_validator.py]
    L --> M{Config Valid?}
    M -->|No| N[Fix Configuration]
    N --> L
    M -->|Yes| O[run_pipeline.py --test-config]
    
    %% Bootstrap QA Validation Branch (DEFAULT)
    O --> OA{Bootstrap QA?}
    OA -->|No| P
    OA -->|Yes| OB[bootstrap_qa_validator.py create]
    OB --> OC[Run Wave 1 (20%)]
    OC --> OD[Run Wave 2 (20%)]
    OD --> OE[bootstrap_qa_validator.py validate]
    OE --> OF{QA Stable?}
    OF -->|Poor| OG[⚠️ Adjust Parameters]
    OF -->|Good| P
    OG --> OB
    
    %% Main Pipeline Steps
    O --> P[🔬 STEP 01: Connectivity Extraction]
    P --> Q[extract_connectivity_matrices.py]
    Q --> R[DSI Studio Processing]
    R --> S[organized_matrices/]
    S --> T[aggregate_network_measures.py]
    T --> U[aggregated_network_measures.csv]
    
    U --> V[⚙️ STEP 02: Quality Optimization]
    V --> W[metric_optimizer.py]
    W --> X[optimization_results/]
    
    X --> Y[🎯 STEP 03: Selection]
    Y --> Z[optimal_selection.py]
    Z --> AA[selected_combinations/]
    
    AA --> BB[📊 STEP 04: Statistical Analysis]
    BB --> CC[statistical_analysis.py]
    CC --> DD[statistical_results/]
    
    %% Quality Checks (Integrated)
    U --> EE[quick_quality_check.py]
    EE --> FF{Quality OK?}
    FF -->|Issues| GG[⚠️ Quality Warnings]
    FF -->|Good| V
    GG --> V
    
    %% Manual Step-by-Step Path
    I --> HH[Individual Steps]
    HH --> II[python run_pipeline.py --step 01]
    HH --> JJ[python run_pipeline.py --step 02]
    HH --> KK[python run_pipeline.py --step 03]
    HH --> LL[python run_pipeline.py --step 04]
    
    II --> Q
    JJ --> W
    KK --> Z
    LL --> CC
    
    %% Legacy Shell Script Path
    J --> MM[01_extract_connectome.sh]
    MM --> Q
    
    %% Configuration Files
    NN[📋 Configuration Files] --> OO[01_working_config.json<br/>optimal_config.json<br/>sweep_config.json<br/>production_config.json]
    OO --> Q
    OO --> O
    
    %% Utility Scripts
    PP[🛠️ Utility Scripts] --> QQ[json_validator.py<br/>quick_quality_check.py<br/>validate_setup.py<br/>statistical_metric_comparator.py<br/>verify_parameter_uniqueness.py]
    
    %% Windows Support
    RR[🪟 Windows Users] --> SS[00_install_windows.bat]
    SS --> C
    
    %% Styling
    classDef setupNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef mainStep fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef scriptNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef configNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef resultNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef utilityNode fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef qualityNode fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    
    class A,B,C,D,E,F,RR,SS setupNode
    class P,V,Y,BB mainStep
    class Q,W,Z,CC,O,II,JJ,KK,LL,MM scriptNode
    class K,NN,OO configNode
    class S,U,X,AA,DD resultNode
    class PP,QQ,L utilityNode
    class EE,FF,GG qualityNode
```

## 📋 Script Categories

### 🛠️ **Setup & Installation**
- `00_install_new.sh` - Environment setup (Unix/macOS)
- `00_install_windows.bat` - Environment setup (Windows)
- `validate_setup.py` - Validate DSI Studio and environment

### 🚀 **Main Pipeline Scripts**
- `run_pipeline.py` - Main orchestrator (JSON test configs + individual steps)
- `extract_connectivity_matrices.py` - Step 01: Connectivity extraction
- `metric_optimizer.py` - Step 02: Quality optimization
- `optimal_selection.py` - Step 03: Selection of best combinations
- `statistical_analysis.py` - Step 04: Statistical analysis

### 🔧 **Utility Scripts**
- `bootstrap_qa_validator.py` - Bootstrap QA validation with cross-validation (RECOMMENDED)
- `json_validator.py` - Validate JSON configurations
- `quick_quality_check.py` - Quality analysis and outlier detection
- `qa_cross_validator.py` - Compare QA results between random subsets (legacy)
- `aggregate_network_measures.py` - Combine network measures
- `statistical_metric_comparator.py` - Compare different parameter sets
- `verify_parameter_uniqueness.py` - Check parameter diversity

### 📋 **Configuration Files**
- `test_full_pipeline.json` - Complete 4-step test (5 subjects)
- `test_all_subjects.json` - Production run (all subjects)
- `test_extraction_only.json` - Step 01 only test
- `test_qa_validation_set_a.json` - QA validation subset A (4 subjects)
- `test_qa_validation_set_b.json` - QA validation subset B (4 subjects)
- `01_working_config.json` - DSI Studio extraction parameters
- `optimal_config.json` - Optimized parameters
- `sweep_config.json` - Parameter sweep configurations

### 🗂️ **Legacy Scripts**
- `01_extract_connectome.sh` - Legacy shell script for extraction
- `show_workflow.py` - Deprecated workflow display (can be removed)

## 🎯 **Recommended Workflow Paths**

### Path 1: Integrated Bootstrap QA Validation (RECOMMENDED - DEFAULT)
```bash
# Production run with integrated bootstrap QA (RECOMMENDED)
python run_pipeline.py --test-config test_all_subjects.json --enable-bootstrap-qa

# What happens automatically:
# 1. Creates bootstrap configurations (20% in 2 waves)  
# 2. Runs bootstrap wave 1 pipeline
# 3. Runs bootstrap wave 2 pipeline
# 4. Validates QA stability across waves
# 5. Proceeds with full dataset if QA passes
```

### Path 2: Simple Test Configuration (DEVELOPMENT)
```bash
./00_install_new.sh
python validate_setup.py --config 01_working_config.json
python run_pipeline.py --test-config test_full_pipeline.json
```

### Path 3: Manual Bootstrap QA (LEGACY)
```bash
# Manual bootstrap QA validation (legacy approach)
python bootstrap_qa_validator.py create /path/to/data
python run_pipeline.py --test-config bootstrap_configs/bootstrap_qa_wave_1.json
python run_pipeline.py --test-config bootstrap_configs/bootstrap_qa_wave_2.json
python bootstrap_qa_validator.py validate bootstrap_results_*
python run_pipeline.py --test-config test_all_subjects.json
```

### Path 3: Production Run (DIRECT)
```bash
python run_pipeline.py --test-config test_all_subjects.json
```

### Path 4: Individual Steps
```bash
python run_pipeline.py --step 01 --data-dir /data --extraction-config optimal_config.json
python run_pipeline.py --step 02
python run_pipeline.py --step 03
python run_pipeline.py --step 04
```

## 🔍 **Quality Assurance Integration**

Quality checks are automatically integrated:
- Parameter uniqueness validation
- Outlier detection across subjects
- Data completeness verification
- Network topology validation

The diagram shows how `quick_quality_check.py` is integrated into the main workflow for automatic quality assurance.
