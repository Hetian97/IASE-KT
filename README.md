# IASE-KT: Multi-Concept Knowledge Tracing with State-Dependent Feedback Allocation and Adaptive Computation

IASE-KT is a PyTorch knowledge tracing model for predicting item-level
correctness when an exercise involves multiple knowledge components
(KCs). It allocates item-level feedback across the KCs in the current
item, separates short-term response information from long-term knowledge
states, and selectively performs per-KC comparisons when the initial
allocation is ambiguous.

This repository provides the model implementation, data preparation,
training and evaluation scripts, experiment settings, and executable
protocol checks.

-   **Code:** https://github.com/Hetian97/IASE-KT.git
-   **Manuscript experiment environment:** Ubuntu 24.04, Python 3.11,
    PyTorch 2.7.1+cu128, NVIDIA RTX 5090D, Ryzen 9 9950X, and 128 GB
    RAM.
-   **Local protocol-check environment:** Windows, Python 3.12, and
    PyTorch 2.7.1 CPU.
-   **Data:** Obtained from the original public sources; learner records
    are not redistributed in this repository.

## 1. Method Overview

IASE-KT predicts the probability of a correct response from the
learner's preceding response history, the current item, and its complete
KC set. After the observed outcome becomes available, the model updates
the states of the KCs associated with that item.

  -----------------------------------------------------------------------
  Component                           Computation and purpose
  ----------------------------------- -----------------------------------
  **Relative Contribution Allocation  Uses local likelihood sensitivity
  (RCA)**                             to estimate how each KC state
                                      affects the long-term prediction,
                                      producing initial
                                      feedback-allocation weights.

  **Prediction Timescale Separation   Combines long- and short-term
  (PTS)**                             logits for prediction and scales
                                      the long-term residual by the
                                      magnitude of the short-term output.

  **Importance-Aware Weighting        Combines allocation entropy with
  (IAW)**                             Jaccard relations between KCs
                                      calculated from training items to
                                      select ambiguous responses for
                                      refinement and regulate update
                                      coefficients.

  **Selective per-KC refinement**     Resets one KC state to its initial
                                      value at a time and recomputes the
                                      long-term likelihood; this extra
                                      comparison is used only when the
                                      refinement criterion is met.
  -----------------------------------------------------------------------

The current response is predicted before its outcome enters the
learner's history or updates knowledge states. In EdNet, items in the
same solving group are predicted from the same pre-feedback state
snapshot; feedback is then processed in item-ID and original-row order.

## 2. Repository Structure

``` text
IASE-KT/
├── config.py          # Model and experiment configurations
├── data.py            # Response ingestion, data splits, and KC relations
├── model.py           # Causal encoder, RCA, PTS, IAW, and learner state
├── losses.py          # Event-masked losses and parameter regularization
├── train.py           # Training and validation-based checkpoint selection
├── evaluate.py        # AUC, accuracy, FC-AUC, and checkpoint assessment
├── validate.py        # Numerical and protocol checks
├── requirements.txt   # Python dependencies
├── README.md
└── validation/        # Validation records and final check results
```

## 3. Datasets

The experiments use XES3G5M, EdNet-KT1-50K, and ASSIST2009-SB. Download
each dataset from its original source and comply with its access
conditions and terms of use.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Dataset             Statistics        Student split (train Source
                      reported for the  / validation / test) 
                      study                                  
  ------------------- ---------------- --------------------- -----------------------------------------------------------------------------------------------------------
  **XES3G5M**         18,066 students,      12,646 / 1,807 / https://github.com/ai4ed/XES3G5M
                      7,652 items, 865                 3,613 
                      KCs, 5,549,635                         
                      interactions                           

  **EdNet-KT1-50K**   Sample of 50,000      35,000 / 5,000 / https://github.com/riiid/ednet
                      students from                   10,000 
                      EdNet-KT1; full                        
                      dataset: 784,309                       
                      students, 13,169                       
                      items, 188 KCs,                        
                      95,293,926                             
                      interactions                           

  **ASSIST2009-SB**   4,151 students,      2,905 / 415 / 831 https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
                      16,891 items,                          
                      110 KCs, 325,637                       
                      interactions                           
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Data preparation rules

-   Each canonical row represents one complete response, including its
    full KC set and one binary correctness outcome.
-   Rows sharing a learner and event ID are merged only when item,
    outcome, position, group, and original-row metadata agree. Their KC
    sets are combined. Different event IDs remain separate, including
    repeated attempts on the same item.
-   Students are split into training, validation, and test sets.
    Test-student responses do not contribute to training losses or
    training-set KC relations.
-   Sampling and splitting use SHA-256 ordering. Sampling uses
    `IASEKT-v1|sample|dataset name|student ID`; splitting substitutes
    `split` for `sample`. Sampling precedes partitioning, and all random
    seeds use the same student partitions.
-   Preparation commands write `canonical.csv`, `vocab.json`, and
    `split.json`. The vocabulary maps original item and KC IDs to
    compact integer IDs while retaining the original mappings.

### Dataset-specific notes

**XES3G5M.** The public preprocessed sequence files do not contain
unique interaction identifiers, so the complete original event stream
cannot be reliably reconstructed from those files alone. This
implementation expects a canonical event export that preserves complete
KC annotations and unique event IDs.

**EdNet-KT1-50K.** The adapter reads one KT1 CSV file per learner and
the official `questions.csv` metadata. It selects learners before
parsing their response files, retains solving groups and original row
indices, and derives correctness from `user_answer` and
`correct_answer`.

**ASSIST2009-SB.** The adapter reads a raw Skill Builder CSV containing
`user_id`, `problem_id`, `skill_id`, `correct`, and `order_id`. It
combines multi-KC rows belonging to the same response, orders events by
`order_id` and original row, and retains independent repeated responses.
ASSIST versions differ in filtering and duplicate handling; the
exact-count command requires the 4,151-student version and rejects
mismatched counts rather than changing the data to force a match.

### Canonical data format

Each row in `canonical.csv` represents one complete response.

  -----------------------------------------------------------------------
  Column                  Type                    Description
  ----------------------- ----------------------- -----------------------
  `student_id`            String                  Learner identifier;
                                                  surrounding whitespace
                                                  is removed.

  `item_id`               Integer                 Item identifier.

  `concept_ids`           String                  Complete set of
                                                  distinct KCs, separated
                                                  by `|`.

  `outcome`               `0` or `1`              Observed item-level
                                                  correctness.

  `position`              Integer                 Response order or
                                                  ordering timestamp.

  `event_id`              String                  Unique response
                                                  identifier within a
                                                  learner.

  `group_id`              Optional string         Atomic solving-group
                                                  identifier; empty for
                                                  independent responses.

  `original_row`          Optional integer        Original source-row
                                                  index used to resolve
                                                  update order.
  -----------------------------------------------------------------------

## 4. Requirements and Installation

The manuscript experiments used Ubuntu 24.04, Python 3.11, and PyTorch
2.7.1+cu128. Local protocol checks were also run on Windows with Python
3.12 and PyTorch 2.7.1 CPU; these are separate execution settings.
Dependencies include PyTorch and NumPy, with SciPy for the optional
paired-seed comparison. See `requirements.txt` for the package list.

``` bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

``` bash
source .venv/bin/activate
```

Install dependencies and run local checks:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python validate.py --report validation/local_checks.json
```

For CUDA 12.8, install the matching PyTorch build according to the
official PyTorch installation instructions. The following command
matches the version used in the manuscript environment:

``` bash
python -m pip install torch==2.7.1+cu128 --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128
```

## 5. Prepare the Datasets

Replace example paths with the locations of downloaded files.

**XES3G5M** --- use a complete event export that preserves unique event
IDs and complete KC annotations:

``` bash
python data.py canonical --input /path/to/xes_complete_events.csv \
  --output data/XES3G5M --dataset XES3G5M \
  --counts 12646 1807 3613
```

The preprocessed public sequence files alone are insufficient to
reconstruct unique original events reliably.

**EdNet-KT1-50K:**

``` bash
python data.py ednet --input /path/to/KT1 \
  --questions /path/to/contents/questions.csv \
  --output data/EdNet-KT1-50K --dataset EdNet-KT1-50K \
  --sample-size 50000 --counts 35000 5000 10000
```

**ASSIST2009-SB:**

``` bash
python data.py assist --input /path/to/skill_builder_data.csv \
  --output data/ASSIST2009-SB --dataset ASSIST2009-SB \
  --counts 2905 415 831
```

The ASSIST adapter requires the raw Skill Builder columns listed above.
Preprocessed three-line sequences from the AKT repository do not contain
the raw event identifiers required by this adapter; use a canonical
export if those identifiers were preserved during preprocessing.

## 6. Train and Evaluate

Train with the five manuscript seeds (`2022`--`2026`):

``` bash
python train.py --data data/EdNet-KT1-50K/canonical.csv \
  --dataset EdNet-KT1-50K --output runs/EdNet-KT1-50K \
  --seeds 2022 2023 2024 2025 2026
```

For compatible CUDA hardware, BF16 backbone execution can be selected
explicitly:

``` bash
python train.py --data data/EdNet-KT1-50K/canonical.csv \
  --dataset EdNet-KT1-50K --output runs/EdNet-KT1-50K \
  --device cuda --precision bfloat16
```

Evaluate a selected checkpoint:

``` bash
python evaluate.py \
  --checkpoint runs/EdNet-KT1-50K/seed_2022/best.pt \
  --data data/EdNet-KT1-50K/canonical.csv --split test \
  --output runs/EdNet-KT1-50K/seed_2022/assessment.json
```

For Windows PowerShell, use one-line commands or replace Bash
line-continuation backslashes with PowerShell backticks.

### Computation branch options

The default `adaptive` mode applies the entropy/co-occurrence trigger.
`base_only` uses the initial sensitivity weights, while `refine_only`
refines every multi-concept response. Single-KC responses always use the
base branch.

``` bash
python train.py --data data/EdNet-KT1-50K/canonical.csv \
  --dataset EdNet-KT1-50K --output runs/base \
  --branch-mode base_only --seeds 2022

python train.py --data data/EdNet-KT1-50K/canonical.csv \
  --dataset EdNet-KT1-50K --output runs/refinement \
  --branch-mode refine_only --seeds 2022
```

## 7. Experimental Configuration and Evaluation Protocol

Unless overridden by the command line, the manuscript's main settings
are:

  -----------------------------------------------------------------------
  Setting                                                           Value
  ------------------------------ ----------------------------------------
  State and interaction                                               128
  dimension                      

  Causal encoder layers /                                           2 / 4
  attention heads                

  Encoder feed-forward dimension                                      512

  Dropout                                                             0.1

  First segment / new events in                           200 / 100 / 100
  later segments / retained      
  history                        

  Short-term response window                                           20

  Attribution temperature, `T_a`                                      0.7

  Attenuation temperature, `T_r`                                      1.0

  Refinement threshold, `tau`                                        0.55

  Shared refinement-map                                                 2
  iterations                     

  Entropy exponent, `gamma`                                             1

  Optimizer / learning rate                                 AdamW / 0.001

  Effective batch size                                      128 sequences

  Learning-rate warm-up               First 5% of planned optimizer steps

  Maximum epochs /                                               100 / 10
  early-stopping patience        

  Maximum global gradient norm                                        1.0

  Separation / consistency /                   0.03 / 0.1 / 0.01 / 0.0001
  structural / L2 loss weights   
  -----------------------------------------------------------------------

Evaluation conventions:

-   The first valid response initializes the learner trajectory and is
    excluded from prediction losses and metrics. Its feedback updates
    state and history.
-   Each subsequent valid response receives one prediction and one state
    update. Overlapping windows carry historical context without
    repeating feedback updates; learner states persist across windows,
    while gradients stop at segment boundaries.
-   Predictions are saved before the current outcome is revealed. EdNet
    items in the same solving group share a pre-feedback snapshot;
    updates follow item ID and original-row order.
-   KC Jaccard relations count distinct training items and remain fixed
    during validation and testing.
-   The best checkpoint is selected by validation AUC. Test assessment
    occurs after checkpoint restoration.
-   Local sensitivity requires autograd during assessment; use
    `torch.no_grad()` rather than `torch.inference_mode()`.
-   The objective includes explicit L2 regularization. AdamW uses
    `weight_decay=0` to avoid adding a second decoupled weight penalty.
-   Loss terms are normalized over valid events in the complete
    optimizer batch. The centered-logit separation statistic is
    calculated over that same batch, so changing `--batch-size` changes
    the experimental configuration.

The manuscript does not fully specify training initialization for the
first response. The implementation convention is described below; report
this choice when reproducing or modifying the protocol.

## 8. Outputs

Each seed creates a directory containing the selected checkpoint and
records:

``` text
seed_2022/
├── best.pt       # Parameters, optimizer state, configuration, splits, and KC relations
├── history.json  # Epoch losses and validation measurements
└── result.json   # Assessment of the selected checkpoint
```

The run-level `summary.json` aggregates seed results using sample
standard deviations. ROC AUC is recorded as `null` when only one outcome
class is present; validation checkpoint selection requires both classes.

FC-AUC evaluates discrimination on matched subsequent single-KC
responses. For each multi-KC event/concept pair, the protocol finds the
first subsequent single-KC response containing that concept within 20
valid interactions and uses its online prediction before outcome
revelation. The matching protocol retains its specified event/concept
multiplicity.

## 9. Implementation Notes

The following conventions cover details not fully specified by the
manuscript equations:

-   Learned position embeddings index retained history locally from zero
    and are reassigned when the context window changes.
-   The causal encoder uses post-normalized attention blocks and reads
    the last history token. Explicit attention-matrix operations support
    higher-order derivatives.
-   The prediction hidden width and default refinement hidden width
    equal the state dimension. `refinement_dim` remains configurable.
-   Each KC's initial reset anchor is cached at its first encounter in a
    learner trajectory. States and anchors are detached at segment
    boundaries.
-   The first response updates state and history but does not contribute
    prediction loss. Its training initialization is not fully specified
    in the manuscript.
-   `differentiate_sensitivity=True` allows training losses to
    differentiate through the likelihood gradient. Setting it to `False`
    stops the gradient-of-gradient path while retaining the
    state-displacement path.
-   At the zero entropy-factor boundary, the implementation uses a
    finite zero derivative for positive exponents; exponent zero has
    unit scale.
-   Ordinary responses reuse saved prediction tensors for feedback
    calculations. Solving groups recompute update quantities after label
    revelation, while prediction and separation losses use saved
    pre-feedback logits.
-   The actual parameter count follows tensor dimensions and vocabulary
    sizes; unused parameters are not added to match a reported total.

These conventions form part of the executable configuration. Changing
them requires a new benchmark run, and the changed settings should be
reported.

## 10. Validation and Reproducibility Status

Run the protocol checks with:

``` bash
python validate.py --report validation/local_checks.json
```

The checks cover numerical loss calculations, likelihood sensitivity
against finite differences, prediction causality, learner isolation,
solving-group timing and ordering, state-update scope, entropy
boundaries, CPU BF16, deterministic partitions, checkpoint restoration,
and independent pairwise ROC AUC calculations.

Review records and the final check inventory are stored in
`validation/review.json` and `validation/final_checks.json`. These
deterministic fixture checks assess executable behavior; they do not
establish full benchmark reproduction.

**Benchmark status:** The repository's executable protocol checks are
available. Full training on XES3G5M, EdNet-KT1-50K, and ASSIST2009-SB,
reproduction of the manuscript's AUC results, and CUDA latency/MAC
measurements have not been reproduced in this release. Do not treat
fixture scores as dataset benchmark results.

## 11. References and Citation

If you use the datasets, cite their original publications and follow the
source repositories' terms.

-   **XES3G5M:** Liu, Z. et al. (2023). "XES3G5M: A knowledge tracing
    benchmark dataset with auxiliary information." *Advances in Neural
    Information Processing Systems*, 36, 32958--32970. Dataset:
    https://github.com/ai4ed/XES3G5M
-   **EdNet:** Choi, Y. et al. (2020). "EdNet: A large-scale
    hierarchical dataset in education." In *Artificial Intelligence in
    Education*, LNCS 12164, pp. 69--73. Dataset:
    https://github.com/riiid/ednet
-   **ASSISTments 2009--2010 Skill Builder:** Dataset and access
    information:
    https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

Associated manuscript: Wang, Hetian; Liu, Chunhui; Zhao, Li.
"Multi-concept knowledge tracing with state-dependent feedback
allocation and adaptive computation." Add the final venue, year, and DOI
here when publication metadata are available.

## 12. License and Contributions

A repository license is not identified in the supplied project
materials. Check the repository for a `LICENSE` file before reusing or
redistributing the code; dataset terms are separate from the code
license.

Bug reports and reproducibility corrections can include the operating
system, Python and PyTorch versions, exact command, configuration,
random seed, and relevant logs. Contributions that change data
processing, model behavior, or evaluation should include corresponding
validation checks and clearly document any protocol changes.
