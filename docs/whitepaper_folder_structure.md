# Project Albedo: Whitepaper Folder Structure

The following is the Overleaf (LaTeX) folder structure for the white paper to be released along with a research paper (either published or pre-print (undecided)).

```bash
Project-Albedo-Whitepaper/
│
├── main.tex                     % Main compilation file
├── preamble.tex                 % Packages, macros, theorem styles
├── references.bib               % Bibliography
├── glossary.tex                 % Acronyms & glossary
├── notation.tex                 % Mathematical notation
├── acknowledgements.tex
├── abstract.tex
│
├── frontmatter/
│   ├── cover.tex
│   ├── titlepage.tex
│   ├── copyright.tex
│   ├── preface.tex
│   └── executive_summary.tex
│
├── chapters/
│   │
│   ├── 01_introduction/
│   │   ├── chapter.tex
│   │   ├── motivation.tex
│   │   ├── bottleneck.tex
│   │   ├── objectives.tex
│   │   └── images/
│   │
│   ├── 02_rendering_foundations/
│   │   ├── chapter.tex
│   │   ├── light_transport.tex
│   │   ├── rendering_equation.tex
│   │   ├── monte_carlo.tex
│   │   ├── gbuffers.tex
│   │   └── images/
│   │
│   ├── 03_mathematical_foundations/
│   │   ├── chapter.tex
│   │   ├── linear_algebra.tex
│   │   ├── multivariable_calculus.tex
│   │   ├── probability.tex
│   │   ├── optimization.tex
│   │   ├── ode.tex
│   │   ├── stochastic_processes.tex
│   │   ├── differential_geometry.tex
│   │   └── images/
│   │
│   ├── 04_diffusion_models/
│   │   ├── chapter.tex
│   │   ├── score_matching.tex
│   │   ├── ddpm.tex
│   │   ├── probability_flow.tex
│   │   ├── conditional_diffusion.tex
│   │   ├── limitations.tex
│   │   └── images/
│   │
│   ├── 05_rectified_flow/
│   │   ├── chapter.tex
│   │   ├── theory.tex
│   │   ├── derivation.tex
│   │   ├── loss.tex
│   │   ├── inference.tex
│   │   ├── comparison.tex
│   │   └── images/
│   │
│   ├── 06_system_architecture/
│   │   ├── chapter.tex
│   │   ├── overview.tex
│   │   ├── data_flow.tex
│   │   ├── synchronization.tex
│   │   ├── memory_layout.tex
│   │   └── images/
│   │
│   ├── 07_client_transport/
│   │   ├── chapter.tex
│   │   ├── react.tex
│   │   ├── websocket.tex
│   │   ├── protocol.tex
│   │   ├── streaming.tex
│   │   └── images/
│   │
│   ├── 08_java_engine/
│   │   ├── chapter.tex
│   │   ├── procedural_generation.tex
│   │   ├── marching_cubes.tex
│   │   ├── serialization.tex
│   │   ├── grpc.tex
│   │   └── images/
│   │
│   ├── 09_gpu_engine/
│   │   ├── chapter.tex
│   │   ├── taichi.tex
│   │   ├── cuda.tex
│   │   ├── bvh.tex
│   │   ├── pytorch.tex
│   │   ├── zero_copy.tex
│   │   ├── flow_model.tex
│   │   └── images/
│   │
│   ├── 10_hardware_optimizations/
│   │   ├── chapter.tex
│   │   ├── vram_budget.tex
│   │   ├── fp16.tex
│   │   ├── tensor_cores.tex
│   │   ├── threading.tex
│   │   ├── profiling.tex
│   │   └── images/
│   │
│   ├── 11_design_decisions/
│   │   ├── chapter.tex
│   │   ├── rectified_flow_vs_ddpm.tex
│   │   ├── taichi_vs_optix.tex
│   │   ├── java_vs_python.tex
│   │   ├── websocket_design.tex
│   │   ├── canvas_vs_webgl.tex
│   │   └── images/
│   │
│   ├── 12_training_pipeline/
│   │   ├── chapter.tex
│   │   ├── dataset.tex
│   │   ├── synthetic_generation.tex
│   │   ├── training_loop.tex
│   │   ├── evaluation.tex
│   │   └── images/
│   │
│   ├── 13_results/
│   │   ├── chapter.tex
│   │   ├── fps.tex
│   │   ├── latency.tex
│   │   ├── quality_metrics.tex
│   │   ├── ablation.tex
│   │   ├── comparisons.tex
│   │   └── images/
│   │
│   ├── 14_future_work/
│   │   ├── chapter.tex
│   │   ├── reflow.tex
│   │   ├── multi_gpu.tex
│   │   ├── physics.tex
│   │   ├── tensorrt.tex
│   │   └── images/
│   │
│   └── appendices/
│       ├── appendix_a_math.tex
│       ├── appendix_b_proofs.tex
│       ├── appendix_c_hyperparameters.tex
│       ├── appendix_d_api.tex
│       ├── appendix_e_glossary.tex
│       └── images/
│
├── figures/
│   ├── architecture/
│   ├── math/
│   ├── rendering/
│   ├── benchmarks/
│   ├── ui/
│   └── logos/
│
├── tables/
│   ├── hardware.tex
│   ├── datasets.tex
│   ├── notation.tex
│   └── abbreviations.tex
│
├── tikz/
│   ├── architecture.tex
│   ├── pipeline.tex
│   ├── rendering_equation.tex
│   ├── flow_matching.tex
│   └── training_pipeline.tex
│
├── algorithms/
│   ├── ray_tracing.tex
│   ├── flow_matching.tex
│   ├── training_loop.tex
│   └── inference.tex
│
└── assets/
    ├── fonts/
    ├── icons/
    └── diagrams/

```
