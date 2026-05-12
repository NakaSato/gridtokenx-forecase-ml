# GridTokenX: PoC Technical Tools Documentation

This document provides a technical overview of the Proof of Concept (PoC) tools available in GridTokenX for microgrid optimization, grid physics simulation, and resilience testing.

## 1. Optimization Engines

### Coordinated Cluster MILP (`optimizer/run_optimization.py`)
A Mixed-Integer Linear Programming (MILP) engine that optimizes diesel generation and BESS dispatch across the Ko Tao-Phangan-Samui cluster.
- **Goal**: Minimize fuel consumption and carbon emissions while maintaining N-1 security.
- **Constraints**: Ramp rates, minimum up/down time, cable capacity, and BESS SoC limits.
- **Usage**: `just optimize`

### Distributed ADMM Resilience (`optimizer/admm_resilience.py`)
Implements the Alternating Direction Method of Multipliers (ADMM) for distributed grid coordination.
- **Purpose**: Enables islands to coordinate dispatch without a central controller, crucial for resilience during communication failures.
- **Feature**: Converges on optimal power flow while respecting local island constraints.

### Integrated Social Cockroach Algorithm (ISCA) (`optimizer/isca.py`)
A metaheuristic optimizer designed for complex, non-convex dispatch problems.
- **Usage**: Typically used for global hyperparameter search or complex dispatch scenarios where MILP might struggle with non-linear generator curves.

## 2. Grid Physics & Simulation

### Optimal Power Flow (OPF) (`research/pypsa_model.py`)
Utilizes the **PyPSA** (Python for Power System Analysis) library to perform AC power flow and OPF calculations.
- **Topology**: Models the 115 kV submarine cable ring and radial 33 kV links.
- **Usage**: `just opf-test`

### Pandapower Grid Engine (`research/pandapower_model.py`)
A persistent grid physics engine used for real-time contingency analysis.
- **Fidelity**: High-performance AC power flow simulation.
- **Integration**: Used by the `Early Warning System` to detect voltage or thermal violations in the next 6 hours.

## 3. Resilience & Stress Testing

### N-1 Contingency Stress Test (`research/stress_test.py`)
Simulates the total failure of the 115 kV mainland connector (a known risk for islanded microgrids).
- **Metric**: Evaluates how long the local 10 MW diesel backup can sustain the island load without a BESS.
- **Usage**: `just stress-test`

### Monte Carlo Stochastic Engine (`research/stochastic_test.py`)
Runs hundreds of simulations (N=500) with randomized weather, holiday spikes, and equipment failures.
- **Output**: Generates a resilience distribution and identifies "tail risks" for the grid.
- **Usage**: `just stochastic-test`

### Cascading Failure Analysis (`research/cascading_failure.py`)
Models the ripple effect of cable overloads. If one circuit trips due to congestion, this tool evaluates if the remaining links will also fail, leading to a cluster-wide blackout.

## 4. Operational Tools

### Early Warning System (`optimizer/early_warning.py`)
A proactive monitoring tool that uses the latest forecast to run lookahead physics simulations.
- **Lookahead**: 6 hours.
- **Alerts**: Triggers if the forecasted load will cause a cable bottleneck or voltage drop at distal nodes (like Ko Tao).

### PEA AWS Sandbox Onboarding (`data/pea_onboard.py`)
A specialized tool for calibrating synthetic models with real SCADA data from the Provincial Electricity Authority (PEA).
- **Usage**: `just pea-onboard`
