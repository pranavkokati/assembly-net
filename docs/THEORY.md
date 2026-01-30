# Theoretical Foundation

This document provides the formal mathematical framework for Assembly-Net.

## Assembly Non-Equivalence Theorem

### Statement

There exist pairs of assembly trajectories T₁, T₂ such that:

```
G₁(T_final) ≅ G₂(T_final)  (isomorphic final structures)
```

but

```
P(T₁) ≠ P(T₂)  (different emergent properties)
```

### Required Assumptions

The theorem holds under the following assumptions:

| ID | Name | Statement | Physical Basis |
|----|------|-----------|----------------|
| A1 | Irreversibility | k_off << k_on | Bonds persist once formed |
| A2 | Finite Valency | Max coordination numbers | Geometric frustration |
| A3 | Non-equilibrium | Gillespie kinetics | Kinetic control |
| A4 | Path-dependent Bonds | Strength depends on time | Early bonds stronger |
| A5 | Topology-Property | Properties depend on timing | ELDI matters |

### When the Theorem Does NOT Hold

**If A1 violated (reversibility allowed):**
- System reaches thermodynamic equilibrium
- All trajectories to same final structure become equivalent
- P(T₁) = P(T₂) for G₁(T) ≅ G₂(T)

**If A2 violated (unlimited valency):**
- No geometric frustration
- All assembly paths equivalent

### Proof Sketch

The mechanical score M(T) depends on the Early Loop Dominance Index (ELDI):

```
M(T) = 0.5 × ELDI + 0.3 × (1 - t_perc/T) + 0.2 × min(1, β₁/20)
```

Two trajectories can reach isomorphic final graphs with identical β₁, but with different loop formation times:

- T₁: Loops form at t ∈ [0, T/2] → ELDI₁ ≈ 1
- T₂: Loops form at t ∈ [T/2, T] → ELDI₂ ≈ 0

Since M depends on ELDI: M(T₁) ≠ M(T₂). ∎

## Information-Theoretic Limitation

### Theorem 2

Let f: G → ℝ be any function computable from the final graph G(T) alone.
Let M: T → ℝ be the mechanical score function.

There exist trajectories T₁, T₂ with G(T₁, T) ≅ G(T₂, T) such that:

```
f(G(T₁, T)) = f(G(T₂, T))    (static features identical)
M(T₁) ≠ M(T₂)                (mechanical scores differ)
```

**Consequence:** No static model can perfectly predict mechanical properties
for assembly-driven materials. Temporal information is necessary.

## Topological Invariants

### Betti Numbers

For a graph G = (V, E):

- **β₀(G)**: Number of connected components
  - β₀ = |V| - rank(L), where L is the graph Laplacian

- **β₁(G)**: Number of independent cycles (cyclomatic complexity)
  - β₁ = |E| - |V| + β₀ (Euler characteristic)

### Early Loop Dominance Index (ELDI)

Let τ₁, τ₂, ..., τₖ be the times when cycles form (β₁ increases).

```
ELDI = |{τᵢ : τᵢ < T/2}| / k
```

ELDI ∈ [0, 1] quantifies what fraction of loops formed in the first half of assembly.

**Key insight:** ELDI is a history-dependent feature that CANNOT be computed from the final structure alone.

## Physical Justification

### Mechanical Properties

The mechanical robustness of a coordination network depends on:

1. **Loop density (β₁ / |V|)**: More cycles = more redundant load paths

2. **Early loop formation**: Cycles formed early become "load-bearing" as subsequent bonds form around them

3. **Percolation timing**: Early percolation allows stress distribution across the entire network

### Literature Support

- Meakin (1988): DLA produces fractal, branched structures
- Lin et al. (1989): RLA produces compact structures
- Ejima et al. (2013): Metal-phenolic network assembly kinetics
- Caruso et al. (2009): Mechanically-induced changes in polymers

## References

1. Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. J. Phys. Chem. 81(25), 2340-2361.
2. Edelsbrunner, H., & Harer, J. (2008). Persistent homology - a survey. Contemporary Mathematics, 453, 257-282.
3. Thorpe, M. F. (1983). Continuous deformations in random networks. J. Non-Cryst. Solids, 57(3), 355-370.
