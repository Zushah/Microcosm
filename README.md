<p align="center"><img src="./assets/logo.png" width="25%"></p>
<hr>
<h1 align="center">M I C R O C O S M</h1>
<p align="center">Life simulation of evolving unicellular organisms in a stochastic artificial chemistry.<br><a href="https://zushah.github.io/Microcosm">https://zushah.github.io/Microcosm</a></p>
<hr>

## 1. Atomic Theory and Thermodynamics
The world is governed by a strict conservation of mass and energy. Matter consists of six fundamental elements (**A**, **B**, **C**, **D**, **E**, and **X**), each defined by distinct atomic properties including mass, polarity, and intrinsic potential energy.
- High-Energy Isotopes: Elements **D** and **E** possess high potential energy, acting as the primary chemical fuel for biological systems.
- Waste Products: Element **X** represents a low-energy state, often accumulating as a metabolic byproduct of catabolism.
- Molecular Stability: Atoms aggregate into molecules. The stability $`S`$ of a compound is a function of its atomic count $`N`$ and net polarity $`P`$, calculated by the equation:
```math
S = \max(0, 1 - 0.08N - |P - 0.5|).
```
Large, non-polar molecules tend to destabilize (i.e. $`S \to 0`$), driving the system toward entropic decay unless maintained by biological activity.

## 2. Enzymology and Metabolism
Life is defined by the possession of a genome that encodes functional proteins called **enzymes**. These biological catalysts lower the activation energy required for chemical transformations. Metabolic pathways are classified into three primary categories:
### Anabolase
Anabolase enzymes perform anabolism by facilitating the polymerization of simple substrates into complex molecules. This process is endothermic; it consumes cellular energy reserves to forge new chemical bonds. The energy cost $`E_{cost}`$ is proportional to the difference in bond enthalpy $`\Delta H`$ between products and reactants:
```math
E_{cost} = \Delta H + E_{activation} \quad (\text{where } \Delta H > 0).
```
### Catabolase
Catabolase enzymes perform catabolism by degrading complex molecules into simpler constituents. This process is exothermic, releasing bond energy. The cell operates with a thermodynamic efficiency $`\eta`$ (typically $`0.85`$), capturing useful energy $`E_{cell}`$ while releasing the remainder as entropy:
```math
E_{cell} = \eta \Delta E_{total},\\
Q_{heat} = (1 - \eta) \Delta E_{total}.
```
This generates thermal waste $`Q_{heat}`$ and may occasionally transmute useful substrates into inert **X** waste, simulating oxidative stress or molecular damage.
### Transportase
Unlike passive diffusion, Transportase enzymes actively pump high-value isotopes (specifically **D** and **E**) from the environment into the cell against the concentration gradient. This requires a direct expenditure of metabolic energy proportional to the mass moved, i.e. $`E_{cost} \propto N_{atoms}`$.

## 3. Reaction Kinetics
Enzymatic activity is not constant but is strictly regulated by environmental conditions. The reaction rate $`R`$ follows a modified Arrhenius equation, combining a Gaussian thermal dependence with a Laplacian pH dependence:
```math
R(T, pH) = R_{max} e^{\left(-\frac{(T - T_{opt})^2}{2\sigma^2} -|pH - pH_{opt}|\right)}.
```
- Thermal Dependence: Reaction rates follow a Gaussian distribution where $`\sigma`$ represents the enzyme's thermal tolerance width. Deviation from $`T_{opt}`$ results in a sharp decline in metabolic efficiency.
- pH Sensitivity: Similarly, enzymes are sensitive to environmental acidity. Extreme pH levels denature these proteins, halting metabolism.

## 4. Cellular Physiology and Homeostasis
A cell is modeled as a thermodynamic system maintaining a state of non-equilibrium.
- Basal Metabolic Rate: Cells constantly consume energy to combat entropy. If internal energy stores are depleted or accumulated stress exceeds the decay threshold, the cell dies. The metabolic stress accumulation $`\Delta S`$ is non-linear with respect to thermal deviation, modeled as:
```math
\Delta S \propto |T - T_{opt}|^{1.6}.
```
- Thermoregulation: Metabolic reactions (particularly catabolism) are exothermic. In dense colonies, the collective heat generation $`Q_{metabolic}`$ can locally raise the ambient temperature. This creates a feedback loop where cells must evolve higher thermal tolerances to survive their own metabolic waste heat. The local temperature field $`T(x,y)`$ evolves according to a discrete diffusion-relaxation equation:
```math
T_{t+1} = (1-\alpha)T_t + \alpha \bar{T}_{neighbors} + \lambda(T_{base} - T_t) + Q_{metabolic}.
```
- Reproduction: Once a cell accumulates sufficient energy and biomass relative to its volume, it undergoes binary fission. The genome is replicated with a non-zero error rate (mutation), altering enzyme affinities or thermal optima in the progeny. This descent with modification drives the Darwinian evolution of the population.
