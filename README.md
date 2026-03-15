<p align="center"><img src="./assets/logo.png" width="25%"></p>
<hr>
<h1 align="center">M I C R O C O S M</h1>
<p align="center">Life simulation of evolving unicellular organisms in a stochastic artificial chemistry.<br><a href="https://zushah.github.io/Microcosm">https://zushah.github.io/Microcosm</a></p>
<hr>

## 1. Atomic Theory and Chemistry
Microcosm models matter as a discrete artificial chemistry built from six fundamental elements (**A**, **B**, **C**, **D**, **E**, and **F**), each with its own mass, polarity, and intrinsic energetic value. Rather than attempting biochemical realism, the simulation uses a compact state space in which higher-order ecological behavior emerges from simple compositional rules.
- Energetic Isotopes: Elements **D** and **E** carry high elemental energy and remain the most concentrated chemical fuel available to evolving cells.
- Waste State: Element **F** is a low-energy terminal state and commonly accumulates as a byproduct of destructive metabolism.
- Molecular Energetics: A molecule is represented by an elemental composition together with a bond multiplier $`\beta`$. Its total stored energy $`E_{mol}`$ is:
```math
E_{mol} = \beta \sum_i n_i E_i.
```
- Molecular Mobility: Diffusion is stochastic and depends on molecular size $`N`$ and mean polarity $`P`$. The diffusion parameter $`D`$ is:
```math
D = \min\left(0.04,\ 0.01 + \frac{P}{N + 1} \cdot 0.04\right).
```
Small, polar molecules therefore migrate through the lattice more rapidly than large or weakly polar compounds.

## 2. Enzymology and Metabolism
Life is defined by the possession of a genome encoding functional proteins called **enzymes** together with a scalar environmental optimum called **optimal enval**. Enzymes catalyze transformations over both extracellular and intracellular molecules, while the cell's optimal enval determines which sign of the ambient field it can exploit most efficiently.
### Anabolase
Anabolase enzymes perform anabolism by polymerizing accepted substrates into a single higher-bond product. This process is endergonic: the cell must pay the energetic difference between the bonded product and its unbound elemental constituents, together with a fixed activation cost. In the current implementation, the net cellular energy change is:
```math
\Delta E_{cell} = E_{enval} - \left[(\beta - 1)\sum_i n_i E_i + E_{act}\right].
```
Thus anabolism is often only sustainable when either prior catabolism or aligned enval flux subsidizes the reaction.
### Catabolase
Catabolase enzymes perform catabolism by selecting a substrate, releasing its bond energy, and fragmenting it into lower-bond byproducts. In addition, catabolism may transmute energetic isotopes **D** and **E** into low-energy **F**, extracting additional chemical free energy from the conversion. The chemically available yield is therefore:
```math
\Delta E_{chem} = \eta\left(E_{bond} + E_{transmute} - E_{act}\right),
```
where $`\eta`$ is the enzyme harvest fraction. The total cellular gain is then $`\Delta E_{chem} + E_{enval}`$.
### Transportase
Transportase enzymes actively import **D** and **E** atoms from the local tile into the cell, paying an energy cost proportional to the transported mass. This allows a lineage to concentrate scarce high-energy isotopes internally even when passive access would be insufficient.
### Energy Transduction
All enzyme classes may couple metabolism to the signed environmental scalar **`enval`**. The sign of the cell's genomic optimum determines the favorable direction of flux: cells with positive optimal enval consume positive enval and emit negative enval, whereas cells with negative optimal enval do the converse. If aligned field is available, the harvested magnitude $`I`$ is:
```math
I = \min(V_{\parallel}, \tau \phi),
```
where $`V_{\parallel}`$ is the locally available aligned enval, $`\tau`$ is the enzyme throughput, and $`\phi`$ is the enval-matching efficiency from Section 3. The default partition is:
```math
E_{enval} = \frac{2}{3}I, \qquad V_{out} = -\operatorname{sgn}(V_{opt})\frac{1}{3}I.
```
This makes enval neither intrinsically favorable nor intrinsically hostile; whichever sign is abundant can become metabolically useful for the phenotype adapted to it.

## 3. Reaction Kinetics
The primary environmental variable is a single unbounded signed scalar called **`enval`**. At initialization the world begins with a random base enval sampled from the interval $`[-1, +1]`$, after which diffusion and metabolism are free to push the field to any real value. Enzymatic efficiency $\phi$ is governed by a Gaussian dependence on local enval mismatch.
```math
\phi(V_{loc}, V_{opt}) = \exp\left(-\frac{(V_{loc} - V_{opt})^2}{2\sigma^2}\right).
```
- Local Sensing: $`V_{loc}`$ is the mean enval over the $`5 \times 5`$ neighborhood surrounding the cell (Chebyshev radius of two).
- Genomic Target: $`V_{opt}`$ is stored in the cell genome as **optimal enval**.
- Enzyme Tolerance: $`\sigma`$ is stored per enzyme as its enval response width.

The reaction attempt probability is then scaled by this efficiency term:
```math
p_{rxn} = \min(1, 1.2R_{base})\,\phi.
```
Strong mismatch simultaneously suppresses reaction probability and enval harvesting throughput, so maladapted cells suffer a rapid collapse in metabolic performance.

## 4. Cellular Physiology, Field Dynamics, and Evolution
A cell is modeled as a non-equilibrium metabolic agent coupled to a diffusing environmental field.
- Basal Maintenance: Each cell continuously pays a maintenance cost in stored energy. If energy reaches zero, the cell dies and releases its internal molecular inventory.
- Enval Stress: In addition to basal maintenance, mismatch between local and optimal enval increases the cell's starvation/decay timer according to:
```math
\Delta t_{stress} = k\,|V_{loc} - V_{opt}|^{1.6}\max(1, n_{enz}),
```
where $`k`$ is the genome-level stress factor and $`n_{enz}`$ is enzyme count. If the accumulated timer exceeds the decay threshold, the cell dies.
- Spatial Field Dynamics: The enval field diffuses across a toroidal lattice by discrete local averaging:
```math
V_{t+1}(x, y) = (1 - \alpha)V_t(x, y) + \alpha\,\bar{V}_{3 \times 3}(x, y), \qquad \alpha = 0.18.
```
Reaction-generated enval is deposited into the local Moore neighborhood, allowing successful metabolism to reshape the surrounding field.
- Reproduction: Once a cell's stored energy exceeds its reproduction threshold, it undergoes binary fission into a nearby empty tile. Energy and internal molecules are partitioned stochastically between parent and offspring.
- Founder Initialization: Randomly spawned founder cells receive an optimal enval sampled from a narrow band centered on the current world-average enval:
```math
V_{opt}^{(0)} \sim U(\bar{V}_{world} - 0.1,\ \bar{V}_{world} + 0.1).
```
- Heritable Enval Adaptation: During division, the offspring's optimal enval is inherited by a biased midpoint mutation rule. Let $`P`$ be the parental optimal enval and $`W`$ the current world-average enval. Define:
```math
\delta = \max\left(\frac{|P - W|}{2}, \varepsilon\right), \qquad \varepsilon = 0.03.
```
Then the offspring receives:
```math
V'_{opt} =
\begin{cases}
P & \text{with probability } 0.50,\\
P + \operatorname{sgn}(W - P)\delta & \text{with probability } 0.25,\\
P - \operatorname{sgn}(W - P)\delta & \text{with probability } 0.25.
\end{cases}
```
Other genome parameters (including enzyme affinity, secretion behavior, enval tolerance width, enval throughput, reproduction threshold, and decay time) also mutate stochastically.

The central ecological consequence is that successful lineages tend to destabilize the very field regime they exploit. Positive-adapted cells deplete positive enval and produce negative enval; negative-adapted cells deplete negative enval and produce positive enval. The intended macrodynamics are therefore cyclical succession, lineage turnover, and repeated ecological boom-bust behavior.
