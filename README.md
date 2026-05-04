<p align="center"><img src="./assets/logo.png" width="25%"></p>
<hr>
<h1 align="center">M I C R O C O S M</h1>
<p align="center">Life simulation of evolving unicellular organisms in a stochastic artificial chemistry.<br><a href="https://zushah.github.io/Microcosm">https://zushah.github.io/Microcosm</a></p>
<hr>

## 1. Atomic Theory, Molecules, and Space
Microcosm models matter as a discrete artificial chemistry built from six fundamental elements (**A**, **B**, **C**, **D**, **E**, and **F**). Each element carries intrinsic polarity and intrinsic elemental energy; masses are also defined as metadata, but the active dynamics in the current simulation are driven by composition, bond state, polarity, and enval coupling rather than by explicit mechanics.
- Energetic Species: **D** and **E** are the most energy-rich elements and therefore remain the most concentrated atomic fuel available to evolving cells.
- Terminal Sink: **F** is a low-energy terminal state and commonly accumulates as the endpoint of downhill transmutation.
- Molecular Energetics: A molecule is represented by an elemental composition together with a bond multiplier $`\beta`$. Its total stored energy $`E_{\mathrm{mol}}`$ is:
```math
E_{\mathrm{mol}} = \beta \sum_i n_i E_i.
```
- Molecular Mobility: A molecule of size $`N`$ and mean polarity $`\bar{P}`$ receives a diffusion parameter $`D`$ given by:
```math
D = \min\left(0.04,\ 0.01 + \frac{\bar{P}}{N + 1} \cdot 0.04\right).
```
Larger molecules therefore move more slowly, while smaller and more polar molecules hop more readily through the lattice.
- Spatial Substrate Field: The world is a toroidal lattice. Each tile begins with a shared base enval and a seeded molecular inventory: **A** is always present, while **B**, **C**, **D**, **E**, **F**, and a small **BC** dimer appear stochastically.
- Diffusion Mode: Molecular diffusion is asynchronous and event-scheduled. Each diffusion event moves one molecule to a cardinal neighbor with locally lower composition density (with tie-rotation and short-term backtracking suppression), rather than globally re-sampling every molecule each tick.

## 2. Enzymology, Specificity, and Metabolism
Life is defined by the possession of a genome encoding functional proteins called **enzymes** together with a scalar environmental optimum called **optimal enval**. Each enzyme stores a class, a specificity mask over the six elements, an enval response width $`\sigma`$, an enval throughput $`\tau`$, and a secretion probability.
- Specificity Rule: Let $`X(M)`$ be the set of elements present in molecule $`M`$, and let $`S`$ be the enzyme specificity mask. For anabolase and catabolase,
```math
X(M) \subseteq S.
```
For transmutase,
```math
X(M) \cap S \neq \varnothing.
```
Thus synthetic and degradative enzymes are exclusive over their allowed alphabet, whereas transmutases only require access to at least one allowed source element.

### Anabolase
Anabolase enzymes perform anabolism by sampling accepted substrates from the extracellular and intracellular pools and polymerizing them into a single higher-bond product. The product composition is simply the summed elemental composition of the substrates, but its bond multiplier is elevated above unity. The energetic cost is the increase in bond storage, scaled by a bond-cost fraction $`\lambda`$, together with a fixed catalytic cost $`E_{\mathrm{act}}`$:
```math
\Delta E_{\mathrm{cell}} = E_{\mathrm{enval}} - \left(\lambda\,\max\left(0, E_{\mathrm{bond}}^{\mathrm{prod}} - \sum_j E_{\mathrm{bond}}^{(j)}\right) + E_{\mathrm{act}}\right).
```
This makes anabolism endergonic unless it is subsidized by previously stored energy or by aligned enval harvesting.

### Catabolase
Catabolase enzymes perform catabolism by selecting one accepted substrate of size at least two and fragmenting it into two lower-bond products. The current implementation does **not** perform the old catabolic isotope-transmutation step; catabolase now extracts only released bond energy, scaled by a harvest fraction $`h_b`$ (currently defaulted to 1.0), minus catalytic cost:
```math
\Delta E_{\mathrm{cell}} = h_b\,\max\left(0, E_{\mathrm{bond}}^{\mathrm{sub}} - \sum_k E_{\mathrm{bond}}^{(k)}\right) - E_{\mathrm{act}} + E_{\mathrm{enval}}.
```
If the net usable gain is non-positive, the reaction is discarded.

### Transmutase
Transmutase performs transmutation by acting on one accepted molecule and changing exactly one atom at a time along a directed artificial transmutation ladder.
- Uphill ladder:
```math
F \to A \to C \to B \to E \to D
```
- Downhill collapse:
```math
A, B, C, D, E \to F
```
If the raw elemental change is $`\Delta E_{\mathrm{raw}}`$, then downhill transmutation only harvests a fraction $`h_d`$ of that release, while uphill transmutation pays the full energetic penalty:
```math
\Delta E_{\mathrm{trans}} =
\begin{cases}
h_d\,\Delta E_{\mathrm{raw}}, & \Delta E_{\mathrm{raw}} \ge 0, \\
\Delta E_{\mathrm{raw}}, & \Delta E_{\mathrm{raw}} < 0.
\end{cases}
```
The full cellular effect is then:
```math
\Delta E_{\mathrm{cell}} = \Delta E_{\mathrm{trans}} - E_{\mathrm{act}} + E_{\mathrm{enval}}.
```
When both directions are available, transmutase biases more strongly toward uphill moves when aligned enval is abundant and the cell already has substantial energy reserves.

### Enval Coupling
All enzyme classes may couple metabolism to the signed environmental scalar **`enval`**. Let $`s`$ denote the cell polarity induced by genomic optimal enval ($`s = +1`$ for positive-adapted cells, $`s = -1`$ for negative-adapted cells, and random sign if the optimum is exactly zero). If the local field is aligned with that polarity, the harvested input magnitude is:
```math
I = \min\left(\max(0, sV_{\mathrm{loc}}),\ \tau\right).
```
By default the harvested energy term is:
```math
E_{\mathrm{enval}} = \frac{2}{3}I.
```
The reaction simultaneously pumps opposite-sign enval back into the environment. With baseline pump magnitude $`P`$ and recycle fraction $`f_r`$ (default $`P = 0.3`$, $`f_r = 1/3`$), the emitted field term is:
```math
V_{\mathrm{out}} = -s\left(P + f_r I\right).
```
Operationally, aligned enval is removed from the local tile while opposite-sign enval is deposited into one randomly sampled site in the local Moore neighborhood (including the source tile). Successful metabolism therefore does not merely consume a pre-existing field; it actively reshapes that field.

## 3. Reaction Kinetics and Field Dynamics
The primary environmental variable is a single unbounded signed scalar called **`enval`**. At initialization the world begins with a random base enval sampled from the interval $`[-1, +1]`$, after which diffusion and metabolism are free to push the field to any real value. Enzymatic efficiency $`\phi`$ is governed by a Gaussian dependence on local enval mismatch:
```math
\phi(V_{\mathrm{loc}}, V_{\mathrm{opt}}) = \exp\left(-\frac{(V_{\mathrm{loc}} - V_{\mathrm{opt}})^2}{2\sigma^2}\right).
```
- Local Sensing: $`V_{\mathrm{loc}}`$ is the mean enval over the $`5 \times 5`$ neighborhood surrounding the cell (Chebyshev radius of two).
- Genomic Target: $`V_{\mathrm{opt}}`$ is stored in the cell genome as **optimal enval**.
- Enzyme Tolerance: $`\sigma`$ is stored per enzyme as its enval response width.

The reaction attempt probability is scaled by this efficiency term:
```math
p_{\mathrm{rxn}} = \min(1, 1.2R_{\mathrm{base}})\,\phi.
```
Current base rates are 0.85 for anabolase, 0.98 for catabolase, and 0.30 for transmutase. Strong mismatch therefore suppresses metabolism primarily by preventing reactions from firing at all; enval harvesting falls indirectly because fewer reactions occur.

The enval field itself diffuses across the toroidal lattice by discrete local averaging:
```math
V_{t+1}(x, y) = (1 - \alpha)V_t(x, y) + \alpha\,\bar{V}_{3 \times 3}(x, y), \qquad \alpha = 0.18.
```
Because reactions continuously remove aligned enval and emit opposite-sign enval, the field is both smoothed by diffusion and destabilized by metabolic feedback.

## 4. Cellular Physiology and Evolution
A cell is a stationary metabolic agent that occupies a tile, stores free energy, stores an internal molecular inventory, and executes its genome each tick.
- Basal Maintenance: Each cell continuously pays a maintenance cost in stored energy. If energy reaches zero, the cell dies and releases its internal molecules into the environment.
- Opportunistic Uptake: If internal elemental reserves fall below the desired reserve target, the cell passively grabs one random molecule from the tile into its internal pool.
- Enval Stress: In addition to basal maintenance, mismatch between local and optimal enval increases the cell's starvation/decay timer according to:
```math
\Delta t_{\mathrm{stress}} = k\,|V_{\mathrm{loc}} - V_{\mathrm{opt}}|^{1.6}\max(1, n_{\mathrm{enz}}),
```
where $`k`$ is the genome-level stress factor and $`n_{\mathrm{enz}}`$ is enzyme count. If the accumulated timer exceeds the decay threshold, the cell dies.
- Reproduction: Once stored free energy exceeds the reproduction threshold, the cell divides into a nearby empty tile within radius two. Energy is split stochastically around a 50/50 partition, and each internal molecule independently has a 50% chance to go to the offspring.
- Lineage Identity: Offspring inherit the parental lineage identifier, so lineage structure tracks ecological descent rather than instantaneous genotype identity.

### Predator-Prey Dynamics
Cells may now carry two non-metabolic combat enzymes: **defensase** and **attackase**. Founders always begin with one defensase, while attackase is absent from founder genomes and must appear later by mutation during reproduction. Each combat enzyme carries an integer **level** rather than an element-specificity mask. Founder defensase levels are drawn from a rounded Gaussian distribution centered on 100 (current implementation: $`\sigma = 10`$, clamped to at least 1). During reproduction, inherited combat enzymes keep their current level with 50% probability, step downward by a small integer with 25% probability, or step upward by a small integer with 25% probability; combat enzymes can also still be gained, lost, or class-switched by the broader genome-mutation machinery.

Predation is evaluated between edge-adjacent cells of different lineages after the ordinary metabolic step. Let $`A_{\mathrm{atk}}`$ and $`A_{\mathrm{def}}`$ denote the summed attackase and defensase levels of cell $`A`$, and similarly for cell $`B`$.
- If only one cell satisfies $`\text{attack} > \text{opponent defense}`$, that cell engulfs the other.
- If both cells satisfy that inequality, the larger margin wins: compare $`A_{\mathrm{atk}} - B_{\mathrm{def}}`$ against $`B_{\mathrm{atk}} - A_{\mathrm{def}}`$.
- If neither cell can exceed the opponent's defense, or if the two margins are equal, nothing happens.

Successful predation performs a phagocytosis-like assimilation event: the loser is removed from the world, while the winner absorbs the loser's stored free energy, internal molecules, and enzyme inventory. Because absorbed enzymes are appended to the predator genome, combat and metabolic repertoires can both expand through ecological interactions rather than by descent alone.

Founders are seeded with an optimal enval drawn from a narrow band around the current world-average enval and initially carry an **anabolase**, a **catabolase**, and one Gaussian-sampled **defensase**. The metabolic founders still begin with **ABC** specificity, while attackase remains absent until it appears later by mutation or by enzyme class switching.

During division, the offspring optimal enval is inherited by a biased midpoint rule. Let $`P`$ be the parental optimal enval and $`L`$ the dividing cell's current local-average enval. Define:
```math
\delta = \max\left(\frac{|P - L|}{2},\ \varepsilon\right), \qquad \varepsilon = 0.03.
```
Then the offspring receives:
```math
V'_{\mathrm{opt}} =
\begin{cases}
P, & \text{with probability } 0.50, \\
P + \mathrm{sgn}(L - P)\,\delta, & \text{with probability } 0.25, \\
P - \mathrm{sgn}(L - P)\,\delta, & \text{with probability } 0.25.
\end{cases}
```
Other genome parameters also mutate stochastically: reproduction threshold, decay time, default secretion probability, enval stress factor, enzyme specificity masks, enzyme class, combat-enzyme level, enval response width, secretion probability, enval throughput, anabolase bond multiplier, and transmutase downhill harvest fraction. Enzyme count may also increase or decrease over evolutionary time.

The central ecological consequence is that successful lineages tend to destabilize the very field regime they exploit. Positive-adapted cells deplete positive enval and emit negative enval; negative-adapted cells deplete negative enval and emit positive enval. The intended macrodynamics are therefore cyclical succession, lineage turnover, and repeated ecological boom-bust behavior.

## 5. Runtime and Interface
Microcosm is a browser-native ES-module application with no backend and no build step in the repository itself. The runtime is organized into a small set of source files:
- `./src/sim/bio.js` defines enzyme classes and reaction execution.
- `./src/sim/cell.js` defines cellular physiology, reaction logging, division, and mutation.
- `./src/sim/chem.js` defines the artificial chemistry and molecule construction.
- `./src/sim/eco.js` defines predator-prey interactions and combat mechanisms.
- `./src/sim/rng.js` defines a seeded pseudorandom number generator and its helpers.
- `./src/sim/world.js` defines the toroidal world, molecule diffusion, enval diffusion, and spawning.
- `./src/render/canvas.js` defines the WebGL2 renderer.
- `./src/main.js` wires together the simulation loop, statistics, interaction, and inspector UI.

At startup the simulation builds a 320 x 240 world, spawns 32 founder cells, and advances the world at a fixed step of 10 ms. After initialization, additional founder cells are injected stochastically with a population-dependent spawn probability that declines toward a floor as occupancy rises. The renderer colors the background by local enval and colors cells by lineage identity; an interaction panel can switch tile coloring to per-element concentration channels (**A** through **F**). The HUD reports frame rate, tick rate, population, average energy, lineage count, enzyme functional diversity, elemental totals, and average enval. Left-clicking a cell opens a detailed inspector; in Explore mode right-clicking a cell selects its lineage, while in Edit mode right-click drag applies a rectangular enval brush. The inspector exposes internal molecules, genome parameters, recent reactions, and reaction-level energetic bookkeeping, and a button copies the recent reaction log to the clipboard.
