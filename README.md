<p align="center"><img src="./assets/logo.png" width="25%"></p>
<hr>
<h1 align="center">M I C R O C O S M</h1>
<p align="center">Life simulation of evolving unicellular organisms in a stochastic artificial chemistry.<br><a href="https://zushah.github.io/Microcosm">https://zushah.github.io/Microcosm</a></p>
<hr>

## 1. Atomic Theory, Molecules, and Space
Microcosm models matter as a discrete artificial chemistry built from six fundamental elements (**A**, **B**, **C**, **D**, **E**, and **F**). Each element carries intrinsic polarity and intrinsic elemental energy; masses are also defined as metadata, but the active dynamics in the current simulation are driven by composition, bond state, polarity, and enval coupling rather than by explicit mechanics.
- **D** and **E** have the highest positive elemental energies (4.0 and 3.0, respectively) and therefore provide the highest-energy atomic substrates; their abundance remains controlled by seeding, diffusion, and reactions.
- **F** is a low-energy terminal state and commonly accumulates as the endpoint of downhill transmutation.
- A molecule is represented by an elemental composition together with a bond multiplier $`\beta`$. Its total stored energy $`E_{\mathrm{mol}}`$ is:
```math
E_{\mathrm{mol}} = \beta \sum_i n_i E_i.
```
- A molecule of size $`N`$ and mean polarity $`\bar{P}`$ receives a diffusion parameter $`D`$, and therefore larger molecules move more slowly, while smaller and more polar molecules hop more readily through the lattice, and this diffusion is given by:
```math
D = \min\left(0.04,\ 0.01 + \frac{\bar{P}}{N + 1} \cdot 0.04\right).
```
- The world is a toroidal lattice. Each tile begins with a shared base enval and a seeded molecular inventory: **A** is always present, while **B**, **C**, **D**, **E**, **F**, and a small **BC** dimer are seeded stochastically with default probabilities 0.60, 0.45, 0.12, 0.08, 0.05, and 0.05, respectively.
- Molecular diffusion is asynchronous and event-scheduled. Each diffusion event moves one molecule to a cardinal neighbor with locally lower composition density (with tie-rotation and short-term backtracking suppression), rather than globally re-sampling every molecule each tick.

## 2. Enzymology, Specificity, and Metabolism
Life is defined by the possession of a genome encoding functional proteins called enzymes together with a scalar environmental value called the optimal enval. Each enzyme stores a class, a specificity mask over the six elements, an enval response width $`\sigma`$, an enval throughput $`\tau`$, and a secretion probability.

Let $`X(M)`$ be the set of elements present in molecule $`M`$, and let $`S`$ be the enzyme specificity mask. For anabolase and catabolase,
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
Catabolase enzymes perform catabolism by selecting one accepted substrate of size at least two and fragmenting it into two lower-bond products. Catabolase does not alter elemental identities during fragmentation; it extracts only released bond energy, scaled by a harvest fraction $`h_b`$ (currently defaulted to 1.0), minus catalytic cost:
```math
\Delta E_{\mathrm{cell}} = h_b\,\max\left(0, E_{\mathrm{bond}}^{\mathrm{sub}} - \sum_k E_{\mathrm{bond}}^{(k)}\right) - E_{\mathrm{act}} + E_{\mathrm{enval}}.
```
If the net usable gain is non-positive, the reaction is discarded.

### Transmutase
Transmutase performs transmutation by acting on one accepted molecule and changing exactly one atom at a time along a directed artificial transmutation ladder.

Uphill ladder:
```math
F \to A \to C \to B \to E \to D
```

Downhill collapse:
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
All metabolic enzyme classes may couple metabolism to the signed environmental scalar $`\mathrm{enval}`$. Let $`s`$ denote the cell polarity induced by genomic optimal enval ($`s = +1`$ for positive-adapted cells, $`s = -1`$ for negative-adapted cells, and random sign if the optimum is exactly zero). If the local field is aligned with that polarity, the harvested input magnitude is:
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
The primary environmental variable is a single unbounded signed scalar called $`\mathrm{enval}`$. At initialization the world begins with a random base enval sampled from the interval $`[-1, +1]`$, after which diffusion and metabolism are free to push the field to any real value. Enzymatic efficiency $`\phi`$ is governed by a Gaussian dependence on local enval mismatch:
```math
\phi(V_{\mathrm{loc}}, V_{\mathrm{opt}}) = \exp\left(-\frac{(V_{\mathrm{loc}} - V_{\mathrm{opt}})^2}{2\sigma^2}\right).
```
- $`V_{\mathrm{loc}}`$ is the mean enval over the $`5 \times 5`$ neighborhood surrounding the cell (Chebyshev radius of two).
- $`V_{\mathrm{opt}}`$ is stored in the cell genome as the optimal enval.
- $`\sigma`$ is stored per enzyme as its enval response width.

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
- Each cell continuously pays a maintenance cost in stored energy. If energy reaches zero, the cell dies and releases its internal molecules into the environment.
- If the internal atom count falls below twice the genome's desired reserve target, the cell grabs one random molecule from its tile into its internal pool.
- In addition to basal maintenance, mismatch between local and optimal enval increases the cell's starvation/decay timer, such that the increase depends on the genome-level stress factor $`k`$ and enzyme count $`n_{\mathrm{enz}}`$, and if the accumulated timer exceeds the decay threshold, the cell dies, according to:
```math
\Delta t_{\mathrm{stress}} = k\,|V_{\mathrm{loc}} - V_{\mathrm{opt}}|^{1.6}\max(1, n_{\mathrm{enz}}).
```
- Once stored free energy exceeds the reproduction threshold, the cell divides into a nearby empty tile within radius two. Energy is split stochastically around a 50/50 partition, and each internal molecule independently has a 50% chance to go to the offspring.
- Offspring inherit the parental lineage identifier, so lineage structure tracks ecological descent rather than instantaneous genotype identity.

### Predator-Prey Dynamics
Cells may carry two non-metabolic combat enzymes: defensase and attackase. Founders always begin with one defensase, while attackase is absent from founder genomes and must appear later by mutation during reproduction. Each combat enzyme carries an integer level rather than an element-specificity mask. Founder defensase levels are drawn from a rounded Gaussian distribution centered on 100 (current implementation: $`\sigma = 10`$, clamped to at least 1). During reproduction, inherited combat enzymes keep their current level with 50% probability, step downward by a uniformly sampled integer from 1 to 4 with 25% probability, or step upward by such an integer with 25% probability; combat enzymes can also still be gained, lost, or class-switched by the broader genome-mutation machinery.

Predation is evaluated between Moore-neighbor cells (orthogonal or diagonal adjacency) of different lineages after the ordinary metabolic step. Let $`A_{\mathrm{atk}}`$ and $`A_{\mathrm{def}}`$ denote the summed attackase and defensase levels of cell $`A`$ and similarly for cell $`B`$.
- If only one cell satisfies $`\text{attack} > \text{opponent defense}`$, that cell engulfs the other.
- If both cells satisfy that inequality, the larger margin wins: compare $`A_{\mathrm{atk}} - B_{\mathrm{def}}`$ against $`B_{\mathrm{atk}} - A_{\mathrm{def}}`$.
- If neither cell can exceed the opponent's defense, or if the two margins are equal, nothing happens.

Successful predation performs a phagocytosis-like assimilation event: the loser is removed from the world, while the winner absorbs the loser's stored free energy, internal molecules, and enzyme inventory. Transferred enzymes fill free genome slots and may replace existing enzymes when the predator is at capacity, allowing combat and metabolic repertoires to change through ecological interactions as well as descent.

Founders are seeded with an optimal enval drawn from a narrow band around the current world-average enval and initially carry an anabolase, a catabolase, and a defensase. The metabolic founders still begin with **ABC** specificity, while attackase remains absent until it appears later by mutation or by enzyme class switching.

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
The current browser implementation is a browser-hosted Rust simulation compiled to WebAssembly and visualized through WasmGPU. The checked-in WebAssembly module is loaded from `./src/rust/microcosm.wasm`, while its source is maintained in the Rust workspace under `./src/rust`. No backend is required as the browser hosts the simulation locally and communicates with it through a narrow interface.
- `./src/rust/core/src/config.rs`, `chem.rs`, and `molecule.rs` define the world configuration, elemental chemistry, fixed-composition molecules, bond energetics, and event-scheduled molecular diffusion.
- `./src/rust/core/src/genome.rs`, `bio.rs`, and `cell.rs` define genomes, metabolic and combat enzymes, reaction execution, cellular energy accounting, mutation, division, and lineage inheritance.
- `./src/rust/core/src/world.rs` defines the toroidal world and the authoritative step order: molecular diffusion, cellular metabolism and physiology, Moore-neighborhood predation, enval diffusion, and time advancement.
- `./src/rust/core/src/stats.rs` provides aggregate measurements, `render_buffers.rs` provides compact typed render data, `world.rs` provides authoritative bounded state inspection, and `snapshot.rs` provides serialized snapshot support. The Rust workspace also contains a native CLI for reproducible runs, benchmarks, invariant checks, statistics, and snapshot operations.
- `./src/rust/wasm/src/lib.rs` exposes creation, reset, fixed-step advancement, render-buffer refresh, statistics, bounded inspection queries, enval editing, and genome editing through the WebAssembly ABI. Frequently updated render arrays are read through typed views; structured inspection and edit results are returned as bounded JSON payloads.
- `./src/wasmgpu/bridge.js` validates and connects the WebAssembly exports to WasmGPU; `runtime.js` owns the simulation handle and memory views; `render.js` draws the tile field and cell overlay; `interact.js` manages selection and editing; `gui.js` presents diagnostics and inspection; and `./src/main-beta.js` coordinates initialization, fixed-step scheduling, reset, pause, and browser events.

The main entry point supplies a 320 x 240 toroidal world with seed `42` unless the `seed` URL parameter is provided, 32 founder cells, and predation enabled. Every tile is initialized with the configured substrate chemistry, and no additional founder cells are injected after startup; population change thereafter arises from reproduction, mortality, and predation. The simulation advances at a fixed internal step of 10 milliseconds. The browser scheduler consumes those fixed steps and limits work to six simulation steps per rendered frame, while allowing pausing and single-step advancement.

WasmGPU renders the signed enval field, tile occupancy, molecular mass, molecule count, or element-presence channels (**A** through **F**) as selectable tile views, and renders cells with lineage-dependent colors. The interface provides seed control, reset, pause, single-step advancement, fullscreen display, exploration, enval editing, genome editing, tile and cell selection, lineage selection, hover diagnostics, genome and enzyme inspection, internal-molecule and recent-reaction inspection, structured detail display, and JSON copying for inspection results. The visualization is therefore a presentation and control surface for the Rust world state, while the simulation rules and numerical state remain authoritative in the WebAssembly module.
