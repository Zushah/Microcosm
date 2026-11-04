<p align="center"><img src="./assets/logo.png" width="25%"></p>
<hr>
<h1 align="center">M I C R O C O S M</h1>
<p align="center">Life simulation of evolving unicellular organisms shaping and shaped by an artificial chemistry.<br><a href="https://zushah.github.io/Microcosm">https://zushah.github.io/Microcosm</a></p>
<hr>

## 1. Elements and Fields
Matter consists of six elements (**A**, **B**, **C**, **D**, **E**, and **F**) held as continuous amounts. Each carries an intrinsic energy and a polarity; energy sets what reactions yield, polarity sets how easily an element crosses a membrane, and mass is used only for display.
- Energies: **A** 0.5, **B** 1.0, **C** 0.9, **D** 4.0, **E** 3.0, and **F** -0.2.
- Polarities: **A** 0.9, **B** 0.4, **C** 0.2, **D** 0.1, **E** 1.0, and **F** 0.6.

The world is a 320 x 240 toroidal grid advanced in fixed ticks of $`\Delta t = 0.01`$ s. Each tile stores one concentration per element, starting near **A** 1.00, **B** 0.65, **C** 0.50, **D** 0.12, **E** 0.08, and **F** 0.05 with seeded smooth noise that preserves these means. Each element spreads over the tile's 3 x 3 neighborhood with its own diffusivity $`D`$ (**A** 2.8, **B** 1.8, **C** 1.4, **D** 1.2, **E** 3.0, and **F** 2.2):
```math
u_i' = u_i + \frac{D\,\Delta t}{9}\left(\sum_{j \in N_9(i)} u_j - 9u_i\right).
```
Elements are conserved: every transfer moves an amount between a tile and a cell, or between cells.

Enval is a signed scalar field that acts as the world's energy source, and every tile has a target $`T_i`$ in a seeded, periodic, smooth landscape that peaks at magnitude 1.0. The field starts on the landscape, diffuses by the same stencil at rate 18, and relaxes back toward its target at rate $`\lambda = 0.2`$:
```math
E_i' = E_i + \left(1 - e^{-\lambda\,\Delta t}\right)\left(T_i - E_i\right).
```
Enval also recharges spent matter. On a tile with $`E \neq 0`$, **F** converts to **D** where enval is positive and to **E** where it is negative, at a rate proportional to $`0.1\,|E|\,c_F`$. Each converted unit spends enval magnitude equal to the energy it gains, 4.2 for **D** and 3.2 for **E**.

## 2. Enzymes and Metabolism
A genome holds 1 to 10 enzymes. Each is a stoichiometric catalyst with reactant and product vectors $`r`$ and $`p`$, a rate $`k`$, a half-saturation $`K`$, routing fractions, an enval niche, and an exposure. Enzymes are not consumed, and no enzyme has a fixed class: what one does follows from its numbers.

Each reaction draws on a pool of amounts $`n_i`$ spread over an extent $`\Omega`$, with a saturating response to its limiting reactant. A cell's local enval $`L`$ is the 5 x 5 tile average at the start of the cell phase, sampled bilinearly at its position, and an enzyme with optimum $`\mu`$ and width $`\sigma`$ runs at the Gaussian factor $`f`$:
```math
c = \min_i \frac{n_i}{\Omega\,r_i}, \qquad f = \exp\left(-\frac{(L - \mu)^2}{2\sigma^2}\right), \qquad \xi_{\mathrm{req}} = \min\left(k\,\Delta t\,f\,\frac{c}{K + c}\,\frac{A}{A_0}\,\phi,\ \min_i \frac{n_i}{r_i}\right).
```
Here $`A`$ is the cell's area, $`A_0`$ is the area of a cell of radius 0.45, and $`\phi`$ is the share of the enzyme's capacity acting on this pool.

The sign of $`\mu`$ sets an enzyme's coupling to enval, and an optimum of exactly zero has none. A coupled enzyme takes up aligned enval $`I = \min(\max(0, \mathrm{sgn}(\mu)L),\ \tau\xi)`$ with throughput $`\tau`$, and converts a fraction $`\eta`$ of it into energy. It releases enval of the opposite sign, made up of a fraction $`\rho`$ of the input and a pumped amount $`\psi\xi`$ that costs one energy per unit. Both exchanges are summed over the cell step and applied through the cell's own bilinear stencil.

Energy is conserved by an exact ledger. A reaction with harvest fraction $`h`$ changes the cell's energy by:
```math
\Delta e = \Delta e_{\mathrm{chem}} + \eta I - \psi\xi - \Delta e_{\mathrm{asm}}, \qquad \Delta e_{\mathrm{chem}} = \begin{cases} h\,(U_r - U_p)\,\xi, & U_r \geq U_p, \\ (U_r - U_p)\,\xi, & U_r < U_p, \end{cases}
```
where $`U_r`$ and $`U_p`$ are the intrinsic energies of the reactant and product vectors. The executed extent $`\xi \leq \xi_{\mathrm{req}}`$ is the largest the cell can pay for from its energy and this reaction's enval harvest, so a cell never overdraws.

Products are routed among four destinations: a fraction $`s`$ is secreted to the field, fractions $`b\,a`$ and $`d\,a`$ build the cell's body and boundary, and the rest stays in the cytoplasm. Building saturates with the cell's energy density, $`a = (e/A) / (2 + e/A)`$, and costs $`\Delta e_{\mathrm{asm}} = 0.5\,(b + d)\,a\,\xi \sum_i p_i`$.

An enzyme's exposure $`\varepsilon`$ divides its capacity between the cytoplasm, $`\phi = 1 - \varepsilon`$, and the cell surface. A surface enzyme acts on whatever fills each part of the perimeter. A neighbor at distance $`d`$ within the contact gap $`g = 0.1`$ occupies a share of the perimeter:
```math
w = \mathrm{clamp}\left(\frac{r + r' + g - d}{g},\ 0,\ 1\right)\frac{\arcsin\left(\min(1,\ r'/d)\right)}{\pi}.
```
Shares summing above 1 are scaled to 1, and any remainder faces the field. On the field, an enzyme reacts with the concentrations at the cell's stencil, with $`\Omega = 1`$ and $`\phi = \varepsilon(1 - \sum w)`$. On a neighbor, it reacts with that neighbor's boundary, with $`\Omega = 2\pi r'`$ and $`\phi = \varepsilon w`$, and routes the products as its own.

## 3. Cells and Physiology
A cell is a disk at a continuous position with three element reservoirs: a body, a boundary, and a cytoplasm. Founders start with energy 2.0, material drawn from the field beneath them, and radius 0.45. Geometry follows from composition:
- The body sets the size, with area $`A = \sum_i B_i`$ and radius $`r = \sqrt{A / \pi}`$ at body density 1.
- The boundary sets the strain $`\varsigma = \sum_i S_i / (0.2 \cdot 2\pi r)`$.
- The boundary's mean element polarity is the membrane polarity $`\bar{P}`$, or 0.5 for an empty boundary.

Elements cross the membrane down their concentration gradient, driven by the permeability $`P = 4`$. Each element's transfer depends on how closely its polarity $`P_i`$ matches the membrane's, and moves only what is available:
```math
\Delta n_i = \min\left(0.25,\ \frac{P\,2\pi r\,\Delta t}{2A}\left(0.25 + 1.5\,e^{-(P_i - \bar{P})^2 / (2 \cdot 0.3^2)}\right)\right)\left(c_i - \frac{n_i}{A}\right)A.
```
Each tick, a cell with maintenance coefficient $`m`$ (0.0786 in founders) and $`N`$ enzymes pays $`(mA + 0.01N)\,\Delta t`$ in energy. There are no other time limits on a cell. A cell dies when its energy reaches zero, and it lyses when its strain falls below 0.7; either way, all its elements return to the field. A cell undergoes fission when its strain reaches 1.485 and both daughters would keep a body of at least 0.05. A uniformly drawn fraction in [0.45, 0.55] of its body, boundary, cytoplasm, and energy forms a daughter, the two separate along a random axis, and the daughter's genome is a mutated copy.

Cells also drift. Each tick, a cell moves by a Gaussian step in each axis with $`\sigma = \sqrt{2 D_c\,\Delta t}`$ and $`D_c = 0.2 \cdot 0.45 / r`$, so larger cells drift less. Up to two relaxation passes then push overlapping pairs apart, each cell taking half of the penetration.

## 4. Heredity and Evolution
Founders share one genome design and differ in their seeded position, the local composition of their material, one optimum for all their enzymes (the world's average enval plus a uniform offset of up to 0.1), and their lineage label. Each carries three enzymes: downhill **D** to **A**, and an enval engine of **B** to **C** with its reverse **C** to **B**. They have no exposure and a mutation rate $`\nu = 0.06`$. No further cells are added after startup.

Variation arises only at fission and is blind: no draw depends on the local environment or on fitness. In a daughter's genome:
- With probability $`0.2\nu`$, an enzyme is replaced by a random one that keeps its optimum.
- Otherwise, each of its numbers mutates independently with probability $`\nu`$: one stoichiometric coefficient of each vector, the rate, half-saturation, harvest and routing fractions, enval width, throughput, energy and release fractions, pump, exposure, and optimum. The optimum takes an unbounded Gaussian step of 0.3, and products are rescaled to balance total reactants.
- With probability $`0.3\nu`$, a random enzyme is removed; failing that, with probability $`0.3\nu`$, a random enzyme is added with the optimum of a random donor.

A lineage is a label that a daughter inherits from its parent, used for coloring and statistics. No rule reads it to decide what a cell does. Selection is whatever differences in survival and fission follow from these rules, and no outcome is assured: results vary with the seed.

The authoritative order of one tick is:
1. Diffuse the element fields.
2. Step each living cell in store order: react internally, deposit secretions, react at the surface, exchange enval, update the radius, pay maintenance, die if out of energy, transport elements, and then lyse or divide.
3. Drift the cells, then resolve overlaps.
4. Diffuse enval, then recharge and relax each tile.
5. Advance the clock.

## 5. Runtime and Interface
The simulation is written in Rust and compiled to `./rust/microcosm.wasm`, which the browser hosts with no backend. WasmGPU renders it and Chalkboard computes the analysis panel. The same seed gives the same trajectory on a given build, one tick at a time or in batches, and across a snapshot resume.
- `./rust/core/src/` holds the model: `chem.rs` (elements), `environment.rs` (landscape and recharge), `genome.rs` (enzymes and mutation), `bio.rs` (reaction kinetics), `cell.rs`, `interaction.rs` (contact shares), `spatial.rs`, `config.rs`, `stats.rs`, `render_buffers.rs`, `snapshot.rs`, `fingerprint.rs`, and `rng.rs`.
- `./rust/core/src/world/` holds the world: `mod.rs` (state and tick order), `fields.rs`, `cells.rs`, `mechanics.rs`, `census.rs`, `inspect.rs`, and `invariants.rs`.
- `./rust/cli/src/` holds the native `microcosm` tool, and `./rust/wasm/src/lib.rs` holds the WebAssembly interface.
- `./wasmgpu/` holds the WasmGPU engine, where `bridge.js` connects the exports, `runtime.js` owns the simulation handle and memory views, `render.js` draws, `interact.js` handles selection and editing, and `gui.js` presents diagnostics and analysis.
- `./main.js` coordinates startup and the frame schedule, and `index.html` holds the page.

The native tool has four commands, run with `cargo run --release --manifest-path rust/Cargo.toml -p microcosm-cli --`:
- `run` advances a world, with optional invariant checks, CSV and JSON output, snapshots, profiling, and a state fingerprint.
- `bench` measures throughput.
- `inspect` summarizes a snapshot and checks its invariants.
- `viability` runs several seeds against fixed health criteria.

The page starts a world with seed `42`, or the `seed` URL parameter, and 32 founders. Each frame advances up to 6 ticks, one at a time, until the simulated time is caught up or 12 ms of stepping has been spent, so a crowded world slows its clock rather than its frame rate. Full statistics refresh 5 times a second and immediately on reset, pause, single-step, and edits.

Render data is uploaded directly from WebAssembly memory. The view shows enval, tile occupancy, mass, total elements, or any one element, with cells overlaid by lineage color. The interface offers seed control, reset, pause, single-step, fullscreen, enval and genome editing, tile, cell, and lineage selection, and inspection of genomes, enzymes, element reservoirs, and recent reactions. The Population analysis panel uses Chalkboard to plot the recent history of live, exposed, and mixed-polarity cells, and a live histogram with summary statistics of cell radius, energy, age, or enzyme count.
