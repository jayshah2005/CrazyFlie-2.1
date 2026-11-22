# AI Day poster figures

Figures for a poster on the learned controller: a neural network, trained only
in simulation, that flies the Crazyflie 2.1 Brushless with no PID.

`figures/` holds each figure twice: **SVG** (vector, use this for print; text
stays editable) and **PNG** (300 dpi). To rebuild them:

```bash
python poster/make_figures.py
python poster/make_figures.py --flight logs/rl_trigger_flight_<stamp>.csv   # also makes Fig 7
```

## What drone RL papers put on the page

I looked at how three closely related papers present their work:

- **Learning to Fly in Seconds** (Eschmann et al., 2024), the method this
  project builds on. It shows a ladder of control levels (position down to motor
  RPM), a training and inference diagram, training curves (mean ± spread over 50
  seeds), real flights plotted against the reference in black, and a
  long-exposure photo of the drone flying.
- **Swift** (Kaufmann et al., Nature 2023), RL drone racing. It opens with
  photos, then a system diagram with the policy labelled as an MLP, then results
  as distributions, then flown trajectories compared.
- **Nano-drone system-identification benchmark** (Busetto et al., 2025), on the
  same Crazyflie 2.1 Brushless. It shows a photo of the drone, the axis
  convention, reference paths next to long-exposure photos of the real flights,
  and measured versus predicted time series.

The same five visuals appear in all of them: a **photo**, a **system diagram**,
the **network or control structure**, **real flights against the reference**,
and **quantitative results**. Figures 1–10 below cover four of the five.
**Photos and training curves need your data**; see "Still to add".

## Print standards used

- **Size:** A0 portrait (841 × 1189 mm), three columns. Each figure is drawn at
  its **printed size**, so place it at 100 % and do not rescale it. One column
  is 250 mm wide, two columns are 515 mm and the full width is 780 mm.
- **Text:** labels at 19–22 pt, box titles at 26 pt, so they are readable from
  1–2 m. Suggested sizes for the rest of the poster: title 85–100 pt, headings
  48–60 pt, body text 32 pt or more, captions 18 pt or more.
- **Little text:** 300–800 words in total. Each figure makes one point and has
  no title of its own. Put the suggested headline below in the poster's heading
  style.
- **#betterposter layout:** one main finding in plain language, big, across the
  middle. Supporting figures go in the side columns.
- **Colour:** blue means simulation or the network, orange means the real drone,
  in every figure. The palette was checked with a colour-vision-deficiency
  validator and every series has a text label. In flight plots the reference is
  drawn in black, as in Learning to Fly in Seconds. The status colours (green,
  amber, red) are used only in Figure 10, always with an icon and a label.
- **Font:** Arial throughout. Use the same font for the rest of the poster.

## Suggested layout (A0 portrait)

The full set is too much for one A0 poster. Use about 6–7 figures. The "Must"
column below marks the ones I would keep.

| Row | Left + middle columns | Right column |
| --- | --- | --- |
| Top | Title, names, main finding, **photo of the drone** | QR code (repo or video) |
| 1 | Fig 1 Pipeline (spans all three columns) | |
| 2 | Fig 4 Network | Fig 2 Control levels |
| 3 | Fig 7 Flight vs target (or Fig 5 Mission) | Fig 6 Results |
| 4 | Fig 9 Sim-to-real gaps | Fig 10 Safety (or long-exposure photo) |

## The figures

| Fig | File | Size (mm) | Must |
| --- | --- | --- | --- |
| 1 | `fig1_pipeline` | 780 × 185 | yes |
| 2 | `fig2_control_levels` | 250 × 302 | yes |
| 3 | `fig3_control_loop` | 515 × 330 | |
| 4 | `fig4_network` | 515 × 250 | yes |
| 5 | `fig5_mission` | 515 × 205 | if no flight log |
| 6 | `fig6_results` | 250 × 205 | yes |
| 7 | `fig7_flight` (needs `--flight`) | 515 × 205 | yes |
| 8 | `fig8_comparison` | 250 × 150 | only once the citation is confirmed |
| 9 | `fig9_sim_to_real` | 515 × 295 | yes |
| 10 | `fig10_safety` | 515 × 238 | |

**Fig 1: Pipeline.** *Headline:* "Trained in simulation, flown for real."
(1) Fly the stock PID and log motor speeds. (2) Fit a brushless simulator to
those flights. (3) Train with reinforcement learning (TD3). (4) Score every 100k
checkpoint in a simulated mission, because the last checkpoint is often not the
best. (5) Compile the network into the firmware. (6) Fly. The return arrow is
the main lesson: each real flight exposed a gap that was fixed in the simulator.

**Fig 2: Control levels.** *Headline:* "One network replaces the whole controller."
The stock firmware chains hand-tuned controllers: position, velocity, attitude,
body rate, then the motor mixer. Our network sits at the lowest level and
outputs motor speeds directly. Learning to Fly in Seconds names this the
hardest level to learn, and Swift, for comparison, outputs thrust and body
rates. This is the "why it's hard" figure.

**Fig 3: Onboard control loop.** *Headline:* "Everything runs on a 41 g drone."
Sensors → Kalman filter → network (100 Hz) → battery-compensated motor commands
→ motors (1 kHz). The laptop only moves the target over radio (50 Hz) and runs
the safety logic. Fig 4 covers most of this, so use it only if you have space.

**Fig 4: The network.** *Headline:* "149 numbers in, 4 motor speeds out."
It's a small fully connected network (multilayer perceptron): 2 hidden layers of
64 tanh neurons, as in the upstream `actor_and_critic.h`, which our patch
doesn't change. 128 of the 149 inputs are its own last 32 motor commands. They
stand in for motor speeds it cannot measure. One pass takes ~0.7 ms on the
drone's microcontroller.

**Fig 5: The mission.** *Headline:* "One network: take-off, circle, landing."
This is the target that `rl_trigger.py --mode mission` sends for the 1 m flight,
computed from the same profile. Fig 7 replaces it once you have a flight log.

**Fig 6: Real-flight results.** *Headline:* "Within a few centimetres, from 0.6 m
to 1.5 m." The tracking error in each phase for the flown policy I1, from the
results table in `l2f/README.md`. The 0.6 m bars are the midpoint of repeated
flights, and the whisker shows the range. The other heights are single flights.
The 10.9 cm at 1.5 m includes one 36 cm Lighthouse position jump.

**Fig 7: Flown vs target.** *Headline:* "What the real drone actually flew."
This is the standard results figure in these papers: the real path in orange
against the target in black, as height over time plus a top view, with the mean
error. It needs a mission CSV from the Windows flight laptop
(`logs/rl_trigger_flight_*.csv`). Pass it with `--flight`. It plots only the
samples where the policy was in control. The script exits with an error if the
log has fewer than 50 such samples.

**Fig 8: Comparison with published controllers.** *Headline:* "Hover accuracy on
par, smoothness still to win." **Check the source before printing.** These
numbers come from `l2f/README.md`, which credits a "brushless-modelling paper,
ICRA 2026". They are **not** in Busetto et al. (arXiv 2512.14450), which is a
system-identification benchmark with no controller comparison. My earlier draft
wrongly credited that paper. Find the actual paper and cite it in the caption.
If you can't, leave this figure out. It is also not a head-to-head test:
different setups, and our hover is one phase of a full mission.

**Fig 9: Closing the sim-to-real gap.** *Headline:* "What the simulator was
missing." Each row is a failure on the real drone and the fix, tagged Sim or
Firmware. The 35× gyro-noise gap is the strongest single talking point.

**Fig 10: Safety.** *Headline:* "Fails safe: it lands instead of falling."
Releasing Enter in the air, a low battery, or lost telemetry triggers an
automatic landing. Ctrl+C, or 1 s without keep-alive packets, cuts the motors.
The diagram leaves out one case: releasing Enter on the floor aborts the
mission.

## Still to add (needs your data)

1. **A photo of the drone**, top of the poster. Every paper above opens with one.
2. **A long-exposure photo of the mission circle.** Learning to Fly in Seconds
   and the IDSIA benchmark both use one. Put LEDs on the drone, use a dim room,
   and set a 20–30 s exposure on a tripod.
3. **A training curve:** evaluation score against training steps for the
   checkpoints `train_mission.sh` compares. It's the most common RL figure, but
   the data is in WSL. Send me the evaluation output and I'll plot it.
4. **Fig 7**, from a real mission log, as above.

## Sources

Papers:
- [Learning to Fly in Seconds (arXiv 2311.13081)](https://arxiv.org/html/2311.13081v2)
- [learning-to-fly code: network size](https://github.com/arplaboratory/learning-to-fly/blob/master/src/config/actor_and_critic.h)
- [Champion-level drone racing using deep RL (Nature 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- [Nonlinear System Identification Nano-drone Benchmark (arXiv 2512.14450)](https://arxiv.org/html/2512.14450)

Poster guidance:
- [UCL: academic poster guidance](https://www.ucl.ac.uk/brand-and-experience/brand/using-our-brand/print/academic-posters)
- [NTU: poster text and fonts](https://libguides.ntu.edu.sg/posterdesign/text-fonts)
- [Animate Your Science: poster font sizes](https://www.animateyour.science/post/how-to-choose-appropriate-font-sizes-for-your-scientific-poster)
- [Better Posters: the Morrison billboard poster](https://betterposters.beehiiv.com/p/the-morrison-billboard-poster-part-19-05-04)
- [Creating a better science conference poster](https://dionhaefner.github.io/2021/09/creating-a-better-science-conference-poster)
