# PromptMiner-SAM2: ICASSP paper blueprint

## 1. Paper identity

### Working title

**PromptMiner-SAM2: Proposal-Conditioned Explicit Prompt Mining for Overhead Instance Segmentation**

Alternative, if the boundary module is experimentally decisive:

**From Proposals to Prompts: Boundary-Aware Prompt Mining for SAM2 Instance Segmentation**

### One-sentence thesis

A detector proposal should not be treated as only a box prompt: an explicitly supervised, proposal-conditioned coarse shape can be mined into complementary native SAM2 prompts, and a prediction-gated high-resolution correction can improve the shape before it is encoded.

### Research question

For automatic overhead instance segmentation, can a two-stage detector convert each proposal into **interpretable point, box, and dense-mask prompts** that use SAM2's native interface more effectively than using the proposal box or learned prompt features alone?

### What this paper is not

- It is not a satellite--UAV fusion paper.
- It is not a new dataset paper.
- It does not claim that SAM2 alone solves remote-sensing instance segmentation.
- It does not claim a boundary improvement without a boundary metric or suitably cautious evidence.

## 2. The story reviewers should retain

```
Detection proposal
      |
      v
Explicit, supervised coarse shape
      |
      +-- interior / exterior --> foreground-background points
      +-- proposal extent -----> box prompt
      +-- spatial shape --------> dense-mask prompt
      |
      v
Prediction-gated P2 local refinement
      |
      v
Native SAM2 prompt encoder + mask decoder --> final instance mask
```

The key contrast is **explicit prompt mining versus opaque prompt adaptation**. The intermediate shape is supervised, visible, and can be diagnosed. The three prompt types are complementary rather than redundant:

| Prompt | Information it supplies | Failure it addresses |
|---|---|---|
| Foreground/background points | Instance interior and exclusion | Leakage into nearby instances |
| Proposal box | Instance extent | Ambiguous spatial scope |
| Dense coarse mask | Global object shape | Box-only prompts that include background |
| P2 refinement | Local high-resolution correction | Blunt or shifted coarse boundaries |

## 3. Contribution claims

Use these claims only after the planned experiments support them.

1. **Explicit prompt bridge.** A proposal-conditioned, supervised coarse-mask branch converts detector features into the sparse and dense prompts used by SAM2, instead of directly producing unconstrained prompt embeddings.
2. **Complementary native prompts.** A confidence- and topology-aware miner produces foreground/background points from the coarse shape; the same proposal anchors the box and dense-mask prompts in one shared coordinate system.
3. **Prediction-gated local refinement.** P2BoundaryRefiner injects high-resolution features only within a search region computed from the prediction, avoiding target leakage at inference and avoiding an unconstrained second mask head.
4. **Evidence package.** Cross-dataset comparisons, prompt-path ablations, refinement-control ablations, qualitative cases, and (if feasible) a boundary-sensitive metric test the mechanism rather than only the final AP.

## 4. Four-page ICASSP narrative

ICASSP permits four pages of technical content; reserve a fifth page for references if needed. The target is a dense, coherent four-page article, not a compressed journal section.

| Space | Content | Reader question answered |
|---|---|---|
| Page 1, left | Title, 120--150-word abstract, two-paragraph introduction, contributions | Why are proposal boxes insufficient for promptable segmentation? |
| Page 1, right + Page 2 | Fig. 1 and Method: detector/SAM2 coupling; explicit coarse-shape branch; coordinate convention | What is the actual bridge from proposal to prompt? |
| Page 2 | Prompt mining and P2 refinement; one compact objective equation | Why are these prompts trustworthy and complementary? |
| Page 3 | Experimental protocol; Table 1 main comparison; Fig. 2 qualitative prompt-to-mask examples | Does the method generalize and what visibly changes? |
| Page 4 | Table 2 ablations; boundary/refinement evidence; limitations; conclusion | Which design choices cause the gain, and what is not established? |
| Optional Page 5 | References and acknowledgements only | Does the paper comply with the venue rule? |

## 5. Proposed section-level outline

### Abstract (120--150 words)

Use exactly four moves: task difficulty; missing bridge between proposals and SAM2 prompts; method; verified results. Do not introduce undefined abbreviations. Do not mention dual-stream data or unrelated benchmark construction.

### 1. Introduction (roughly 0.55 page)

Paragraph 1: Building instance masks require instance separation and precise contours; detector boxes alone contain extent but not object shape.

Paragraph 2: SAM/SAM2 accepts meaningful prompt modalities, but automatic instance-segmentation pipelines commonly use a box or opaque learned adaptation. State the gap: a detector needs an explicit way to translate proposal evidence into complementary prompts.

Paragraph 3: State the purpose directly: *This paper investigates whether a proposal-conditioned coarse shape can serve as the bridge between two-stage detection and native SAM2 prompting.* Briefly introduce the prompt miner and local refiner.

End with three concise contributions. Cite only literature needed to establish the gap: Mask R-CNN/two-stage segmentation, SAM, SAM2, and relevant remote-sensing prompt methods.

### 2. Method (roughly 1.6 pages)

#### 2.1 Overview and shared proposal geometry

Define image, FPN features, candidate proposal $b_r$, RoI feature $X_r$, and final mask. Explain once that $b_r$ is the common coordinate frame for RoIAlign, target crop, point mapping, dense-prompt placement, and box prompting. This is the central design invariant.

#### 2.2 Explicit coarse-shape branch

Give the $64\times64$ coarse-logit equation and BCE+Dice supervision. State why an explicit intermediate target is chosen: it is inspectable and supplies spatial evidence; it is not merely an auxiliary segmentation output.

#### 2.3 Boundary-gated P2 refinement

Describe the prediction-derived uncertainty/boundary band, the validity gate, local $P_2$ RoI feature, and bounded residual. Use a single compact equation for the refined logit. Explicitly state the anti-leakage property: ground truth affects the training loss only, never the forward support.

#### 2.4 Prompt miner and mask decoder

Explain the 2P2N point policy in words, not an excessive morphological derivation: foreground points come from confident interiors; background points from the exterior; unreliable slots are neutralized. Describe dense-prompt rasterization and joint native PromptEncoder invocation. End with one total-loss equation.

### 3. Experiments (roughly 1.45 pages)

#### 3.1 Setup

Report datasets, exact splits, input size, initialization, frozen/trainable blocks, optimizer, schedule, batch size, epochs, checkpoint selection, and COCO mask metrics. Include every setting needed to reproduce the reported comparison.

#### 3.2 Main comparison

Use two public single-view datasets with different difficulty profiles if results are available. The main table must include strong detector-based, remote-sensing, and SAM-prompt baselines under a consistent protocol. If protocols are not comparable, place those values in a separate block and say so.

#### 3.3 Mechanism analysis

First show a cumulative prompt ablation: Point; Point+Box; Point+Box+Dense; Full. Then isolate the refiner: no P2; unrestricted/global P2 correction; prediction-gated P2 correction. Report AP, AP50, AP75; add Boundary IoU or boundary F-score if masks permit it.

#### 3.4 Qualitative and failure analysis

Show at least three cases: adjacent buildings, weak boundary/shadow, and a failure case. Each panel should contain input, proposal, coarse shape, P2-refined shape, selected points, and final mask. Explain a failure honestly (for example, proposal misses an object or coarse topology is wrong), because prompt mining cannot repair missing-instance recall.

### 4. Conclusion (roughly 0.15 page)

Restate the contribution as a proposal-to-prompt bridge, not as a generic foundation-model improvement. Report only the validated empirical conclusion and one specific limitation.

## 6. Required visual and empirical assets

### Figure 1: Method diagram (full width, early in paper)

Redraw the current architecture figure for readability at single-paper scale. It should visually emphasize the dataflow below; remove all dual-stream elements.

```
image -> SAM2 Hiera -> FPN -> RPN/RoI -> coarse shape -> P2 refiner
                                      |                      |
                                      +---- shared box -------+
                                                             |
                          points + box + dense mask ---------+
                                                             v
                                  SAM2 prompt encoder + mask decoder -> mask
```

### Figure 2: Prompt formation and qualitative result (two columns)

For each of 3--4 examples: crop/input; $C_r$; $\widehat C_r$ with refinement support; foreground/background point overlay; final mask vs ground truth. Use a consistent legend and make grayscale legible.

### Table 1: Main results

Use two dataset blocks. Columns: mask AP, AP50, AP75; add box AP only if it serves a stated argument. Do not include rows with missing results. Include number of runs or uncertainty when feasible.

### Table 2: Mechanism ablations

Rows: Point; Point+Box; Point+Box+Dense; Full; Full without P2; Full with unrestricted P2. Columns: AP, AP50, AP75, boundary metric, parameters/FLOPs if the refiner cost is material.

### Optional small table or text line: efficiency

Report inference latency and trainable parameters only if measured under identical hardware and resolution. Do not estimate them post hoc.

## 7. Evidence checklist before drafting results prose

- [ ] Main results are from fixed splits and a documented checkpoint-selection rule.
- [ ] All ablations differ in exactly the stated component(s).
- [ ] Prompt mining is applied identically at training and inference, except for target-only losses.
- [ ] Points are derived from predictions at inference; no ground-truth prompt information leaks.
- [ ] Baseline implementations and input sizes are reported.
- [ ] Qualitative panels use held-out images, not selected training samples.
- [ ] Boundary claims are supported by a boundary-specific metric or softened to an AP75 observation.
- [ ] Every numerical claim has a table, figure, or reproducible log behind it.

## 8. Drafting sequence

1. Freeze the experiments and generate Table 1, Table 2, and Figure 2 first.
2. Redraw Figure 1 around the proposal-to-prompt bridge.
3. Write Method from the dataflow, using only notation needed by the figures.
4. Write experiments around the questions each table answers.
5. Write the introduction and abstract last, after the numerical story is fixed.
6. Fit and compile in the official ICASSP 2026 template; then perform a page-by-page visual inspection.

## 9. Decisions that must be made before a final TeX rewrite

1. Which public datasets have complete, defensible single-stream results available now?
2. Do we have (or can we compute) a boundary metric? If not, remove strong boundary-improvement language.
3. Is the P2 refinement ablation available with a no-P2 and an unrestricted-P2 control? If not, present P2 as an engineering choice, not a causal conclusion.
4. Is the target submission ICASSP 2027? The downloaded ICASSP 2026 template is official for 2026 but must be replaced if ICASSP 2027 publishes a venue-specific kit.
