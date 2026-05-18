# Related Work: FocusUI vs. GAP

| Axis | FocusUI | GAP (ours) |
|---|---|---|
| Task | UI grounding (instruction -> bbox) | Bug-type classification (K+1-way) |
| Training requirement | Patch-level supervision required | **Fully training-free** |
| Instruction dependency | Requires instruction embedding | **Instruction-free** (CI/CD friendly) |
| Signals | Instruction-cond. score + UI-graph score | Attention + (1-entropy) + edge density |
| Position handling | PosPad strategy | Plain index selection + CLS preservation |
| Evaluation focus | ScreenSpot-Pro, retention vs. accuracy | **Bug-type collapse-point sensitivity** |

FocusUI is the most direct adjacent UI-token-pruning method because it also targets UI screenshots and tries to preserve the visual tokens that matter most. Its objective, however, is UI grounding: given an instruction, retain tokens that help recover a bounding box. GAP is framed for CI/CD visual bug detection, where the model receives a screenshot and must classify it as clean or as one of the visual bug types. This changes the supervision target, the available inference-time context, and the failure analysis we care about.

The CI/CD setting makes instruction-free and training-free behavior important. Automated UI regression jobs usually do not have a natural language instruction for each screenshot, and teams rarely have patch-level bounding-box labels for every bug type. GAP therefore uses signals already available from the image/model path: early-layer attention, inverse color entropy, and edge density. This keeps the method deployable on unlabeled screenshots while still exposing interpretable per-bug collapse points as drop rate increases.

FocusUI also handles position preservation through PosPad, while GAP deliberately uses plain index selection with explicit token-0 preservation. That choice keeps the implementation small and makes the comparison against generic pruning baselines direct: Random drop tests whether any pruning works, FastV tests text-conditioned language-layer attention, FocusUI represents instruction-conditioned UI grounding, and GAP tests whether GUI-aware image signals are enough for bug-type classification.
