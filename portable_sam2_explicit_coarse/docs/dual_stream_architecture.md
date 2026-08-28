# Dual-Stream (UAV→Satellite) Architecture Notes

Design intent: use UAV oblique views of the *same scene* to enhance
satellite-view instance segmentation. All fusion is geometry-faithful (known
intrinsics/extrinsics, no learned alignment) and zero-gated (baseline exactly
preserved at init), so any metric change is attributable to UAV content.

## Streams

### Satellite stream (trainable)
1. **Encoder** — SAM2 base_plus + LoRA (r=16, qkv/proj, ~1.49M / 70.6M trainable).
   Outputs: stride-16 `image_embeddings` (64×64×256), backbone-FPN high-res
   features (s4/s8), hidden states.
2. **Neck** — `RSSAM2PAFPN`: five 512-ch maps at strides 4/8/16/32/64
   (P2…P6). P2 = 256×256, P3 = 128×128 for 1024² input.
3. **Heads** — RPN (all strides) → proposals; BBox head (RoIAlign s4–32);
   RoI mask head: point prompts (`point_emb` on mask feats), shape-prior
   injector, P2 boundary refiner (uses P2 directly), SAM2 mask decoder +
   quality head. The decoder attends to `image_embeddings` — the only direct
   feature path to final mask logits.

### UAV stream (frozen)
4. **Encoder** — a second SAM2 base_plus, frozen, same pretrained weights as
   satellite init. Consumes 5 oblique 512² views (yaw 0/72/144/216/288°).
   Produces a 4-level 256-ch FPN.

### Geometry fusion
5. **GeometrySparseBEV** — builds a 128×128 ground grid covering the same
   360 m square as the satellite tile; projects each ground point through
   W2C + K into the 5 views (FP32, autocast disabled — cm-scale world
   coordinates overflow fp16); samples K/V tokens in windows 7/5/3/3 across
   all views/levels; cross-attends with satellite-P3 queries →
   BEV 128×128×128.
6. **P2P3ZeroGate** — `γ_p2`, `γ_p3` (zero-init scalars) × 1×1 convs inject
   the BEV residual into P2 (bilinear-upsampled) and P3.
7. **EmbeddingZeroGate** (r3 variant, `UE_DUAL_INJECT_EMBEDDING=1`) — same
   contract into the stride-16 `image_embeddings` (1×1 conv 128→256,
   zero-init γ_emb). Puts UAV information on the mask decoder's main canvas.

## Loss
`L = L_rpn_cls + L_rpn_bbox + L_cls + L_bbox + L_mask + L_shape + L_p2br (+ L_quality)`

## Optimizer groups
sat_backbone (LoRA) · mask_decoder · shape_prior · p2_boundary_refiner ·
quality_head · sat_other · **drone_guidance** (geometry_bev + p2p3_gate +
embedding_gate, lr×1.0). UAV encoder excluded (frozen).

## Monitors (per epoch)
`dual-stream monitor: valid=… point_valid=… bev[mean/max]=… γ[p2/p3]=…
γ_emb=… injected_ratio[p2/p3]=… injected_ratio_emb=…`

r1 lesson: valid=0% / γ=0 for the whole run (fp16 overflow) — these fields
exist so a dead fusion branch is visible in epoch 1.

## Mermaid

```mermaid
flowchart TB
    SAT["🛰 Satellite image 1024×1024×3"]
    UAVIN["🚁 5 UAV views 512×512×3<br/>K 5×3×3 · W2C 5×4×4 · scene center (cm)"]

    subgraph SATSTREAM["Satellite stream (trainable)"]
        ENC["SAM2 Base+ encoder<br/>+ LoRA r=16 (1.49M/70.6M)"]
        IMGEMB["image embedding<br/>64×64×256 (stride 16)"]
        NECK["RSSAM2PAFPN neck"]
        PYR["FPN pyramid 512 ch<br/>P2 256² · P3 128² · P4 64² · P5 32² · P6 16²"]
    end

    subgraph UAVSTREAM["UAV stream (frozen)"]
        DENC["SAM2 Base+ encoder (frozen)"]
        DFPN["4-level FPN 256 ch"]
    end

    subgraph FUSION["Geometry fusion (zero-gated)"]
        BEV["GeometrySparseBEV<br/>1) ground grid 128×128 (360 m, sat-aligned)<br/>2) project W2C·K → uv (FP32)<br/>3) window sampling 7/5/3/3 × 5 views<br/>4) cross-attn: Q=sat P3, K/V=UAV<br/>⇒ BEV 128×128×128"]
        GATE["P2P3ZeroGate<br/>γ_p2, γ_p3 · 1×1 conv (0-init)"]
        EGATE["EmbeddingZeroGate<br/>γ_emb · 1×1 conv 128→256 (0-init)"]
    end

    subgraph HEADS["Detection heads"]
        RPN["RPN (s4–64)"]
        BBOX["BBox head"]
        MH["RoI mask head<br/>point prompts · shape prior<br/>P2 boundary refiner<br/>SAM2 mask decoder + quality head"]
        OUT["Instance masks"]
    end

    SAT --> ENC
    ENC -->|"stride 16"| IMGEMB
    ENC -->|"hidden states"| NECK
    NECK --> PYR
    UAVIN --> DENC
    DENC --> DFPN

    PYR -->|"queries (P3)"| BEV
    DFPN -->|"K / V tokens"| BEV
    UAVIN -.->|"K, W2C, center"| BEV

    BEV --> GATE
    BEV --> EGATE
    GATENOTE["⊕ residual"]
    GATE -.->|"⊕ into P2, P3"| PYR
    EGATE -.->|"⊕ into embedding"| IMGEMB

    PYR --> RPN
    RPN --> BBOX
    RPN -.->|"proposals"| MH
    PYR -->|"mask feats P2–P5"| MH
    IMGEMB -->|"decoder canvas (+BEV)"| MH
    MH --> OUT

    classDef sat fill:#dbeafe,stroke:#1e40af,color:#1e3a8a
    classDef uav fill:#ffedd5,stroke:#9a3412,color:#7c2d12,stroke-dasharray:6 3
    classDef fus fill:#dcfce7,stroke:#166534,color:#14532d
    classDef hd fill:#ede9fe,stroke:#5b21b6,color:#4c1d95
    class SAT,ENC,IMGEMB,NECK,PYR sat
    class UAVIN,DENC,DFPN uav
    class BEV,GATE,EGATE,GATENOTE fus
    class RPN,BBOX,MH,OUT hd
```

Variants: **r1** (dead fusion, fp16 overflow — superseded) · **r2**
(P2P3ZeroGate only) · **r3/emb** (P2P3 + EmbeddingZeroGate). All share seed
44 / 128 train scenes / 259 val scenes with the single-stream control
(best segm/mAP 0.7312).
