/** Optimizer-step estimation mirror (single epoch → total steps).
 *
 *  SYNC WITH `runtime/training/phases/optimizer.py:62`:
 *
 *      _dl_len = len(ctx.dataloader)
 *      ctx.steps_per_epoch = (_dl_len + args.grad_accum - 1) // args.grad_accum
 *
 *  The subtle part is `_dl_len`: DDP sharding happens **inside the sampler**
 *  (`BucketBatchSampler` / `NavitPackBatchSampler` both call `_ddp_shard`, see
 *  `runtime/training/dataset.py:1796`), so `len(dataloader)` is already the
 *  per-rank batch count — batch is split across cards on top of being split
 *  into batches. Any estimate that divides only by (batch × ga) overshoots by
 *  exactly the card count.
 *
 *  This lives in its own module (rather than inline in `Train.tsx`) because it
 *  is pure arithmetic that must track a backend formula — that combination
 *  needs tests, and a value buried in a component can't have them. The
 *  ddp-divisor bug was silently wrong in the panel for exactly that reason.
 */

/** Inputs for the estimate. All counts are **global** (pre-sharding). */
export interface StepEstimateInput {
  /** Effective sample count = Σ repeat × imgs × resoCount. Ignored in navit mode. */
  totalEffective: number
  /** config.batch_size — per-card batch (PyTorch DDP convention). */
  batchSize: number
  /** config.grad_accum — serial accumulation, does not shard. */
  gradAccum: number
  /** config.ddp_num_processes — card count; 1 or 0 means single-card. */
  ddpProcs: number
  /** config.epochs. 0 → no natural total. */
  epochs: number
  /** config.max_steps. 0 means "unlimited" (schema convention). */
  maxSteps: number
  /** config.navit_packing. */
  navitOn: boolean
  /**
   * Backend pack simulation result — **global** pack count for one epoch.
   * `null` when the simulation hasn't arrived yet (or navit is off).
   * Note the producer (`studio/services/projects/versions.py:897`) has no DDP
   * awareness, so this is a pre-sharding number just like `totalEffective`.
   */
  packsPerEpoch: number | null
}

export interface StepEstimate {
  /** Per-rank optimizer steps in one epoch. `null` = cannot estimate yet. */
  stepsPerEpoch: number | null
  /** stepsPerEpoch × epochs, before the max_steps cap. */
  naturalTotal: number | null
  /** What the run will actually do: min(maxSteps, naturalTotal) when capped. */
  finalTotal: number | null
  /** True when max_steps cuts the run short of `epochs` full passes. */
  maxStepsTruncates: boolean
  /** The divisor actually applied to the sample/pack count (for display). */
  divisor: number
}

/** Coerce a possibly-absent config field to a positive int, `fallback` on junk.
 *
 *  `Number(undefined)` is NaN and `NaN || fallback` → fallback, which is why the
 *  `|| fallback` idiom works for the 1-defaults. Kept explicit so a 0 in the
 *  config (legal for max_steps / epochs) doesn't silently become 1.
 */
export function positiveInt(v: unknown, fallback: number): number {
  const n = Number(v)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback
}

/** Steps for one rank in one epoch, applying the three stages in backend order.
 *
 *  The order matters and a single combined division gets it wrong:
 *
 *    1. batch:  ceil(samples / batch)  — partial last batch still runs
 *    2. shard:  floor(batches / cards) — `_ddp_shard` **truncates**, because
 *               ranks must have strictly equal batch counts or the short rank
 *               exits the loop early and the others hang forever on the next
 *               collective (`runtime/training/dataset.py:863-871`)
 *    3. accum:  ceil(perRank / ga)     — a partial accumulation window still
 *               takes an optimizer step (optimizer.py:62 rounds up)
 *
 *  `ceil(samples / (batch·ga·cards))` collapses all three and can overshoot by
 *  one: 10 samples, batch 1, ga 1, 3 cards → backend 3, combined formula 4.
 */
function stepsForOneRank(count: number, batch: number, ga: number, cards: number): number {
  const batches = Math.ceil(count / batch)
  const perRank = Math.floor(batches / cards)
  // per_rank == 0 means the epoch trains nothing at all (dataset.py:881 returns
  // [] and warns). Report 0 rather than 1 so the panel can't imply progress.
  if (perRank === 0) return 0
  return Math.ceil(perRank / ga)
}

/** Estimate per-epoch and total optimizer steps.
 *
 *  Regular path:  samples, batched by `batch_size`.
 *  NaViT path:    packs, and `batch_size` does not participate — the pack
 *                 sampler fills each step from a token budget, so one step is
 *                 one pack (batch = 1 in the staged formula).
 *
 *  Not modelled: AR-bucketing loss. `BucketBatchSampler` batches **within each
 *  bucket**, so K buckets can produce up to K partial batches, while this
 *  treats the dataset as one pool. Under a same-AR dataset the error is < 5%;
 *  with many buckets the real count is somewhat higher. The backend counts real
 *  batches and is always the authority — this is a planning aid.
 */
export function estimateSteps(input: StepEstimateInput): StepEstimate {
  const bs = positiveInt(input.batchSize, 1)
  const ga = positiveInt(input.gradAccum, 1)
  const cards = positiveInt(input.ddpProcs, 1)
  const epochs = positiveInt(input.epochs, 0)
  const maxSteps = positiveInt(input.maxSteps, 0)

  // batch_size does not participate in navit batching — the pack sampler
  // decides how many images go in a step from the token budget.
  const divisor = input.navitOn ? ga * cards : bs * ga * cards

  let stepsPerEpoch: number | null = null
  if (input.navitOn) {
    // Show nothing rather than a wrong number while the simulation is in
    // flight (宁缺毋假) — the frontend can't reproduce the packer.
    if (input.packsPerEpoch !== null && input.packsPerEpoch > 0) {
      stepsPerEpoch = stepsForOneRank(input.packsPerEpoch, 1, ga, cards)
    }
  } else if (input.totalEffective > 0) {
    stepsPerEpoch = stepsForOneRank(input.totalEffective, bs, ga, cards)
  }

  const naturalTotal =
    stepsPerEpoch !== null && epochs > 0 ? stepsPerEpoch * epochs : null
  const finalTotal =
    naturalTotal !== null && maxSteps > 0 ? Math.min(maxSteps, naturalTotal) : naturalTotal
  const maxStepsTruncates =
    maxSteps > 0 && naturalTotal !== null && maxSteps < naturalTotal

  return { stepsPerEpoch, naturalTotal, finalTotal, maxStepsTruncates, divisor }
}

/** Global batch = per-card batch × grad_accum × cards.
 *
 *  Mirrors the wording in `studio/domain/training.py:395-404`. Exported so the
 *  panel and the ddp field hint can't drift apart.
 */
export function globalBatch(batchSize: number, gradAccum: number, ddpProcs: number): number {
  return positiveInt(batchSize, 1) * positiveInt(gradAccum, 1) * positiveInt(ddpProcs, 1)
}
