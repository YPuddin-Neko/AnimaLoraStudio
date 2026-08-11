import { describe, expect, it } from 'vitest'
import { estimateSteps, globalBatch, positiveInt, type StepEstimateInput } from './trainSteps'

/** Defaults = single-card, no navit, nothing capped. Override per test. */
function inp(over: Partial<StepEstimateInput> = {}): StepEstimateInput {
  return {
    totalEffective: 0,
    batchSize: 1,
    gradAccum: 1,
    ddpProcs: 1,
    epochs: 0,
    maxSteps: 0,
    navitOn: false,
    packsPerEpoch: null,
    ...over,
  }
}

describe('estimateSteps — DDP divisor (regression)', () => {
  // These two cases are the bug that motivated this module. Both numbers come
  // from a real 2-card DCU run: the panel showed 71 while the trainer logged
  // 18/36 — off by exactly the card count, because ddp_num_processes was
  // missing from the denominator.
  it('divides by card count: 142 samples, bs=2 ga=2 on 2 cards → 18 (not 36)', () => {
    const r = estimateSteps(inp({ totalEffective: 142, batchSize: 2, gradAccum: 2, ddpProcs: 2 }))
    // Staged, exactly as the backend: ceil(142/2)=71 batches → 71//2=35 per
    // rank (truncated) → ceil(35/2)=18 steps. 35 and 18 are the numbers the
    // real 2-card run logged.
    expect(r.stepsPerEpoch).toBe(18)
    expect(r.divisor).toBe(8)
  })

  it('divides by card count: 142 samples, bs=1 ga=2 on 2 cards → 36 (not 71)', () => {
    const r = estimateSteps(inp({ totalEffective: 142, batchSize: 1, gradAccum: 2, ddpProcs: 2 }))
    expect(r.stepsPerEpoch).toBe(36)
    expect(r.divisor).toBe(4)
  })

  it('single card is unchanged by the fix: 142 / (1×2×1) = 71', () => {
    const r = estimateSteps(inp({ totalEffective: 142, batchSize: 1, gradAccum: 2, ddpProcs: 1 }))
    expect(r.stepsPerEpoch).toBe(71)
  })

  it('ddpProcs of 0 / undefined / NaN is treated as single-card', () => {
    for (const bad of [0, undefined, NaN, -3, 'x']) {
      const r = estimateSteps(
        inp({ totalEffective: 100, ddpProcs: bad as unknown as number }),
      )
      expect(r.stepsPerEpoch).toBe(100)
    }
  })

  it('N cards scales the estimate down by exactly N', () => {
    // 960 is divisible by every card count tested, so the roundings can't mask
    // an off-by-one and the ratio has to be exact.
    const base = estimateSteps(inp({ totalEffective: 960, ddpProcs: 1 })).stepsPerEpoch!
    for (const n of [2, 3, 4, 8]) {
      const got = estimateSteps(inp({ totalEffective: 960, ddpProcs: n })).stepsPerEpoch!
      expect(got).toBe(base / n)
    }
  })
})

describe('estimateSteps — staged rounding matches the backend', () => {
  it('shard truncates, so the estimate is not ceil(samples / (bs·ga·cards))', () => {
    // 10 batches over 3 cards: _ddp_shard hands each rank 10//3 = 3 and drops
    // the tail, because unequal batch counts deadlock DDP (dataset.py:863-871).
    // The single-division formula would say ceil(10/3) = 4.
    const r = estimateSteps(inp({ totalEffective: 10, batchSize: 1, gradAccum: 1, ddpProcs: 3 }))
    expect(r.stepsPerEpoch).toBe(3)
  })

  it('partial last batch still counts (batching rounds up)', () => {
    // 7 samples at batch 2 → 4 batches, the last holding a single sample.
    const r = estimateSteps(inp({ totalEffective: 7, batchSize: 2 }))
    expect(r.stepsPerEpoch).toBe(4)
  })

  it('partial accumulation window still takes a step (optimizer.py:62 rounds up)', () => {
    // 5 batches at ga=2 → 3 steps: the third window has only one batch in it.
    const r = estimateSteps(inp({ totalEffective: 5, batchSize: 1, gradAccum: 2 }))
    expect(r.stepsPerEpoch).toBe(3)
  })

  it('reports 0 when there are fewer batches than cards (empty epoch)', () => {
    // dataset.py:881 returns [] and warns — every rank gets nothing, so the
    // epoch trains nothing. Must not round up to 1 and imply progress.
    const r = estimateSteps(inp({ totalEffective: 3, batchSize: 1, ddpProcs: 4, epochs: 10 }))
    expect(r.stepsPerEpoch).toBe(0)
    expect(r.naturalTotal).toBe(0)
  })
})

describe('estimateSteps — navit path', () => {
  it('ignores batch_size (pack sampler decides step contents)', () => {
    const a = estimateSteps(inp({ navitOn: true, packsPerEpoch: 80, batchSize: 1, gradAccum: 2 }))
    const b = estimateSteps(inp({ navitOn: true, packsPerEpoch: 80, batchSize: 8, gradAccum: 2 }))
    expect(a.stepsPerEpoch).toBe(40)
    expect(b.stepsPerEpoch).toBe(40)
  })

  it('still divides by card count — packs_per_epoch is a global count', () => {
    // versions.py:897 simulates packing with no DDP awareness, and the pack
    // sampler shards through _ddp_shard just like the bucket sampler does.
    const r = estimateSteps(inp({ navitOn: true, packsPerEpoch: 80, gradAccum: 2, ddpProcs: 2 }))
    expect(r.stepsPerEpoch).toBe(20)
    expect(r.divisor).toBe(4)
  })

  it('shows nothing until the backend simulation arrives', () => {
    const r = estimateSteps(inp({ navitOn: true, packsPerEpoch: null, totalEffective: 142 }))
    expect(r.stepsPerEpoch).toBeNull()
    expect(r.naturalTotal).toBeNull()
    expect(r.finalTotal).toBeNull()
  })

  it('does not fall back to totalEffective when packs are 0', () => {
    const r = estimateSteps(inp({ navitOn: true, packsPerEpoch: 0, totalEffective: 142 }))
    expect(r.stepsPerEpoch).toBeNull()
  })
})

describe('estimateSteps — totals and the max_steps cap', () => {
  it('naturalTotal = stepsPerEpoch × epochs', () => {
    const r = estimateSteps(inp({ totalEffective: 142, gradAccum: 2, ddpProcs: 2, epochs: 200 }))
    expect(r.stepsPerEpoch).toBe(36)
    expect(r.naturalTotal).toBe(7200)
    expect(r.finalTotal).toBe(7200)
    expect(r.maxStepsTruncates).toBe(false)
  })

  it('max_steps=0 means unlimited, not "zero steps"', () => {
    const r = estimateSteps(inp({ totalEffective: 100, epochs: 10, maxSteps: 0 }))
    expect(r.finalTotal).toBe(1000)
    expect(r.maxStepsTruncates).toBe(false)
  })

  it('max_steps below the natural total truncates', () => {
    const r = estimateSteps(inp({ totalEffective: 100, epochs: 10, maxSteps: 400 }))
    expect(r.finalTotal).toBe(400)
    expect(r.maxStepsTruncates).toBe(true)
  })

  it('max_steps above the natural total does not inflate it', () => {
    const r = estimateSteps(inp({ totalEffective: 100, epochs: 10, maxSteps: 99999 }))
    expect(r.finalTotal).toBe(1000)
    expect(r.maxStepsTruncates).toBe(false)
  })

  it('epochs=0 gives no total (steps/epoch alone is still shown)', () => {
    const r = estimateSteps(inp({ totalEffective: 100, epochs: 0 }))
    expect(r.stepsPerEpoch).toBe(100)
    expect(r.naturalTotal).toBeNull()
  })

  it('no samples → no estimate at all', () => {
    const r = estimateSteps(inp({ totalEffective: 0, epochs: 10 }))
    expect(r.stepsPerEpoch).toBeNull()
    expect(r.finalTotal).toBeNull()
  })
})

describe('globalBatch', () => {
  it('multiplies per-card batch by ga and cards (studio/domain/training.py:395-404)', () => {
    expect(globalBatch(1, 2, 2)).toBe(4)
    expect(globalBatch(2, 2, 2)).toBe(8)
    expect(globalBatch(1, 1, 1)).toBe(1)
  })

  it('treats missing card count as 1', () => {
    expect(globalBatch(2, 2, 0)).toBe(4)
  })
})

describe('positiveInt', () => {
  it('keeps positive ints, falls back on everything else', () => {
    expect(positiveInt(4, 1)).toBe(4)
    expect(positiveInt('4', 1)).toBe(4)
    expect(positiveInt(0, 1)).toBe(1)
    expect(positiveInt(0, 0)).toBe(0)
    expect(positiveInt(-2, 1)).toBe(1)
    expect(positiveInt(undefined, 1)).toBe(1)
    expect(positiveInt(null, 7)).toBe(7)
    expect(positiveInt('abc', 1)).toBe(1)
    expect(positiveInt(Infinity, 1)).toBe(1)
  })

  it('floors fractional input (a step count is an integer)', () => {
    expect(positiveInt(2.9, 1)).toBe(2)
  })
})
