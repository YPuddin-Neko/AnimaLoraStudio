import { describe, expect, it } from 'vitest'
import { applyAutoMaskRegions } from './InpaintCanvas'

describe('automatic mask rasterization', () => {
  it('uses the exact rectangular interior and independent x/y feather distances', () => {
    const width = 6
    const height = 6
    const pixels = new Uint8ClampedArray(width * height * 4)
    applyAutoMaskRegions(pixels, width, height, [
      { x1: 2, y1: 2, x2: 4, y2: 4, feather_x: 2, feather_y: 1 },
    ])
    const alpha = (x: number, y: number) => pixels[(y * width + x) * 4 + 3]

    expect(alpha(2, 2)).toBe(255)
    expect(alpha(3, 3)).toBe(255)
    expect(alpha(1, 2)).toBe(128)
    expect(alpha(0, 2)).toBe(0)
    expect(alpha(2, 1)).toBe(0)
    expect(alpha(4, 2)).toBe(128)
    expect(alpha(5, 2)).toBe(0)
  })

  it('unions overlapping automatic regions without weakening an existing mask', () => {
    const width = 5
    const height = 3
    const pixels = new Uint8ClampedArray(width * height * 4)
    pixels[(1 * width + 1) * 4 + 3] = 220
    applyAutoMaskRegions(pixels, width, height, [
      { x1: 2, y1: 1, x2: 3, y2: 2, feather_x: 2, feather_y: 0 },
      { x1: 3, y1: 1, x2: 4, y2: 2, feather_x: 1, feather_y: 0 },
    ])
    const alpha = (x: number, y: number) => pixels[(y * width + x) * 4 + 3]

    expect(alpha(1, 1)).toBe(220)
    expect(alpha(2, 1)).toBe(255)
    expect(alpha(3, 1)).toBe(255)
    expect(alpha(4, 1)).toBe(0)
  })
})
