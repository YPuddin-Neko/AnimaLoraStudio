import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SystemStats from './SystemStats'
import { api, type SystemStats as Stats } from '../api/client'

// useEventStream 在 jsdom 下不会真起 EventSource (源码有 typeof 守卫)，所以
// 这里测的主要是：mount 时 GET 一次冷启动 + 各种 stats 形态下的渲染。SSE
// delta 的合并行为另测（手动验证或 e2e）。

function makeStats(overrides: Partial<Stats> = {}): Stats {
  return {
    cpu_pct: 12.5,
    ram_used_gb: 8.0,
    ram_total_gb: 32.0,
    gpu: [
      {
        index: 0,
        name: 'Test GPU',
        util_pct: 50,
        vram_used_gb: 4.0,
        vram_total_gb: 24.0,
        temp_c: 55,
      },
    ],
    ...overrides,
  }
}

describe('SystemStats', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders nothing before first fetch resolves', () => {
    vi.spyOn(api, 'systemStats').mockReturnValue(new Promise(() => {}))
    const { container } = render(<SystemStats />)
    expect(container.firstChild).toBeNull()
  })

  it('shows CPU / MEM / GPU / VRAM pills with values after mount fetch', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.getByText('13%')).toBeInTheDocument()
    expect(screen.getByText('MEM')).toBeInTheDocument()
    expect(screen.getByText('8.0/32G')).toBeInTheDocument()
    expect(screen.getByText('GPU')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('VRAM')).toBeInTheDocument()
    expect(screen.getByText('4.0/24G')).toBeInTheDocument()
  })

  it('hides GPU / VRAM when stats.gpu is null', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: null }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('hides GPU / VRAM when stats.gpu is empty array', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: [] }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('shows high-tone class when util exceeds 90%', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ cpu_pct: 95 }))
    render(<SystemStats />)
    const el = await screen.findByText('95%')
    expect(el.className).toContain('text-err')
  })

  // ── 多卡 ────────────────────────────────────────────────────────────
  // 真机场景：海光 DCU 双卡 BW（各 64GB）。原实现只显示 gpu[0]，等于一半显存
  // 看不见 —— 这几条钉住「显存加总、利用率取均值、逐卡明细进 tooltip」。

  // 显存数值刻意避开 x.5% 的四舍五入边界（如 40/64 = 62.5%）：那种值的取整
  // 结果依赖 JS toFixed 的 half-away-from-zero 语义，断言会变得脆且难读。
  function twoCards(): Stats {
    return makeStats({
      gpu: [
        { index: 0, name: 'BW', util_pct: 90, vram_used_gb: 32.0, vram_total_gb: 64.0, temp_c: 70 },
        { index: 1, name: 'BW', util_pct: 10, vram_used_gb: 4.0, vram_total_gb: 64.0, temp_c: 50 },
      ],
    })
  }

  it('sums VRAM across cards and averages utilization', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    // 显存可加：32 + 4 = 36 已用，64 + 64 = 128 总量
    await waitFor(() => expect(screen.getByText('36.0/128G')).toBeInTheDocument())
    // 利用率取平均：(90 + 10) / 2 = 50%，**不是** 100%（相加无意义）
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('marks multi-card pills with card count in the label', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    // label 带卡数让「这是均值/合计而非单卡」一眼可见
    await waitFor(() => expect(screen.getByText('GPU×2')).toBeInTheDocument())
    expect(screen.getByText('VRAM×2')).toBeInTheDocument()
    expect(screen.queryByText('GPU')).toBeNull()
  })

  it('lists every card in the tooltip', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    const vram = await screen.findByText('36.0/128G')
    const tip = vram.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).toContain('显存合计 36.0 / 128 GB (28%) · 2 卡')
    expect(tip).toContain('#0 BW  32.0/64G (50%) · 90% · 70°C')
    expect(tip).toContain('#1 BW  4.0/64G (6%) · 10% · 50°C')
  })

  it('keeps single-card labels unchanged', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    // 单卡不加 ×N 后缀 —— NVIDIA 单卡用户看到的与改动前一致
    await waitFor(() => expect(screen.getByText('GPU')).toBeInTheDocument())
    expect(screen.getByText('VRAM')).toBeInTheDocument()
    expect(screen.queryByText('GPU×1')).toBeNull()
  })

  it('averages only cards that report utilization', async () => {
    // DCU 上 smi 解析可能只覆盖部分卡；缺失的不该被当 0 拉低均值
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'BW', util_pct: 80, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: null },
        { index: 1, name: 'BW', util_pct: null, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: null },
      ],
    }))
    render(<SystemStats />)
    // 80 而非 40 —— 只对有读数的卡求平均
    await waitFor(() => expect(screen.getByText('80%')).toBeInTheDocument())
    // 显存仍全卡加总
    expect(screen.getByText('20.0/128G')).toBeInTheDocument()
  })

  it('hides the GPU pill when no card reports utilization', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'BW', util_pct: null, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: null },
        { index: 1, name: 'BW', util_pct: null, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: null },
      ],
    }))
    render(<SystemStats />)
    // VRAM 照常显示，利用率 pill 整个隐藏（不显示 "0%" 假读数）
    await waitFor(() => expect(screen.getByText('20.0/128G')).toBeInTheDocument())
    expect(screen.queryByText('GPU×2')).toBeNull()
    expect(screen.queryByText('GPU')).toBeNull()
  })

  it('only fetches once on mount (SSE 化后无轮询)', async () => {
    const spy = vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
    // 等一段实际时间让任何潜在的轮询有机会触发
    await new Promise((r) => setTimeout(r, 200))
    expect(spy).toHaveBeenCalledTimes(1)
  })
})
