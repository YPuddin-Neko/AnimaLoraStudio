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

  it('lists per-card VRAM in the VRAM tooltip', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    const vram = await screen.findByText('36.0/128G')
    const tip = vram.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).toContain('显存合计 36.0 / 128 GB (28%) · 2 卡')
    expect(tip).toContain('#0 BW  32.0/64G (50%)')
    expect(tip).toContain('#1 BW  4.0/64G (6%)')
  })

  it('keeps utilization out of the VRAM tooltip', async () => {
    // 两个 tooltip 各只列自己那项指标。早期版本共用一份合并行，于是悬停利用率
    // 时满屏是显存数字、要找的利用率被夹在中间。
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    const vram = await screen.findByText('36.0/128G')
    const tip = vram.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).not.toContain('90%')
    expect(tip).not.toContain('70°C')
  })

  it('lists per-card utilization and temperature in the GPU tooltip', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    const util = await screen.findByText('50%')
    const tip = util.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).toContain('GPU 利用率均值 50% · 2 卡')
    expect(tip).toContain('#0 BW  90% · 70°C')
    expect(tip).toContain('#1 BW  10% · 50°C')
    // 显存不该混进来
    expect(tip).not.toContain('32.0/64G')
  })

  it('marks unavailable utilization explicitly in the GPU tooltip', async () => {
    // 部分卡拿不到利用率时，那一行要写明「不可用」而不是留空或填 0 ——
    // 空行看着像渲染 bug，0% 是合法读数会误导。
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'BW', util_pct: 80, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: 60 },
        { index: 1, name: 'BW', util_pct: null, vram_used_gb: 10.0, vram_total_gb: 64.0, temp_c: 55 },
      ],
    }))
    render(<SystemStats />)
    const util = await screen.findByText('80%')
    const tip = util.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).toContain('#0 BW  80% · 60°C')
    expect(tip).toContain('#1 BW  利用率不可用 · 55°C')
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

  it('never marks an "active" card in tooltips, and shows every card', async () => {
    // 后端会给 active（上游 #491：让 pill 只显示 torch 在用那张），但本分支的 pill
    // 是全卡汇总，逐卡明细里**不该**标「在用」：
    //   - topbar 是 server 进程报的，current_device() 只反映 server 自己那张，
    //     与训练子进程用哪张无关；
    //   - 多卡 DDP 下每个 rank 各占一张，「在用」根本不是单张。
    // 真机上标出来就是「#0 0% ←在用 / #1 100%」，看着像读数矛盾。
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'BW', util_pct: 0, vram_used_gb: 1.1, vram_total_gb: 64.0, temp_c: 54, active: true },
        { index: 1, name: 'BW', util_pct: 100, vram_used_gb: 53.0, vram_total_gb: 64.0, temp_c: 62, active: false },
      ],
    }))
    render(<SystemStats />)
    const vram = await screen.findByText('54.1/128G')
    const tip = vram.closest('[title]')?.getAttribute('title') ?? ''
    expect(tip).not.toContain('在用')
    // 两张卡都要出现在明细里（全卡汇总的意义就在这）
    expect(tip).toContain('1.1/64G')
    expect(tip).toContain('53.0/64G')
  })

  it('only fetches once on mount (SSE 化后无轮询)', async () => {
    const spy = vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
    // 等一段实际时间让任何潜在的轮询有机会触发
    await new Promise((r) => setTimeout(r, 200))
    expect(spy).toHaveBeenCalledTimes(1)
  })

  // ── 功耗 pill（DCU 专有）─────────────────────────────────────────────
  //
  // 起因：容器里 hy-smi 的 AvgPwr 列报 79W/95W，被读成「卡没跑满」，而 sysfs
  // 同一时刻是 564W/563W。后端改读 sysfs 后，这个 pill 把真值显示出来，
  // hover 给频率 —— 判断卡有没有被限制靠的是频率档位，不是瓦数。

  function twoCardsWithPower(): Stats {
    return makeStats({
      gpu: [
        {
          index: 0, name: 'BW', util_pct: 100, vram_used_gb: 53.0, vram_total_gb: 63.0,
          temp_c: 58, power_w: 564, power_cap_w: 1000, sclk_mhz: 1500, sclk_max_mhz: 1500,
        },
        {
          index: 1, name: 'BW', util_pct: 100, vram_used_gb: 53.0, vram_total_gb: 63.0,
          temp_c: 61, power_w: 563, power_cap_w: 1000, sclk_mhz: 1500, sclk_max_mhz: 1500,
        },
      ],
    })
  }

  it('sums power across cards (physical quantity, not averaged)', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCardsWithPower())
    render(<SystemStats />)
    // 功率可加，与显存同口径：564 + 563 = 1127W。**不是**取均值 563。
    await waitFor(() => expect(screen.getByText('1127W')).toBeInTheDocument())
    expect(screen.getByText('PWR×2')).toBeInTheDocument()
  })

  it('shows per-card clock in the power tooltip and marks full speed', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCardsWithPower())
    const { container } = render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('1127W')).toBeInTheDocument())
    const titles = Array.from(container.querySelectorAll('[title]'))
      .map((el) => el.getAttribute('title') ?? '')
    const pwr = titles.find((t) => t.includes('1500/1500MHz'))
    expect(pwr).toBeTruthy()
    // 频率与上限都要在 tooltip 里 —— 这是这个 pill 的重点
    expect(pwr).toContain('564W / 1000W')
    expect(pwr).toContain('满频')       // 满频
  })

  it('marks a throttled card as such', async () => {
    const s = twoCardsWithPower()
    s.gpu![0] = { ...s.gpu![0], sclk_mhz: 600, sclk_max_mhz: 1500, power_w: 90 }
    vi.spyOn(api, 'systemStats').mockResolvedValue(s)
    const { container } = render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('653W')).toBeInTheDocument())
    const titles = Array.from(container.querySelectorAll('[title]'))
      .map((el) => el.getAttribute('title') ?? '')
    expect(titles.some((t) => t.includes('600/1500MHz') && t.includes('降频'))).toBe(true)
  })

  it('hides the power pill when no card reports power (NVIDIA / old backend)', async () => {
    // 旧后端不发这些 key（前端字段是可选的）→ 整个 pill 隐藏，不显示 "0W"
    vi.spyOn(api, 'systemStats').mockResolvedValue(twoCards())
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('36.0/128G')).toBeInTheDocument())
    expect(screen.queryByText(/PWR/)).toBeNull()
  })

  it('excludes cards without power from the cap denominator', async () => {
    // 一张报得出、一张报不出：分母只算报得出的那张（1000W），否则比例被拉低一半
    const s = twoCardsWithPower()
    s.gpu![1] = { ...s.gpu![1], power_w: null, power_cap_w: null }
    vi.spyOn(api, 'systemStats').mockResolvedValue(s)
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('564W')).toBeInTheDocument())
  })
})
