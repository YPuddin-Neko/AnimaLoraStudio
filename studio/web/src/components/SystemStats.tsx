import { useEffect, useState } from 'react'
import { api, type SystemStats as SystemStatsData } from '../api/client'
import { useEventStream } from '../lib/useEventStream'

function toneClasses(pct: number): { text: string; bg: string } {
  if (pct >= 90) return { text: 'text-err', bg: 'bg-err-soft' }
  if (pct >= 70) return { text: 'text-warn', bg: 'bg-warn-soft' }
  return { text: 'text-fg-primary', bg: 'bg-accent-soft' }
}

function fmtGb(used: number, total: number): string {
  return `${used.toFixed(1)}/${Math.round(total)}G`
}

interface PillProps {
  label: string
  value: string
  pct: number
  tooltip: string
}

/** 进度条胶囊 — 整个 pill 背景按占用百分比填色 (>=70% warn, >=90% err)，
 *  高度与 topbar 上其他元素 (搜索 icon 32px) 一致。
 *
 *  `min-w-[96px]` + `justify-between` 让 4 个 pill 视觉等宽：CPU/GPU 只占 3-4
 *  字符（"CPU 13%"），MEM/VRAM 占 11 字符（"MEM 35.6/63G"），auto-width 下
 *  宽度差近 1 倍。固定下界 96px (够 "VRAM 80.0/128G" 之类最长情况)，label 左
 *  value 右两端对齐，bg 填充自然居于中间。 */
function Pill({ label, value, pct, tooltip }: PillProps) {
  const tone = toneClasses(pct)
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <div
      className="relative flex items-center justify-between gap-1.5 h-8 min-w-[96px] px-2 rounded-md border border-dim bg-surface overflow-hidden shrink-0"
      title={tooltip}
    >
      <div
        aria-hidden
        className={`absolute inset-y-0 left-0 ${tone.bg} transition-[width] duration-500 ease-out`}
        style={{ width: `${clamped}%` }}
      />
      <span className="relative z-10 text-2xs uppercase tracking-wider text-fg-tertiary">{label}</span>
      <span className={`relative z-10 font-mono text-xs tabular-nums ${tone.text}`}>{value}</span>
    </div>
  )
}

export default function SystemStats() {
  const [stats, setStats] = useState<SystemStatsData | null>(null)

  // mount 时拉一次冷启动 (避免空白等 2.5s 首个 SSE 事件)，之后纯靠后端
  // sampler 通过 SSE 推送。SSE 重连时 onOpen 也补一次冷启动，防漏。
  useEffect(() => {
    let cancelled = false
    api.systemStats().then((s) => {
      if (!cancelled) setStats(s)
    }).catch(() => {/* 首次失败：等 SSE 第一帧就行 */})
    return () => { cancelled = true }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type !== 'system_stats_updated') return
      const payload = evt.payload as SystemStatsData | undefined
      if (payload) setStats(payload)
    },
    {
      onOpen: () => {
        // SSE 重连：补一次冷启动；服务端 sampler 仍在跑，下次 tick 会自然推
        // 上来，但这一次显式 GET 让 UI 立刻刷新
        api.systemStats().then((s) => setStats(s)).catch(() => {})
      },
    },
  )

  if (!stats) return null

  // 上游在这里算过一个 gpu0（`gpu.find(g => g.active) ?? gpu[0]`），给「pill 只显示
  // 一张卡」的旧 UI 用。本分支的 pill 已改成全卡汇总 + tooltip 逐卡明细，没有
  // 「选一张显示」这回事，所以不需要 gpu0。
  //
  // 但 active 这个信息本身有用：多卡机器上「哪张卡在跑训练」不看不出来（torch 与
  // NVML 编号不同构，上游 #491）。所以改成在逐卡明细里给那张打标记，见下面
  // activeMark —— 信息保留，且不必牺牲汇总视图。
  const ramPct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb) * 100 : 0

  // ── 多卡汇总 ────────────────────────────────────────────────────────
  // pill 显示全卡合计，逐卡明细进 tooltip（原实现只显示 gpu[0] + "(+N more)"，
  // 双卡机器上等于一半的显存看不见）。
  //
  // 汇总口径按量的性质分开，不能一律取和或一律取平均：
  // - **显存**：物理量，可加 —— 合计已用 / 合计总量。
  // - **利用率**：百分比，相加无意义（两卡满载会得到 200%）—— 取算术平均。
  //   刻意不按显存或 SM 数加权：topbar 是粗粒度概览，加权在同型号多卡上与
  //   平均等价，混插不同型号时反而更难解释。
  const gpus = stats.gpu ?? []
  const hasGpu = gpus.length > 0
  const multi = gpus.length > 1

  const vramUsed = gpus.reduce((s, g) => s + g.vram_used_gb, 0)
  const vramTotal = gpus.reduce((s, g) => s + g.vram_total_gb, 0)
  const vramPct = vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0

  // 利用率可能整体缺失（DCU 上 smi 解析不到时为 null），只对有读数的卡求平均。
  // 一张都没有 → null，整个 GPU pill 隐藏（0% 是合法读数，不能拿来兜底缺失值）。
  const utilValues = gpus.map((g) => g.util_pct).filter((u): u is number => u != null)
  const utilAvg = utilValues.length > 0
    ? utilValues.reduce((s, u) => s + u, 0) / utilValues.length
    : null

  // 两个 pill 的 tooltip 各自只列**自己那项**指标的逐卡明细。
  // 早期版本两边共用一份「显存 + 利用率 + 温度」的合并行，于是悬停 GPU 利用率时
  // 满屏是显存数字，要找的利用率被夹在中间 —— tooltip 的意义就是「这个 pill 的
  // 数是怎么来的」，混进无关指标反而更难读。
  //
  // 温度跟着利用率而不是显存：它俩都是「卡当前忙不忙」的即时状态，且温度只有
  // 一个数、并进利用率行不会太长；显存那行本身已有 used/total/百分比三个数。

  /** 逐卡行的「在用」标记。多卡下 torch 与 NVML 编号不同构，光看 #0/#1 分不出
   *  哪张在跑（上游 #491）。单卡不标 —— 只有一张，标了是废话。 */
  const activeMark = (g: { active?: boolean }) =>
    multi && g.active ? ' ←在用' : ''

  /** 逐卡显存：`#0 BW  12.3/64G (19%)`。 */
  const perCardVram = gpus.map((g) => {
    const pct = g.vram_total_gb > 0
      ? ` (${((g.vram_used_gb / g.vram_total_gb) * 100).toFixed(0)}%)`
      : ''
    return `#${g.index} ${g.name}  ${g.vram_used_gb.toFixed(1)}/${Math.round(g.vram_total_gb)}G${pct}${activeMark(g)}`
  }).join('\n')

  /** 逐卡利用率 + 温度：`#0 BW  90% · 70°C`。缺失项省略而非填 0 —— 0% 是合法读数。 */
  const perCardUtil = gpus.map((g) => {
    const util = g.util_pct != null ? `${g.util_pct}%` : '利用率不可用'
    const temp = g.temp_c != null ? ` · ${g.temp_c}°C` : ''
    return `#${g.index} ${g.name}  ${util}${temp}${activeMark(g)}`
  }).join('\n')

  // ── 功耗（只有 DCU 有；NVIDIA 侧后端暂不报 → 整个 pill 隐藏）───────────
  //
  // 加了这个 pill 是因为真机上「功率看起来很低」曾被误判成「卡没跑满」：容器里
  // hy-smi 的 AvgPwr 列报 79W/95W，而 sysfs 同一时刻是 564W/563W（差 6-7 倍）。
  // 后端已改成读 sysfs，这里把真实值显示出来。
  //
  // **hover 显示频率**是这个 pill 的重点，而不是附赠信息：功率低本身说明不了问题
  // （上限是天花板不是目标，真实负载点亮芯片不同部分，均值远低于上限很正常），
  // 真正能判断卡有没有被限制的是频率档位 —— 撞功率墙或温度墙的卡会主动降档，
  // 而「当前 = 最高」就是健康状态。所以 tooltip 里频率和上限一起给，让用户能自己
  // 得出结论而不是盯着瓦数猜。
  const powerValues = gpus.map((g) => g.power_w).filter((p): p is number => p != null)
  // 合计而非平均：功率是物理量，可加（与显存同口径，与利用率相反）。
  const powerTotal = powerValues.length > 0
    ? powerValues.reduce((s, p) => s + p, 0)
    : null
  // 上限只累加**报得出功率的那些卡**，否则比例会被没数据的卡拉低：
  // 一张 580W/1000W + 一张读不到 → 若分母算 2000W 就成了 29%，看着像半空闲。
  const powerCapTotal = gpus
    .filter((g) => g.power_w != null && g.power_cap_w != null)
    .reduce((s, g) => s + (g.power_cap_w as number), 0) || null

  /** 逐卡功率 + 频率：`#0 BW  564W / 1000W · 1500/1500MHz 满频`。 */
  const perCardPower = gpus.map((g) => {
    const pw = g.power_w != null ? `${g.power_w}W` : '功率不可用'
    const cap = g.power_cap_w != null ? ` / ${g.power_cap_w}W` : ''
    let clk = ''
    if (g.sclk_mhz != null) {
      clk = g.sclk_max_mhz != null
        ? ` · ${g.sclk_mhz}/${g.sclk_max_mhz}MHz${g.sclk_mhz >= g.sclk_max_mhz ? ' 满频' : ' 降频'}`
        : ` · ${g.sclk_mhz}MHz`
    }
    return `#${g.index} ${g.name}  ${pw}${cap}${clk}${activeMark(g)}`
  }).join('\n')

  // 单卡时不重复显示汇总行（与逐卡行内容完全一样，纯噪音）
  const vramTooltip = hasGpu
    ? (multi
        ? `显存合计 ${vramUsed.toFixed(1)} / ${Math.round(vramTotal)} GB (${vramPct.toFixed(0)}%) · ${gpus.length} 卡\n${perCardVram}`
        : `显存 ${perCardVram}`)
    : ''
  const utilTooltip = hasGpu
    ? (multi && utilAvg != null
        ? `GPU 利用率均值 ${utilAvg.toFixed(0)}% · ${gpus.length} 卡\n${perCardUtil}`
        : `GPU 利用率 · ${perCardUtil}`)
    : ''
  // 末行那句提示是刻意留的：这个 pill 存在的起因就是有人把低功率读成"没跑满"。
  const powerTooltip = hasGpu && powerTotal != null
    ? (multi
        ? `功耗合计 ${powerTotal}W · ${gpus.length} 卡\n${perCardPower}\n\n满频即未受限；上限是天花板不是目标`
        : `功耗 ${perCardPower}\n\n满频即未受限；上限是天花板不是目标`)
    : ''

  return (
    <div className="hidden md:flex items-center gap-2 shrink-0">
      <Pill
        label="CPU"
        value={`${stats.cpu_pct.toFixed(0)}%`}
        pct={stats.cpu_pct}
        tooltip={`CPU 占用 ${stats.cpu_pct.toFixed(1)}%`}
      />
      <Pill
        label="MEM"
        value={fmtGb(stats.ram_used_gb, stats.ram_total_gb)}
        pct={ramPct}
        tooltip={`内存 ${stats.ram_used_gb.toFixed(1)} / ${stats.ram_total_gb.toFixed(1)} GB (${ramPct.toFixed(0)}%)`}
      />
      {hasGpu && (
        <>
          {/* 利用率可能整体拿不到（DCU 上 smi 解析失败时为 null，见 GpuStats.util_pct）
              —— 整个 pill 隐藏而不是显示 "null%" / "0%"。0% 是合法读数，不能拿来
              兜底缺失值。VRAM pill 不受影响：显存在两个后端上都可靠。
              多卡时 label 带卡数（"GPU×2"），让「这是均值不是单卡」一眼可见。 */}
          {utilAvg != null && (
            <Pill
              label={multi ? `GPU×${gpus.length}` : 'GPU'}
              value={`${utilAvg.toFixed(0)}%`}
              pct={utilAvg}
              tooltip={utilTooltip}
            />
          )}
          {/* 功耗 pill：只有 DCU 报得出（NVIDIA 侧后端未接 → 隐藏，不显示 "0W"）。
              pct = 合计功率 / 合计上限，语义与 toneClasses 的阈值正好对得上：
              逼近上限（≥90%）的卡确实会降频，该警示；真机 580/1000 = 58% 走中性色。
              上限拿不到时传 0（不填色）—— 没有比例可算，不如不画。 */}
          {powerTotal != null && (
            <Pill
              label={multi ? `PWR×${gpus.length}` : 'PWR'}
              value={`${powerTotal}W`}
              pct={powerCapTotal != null && powerCapTotal > 0
                ? (powerTotal / powerCapTotal) * 100
                : 0}
              tooltip={powerTooltip}
            />
          )}
          <Pill
            label={multi ? `VRAM×${gpus.length}` : 'VRAM'}
            value={fmtGb(vramUsed, vramTotal)}
            pct={vramPct}
            tooltip={vramTooltip}
          />
        </>
      )}
    </div>
  )
}
