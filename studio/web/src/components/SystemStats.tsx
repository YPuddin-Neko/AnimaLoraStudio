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

  /** 逐卡一行：`#0 BW  12.3/64G (19%) · 45% · 50°C`。缺失项省略而非填 0。 */
  const perCardLines = gpus.map((g) => {
    const pct = g.vram_total_gb > 0
      ? ` (${((g.vram_used_gb / g.vram_total_gb) * 100).toFixed(0)}%)`
      : ''
    const util = g.util_pct != null ? ` · ${g.util_pct}%` : ''
    const temp = g.temp_c != null ? ` · ${g.temp_c}°C` : ''
    return `#${g.index} ${g.name}  ${g.vram_used_gb.toFixed(1)}/${Math.round(g.vram_total_gb)}G${pct}${util}${temp}`
  }).join('\n')

  // 单卡时不重复显示汇总行（与逐卡行内容完全一样，纯噪音）
  const vramTooltip = hasGpu
    ? (multi
        ? `显存合计 ${vramUsed.toFixed(1)} / ${Math.round(vramTotal)} GB (${vramPct.toFixed(0)}%) · ${gpus.length} 卡\n${perCardLines}`
        : `显存 ${perCardLines}`)
    : ''
  const utilTooltip = hasGpu
    ? (multi && utilAvg != null
        ? `GPU 利用率均值 ${utilAvg.toFixed(0)}% · ${gpus.length} 卡\n${perCardLines}`
        : `GPU 利用率 · ${perCardLines}`)
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
