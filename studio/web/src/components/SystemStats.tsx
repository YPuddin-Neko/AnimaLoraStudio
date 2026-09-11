import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
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
  accessibleLabel: string
  value: string
  pct: number
  description: string
}

/** 进度条胶囊 — 整个 pill 背景按占用百分比填色 (>=70% warn, >=90% err)，
 *  高度与 topbar 上其他元素 (搜索 icon 32px) 一致。
 *
 *  `min-w-[96px]` + `justify-between` 让 4 个 pill 视觉等宽：CPU/GPU 只占 3-4
 *  字符（"CPU 13%"），MEM/VRAM 占 11 字符（"MEM 35.6/63G"），auto-width 下
 *  宽度差近 1 倍。固定下界 96px (够 "VRAM 80.0/128G" 之类最长情况)，label 左
 *  value 右两端对齐，bg 填充自然居于中间。 */
function Pill({ label, accessibleLabel, value, pct, description }: PillProps) {
  const tone = toneClasses(pct)
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <div
      role="meter"
      aria-label={accessibleLabel}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={clamped}
      aria-valuetext={description}
      className="relative flex items-center justify-between gap-1.5 h-8 min-w-[96px] px-2 rounded-md border border-dim bg-surface overflow-hidden shrink-0"
      title={description}
    >
      <div
        aria-hidden="true"
        className={`absolute inset-y-0 left-0 ${tone.bg} transition-[width] duration-500 ease-out motion-reduce:transition-none`}
        style={{ width: `${clamped}%` }}
      />
      <span className="relative z-10 text-2xs uppercase tracking-wider text-fg-tertiary">{label}</span>
      <span className={`relative z-10 font-mono text-xs tabular-nums ${tone.text}`}>{value}</span>
    </div>
  )
}

export default function SystemStats() {
  const { t } = useTranslation()
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

  // ── 多卡汇总 ────────────────────────────────────────────────────────
  //
  // 与上游的差别：上游显示 `active` 那一张 + 「(+N more)」，本分支显示**全卡合计**、
  // 逐卡明细进 description。双卡 DCU 上「只显示一张」等于一半显存看不见，而多卡
  // 训练时两张卡的占用都要盯。
  //
  // 也因此不用上游的 gpu0/active 选择：DDP 下每个 rank 各占一张，"在用" 不是单张
  // （topbar 由 server 进程报，current_device() 只反映 server 自己那张，与训练子
  // 进程无关）。
  //
  // 汇总口径按量的性质分开，不能一律取和或一律取平均：
  // - **显存 / 功率**：物理量，可加 —— 合计已用 / 合计总量。
  // - **利用率**：百分比，相加无意义（两卡满载会得到 200%）—— 取算术平均。
  //   刻意不按显存或 SM 数加权：topbar 是粗粒度概览，加权在同型号多卡上与
  //   平均等价，混插不同型号时反而更难解释。
  const gpus = stats.gpu ?? []
  const hasGpu = gpus.length > 0
  const multi = gpus.length > 1

  const ramPct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb) * 100 : 0

  const vramUsed = gpus.reduce((s, g) => s + g.vram_used_gb, 0)
  const vramTotal = gpus.reduce((s, g) => s + g.vram_total_gb, 0)
  const vramPct = vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0

  // 利用率可能整体缺失（DCU 上 smi 解析不到时为 null），只对有读数的卡求平均。
  // 一张都没有 → null，整个 GPU pill 隐藏（0% 是合法读数，不能拿来兜底缺失值）。
  //
  // 上游这版直接 `${gpu0.util_pct}%` / `pct={gpu0.util_pct}` 不判空 —— 在 DCU 上
  // 会渲染出 "null%"、`pct` 变 NaN（宽度 `NaN%` 是无效 CSS）。判空必须保留。
  const utilValues = gpus.map((g) => g.util_pct).filter((u): u is number => u != null)
  const utilAvg = utilValues.length > 0
    ? utilValues.reduce((s, u) => s + u, 0) / utilValues.length
    : null

  // 三个 pill 的 description 各自只列**自己那项**指标的逐卡明细。
  // 早期版本共用一份「显存 + 利用率 + 温度」的合并行，于是悬停 GPU 利用率时
  // 满屏是显存数字，要找的利用率被夹在中间 —— description 的意义就是「这个 pill 的
  // 数是怎么来的」，混进无关指标反而更难读。
  //
  // 温度跟着利用率而不是显存：它俩都是「卡当前忙不忙」的即时状态，且温度只有
  // 一个数、并进利用率行不会太长；显存那行本身已有 used/total/百分比三个数。

  /** 逐卡显存：`#0 BW  12.3/64G (19%)`。 */
  const perCardVram = gpus.map((g) => {
    const pct = g.vram_total_gb > 0
      ? ` (${((g.vram_used_gb / g.vram_total_gb) * 100).toFixed(0)}%)`
      : ''
    return `#${g.index} ${g.name}  ${g.vram_used_gb.toFixed(1)}/${Math.round(g.vram_total_gb)}G${pct}`
  }).join('\n')

  /** 逐卡利用率 + 温度：`#0 BW  90% · 70°C`。缺失项写明不可用而非填 0。 */
  const perCardUtil = gpus.map((g) => {
    const util = g.util_pct != null
      ? `${g.util_pct}%`
      : t('topbar.systemStats.utilUnavailable')
    const temp = g.temp_c != null ? ` · ${g.temp_c}°C` : ''
    return `#${g.index} ${g.name}  ${util}${temp}`
  }).join('\n')

  // ── 功耗（只有 DCU 有；NVIDIA 侧后端暂不报 → 整个 pill 隐藏）───────────
  //
  // 加了这个 pill 是因为真机上「功率看起来很低」曾被误判成「卡没跑满」：容器里
  // hy-smi 的 AvgPwr 列报 79W/95W，而 sysfs 同一时刻是 564W/563W（差 6-7 倍）。
  // 后端已改成读 sysfs，这里把真实值显示出来。
  //
  // description 里连频率一起给（当前 / 最高）：判断卡有没有被限制靠的是频率档位，
  // 撞功率墙或温度墙的卡会主动降档。只摆数据，不在 UI 里写结论。
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

  /** 逐卡功率 + 频率：`#0 BW  564W / 1000W · 1500/1500MHz`。
   *
   *  只给数，不加「满频 / 降频」的判词：`当前/最高` 两个数并排本身就说明了状态，
   *  再补一个词是同义重复。 */
  const perCardPower = gpus.map((g) => {
    const pw = g.power_w != null
      ? `${g.power_w}W`
      : t('topbar.systemStats.powerUnavailable')
    const cap = g.power_cap_w != null ? ` / ${g.power_cap_w}W` : ''
    let clk = ''
    if (g.sclk_mhz != null) {
      clk = g.sclk_max_mhz != null
        ? ` · ${g.sclk_mhz}/${g.sclk_max_mhz}MHz`
        : ` · ${g.sclk_mhz}MHz`
    }
    return `#${g.index} ${g.name}  ${pw}${cap}${clk}`
  }).join('\n')

  // 单卡时不重复显示汇总行（与逐卡行内容完全一样，纯噪音）
  const vramDescription = multi
    ? t('topbar.systemStats.vramMulti', {
        used: vramUsed.toFixed(1),
        total: Math.round(vramTotal),
        percent: vramPct.toFixed(0),
        count: gpus.length,
        perCard: perCardVram,
      })
    : t('topbar.systemStats.vramSingle', { perCard: perCardVram })
  const utilDescription = multi && utilAvg != null
    ? t('topbar.systemStats.gpuMulti', {
        percent: utilAvg.toFixed(0),
        count: gpus.length,
        perCard: perCardUtil,
      })
    : t('topbar.systemStats.gpuSingle', { perCard: perCardUtil })
  const powerDescription = multi
    ? t('topbar.systemStats.powerMulti', {
        watts: powerTotal ?? 0,
        count: gpus.length,
        perCard: perCardPower,
      })
    : t('topbar.systemStats.powerSingle', { perCard: perCardPower })

  return (
    <div className="ui-app-shell-topbar-stats">
      <Pill
        label="CPU"
        accessibleLabel={t('topbar.systemStats.cpuLabel')}
        value={`${stats.cpu_pct.toFixed(0)}%`}
        pct={stats.cpu_pct}
        description={t('topbar.systemStats.cpu', { percent: stats.cpu_pct.toFixed(1) })}
      />
      <Pill
        label="MEM"
        accessibleLabel={t('topbar.systemStats.memoryLabel')}
        value={fmtGb(stats.ram_used_gb, stats.ram_total_gb)}
        pct={ramPct}
        description={t('topbar.systemStats.memory', {
          used: stats.ram_used_gb.toFixed(1),
          total: stats.ram_total_gb.toFixed(1),
          percent: ramPct.toFixed(0),
        })}
      />
      {hasGpu && (
        <>
          {/* 利用率可能整体拿不到（DCU 上 smi 解析失败时为 null）—— 整个 pill 隐藏
              而不是显示 "null%" / "0%"。0% 是合法读数，不能拿来兜底缺失值。
              多卡时 label 带卡数（"GPU×2"），让「这是均值不是单卡」一眼可见。 */}
          {utilAvg != null && (
            <Pill
              label={multi ? `GPU×${gpus.length}` : 'GPU'}
              accessibleLabel={t('topbar.systemStats.gpuLabel')}
              value={`${utilAvg.toFixed(0)}%`}
              pct={utilAvg}
              description={utilDescription}
            />
          )}
          {/* 功耗 pill：只有 DCU 报得出（NVIDIA 侧后端未接 → 隐藏，不显示 "0W"）。
              pct = 合计功率 / 合计上限，语义与 toneClasses 的阈值正好对得上：
              逼近上限（≥90%）的卡确实会降频，该警示；真机 1127/2000 = 56% 走中性色。
              上限拿不到时传 0（不填色）—— 没有比例可算，不如不画。 */}
          {powerTotal != null && (
            <Pill
              label={multi ? `PWR×${gpus.length}` : 'PWR'}
              accessibleLabel={t('topbar.systemStats.powerLabel')}
              value={`${powerTotal}W`}
              pct={powerCapTotal != null && powerCapTotal > 0
                ? (powerTotal / powerCapTotal) * 100
                : 0}
              description={powerDescription}
            />
          )}
          <Pill
            label={multi ? `VRAM×${gpus.length}` : 'VRAM'}
            accessibleLabel={t('topbar.systemStats.vramLabel')}
            value={fmtGb(vramUsed, vramTotal)}
            pct={vramPct}
            description={vramDescription}
          />
        </>
      )}
    </div>
  )
}
