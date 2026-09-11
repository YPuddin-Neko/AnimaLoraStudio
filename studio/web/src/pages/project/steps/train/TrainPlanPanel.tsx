import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api,
  type BucketDistribution,
  type ConfigData,
  type RegStatus,
  type Task,
  type Version,
} from '../../../../api/client'
import { Tabs } from '../../../../components/SelectionGroup'
import Alert from '../../../../components/Alert'
import Badge from '../../../../components/Badge'
import Button from '../../../../components/Button'
import Card from '../../../../components/Card'
import ConfigYamlPanel from '../../../../components/ConfigYamlPanel'
import { parseFolderMeta } from '../../../../lib/folderMeta'
import { estimateSteps, positiveInt } from '../../../../lib/trainSteps'
import type { SaveStatus } from '../../../../lib/SettingsData'

export function LoadError({ message, onRetry }: { message: string; onRetry: () => void | Promise<void> }) {
  const { t } = useTranslation()
  return (
    <Alert
      tone="danger"
      size="sm"
      action={(
        <Button variant="secondary" size="xs" onClick={() => void onRetry()}>
          {t('common.retry')}
        </Button>
      )}
    >
      {message}
    </Alert>
  )
}

function ActiveTrainTaskCard({ task }: { task: Task }) {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const isTraining = (task.task_type ?? 'train') === 'train'
  const timestamp = task.scheduled_at ?? task.started_at ?? task.created_at
  const timeLabel = task.scheduled_at
    ? t('train.activeTask.scheduledAt')
    : task.started_at
      ? t('train.activeTask.startedAt')
      : t('train.activeTask.createdAt')
  const formattedTime = new Date(timestamp * 1000).toLocaleString(
    i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-US',
    { hour12: false },
  )
  return (
    <Card as="section" padding="md" className="shrink-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="caption mb-1">{t(isTraining ? 'train.activeTask.title' : 'train.activeTask.blockingTitle')}</div>
          <div className="font-semibold text-fg-primary truncate">#{task.id} · {task.name}</div>
        </div>
        <Badge tone="accent" active={task.status === 'running'}>
          {t(`status.${task.status}`)}
        </Badge>
      </div>
      <p className="text-xs text-fg-secondary mt-2 mb-0">
        {timeLabel} · {formattedTime}
      </p>
      <p className="text-xs text-fg-secondary mt-1 mb-0">
        {t(isTraining ? 'train.activeTask.snapshotHint' : 'train.activeTask.blockedByResourceHint', { id: task.id })}
      </p>
      <Button
        variant="secondary"
        size="sm"
        className="mt-3 w-full"
        onClick={() => navigate(`/queue/${task.id}`)}
      >
        {t('train.activeTask.open')}
      </Button>
    </Card>
  )
}

export function TrainRunSummary({ config, plan }: { config: ConfigData | null; plan: TrainDatasetPlan }) {
  const { t } = useTranslation()
  if (!config) return null
  // Name the configured file, never the family's current global default.
  const path = String(config.transformer_path ?? '')
  const filename = path.split(/[\\/]/).pop() || '—'
  const baseModel = filename.replace(/\.safetensors$/i, '')
    .replace(/^anima-base-v(\d+(?:\.\d+)*)$/i, 'Anima base $1')
  const rows = [
    [t('train.summary.baseModel'), baseModel, path],
    [t('train.summary.loraType'), String(config.lora_type || '—')],
    [t('train.summary.prefix'), String(config.output_name || '—')],
    [t('train.summary.epochs'), String(config.epochs ?? '—')],
    [t('train.summary.steps'), plan.finalTotal === null ? '—' : `≈ ${plan.finalTotal}`],
  ]

  return (
    <div role="group" className="train-run-summary" aria-label={t('train.summary.label')}>
      <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-field gap-y-related text-sm">
        {rows.map(([label, value, title]) => (
          <div key={label} className="contents">
            <dt className="text-fg-secondary">{label}</dt>
            <dd className="m-0 break-words font-mono text-fg-primary" title={title || value}>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

export default function TrainPlanPanel({
  projectId,
  activeVersion,
  config,
  onEnableMaskedLoss,
  reg,
  regError,
  onRetryReg,
  activeTask,
  taskLoading,
  saveStatus,
  plan,
  previewOpen,
  previewTab,
  onPreviewTabChange,
}: {
  projectId: number
  activeVersion: Version | null
  config: ConfigData | null
  onEnableMaskedLoss: () => void
  reg: RegStatus | null
  regError: string | null
  onRetryReg: () => void | Promise<void>
  activeTask: Task | null
  taskLoading: boolean
  saveStatus: SaveStatus
  plan: TrainDatasetPlan
  previewOpen: boolean
  previewTab: 'stats' | 'config'
  onPreviewTabChange: (tab: 'stats' | 'config') => void
}) {
  const { t } = useTranslation()

  return (
    <aside id="train-preview-panel" hidden={!previewOpen} className={`train-preview min-w-0 min-h-0 flex-col gap-field ${previewOpen ? 'flex' : 'hidden'}`} aria-label={t('train.previewLabel')}>
      {taskLoading && (
        <Alert tone="info" size="sm" className="shrink-0">{t('train.activeTask.checking')}</Alert>
      )}
      {activeTask && <ActiveTrainTaskCard task={activeTask} />}
      {activeTask && <p className="m-0 text-sm font-medium text-fg-secondary">{t('train.nextRunDraft')}</p>}
      <Tabs
        value={previewTab}
        onChange={onPreviewTabChange}
        appearance="segmented"
        layout="content"
        items={[
          { value: 'stats', label: t('train.previewTabStats'), controls: 'train-preview-stats-panel' },
          { value: 'config', label: t('train.previewTabYaml'), controls: 'train-preview-config-panel' },
        ]}
        ariaLabel={t('train.previewLabel')}
        idPrefix="train-preview"
        className="shrink-0 self-end"
      />
      <div
        id="train-preview-stats-panel"
        role="tabpanel"
        aria-labelledby="train-preview-stats"
        tabIndex={0}
        hidden={previewTab !== 'stats'}
        className="train-stats-tab min-h-0 min-w-0 flex-1 overflow-y-auto"
      >
        {config ? (
          <div className="space-y-field">
            <TrainRunSummary config={config} plan={plan} />
            {regError && <LoadError message={t('train.loadRegFailed', { error: regError })} onRetry={onRetryReg} />}
            <DatasetStatsPanel
              projectId={projectId}
              activeVersion={activeVersion}
              reg={reg}
              config={config}
              plan={plan}
              onEnableMaskedLoss={onEnableMaskedLoss}
            />
          </div>
        ) : <p className="m-0 text-sm text-fg-secondary">{t('train.noConfigHint')}</p>}
      </div>
      <div
        id="train-preview-config-panel"
        role="tabpanel"
        aria-labelledby="train-preview-config"
        tabIndex={0}
        hidden={previewTab !== 'config'}
        className={`train-yaml-tab min-h-0 min-w-0 flex-1 flex-col ${previewTab === 'config' ? 'flex' : 'hidden'}`}
      >
        {previewTab === 'config' && (config ? (
          <ConfigYamlPanel
            config={config}
            fileLabel="config.yaml"
            hint={saveStatus.state === 'saving'
              ? t('train.unsavedChanges')
              : saveStatus.state === 'error'
                ? t('train.unsavedSaveError')
                : undefined}
            className="flex flex-1 min-h-0 flex-col"
          />
        ) : <p className="m-0 text-sm text-fg-secondary">{t('train.noConfigHint')}</p>)}
      </div>
    </aside>
  )
}

/** config.resolution 归一成 number[]（schema 是 list[int]，旧 config / 标量也兜底）。 */
function configResolutions(config: ConfigData | null): number[] {
  const r = config?.resolution as unknown
  if (Array.isArray(r)) return r.length ? (r as number[]) : [1024]
  if (typeof r === 'number') return [r]
  return [1024]
}

/** 文件夹有效样本数 = repeat × 图数 × 分辨率档数（px 文件夹固定 1 档；否则跟 config 列表）。 */
function folderEffective(name: string, imageCount: number, resoCount: number): number {
  const { reso, repeat } = parseFolderMeta(name)
  return repeat * imageCount * (reso ? 1 : resoCount)
}

/** reg.files 形如 `5_concept/12345.png` —— 按首段文件夹聚合计数。 */
function aggregateRegFolders(files: string[]): Array<{ name: string; image_count: number }> {
  const m = new Map<string, number>()
  for (const f of files) {
    const idx = f.indexOf('/')
    if (idx < 0) continue
    const folder = f.slice(0, idx)
    m.set(folder, (m.get(folder) ?? 0) + 1)
  }
  return Array.from(m.entries())
    .map(([name, image_count]) => ({ name, image_count }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

/** 训练集 + 正则集分布右栏面板。
 *
 * 显示每个 repeat 文件夹（Kohya 风格 N_label）的 raw 图数 + 有效图数（repeat × imgs），
 * train / reg 分两块汇总，最后给出有效图数总和——这是 anima_train 单 epoch 的实际样本数。
 */
export function useTrainDatasetPlan({
  projectId,
  activeVersion,
  reg,
  config,
}: {
  projectId: number
  activeVersion: Version | null
  reg: RegStatus | null
  config: ConfigData | null
}) {
  const trainFolders = activeVersion?.stats?.train_folders ?? []
  const regFolders = useMemo(
    () => (reg && reg.exists ? aggregateRegFolders(reg.files) : []),
    [reg]
  )

  const resoCount = configResolutions(config).length
  const trainEffective = trainFolders.reduce(
    (s, f) => s + folderEffective(f.name, f.image_count, resoCount),
    0,
  )
  const regEffective = regFolders.reduce(
    (s, f) => s + folderEffective(f.name, f.image_count, resoCount),
    0,
  )
  const totalEffective = trainEffective + regEffective

  // 桶分布 + NaViT 打包预估（后端用真 BucketManager / NavitPackBatchSampler 算）。
  // fetch 在本层做（而非 BucketPreview 内部）：navit 模式下步数公式也吃这份数据。
  const vid = activeVersion?.id ?? 0
  const navitOn = config?.navit_packing === true
  const [dist, setDist] = useState<BucketDistribution | null>(null)
  const [distLoading, setDistLoading] = useState(true)
  const [distError, setDistError] = useState<string | null>(null)
  const [distRetryKey, setDistRetryKey] = useState(0)
  const distSig = JSON.stringify([
    config?.resolution,
    config?.aspect_ratio_limit,
    // 文件夹名单（含 px 前缀 / repeat / 图数）—— 改名加 px 也要触发重取，不能只看总数
    activeVersion?.stats?.train_folders,
    // navit 打包预估的输入 —— 任何一项变了包数都可能变
    config?.navit_packing,
    config?.navit_native_resolution,
    config?.navit_token_budget,
    config?.navit_max_images_per_pack,
    config?.navit_pack_strategy,
    config?.navit_pack_ffd_window,
    config?.navit_drop_last,
    config?.navit_native_over_budget,
    config?.seed,
    reg?.exists,
    reg && reg.exists ? reg.files.length : 0,
  ])
  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    setDistLoading(true)
    setDistError(null)
    const timer = window.setTimeout(() => {
      api.getBucketDistribution(projectId, vid)
        .then((d) => {
          if (!cancelled) {
            setDist(d)
            setDistLoading(false)
          }
        })
        .catch((error) => {
          if (!cancelled) {
            setDist(null)
            setDistError(String(error))
            setDistLoading(false)
          }
        })
    }, 300)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [projectId, vid, distSig, distRetryKey])

  // 单 epoch 优化器步数估算（与 sd-scripts max_train_steps 同语义）。算术在
  // lib/trainSteps.ts —— 它镜像后端 optimizer.py:62 的**分段取整**，必须有测试，
  // 而埋在组件里的表达式没法测（多卡漏除卡数的 bug 就是这么静默存在的）。
  //
  // 这里原本是 `ceil(totalEffective / (bs * ga))`，漏了 ddp_num_processes：
  // batch 在**卡之间**还要再分一次（BucketBatchSampler / NavitPackBatchSampler 都走
  // _ddp_shard 按 batch 粒度分片，各 rank 只拿 1/N），而后端 steps_per_epoch 取
  // len(ctx.dataloader)，那个长度已经是每 rank 的量。真机 2 卡下面板报 71 而后端
  // 实际 36 —— 整整差一个卡数，用户照这个数定 epochs 和调度长度。
  // schema 字段：batch_size / grad_accum / ddp_num_processes / epochs / max_steps。
  const bs = positiveInt(config?.batch_size, 1)
  const ga = positiveInt(config?.grad_accum, 1)
  const ddpProcs = positiveInt(config?.ddp_num_processes, 1)
  const epochs = positiveInt(config?.epochs, 0)
  const maxSteps = positiveInt(config?.max_steps, 0)
  const navitEst = navitOn ? (dist?.navit ?? null) : null
  const { stepsPerEpoch, naturalTotal, finalTotal, maxStepsTruncates } = estimateSteps({
    totalEffective,
    batchSize: bs,
    gradAccum: ga,
    ddpProcs,
    epochs,
    maxSteps,
    navitOn,
    // 后端打包模拟给的是**全局**包数（versions.py 那个函数不知道卡数），分片同样
    // 走 _ddp_shard（dataset.py），除卡数在 estimateSteps 里做。
    packsPerEpoch: navitEst ? navitEst.packs_per_epoch : null,
  })
  // navit 下有效样本以真打包模拟为准（native 收拢多分辨率 fan-out、含 reg），
  // 前端 folderEffective 的 resoCount fan-out 在该模式下会虚算
  const shownEffective = navitEst && navitEst.samples > 0
    ? navitEst.samples
    : totalEffective

  return {
    trainFolders, regFolders, trainEffective, regEffective, resoCount,
    shownEffective, navitOn, navitEst, bs, ga, ddpProcs, epochs, naturalTotal, finalTotal,
    stepsPerEpoch, maxSteps, maxStepsTruncates,
    distLoading, distError, dist,
    retryDistribution: () => setDistRetryKey((value) => value + 1),
  }
}

export type TrainDatasetPlan = ReturnType<typeof useTrainDatasetPlan>

function DatasetStatsPanel({
  projectId,
  activeVersion,
  reg,
  config,
  plan,
  onEnableMaskedLoss,
}: {
  projectId: number
  activeVersion: Version | null
  reg: RegStatus | null
  config: ConfigData | null
  plan: TrainDatasetPlan
  onEnableMaskedLoss: () => void
}) {
  const { t } = useTranslation()
  const {
    trainFolders, regFolders, trainEffective, regEffective, resoCount,
    shownEffective, navitOn, navitEst, bs, ga, ddpProcs, epochs, naturalTotal, finalTotal,
    stepsPerEpoch, maxSteps, maxStepsTruncates, distLoading, distError, dist, retryDistribution,
  } = plan
  return (
    <div className="flex flex-col gap-3 min-w-0">
      <Card as="section" padding="md" aria-labelledby="train-workload-title">
        <h2 id="train-workload-title" className="type-panel-title mb-field">{t('train.statsTitle')}</h2>

        <FolderSection
          title="train/"
          folders={trainFolders}
          effective={trainEffective}
          resoCount={resoCount}
          empty={t('train.noTrainImages')}
        />

        <div className="h-2" />

        <FolderSection
          title="reg/"
          folders={regFolders}
          effective={regEffective}
          resoCount={resoCount}
          empty={reg && !reg.exists ? t('train.regNotBuilt') : t('train.noRegImages')}
        />

        {/* 总计 + 步数估算（不含 AR bucketing 误差；navit 走真打包模拟） */}
        <div className="mt-2.5 pt-2 border-t border-subtle flex flex-col gap-1 text-xs">
          <Row label={t('train.effectiveSamples')} value={String(shownEffective)} bold />
          {navitOn ? (
            navitEst && navitEst.packs_per_epoch > 0 ? (
              <>
                <Row
                  label={t('train.navitPackLine')}
                  value={`≈ ${navitEst.packs_per_epoch}`}
                  dim
                />
                {/* ga=1 单卡时步数就等于包数，这行是重复信息所以藏掉；但只要
                    ga>1 **或** 多卡，步数就已经不等于上面那个包数了，必须显示。 */}
                {(ga > 1 || ddpProcs > 1) && stepsPerEpoch !== null && (
                  <Row
                    label={
                      ddpProcs > 1
                        ? t('train.navitGaCardsLine', { ga, cards: ddpProcs })
                        : t('train.navitGaLine', { ga })
                    }
                    value={`≈ ${stepsPerEpoch} steps/epoch`}
                    dim
                  />
                )}
              </>
            ) : (
              <Row label={t('train.navitEstimating')} value="…" dim />
            )
          ) : (
            stepsPerEpoch !== null && (
              <Row
                label={
                  ddpProcs > 1
                    ? t('train.batchStepFormulaCards', { batch: bs, ga, cards: ddpProcs })
                    : t('train.batchStepFormula', { batch: bs, ga })
                }
                value={`≈ ${stepsPerEpoch} steps/epoch`}
                dim
              />
            )
          )}
          {naturalTotal !== null && (
            <Row
              label={`× epochs (${epochs})`}
              value={`≈ ${naturalTotal} steps`}
              dim
            />
          )}
          {finalTotal !== null && (
            <Row
              label={maxStepsTruncates ? t('train.maxStepsLabel', { n: maxSteps }) : t('train.totalSteps')}
              value={`≈ ${finalTotal}`}
              bold
            />
          )}
        </div>
      </Card>

      {distLoading && (
        <Alert tone="info" size="sm">
          {dist ? t('train.refreshingBucket') : t('train.loadingBucket')}
        </Alert>
      )}
      {distError && (
        <LoadError
          message={t('train.loadBucketFailed', { error: distError })}
          onRetry={retryDistribution}
        />
      )}
      <BucketPreview dist={dist} />

      <MaskedLossHint
        projectId={projectId}
        vid={activeVersion?.id ?? 0}
        maskedLoss={config?.masked_loss === true}
        blocked={config?.leap_enabled === true || config?.navit_packing === true}
        onEnable={onEnableMaskedLoss}
      />
    </div>
  )
}

/** 训练集有 mask 但 masked_loss 关闭时的提示；允许在互斥规则许可时一键启用。 */
function MaskedLossHint({
  projectId, vid, maskedLoss, blocked, onEnable,
}: {
  projectId: number
  vid: number
  maskedLoss: boolean
  blocked: boolean
  onEnable: () => void
}) {
  const { t } = useTranslation()
  const [maskCount, setMaskCount] = useState(0)
  const [maskError, setMaskError] = useState<string | null>(null)
  const [maskRetryKey, setMaskRetryKey] = useState(0)

  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    setMaskError(null)
    api.listCropWorkspaceTrain(projectId, vid)
      .then((r) => {
        if (!cancelled) {
          setMaskCount(r.images.filter((im) => im.mask_mtime != null).length)
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setMaskCount(0)
          setMaskError(String(error))
        }
      })
    return () => { cancelled = true }
  }, [projectId, vid, maskRetryKey])

  if (maskError) {
    return (
      <LoadError
        message={t('train.loadMaskFailed', { error: maskError })}
        onRetry={() => setMaskRetryKey((value) => value + 1)}
      />
    )
  }

  if (maskCount === 0 || maskedLoss) return null
  return (
    <Alert
      tone="warning"
      size="sm"
      action={(
        <Button variant="primary" size="sm" disabled={blocked} onClick={onEnable}>
          {t('train.enableMaskedLoss')}
        </Button>
      )}
    >
      {t('train.maskedLossHint', { n: maskCount })}
      {blocked && ` ${t('train.maskedLossEnableBlocked')}`}
    </Alert>
  )
}

/** 训练集实际分布面板（数据由 DatasetStatsPanel 统一 fetch）。
 *  - 常规 / navit 非 native：ARB 桶分布（后端用真 BucketManager 算）。trainer 用
 *    drop_last=False —— 桶不满只出短 batch、不丢图，所以这里不做丢图警告。
 *  - navit-native：训练绕过 ARB 桶（每图原生尺寸 floor-16px），显示原生尺寸
 *    直方图（真打包模拟返回），不再展示实际不存在的桶。 */
function BucketPreview({ dist }: { dist: BucketDistribution | null }) {
  const { t } = useTranslation()
  if (!dist) return null

  if (dist.navit?.native) {
    const sizes = dist.navit.sizes
    if (sizes.length === 0) return null
    const top = sizes.slice(0, 10)
    const rest = sizes.slice(10)
    const restCount = rest.reduce((s, x) => s + x.count, 0)
    return (
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h2 className="type-panel-title mb-field">{t('train.navitDistTitle')}</h2>
        <div className="flex flex-col gap-0.5">
          {top.map((b) => (
            <div
              key={`${b.w}x${b.h}`}
              className="flex items-baseline gap-1.5 text-xs font-mono pl-1"
            >
              <span className="text-fg-secondary">{b.w}×{b.h}</span>
              <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
              <span className="text-fg-primary">{b.count}</span>
            </div>
          ))}
          {rest.length > 0 && (
            <div className="text-xs font-mono text-fg-secondary pl-1">
              {t('train.navitDistMore', { kinds: rest.length, n: restCount })}
            </div>
          )}
        </div>
        <div className="text-xs text-fg-secondary mt-2">
          {t('train.navitDistHint')}
          {dist.navit.downscaled > 0 && (
            <> {t('train.navitDistDownscaled', { n: dist.navit.downscaled })}</>
          )}
        </div>
      </div>
    )
  }

  if (dist.groups.length === 0) return null

  return (
    <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
      <h2 className="type-panel-title mb-field">
        {t(dist.navit ? 'train.navitBucketTitle' : 'train.bucketDistTitle')}
      </h2>
      <div className="flex flex-col gap-2">
        {dist.groups.map((g) => (
          <div key={g.reso}>
            <div className="text-xs font-mono text-fg-secondary mb-1">{g.reso}px</div>
            <div className="flex flex-col gap-0.5">
              {g.buckets.map((b) => (
                <div
                  key={`${b.w}x${b.h}`}
                  className="flex items-baseline gap-1.5 text-xs font-mono pl-1"
                >
                  <span className="text-fg-secondary">{b.w}×{b.h}</span>
                  <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                  <span className="text-fg-primary">{b.count}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="text-xs text-fg-secondary mt-2">
        {dist.navit ? t('train.navitBucketHint') : t('train.bucketDistHint')}
      </div>
    </div>
  )
}

function FolderSection({
  title,
  folders,
  effective,
  resoCount,
  empty,
}: {
  title: string
  folders: Array<{ name: string; image_count: number }>
  effective: number
  resoCount: number
  empty: string
}) {
  const { t } = useTranslation()
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs mb-1">
        <span className="font-mono text-fg-secondary font-medium">{title}</span>
        {folders.length > 0 && (
          <span className="font-mono text-fg-secondary">∑ {effective}</span>
        )}
      </div>
      {folders.length === 0 ? (
        <div className="text-xs text-fg-secondary pl-1">{empty}</div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {folders.map((f) => {
            const { reso, repeat, label } = parseFolderMeta(f.name)
            const folderResos = reso ? 1 : resoCount
            const eff = repeat * f.image_count * folderResos
            const resoTag = reso ? `${reso}px` : null
            return (
              <div
                key={f.name}
                className="flex items-baseline gap-1.5 text-xs font-mono text-fg-secondary pl-1"
                title={folderResos > 1
                  ? t('train.folderTipReso', { name: f.name, repeat, imgs: f.image_count, resos: folderResos, total: eff })
                  : t('train.folderTip', { name: f.name, repeat, imgs: f.image_count, total: eff })}
              >
                <span className="text-fg-secondary">{label}</span>
                {resoTag && <span className="text-xs text-accent-strong">{resoTag}</span>}
                <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                <span>
                  <span className="text-accent-strong">{repeat}</span>
                  <span className="text-fg-secondary"> × </span>
                  <span className="text-fg-primary">{f.image_count}</span>
                  {folderResos > 1 && (
                    <>
                      <span className="text-fg-secondary"> × </span>
                      <span className="text-accent-strong">{folderResos}</span>
                    </>
                  )}
                  <span className="text-fg-secondary"> = </span>
                  <span className="text-fg-primary font-semibold">{eff}</span>
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function Row({
  label,
  value,
  bold,
  dim,
}: {
  label: string
  value: string
  bold?: boolean
  dim?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-fg-secondary">{label}</span>
      <span className={`font-mono ${bold ? 'text-accent-strong font-semibold' : dim ? 'text-fg-secondary font-medium' : 'text-fg-primary font-medium'}`}>
        {value}
      </span>
    </div>
  )
}
