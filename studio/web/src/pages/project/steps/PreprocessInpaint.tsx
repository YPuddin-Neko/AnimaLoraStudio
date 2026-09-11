import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type HeadMaskProposals,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import ActionGroup from '../../../components/ActionGroup'
import Button from '../../../components/Button'
import Filmstrip from '../../../components/preprocess/Filmstrip'
import AutoHeadMaskPanel from '../../../components/preprocess/AutoHeadMaskPanel'
import InpaintCanvas, {
  renderInpaintedBlob,
  renderMaskBlob,
  type InpaintCanvasHandle,
  type InpaintMode,
  type InpaintStroke,
  type MaskEdit,
} from '../../../components/preprocess/InpaintCanvas'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import { SegmentedControl } from '../../../components/SelectionGroup'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { compareImagePath } from '../../../lib/imageSort'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

type Filter = 'all' | 'pending' | 'edited'

/** 统一编辑历史条目：涂抹与 mask 笔画共用一条时间线。 */
type HistoryEntry =
  | { kind: 'paint'; stroke: InpaintStroke }
  | { kind: 'mask'; edit: MaskEdit }

interface BrushState {
  color: string
  size: number
  hardness: number
}

const DEFAULT_BRUSH: BrushState = { color: '#ffffff', size: 24, hardness: 1 }

function splitRel(name: string): { folder: string; filename: string } {
  const i = name.lastIndexOf('/')
  return {
    folder: i >= 0 ? name.slice(0, i) : '',
    filename: i >= 0 ? name.slice(i + 1) : name,
  }
}

/** 状态模型对齐裁剪页：双数据面双桶（strokesByImage / maskStrokesByImage），
 *  随便切图改动都留在内存，保存按当前模式分发（§9 决策 2）。只有活动图挂
 *  真实 canvas，「保存全部」对非活动图走离屏重放。 */
export default function PreprocessInpaintPage() {
  const { project, activeVersion } = useOutletContext<Ctx>()
  return <InpaintWorkspace key={`${project.id}:${activeVersion?.id ?? 0}`} />
}

function InpaintWorkspace() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  // ────── Workspace data（复用 crop workspace：name + w/h + mtime + mask_mtime）──────
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)

  const refreshWorkspace = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.listCropWorkspaceTrain(project.id, vid)
      if (!mounted.current) return
      setImages([...r.images].sort((a, b) => compareImagePath(a.name, b.name)))
    } catch {
      /* ignore */
    } finally {
      if (mounted.current) setLoading(false)
    }
  }, [project.id, vid])

  useEffect(() => { void refreshWorkspace() }, [refreshWorkspace])

  // ────── Editor state ──────
  // 统一编辑历史：涂抹与 mask 笔画混合入同一时间线 —— 模式只是笔刷，
  // dirty / undo / 保存都跨模式共用，切模式不改变页面状态语义。
  const [mode, setMode] = useState<InpaintMode>('paint')
  // 画笔 / 橡皮跨模式共用：涂抹橡皮擦未保存笔画，遮罩橡皮擦 mask
  const [erase, setErase] = useState(false)
  const [activeName, setActiveName] = useState<string | null>(null)
  const [historyByImage, setHistoryByImage] = useState<Record<string, HistoryEntry[]>>({})
  const [redoByImage, setRedoByImage] = useState<Record<string, HistoryEntry[]>>({})
  const [filter, setFilter] = useState<Filter>('all')
  const [busy, setBusy] = useState(false)
  const [setupOpen, setSetupOpen] = useState(false)
  const [autoBusy, setAutoBusy] = useState(false)

  const [brush, setBrush] = useLocalStorageState<BrushState>(
    'studio:inpaint:brush', DEFAULT_BRUSH,
  )
  const [recentColors, setRecentColors] = useLocalStorageState<string[]>(
    'studio:inpaint:recent_colors', [],
  )

  const canvasRef = useRef<InpaintCanvasHandle | null>(null)

  useEffect(() => {
    if (images.length === 0) return
    if (!activeName || !images.find((im) => im.name === activeName)) {
      setActiveName(images[0].name)
    }
  }, [images, activeName])

  // ────── Derived ──────
  const activeImage = useMemo(
    () => images.find((im) => im.name === activeName) ?? null,
    [images, activeName],
  )
  // useMemo 而不是就地 `?? []`：后者每次渲染都产生新数组引用，会击穿下面两个
  // useMemo（activePaintStrokes / activeMaskStrokes），再让 InpaintCanvas 的
  // `useEffect([strokes])` 每次都跑一遍全画布重绘（clearRect + drawImage 整图 +
  // 合成两个图层）。选中一张还没画过的图时，父组件任何一次重渲染（调笔刷大小、
  // 切模式、图片列表刷新）都会白重绘一次。
  const activeHistory = useMemo(
    () => (activeName ? (historyByImage[activeName] ?? []) : []),
    [activeName, historyByImage],
  )
  const activeRedo = useMemo(
    () => (activeName ? (redoByImage[activeName] ?? []) : []),
    [activeName, redoByImage],
  )
  const activePaintStrokes = useMemo(
    () => activeHistory.filter((h) => h.kind === 'paint').map((h) => h.stroke),
    [activeHistory],
  )
  const activeMaskEdits = useMemo(
    () => activeHistory.filter((h) => h.kind === 'mask').map((h) => h.edit),
    [activeHistory],
  )

  /** dirty 图集合 = 任一数据面有未保存笔画（保存全部 / filter / 计数共用）。 */
  const editedNames = useMemo(
    () => Object.entries(historyByImage)
      .filter(([, h]) => h.length > 0)
      .map(([n]) => n),
    [historyByImage],
  )

  const counts = useMemo(() => {
    const edited = images.filter((im) => (historyByImage[im.name] ?? []).length > 0).length
    return { all: images.length, pending: images.length - edited, edited }
  }, [images, historyByImage])

  const filteredImages = useMemo(() => images.filter((im) => {
    const n = (historyByImage[im.name] ?? []).length
    if (filter === 'pending') return n === 0
    if (filter === 'edited') return n > 0
    return true
  }), [images, filter, historyByImage])

  const incorporateAutoMask = useCallback((result: HeadMaskProposals) => {
    const completed = result.images.filter((item) => (item.status ?? 'done') === 'done')
    const applicable = completed.filter((item) => !item.stale && item.regions.length > 0)
    if (applicable.length > 0) {
      setHistoryByImage((prev) => {
        const next = { ...prev }
        for (const item of applicable) {
          next[item.name] = [
            ...(next[item.name] ?? []),
            {
              kind: 'mask',
              edit: { type: 'auto', regions: item.regions.map((region) => region.mask_region) },
            },
          ]
        }
        return next
      })
      setRedoByImage((prev) => {
        const next = { ...prev }
        for (const item of applicable) next[item.name] = []
        return next
      })
      setMode('mask')
      setErase(false)
    }
    const heads = applicable.reduce((sum, item) => sum + item.regions.length, 0)
    const failed = result.images.filter((item) => item.status === 'failed').length
    const skipped = result.images.filter((item) => item.status === 'skipped').length
    const stale = result.images.filter((item) => item.stale).length
    const summary = applicable.length > 0
      ? t('preprocessInpaint.headMask.completed', { images: applicable.length, heads })
      : t('preprocessInpaint.headMask.completedNone')
    const issues = [
      failed > 0 ? t('preprocessInpaint.headMask.issueFailed', { n: failed }) : '',
      skipped > 0 ? t('preprocessInpaint.headMask.issueSkipped', { n: skipped }) : '',
      stale > 0 ? t('preprocessInpaint.headMask.issueStale', { n: stale }) : '',
    ].filter(Boolean)
    const message = issues.length > 0
      ? t('preprocessInpaint.headMask.completedWithIssues', {
          summary,
          issues: issues.join(t('preprocessInpaint.headMask.issueSeparator')),
        })
      : summary
    toast(message, applicable.length === 0 && (failed > 0 || skipped > 0) ? 'error'
      : issues.length > 0 ? 'info' : 'success')
  }, [t, toast])

  const rawUrl = useCallback((im: CropWorkspaceItem) => {
    const { folder, filename } = splitRel(im.name)
    return api.versionThumbUrl(project.id, vid, 'train', filename, folder, 0)
      + `&_=${im.mtime}`
  }, [project.id, vid])

  const maskBaseUrlFor = useCallback((im: CropWorkspaceItem): string | null => {
    if (im.mask_mtime == null) return null
    return api.maskUrl(project.id, vid, im.name) + `&_=${im.mask_mtime}`
  }, [project.id, vid])

  // ────── Stroke mutations（统一时间线，undo/redo 跨模式）──────
  const pushRecentColor = useCallback((hex: string) => {
    setRecentColors((prev) => [hex, ...prev.filter((c) => c !== hex)].slice(0, 8))
  }, [setRecentColors])

  const pushEntry = useCallback((entry: HistoryEntry) => {
    if (!activeName) return
    setHistoryByImage((prev) => ({
      ...prev,
      [activeName]: [...(prev[activeName] ?? []), entry],
    }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  const onStrokeEnd = useCallback((s: InpaintStroke) => {
    pushEntry({ kind: 'paint', stroke: s })
    pushRecentColor(s.color)
  }, [pushEntry, pushRecentColor])

  const onMaskStrokeEnd = useCallback((s: InpaintStroke) => {
    pushEntry({ kind: 'mask', edit: { type: 'stroke', stroke: s } })
  }, [pushEntry])

  const undo = useCallback(() => {
    if (!activeName) return
    setHistoryByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setRedoByImage((r) => ({
        ...r,
        [activeName]: [...(r[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const redo = useCallback(() => {
    if (!activeName) return
    setRedoByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setHistoryByImage((h) => ({
        ...h,
        [activeName]: [...(h[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const clearActive = useCallback(() => {
    if (!activeName) return
    setHistoryByImage((prev) => ({ ...prev, [activeName]: [] }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== 'z') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  const onPickColor = useCallback((hex: string) => {
    setBrush((prev) => ({ ...prev, color: hex }))
    pushRecentColor(hex)
  }, [setBrush, pushRecentColor])

  // ────── Save（保存 = 该图全部未保存改动，两个数据面一次写完）──────
  /** 保存成功后从历史滤掉对应数据面的 entries（redo 时间线随之作废）。 */
  const clearSavedKind = useCallback((name: string, kind: InpaintMode) => {
    setHistoryByImage((prev) => ({
      ...prev,
      [name]: (prev[name] ?? []).filter((h) => h.kind !== kind),
    }))
    setRedoByImage((prev) => ({ ...prev, [name]: [] }))
  }, [])

  /** 单图两面保存。涂抹先行 —— 产物可能改名（X.jpg→X.png），mask 的 PUT
   *  必须用新 name（旧源文件已删，服务端按 name 校验源图存在）。
   *  返回保存后的 name（无涂抹改动时原样）。 */
  const saveImageBoth = useCallback(async (
    im: CropWorkspaceItem,
    paintStrokes: InpaintStroke[],
    maskEdits: MaskEdit[],
    exporters?: {
      paint: () => Promise<Blob | null>
      mask: () => Promise<{ blob: Blob; coverage: number } | null>
    },
  ): Promise<string> => {
    let name = im.name
    if (paintStrokes.length > 0) {
      const blob = exporters
        ? await exporters.paint()
        : await renderInpaintedBlob(rawUrl(im), im.w, im.h, paintStrokes)
      if (!mounted.current) return name
      if (!blob) throw new Error('canvas not ready')
      const res = await api.saveInpaintTrain(project.id, vid, name, blob)
      if (!mounted.current) return name
      clearSavedKind(im.name, 'paint')
      name = res.name
    }
    if (maskEdits.length > 0) {
      const res = exporters
        ? await exporters.mask()
        : await renderMaskBlob(maskBaseUrlFor(im), im.w, im.h, maskEdits)
      if (!mounted.current) return name
      if (res === null) {
        if (im.mask_mtime != null) await api.deleteMaskTrain(project.id, vid, name)
      } else {
        await api.saveMaskTrain(project.id, vid, name, res.blob)
      }
      if (!mounted.current) return name
      clearSavedKind(im.name, 'mask')
    }
    return name
  }, [project.id, vid, rawUrl, maskBaseUrlFor, clearSavedKind])

  const saveActive = useCallback(async () => {
    if (!activeName || !activeImage) return
    if (activeHistory.length === 0) return
    setBusy(true)
    try {
      // 活动图用挂载中的 canvas 导出（所见即所得），非活动图才走离屏重放
      const newName = await saveImageBoth(
        activeImage, activePaintStrokes, activeMaskEdits,
        {
          paint: () => canvasRef.current?.exportBlob() ?? Promise.resolve(null),
          mask: async () => {
            const r = await canvasRef.current?.exportMaskBlob()
            return r ?? null
          },
        },
      )
      if (!mounted.current) return
      toast(t('preprocessInpaint.toastSaved', { name: newName }), 'success')
      await refreshWorkspace()
      if (!mounted.current) return
      if (newName !== activeName) setActiveName(newName)
      if (mounted.current) void reload()
    } catch (e) {
      if (mounted.current) toast(String(e), 'error')
    } finally {
      if (mounted.current) setBusy(false)
    }
  }, [
    activeName, activeImage, activeHistory.length,
    activePaintStrokes, activeMaskEdits,
    saveImageBoth, refreshWorkspace, reload, toast, t,
  ])

  const saveAll = useCallback(async () => {
    const dirty = editedNames
    if (dirty.length === 0) return
    setBusy(true)
    let ok = 0
    const failed: string[] = []
    try {
      for (const name of dirty) {
        if (!mounted.current) return
        const im = images.find((i) => i.name === name)
        const hist = historyByImage[name] ?? []
        if (!im || hist.length === 0) continue
        try {
          await saveImageBoth(
            im,
            hist.filter((h) => h.kind === 'paint').map((h) => h.stroke),
            hist.filter((h) => h.kind === 'mask').map((h) => h.edit),
          )
          ok++
        } catch {
          failed.push(name)
        }
      }
      if (!mounted.current) return
      toast(
        failed.length > 0
          ? t('preprocessInpaint.toastSavedAllPartial', { ok, failed: failed.length })
          : t('preprocessInpaint.toastSavedAll', { n: ok }),
        failed.length > 0 ? 'error' : 'success',
      )
      await refreshWorkspace()
      if (mounted.current) void reload()
    } finally {
      if (mounted.current) setBusy(false)
    }
  }, [
    editedNames, images, historyByImage, saveImageBoth,
    refreshWorkspace, reload, toast, t,
  ])

  // ────── Render ──────
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  return (
    <StepShell
      title={t('steps.preprocess.title')}
      subtitle={t('preprocessInpaint.subtitle')}
      actions={
        <ActionGroup
          aria-label={t('preprocessInpaint.actionsLabel')}
          secondary={(
            <>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy || autoBusy || images.length === 0}
              title={busy
                ? t('preprocessInpaint.headMask.unavailableSaving')
                : autoBusy
                  ? t('preprocessInpaint.headMask.unavailableRunning')
                  : images.length === 0
                    ? t('preprocessInpaint.headMask.unavailableEmpty')
                    : undefined}
              loading={autoBusy}
              aria-haspopup="dialog"
              onClick={() => setSetupOpen(true)}
            >
              {autoBusy
                ? t('preprocessInpaint.headMask.runningAction')
                : t('preprocessInpaint.headMask.openSetup')}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void saveAll()}
              disabled={busy || autoBusy || editedNames.length === 0}
            >
              {t('preprocessInpaint.saveAll', { n: editedNames.length })}
            </Button>
            </>
          )}
          primary={(
            <Button
              variant="primary"
              size="sm"
              onClick={() => void saveActive()}
              disabled={busy || autoBusy || activeHistory.length === 0}
            >
              {t('preprocessInpaint.saveActive')}
            </Button>
          )}
        />
      }
      belowHeader={<PreprocessToolsBar current="inpaint" projectId={project.id} versionId={vid} />}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
          <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
            {activeImage && (
              <span
                className="min-w-0 truncate text-fg-tertiary text-xs font-mono"
                title={activeImage.name}
              >
                {activeImage.name} · {activeImage.w}×{activeImage.h}
              </span>
            )}
            <span className="flex-1" />
            <Button
              variant="ghost"
              size="sm"
              onClick={undo}
              disabled={!activeName || activeHistory.length === 0}
              title="Ctrl+Z"
            >{t('preprocessInpaint.undo')}</Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={redo}
              disabled={!activeName || activeRedo.length === 0}
              title="Ctrl+Shift+Z"
            >{t('preprocessInpaint.redo')}</Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearActive}
              disabled={!activeName || activeHistory.length === 0}
            >{t('preprocessInpaint.clearActive')}</Button>
          </header>

          <div className="flex-1 min-h-0 overflow-hidden p-3">
            {loading && (
              <p className="text-fg-tertiary text-sm">{t('preprocessInpaint.loading')}</p>
            )}
            {!loading && (
              <div
                className="grid gap-3 h-full min-h-0"
                style={{ gridTemplateColumns: '220px minmax(0, 1fr) 260px' }}
              >
                <Filmstrip
                  items={filteredImages}
                  activeName={activeName}
                  onSelect={setActiveName}
                  thumbUrl={(im) => {
                    const { folder, filename } = splitRel(im.name)
                    return api.versionThumbUrl(
                      project.id, vid, 'train', filename, folder, 256,
                    ) + `&_=${im.mtime}`
                  }}
                  ariaLabel={t('preprocessInpaint.filmstripLabel')}
                  header={(
                    <SegmentedControl
                      items={(['all', 'pending', 'edited'] as const).map((value) => ({
                        value,
                        label: `${t(`preprocessInpaint.filter.${value}`)} ${counts[value]}`,
                      }))}
                      value={filter}
                      onChange={setFilter}
                      ariaLabel={t('preprocessInpaint.filterLabel')}
                      idPrefix="inpaint-image-filter"
                      size="sm"
                      layout="content"
                    />
                  )}
                  itemLabel={(im) => t('preprocessInpaint.imageLabel', { name: im.name })}
                  emptyHint={t(`preprocessInpaint.filmstripEmpty.${filter}`)}
                  renderOverlay={(im) => {
                    const hist = historyByImage[im.name] ?? []
                    const hasPaint = hist.some((h) => h.kind === 'paint')
                    const hasMask = im.mask_mtime != null
                      || hist.some((h) => h.kind === 'mask')
                    if (!hasPaint && !hasMask) return null
                    return (
                      <span className="fs-badge">
                        {hasPaint ? '✎' : ''}{hasMask ? 'M' : ''}
                      </span>
                    )
                  }}
                />

                <div className={`min-w-0 min-h-0 overflow-hidden ${autoBusy || busy ? 'pointer-events-none' : ''}`} aria-busy={autoBusy || busy}>
                  {!activeImage && <p className="text-fg-tertiary text-sm">
                    {t('preprocessInpaint.emptyWorkspace')}{' '}
                    <Link to={`/projects/${project.id}/v/${vid}/preprocess`} className="text-accent hover:underline">{t('preprocessInpaint.goToOverview')}</Link>
                  </p>}
                  {activeImage && <InpaintCanvas
                    key={activeImage.name}
                    ref={canvasRef}
                    imageUrl={rawUrl(activeImage)}
                    imageW={activeImage.w}
                    imageH={activeImage.h}
                    mode={mode}
                    strokes={activePaintStrokes}
                    maskEdits={activeMaskEdits}
                    maskBaseUrl={maskBaseUrlFor(activeImage)}
                    brush={brush}
                    erase={erase}
                    onStrokeEnd={onStrokeEnd}
                    onMaskStrokeEnd={onMaskStrokeEnd}
                    onPickColor={onPickColor}
                  />}
                </div>

                <ToolPanel
                  mode={mode}
                  setMode={setMode}
                  erase={erase}
                  setErase={setErase}
                  brush={brush}
                  setBrush={setBrush}
                  recentColors={recentColors}
                >
                  <AutoHeadMaskPanel
                    projectId={project.id}
                    versionId={vid}
                    activeName={activeName}
                    unsavedCount={editedNames.length}
                    setupOpen={setupOpen}
                    onCloseSetup={() => setSetupOpen(false)}
                    onResults={incorporateAutoMask}
                    onBusyChange={setAutoBusy}
                  />
                </ToolPanel>
              </div>
            )}
          </div>
        </section>
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Tool panel（right side）
// ---------------------------------------------------------------------------

function ToolPanel({
  mode,
  setMode,
  erase,
  setErase,
  brush,
  setBrush,
  recentColors,
  children,
}: {
  mode: InpaintMode
  setMode: (m: InpaintMode) => void
  erase: boolean
  setErase: (v: boolean) => void
  brush: BrushState
  setBrush: (v: BrushState | ((prev: BrushState) => BrushState)) => void
  recentColors: string[]
  children?: React.ReactNode
}) {
  const { t } = useTranslation()
  const [recentOpen, setRecentOpen] = useState(false)
  return (
    <div className="bg-sunken border border-subtle rounded-md flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex flex-col gap-2 p-2.5 flex-1 min-h-0 overflow-y-auto">
        <h3 className="caption">{t('preprocessInpaint.panelTitle')}</h3>
        {/* 模式与工具复用共享分段选择，方向键跟随选择。 */}
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.modeLabel')}</span>
          <SegmentedControl
            items={(['paint', 'mask'] as const).map((value) => ({
              value,
              label: t(`preprocessInpaint.mode.${value}`),
            }))}
            value={mode}
            onChange={setMode}
            ariaLabel={t('preprocessInpaint.modeLabel')}
            idPrefix="inpaint-mode"
            size="sm"
            layout="content"
          />
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.toolLabel')}</span>
          <SegmentedControl
            items={(['brush', 'eraser'] as const).map((value) => ({
              value,
              label: t(`preprocessInpaint.tool.${value}`),
            }))}
            value={erase ? 'eraser' : 'brush'}
            onChange={(value) => setErase(value === 'eraser')}
            ariaLabel={t('preprocessInpaint.toolLabel')}
            idPrefix="inpaint-tool"
            size="sm"
            layout="content"
          />
        </div>

        {mode === 'paint' && (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushColor')}</span>
            <input
              type="color"
              value={brush.color}
              onChange={(e) => setBrush((p) => ({ ...p, color: e.target.value }))}
              className="flex-1 min-w-0 h-7 p-0 border border-subtle rounded cursor-pointer bg-transparent"
              title={t('preprocessInpaint.colorWheel')}
              aria-label={t('preprocessInpaint.colorWheel')}
            />
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setRecentOpen((v) => !v)}
              disabled={recentColors.length === 0}
              aria-expanded={recentOpen}
              aria-controls={recentOpen && recentColors.length > 0 ? 'inpaint-recent-colors' : undefined}
              title={t('preprocessInpaint.recentColors')}
            >
              {t('preprocessInpaint.recentColorsShort')}
            </Button>
          </div>
        )}
        {mode === 'paint' && recentOpen && recentColors.length > 0 && (
          <div
            id="inpaint-recent-colors"
            role="group"
            aria-label={t('preprocessInpaint.recentColors')}
            className="flex items-center gap-1 flex-wrap"
          >
            {recentColors.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => {
                  setBrush((p) => ({ ...p, color: c }))
                  setRecentOpen(false)
                }}
                className={
                  'w-5 h-5 rounded border transition-transform hover:scale-110 ' +
                  (c === brush.color ? 'border-accent' : 'border-dim')
                }
                style={{ backgroundColor: c }}
                title={c}
                aria-label={t('preprocessInpaint.useRecentColor', { color: c })}
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushSize')}</span>
          <input
            type="range"
            min={1} max={400} step={1}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({ ...p, size: Number(e.target.value) }))}
            className="flex-1 min-w-0"
            aria-label={t('preprocessInpaint.brushSizeSlider')}
          />
          <input
            type="number"
            min={1} max={400}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({
              ...p, size: Math.max(1, Math.min(400, Number(e.target.value) || 1)),
            }))}
            className="input input-mono text-sm shrink-0"
            style={{ width: 56, padding: '2px 6px' }}
            aria-label={t('preprocessInpaint.brushSizeValue')}
          />
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushHardness')}</span>
          <input
            type="range"
            min={0} max={100} step={5}
            value={Math.round(brush.hardness * 100)}
            onChange={(e) => setBrush((p) => ({ ...p, hardness: Number(e.target.value) / 100 }))}
            className="flex-1 min-w-0"
            aria-label={t('preprocessInpaint.brushHardnessSlider')}
          />
          <input
            type="number"
            min={0} max={100} step={5}
            value={Math.round(brush.hardness * 100)}
            onChange={(e) => setBrush((p) => ({
              ...p,
              hardness: Math.max(0, Math.min(100, Number(e.target.value) || 0)) / 100,
            }))}
            className="input input-mono text-sm shrink-0"
            style={{ width: 56, padding: '2px 6px' }}
            aria-label={t('preprocessInpaint.brushHardnessValue')}
          />
        </div>
        {children}
      </div>
    </div>
  )
}
