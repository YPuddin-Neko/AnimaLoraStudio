// SettingsData.tsx —— Settings 全局数据层。
//
// 把 secrets / catalog / downloadBusy / SSE 订阅从 SettingsPage 提到根级 Provider，
// 让 SettingsPage 的展示生命周期与数据请求解耦：
// - secrets：一次 fetch，常驻 context；save 后由 SettingsPage 调 setSecrets 更新
// - catalog：reloadCatalog + model_download_changed SSE 订阅常驻，跟下载组件共享
// - downloadBusy：跟 startDownload 配对的 in-flight Set
//
// 这层只持有数据，不渲染 UI。SettingsPage 首次打开后可在 Drawer 内保活，
// 无论 UI 是否挂载，权威数据和订阅都继续由这里持有。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog, type Secrets, type SecretsPatch } from '../api/client'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from './useEventStream'

// 全局「已保存」状态指示（instant-apply 下取代旧的保存按钮 dirty 态）。
export type SaveStatus =
  | { state: 'idle' }
  | { state: 'saving' }
  | { state: 'saved'; at: number }
  | { state: 'error'; error: string }

// 把单字段 patch 浅合并进本地 secrets（乐观更新用）。secrets 是两层结构
// （section → fields），顶层标量字段（如 download_source）直接覆盖。
function mergePatchLocal(base: Secrets, patch: SecretsPatch): Secrets {
  const out = { ...base } as Record<string, unknown>
  for (const key of Object.keys(patch)) {
    const pv = (patch as Record<string, unknown>)[key]
    const bv = out[key]
    if (pv && typeof pv === 'object' && !Array.isArray(pv)
        && bv && typeof bv === 'object' && !Array.isArray(bv)) {
      out[key] = { ...(bv as object), ...(pv as object) }
    } else {
      out[key] = pv
    }
  }
  return out as unknown as Secrets
}

interface SettingsData {
  secrets: Secrets | null
  secretsError: string | null
  setSecrets: (s: Secrets) => void
  reloadSecrets: () => Promise<Secrets | null>
  /** instant-apply 统一写入入口：乐观更新 + 串行 PUT 单字段 patch。 */
  commitSecrets: (patch: SecretsPatch) => void
  /** 包装一次性即时 PUT（下载源 / 主模型 / upscaler 等独立保存），驱动 saveStatus 指示。 */
  runSave: <T>(fn: () => Promise<T>) => Promise<T>
  saveStatus: SaveStatus
  catalog: ModelsCatalog | null
  catalogError: string | null
  reloadCatalog: () => Promise<ModelsCatalog | null>
  downloadBusy: Set<string>
  downloadErrors: Record<string, string>
  startDownload: (model_id: string, variant?: string) => Promise<void>
  /** 下载的逆操作（confirm → DELETE → 刷 catalog），下载中心各区共用。 */
  deleteAsset: (model_id: string, variant: string | undefined, name: string) => Promise<void>
  setDownloadSource: (type: string, source: string) => Promise<void>
}

const Ctx = createContext<SettingsData | null>(null)

export function SettingsDataProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const [secrets, setSecrets] = useState<Secrets | null>(null)
  const [secretsError, setSecretsError] = useState<string | null>(null)
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [downloadBusy, setDownloadBusy] = useState<Set<string>>(new Set())
  const [downloadErrors, setDownloadErrors] = useState<Record<string, string>>({})
  const [headRequested, setHeadRequested] = useState(false)
  const mounted = useRef(true)
  const catalogRequest = useRef(0)
  const starting = useRef(new Set<string>())
  useEffect(() => {
    mounted.current = true
    catalogRequest.current++
    return () => { mounted.current = false }
  }, [])
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ state: 'idle' })
  // 串行 PUT 队列 + in-flight 计数：保证顺序避免后端读改写竞态。
  const saveQueueRef = useRef<Promise<unknown>>(Promise.resolve())
  const pendingRef = useRef(0)

  const reloadSecrets = useCallback(async (): Promise<Secrets | null> => {
    try {
      const loaded = await api.getSecrets()
      setSecrets(loaded)
      setSecretsError(null)
      return loaded
    } catch (e) {
      setSecretsError(String(e))
      return null
    }
  }, [])

  useEffect(() => { void reloadSecrets() }, [reloadSecrets])

  const reloadCatalog = useCallback(async (): Promise<ModelsCatalog | null> => {
    const request = ++catalogRequest.current
    try {
      const c = await api.getModelsCatalog()
      if (!mounted.current || request !== catalogRequest.current) return null
      setCatalog(c)
      setCatalogError(null)
      const headStatus = c.downloads.head_detector?.status
      if (headStatus && !['pending', 'running'].includes(headStatus)) setHeadRequested(false)
      return c
    } catch (e) {
      if (mounted.current && request === catalogRequest.current) setCatalogError(String(e))
      return null
    }
  }, [])

  useEffect(() => { void reloadCatalog() }, [reloadCatalog])

  // model_download_changed 既驱动 catalog 刷新，也是下载失败时的唯一全局信号：
  // 下载在后台线程跑，失败原因（如 gated 仓库缺 token）只进 download status，
  // 不会让 startDownload 的 await 抛错。这里在 failed 时弹一个 error toast，
  // 把后端汇总的可操作 message 顶到用户面前——否则用户只看到卡片上一个红 badge，
  // 原因埋在另一个 tab 的折叠「下载日志」里（甚至只在终端）。
  // Only the head-detector adoption needs recovery polling. SSE remains primary;
  // a stalled/disconnected transfer gets six retreating checks, then manual retry.
  const headStatus = catalog?.downloads.head_detector?.status
  const headActive = headRequested || headStatus === 'pending' || headStatus === 'running'
  useEffect(() => {
    if (!headActive) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    let attempt = 0
    const delays = [1500, 3000, 6000, 10000, 15000, 30000]
    const schedule = () => {
      if (!active || attempt >= delays.length) return
      timer = setTimeout(async () => {
        const c = await reloadCatalog()
        const status = c?.downloads.head_detector?.status
        if (status && !['pending', 'running'].includes(status)) return
        schedule()
      }, delays[attempt++])
    }
    schedule()
    return () => { active = false; clearTimeout(timer) }
  }, [headActive, reloadCatalog])

  useEventStream((evt) => {
    if (evt.type !== 'model_download_changed') return
    if (evt.key === 'head_detector' && ['done', 'failed', 'canceled'].includes(String(evt.status))) {
      // Terminal SSE stops recovery even when the subsequent catalog read fails.
      setHeadRequested(false)
      setCatalog((previous) => {
        const download = previous?.downloads.head_detector
        if (!previous || !download) return previous
        return { ...previous, downloads: { ...previous.downloads, head_detector: {
          ...download, status: evt.status as typeof download.status,
        } } }
      })
    }
    void reloadCatalog().then((c) => {
      if (!mounted.current || evt.status !== 'failed' || !c) return
      const key = String(evt.key ?? '')
      // Head detector reports errors next to the shared retry action, not twice.
      if (key === 'head_detector') return
      const dl = c.downloads[key]
      toast(dl?.message || t('settings.downloadFailed', { error: key }), 'error')
    })
  }, { onOpen: () => { void reloadCatalog() } })

  const startDownload = useCallback(async (model_id: string, variant?: string) => {
    const key = variant ? `${model_id}:${variant}` : model_id
    if (starting.current.has(key)) return
    starting.current.add(key)
    setDownloadErrors((s) => ({ ...s, [key]: '' }))
    setDownloadBusy((s) => new Set(s).add(key))
    try {
      await api.startModelDownload({ model_id, variant })
      if (!mounted.current) return
      if (key === 'head_detector') setHeadRequested(true)
      else toast(t('settings.downloadStarted', { name: key }), 'success')
      await reloadCatalog()
    } catch (e) {
      if (!mounted.current) return
      if (key === 'head_detector') {
        setHeadRequested(false)
        setDownloadErrors((s) => ({ ...s, [key]: String(e) }))
      } else toast(String(e), 'error')
    } finally {
      starting.current.delete(key)
      if (mounted.current) setDownloadBusy((s) => { const n = new Set(s); n.delete(key); return n })
    }
  }, [reloadCatalog, t, toast])

  // 删除已下载资产（下载的逆操作）：confirm → DELETE → 刷 catalog。
  // 下载中心各区（训练模型 / 打标 / eval / 放大器）共用这一份流程。
  const deleteAsset = useCallback(async (model_id: string, variant: string | undefined, name: string) => {
    if (!(await dialog.confirm(t('settings.confirmDeleteAsset', { name }), { tone: 'danger' }))) return
    try {
      await api.deleteModelAsset(model_id, variant)
      toast(t('settings.assetDeleted', { name }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [dialog, reloadCatalog, t, toast])

  // 按类型选下载源：即时存（跟「下载」/ models.root 一样是立即动作，不进表单
  // draft）。刻意不 setSecrets —— 否则会让 SettingsPage 的 draft/server 失同步，
  // 表单 Save 时把这次改动 clobber 回去。dropdown 当前值读 catalog（reloadCatalog
  // 刷新），不依赖表单 secrets。
  // 即时保存包装：给「不进 commitSecrets 队列」的独立 PUT（下载源 / 主模型 /
  // upscaler / auto_sync 等切换类）也驱动右上角 saveStatus 指示，反馈统一。
  const runSave = useCallback(async <T,>(fn: () => Promise<T>): Promise<T> => {
    setSaveStatus({ state: 'saving' })
    try {
      const r = await fn()
      setSaveStatus({ state: 'saved', at: Date.now() })
      return r
    } catch (e) {
      setSaveStatus({ state: 'error', error: String(e) })
      throw e
    }
  }, [])

  const setDownloadSource = useCallback(async (type: string, source: string) => {
    try {
      await runSave(() => api.updateSecrets({ download_sources: { [type]: source } }))
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [runSave, reloadCatalog, toast])

  // instant-apply 统一写入：乐观更新本地 secrets 让控件立即反映，PUT 单字段
  // patch 入串行队列。队列全部清空后用后端权威结果回写一次（拿 validator
  // 规范化 + 敏感字段 mask）——避免连改多个字段时早 PUT 的权威结果覆盖掉
  // 后面字段的乐观值（中途闪回）。失败时重拉 secrets 恢复一致。
  const commitSecrets = useCallback((patch: SecretsPatch) => {
    setSecrets((s) => (s ? mergePatchLocal(s, patch) : s))
    setSaveStatus({ state: 'saving' })
    pendingRef.current += 1
    saveQueueRef.current = saveQueueRef.current
      .then(() => api.updateSecrets(patch))
      .then((authoritative) => {
        pendingRef.current -= 1
        // 每个 PUT 完成都刷新「已保存」时间戳，让连续保存每次都有可见反馈；
        // 权威结果只在队列清空时回写一次，避免中途覆盖后续字段的乐观值。
        if (pendingRef.current === 0) setSecrets(authoritative)
        setSaveStatus({ state: 'saved', at: Date.now() })
      })
      .catch((e) => {
        pendingRef.current -= 1
        setSaveStatus({ state: 'error', error: String(e) })
        toast(String(e), 'error')
        void reloadSecrets()
      })
  }, [reloadSecrets, toast])

  return (
    <Ctx.Provider value={{
      secrets, secretsError, setSecrets, reloadSecrets, commitSecrets, runSave, saveStatus,
      catalog, catalogError, reloadCatalog,
      downloadBusy, downloadErrors, startDownload, deleteAsset, setDownloadSource,
    }}>
      {children}
    </Ctx.Provider>
  )
}

export function useSettingsData(): SettingsData {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettingsData must be used inside <SettingsDataProvider>')
  return ctx
}
