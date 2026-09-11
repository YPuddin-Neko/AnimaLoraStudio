import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '../../i18n'
import { api, type HeadMaskProposals, type Job, type ModelsCatalog } from '../../api/client'
import AutoHeadMaskPanel from './AutoHeadMaskPanel'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  settings: {} as Record<string, unknown>,
  onOpen: undefined as undefined | (() => void),
  onEvent: undefined as undefined | ((event: Record<string, unknown>) => void),
}))

vi.mock('../Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: Record<string, unknown>) => void, options?: { onOpen?: () => void }) => {
    mocks.onEvent = callback
    mocks.onOpen = options?.onOpen
  },
}))
vi.mock('../../lib/SettingsData', () => ({ useSettingsData: () => mocks.settings }))

const job: Job = {
  id: 41, project_id: 2, version_id: 3, kind: 'preprocess', params: '{}',
  params_decoded: { stage: 'head_mask' }, status: 'running', started_at: null,
  finished_at: null, pid: null, log_path: null, error_msg: null,
}

const row = (value: string, label: string, current = false) => ({
  kind: value === 'builtin' ? 'preset' as const : 'local' as const,
  candidate: value === 'builtin' ? null : { kind: 'local' as const, path: value },
  value, label, description: '', download_id: value === 'builtin' ? 'head_detector' : null,
  download_variant: null, status_key: value === 'builtin' ? 'head_detector' : null,
  exists: true, size: 10, files: null, size_estimate: 0, is_current: current,
  removable: value !== 'builtin', deletable: value === 'builtin', extra: {},
})

const catalog = {
  head_detector: {
    id: 'head_detector', name: 'Anime Head Detector', description: '',
    repo: 'deepghs/anime_head_detection', revision: '06604f', target_path: 'model.onnx',
    target_dir: '/models/head_detector', default: 'builtin', current: '/custom.onnx',
    expected_size: 44_585_386, expected_sha256: 'sha', exists: true, valid: true,
    size: 44_585_386, mtime: 1,
  },
  model_sources: { head_detector: [row('builtin', 'Built-in'), row('/custom.onnx', 'Custom detector', true)] },
  downloads: {},
} as unknown as ModelsCatalog

const proposals: HeadMaskProposals = {
  schema_version: 2,
  job_id: 41,
  status: 'complete',
  model: { revision: 'custom', path: '/custom.onnx', input_size: [640, 640], provider: 'CPUExecutionProvider' },
  parameters: { confidence: 0.413, iou_threshold: 0.7, padding_ratio: 0.1, feather_ratio: 0.03 },
  created_at: 1,
  stale_count: 0,
  undo_available: false,
  images: [
    {
      name: '1_data/A.png', status: 'done', size: [100, 100], source_mtime_ns: 1,
      source_file_size: 2, stale: false, stale_reason: null,
      regions: [{ id: 'a', score: 0.9, box: [10, 10, 30, 30], mask_region: { x1: 8, y1: 8, x2: 32, y2: 32, feather_x: 1, feather_y: 1 } }],
    },
    {
      name: '1_data/B.png', status: 'done', size: [100, 100], source_mtime_ns: 1,
      source_file_size: 2, stale: false, stale_reason: null, regions: [],
    },
  ],
}

type Props = React.ComponentProps<typeof AutoHeadMaskPanel>
const results = vi.fn()
const busyChanged = vi.fn()
function Harness({ overrides = {} }: { overrides?: Partial<Props> }) {
  const [open, setOpen] = useState(true)
  return <>
    <button onClick={() => setOpen(true)}>Open setup</button>
    <AutoHeadMaskPanel
      projectId={2}
      versionId={3}
      activeName="1_data/A.png"
      unsavedCount={0}
      setupOpen={open}
      onCloseSetup={() => setOpen(false)}
      onResults={results}
      onBusyChange={busyChanged}
      {...overrides}
    />
  </>
}
function renderPanel(overrides: Partial<Props> = {}) { return render(<Harness overrides={overrides} />) }
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  mocks.settings = { catalog, catalogError: null, downloadBusy: new Set(), downloadErrors: {} }
  vi.spyOn(api, 'getPreprocessStatusTrain').mockResolvedValue({ job: null, log_tail: '', summary: { image_count: 2 } })
  vi.spyOn(api, 'getJob').mockResolvedValue(job)
  vi.spyOn(api, 'startHeadMaskDetection').mockResolvedValue(job)
  vi.spyOn(api, 'getHeadMaskProposals').mockResolvedValue(proposals)
  vi.spyOn(api, 'applyHeadMaskProposals')
  vi.spyOn(api, 'undoHeadMaskApply')
})

describe('Auto mask direct-edit setup', () => {
  it('shows only the simple labelled fields and defaults to the globally selected installed model', () => {
    renderPanel()
    expect(screen.getByRole('dialog', { name: '自动遮罩' })).toBeInTheDocument()
    expect(screen.getByLabelText('识别模型')).toHaveValue('/custom.onnx')
    expect(screen.getByLabelText('检测范围')).toBeInTheDocument()
    expect(screen.getByLabelText('置信度')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '置信度说明' })).toBeInTheDocument()
    expect(screen.getByLabelText('扩展比例')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '扩展比例说明' })).toBeInTheDocument()
    expect(screen.getByLabelText('羽化比例')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '羽化比例说明' })).toBeInTheDocument()
    expect(screen.queryByText(/只生成空间遮罩/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '高级参数' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('NMS IoU')).not.toBeInTheDocument()
    expect(screen.queryByText(/下载头部检测模型/)).not.toBeInTheDocument()
  })

  it('submits current scope, explicit model and internal default IoU, then closes', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.selectOptions(screen.getByLabelText('检测范围'), 'selected')
    await user.selectOptions(screen.getByLabelText('识别模型'), 'builtin')
    fireEvent.change(screen.getByLabelText('置信度'), { target: { value: '0.5' } })
    await user.click(screen.getByRole('button', { name: '开始' }))
    expect(api.startHeadMaskDetection).toHaveBeenCalledWith(2, 3, {
      scope: 'selected', filenames: ['1_data/A.png'], model: 'builtin',
      confidence: 0.5, iou_threshold: 0.7, padding_ratio: 0.1, feather_ratio: 0.03,
    })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('does not transport proposals from an already-terminal initial status', async () => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
      job: { ...job, status: 'done' }, log_tail: 'not rendered', summary: { image_count: 2 },
    })
    renderPanel({ setupOpen: false })
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalled())
    expect(api.getHeadMaskProposals).not.toHaveBeenCalled()
    expect(results).not.toHaveBeenCalled()
    expect(mocks.toast).not.toHaveBeenCalled()
    expect(screen.queryByText(/审核|review|not rendered/i)).not.toBeInTheDocument()
    expect(api.applyHeadMaskProposals).not.toHaveBeenCalled()
    expect(api.undoHeadMaskApply).not.toHaveBeenCalled()
  })

  it.each(['failed', 'canceled'] as const)(
    'does not notify an already-%s job on initial mount',
    async (status) => {
      vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
        job: { ...job, status }, log_tail: '', summary: { image_count: 2 },
      })
      renderPanel({ setupOpen: false })
      await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalled())
      expect(mocks.toast).not.toHaveBeenCalled()
    },
  )

  it.each([
    ['failed', '自动遮罩失败，请查看任务详情后重试。', 'error'],
    ['canceled', '自动遮罩已取消，未添加任何遮罩修改。', 'info'],
  ] as const)('notifies an in-session %s transition exactly once', async (status, message, level) => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValueOnce({
      job, log_tail: '', summary: { image_count: 2 },
    })
    renderPanel({ setupOpen: false })
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalledTimes(1))

    act(() => mocks.onEvent?.({
      type: 'job_state_changed', job_id: job.id, project_id: 2, version_id: 3, status,
    }))
    expect(mocks.toast).toHaveBeenCalledWith(message, level)
    act(() => mocks.onEvent?.({
      type: 'job_state_changed', job_id: job.id, project_id: 2, version_id: 3, status,
    }))
    expect(mocks.toast).toHaveBeenCalledTimes(1)
  })

  it('transports a job enqueued by this mount exactly once', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.click(screen.getByRole('button', { name: '开始' }))
    await waitFor(() => expect(api.startHeadMaskDetection).toHaveBeenCalled())
    act(() => mocks.onEvent?.({
      type: 'job_state_changed', job_id: job.id, project_id: 2, version_id: 3, status: 'done',
    }))
    await waitFor(() => expect(results).toHaveBeenCalledWith(proposals))
    act(() => mocks.onEvent?.({
      type: 'job_state_changed', job_id: job.id, project_id: 2, version_id: 3, status: 'done',
    }))
    expect(api.getHeadMaskProposals).toHaveBeenCalledTimes(1)
    expect(results).toHaveBeenCalledTimes(1)
  })

  it('incorporates only a same-job transition observed by this mount', async () => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValueOnce({
      job, log_tail: '', summary: { image_count: 2 },
    })
    renderPanel({ setupOpen: false })
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalledTimes(1))

    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValueOnce({
      job: { ...job, status: 'done' }, log_tail: '', summary: { image_count: 2 },
    })
    act(() => mocks.onOpen?.())
    await waitFor(() => expect(results).toHaveBeenCalledWith(proposals))
    expect(results).toHaveBeenCalledTimes(1)

    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValueOnce({
      job: { ...job, id: 42, status: 'done' }, log_tail: '', summary: { image_count: 2 },
    })
    act(() => mocks.onOpen?.())
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalledTimes(3))
    expect(api.getHeadMaskProposals).toHaveBeenCalledTimes(1)
  })

  it('reports no installed model with Settings recovery and disables start', () => {
    mocks.settings = {
      ...mocks.settings,
      catalog: { ...catalog, model_sources: { head_detector: [] } },
    }
    renderPanel()
    expect(screen.getByText(/设置 → 预处理/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始' })).toBeDisabled()
  })

  it('keeps the modal open with a local error after enqueue failure', async () => {
    vi.mocked(api.startHeadMaskDetection).mockRejectedValue(new Error('enqueue unavailable'))
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: '开始' }))
    expect(await screen.findByText('Error: enqueue unavailable')).toBeInTheDocument()
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始' })).toBeEnabled()
  })

  it('ignores a proposal response after the version identity changes', async () => {
    const slow = deferred<HeadMaskProposals>()
    vi.mocked(api.getHeadMaskProposals).mockReturnValueOnce(slow.promise)
    const view = renderPanel()
    await userEvent.click(screen.getByRole('button', { name: '开始' }))
    act(() => mocks.onEvent?.({
      type: 'job_state_changed', job_id: job.id, project_id: 2, version_id: 3, status: 'done',
    }))
    await waitFor(() => expect(api.getHeadMaskProposals).toHaveBeenCalled())
    view.rerender(<Harness overrides={{ versionId: 4, setupOpen: false }} />)
    await act(async () => slow.resolve(proposals))
    expect(results).not.toHaveBeenCalled()
  })

  it('renders the simplified modal in English', async () => {
    await i18n.changeLanguage('en')
    try {
      renderPanel()
      expect(screen.getByRole('dialog', { name: 'Auto mask' })).toBeInTheDocument()
      expect(screen.getByLabelText('Recognition model')).toBeInTheDocument()
      expect(screen.getByLabelText('Detection scope')).toBeInTheDocument()
      expect(screen.queryByText('Advanced parameters')).not.toBeInTheDocument()
    } finally {
      await act(async () => { await i18n.changeLanguage('zh') })
    }
  })
})
