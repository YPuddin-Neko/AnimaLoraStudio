import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type HeadMaskProposals, type Job } from '../../api/client'
import { ownsPreprocessJob } from '../../lib/preprocessJob'
import { useSettingsData } from '../../lib/SettingsData'
import { useEventStream } from '../../lib/useEventStream'
import { useToast } from '../Toast'
import HeadMaskSetupModal from './HeadMaskSetupModal'

interface Props {
  projectId: number
  versionId: number
  activeName: string | null
  unsavedCount: number
  setupOpen: boolean
  onCloseSetup: () => void
  onResults: (result: HeadMaskProposals) => void
  onBusyChange?: (busy: boolean) => void
}

/** Key owns all version-local request and job lifetimes. */
export default function AutoHeadMaskPanel(props: Props) {
  return <HeadMaskWorkspace key={`${props.projectId}:${props.versionId}`} {...props} />
}

function HeadMaskWorkspace({
  projectId,
  versionId,
  activeName,
  unsavedCount,
  setupOpen,
  onCloseSetup,
  onResults,
  onBusyChange,
}: Props) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { catalog, catalogError, downloadBusy } = useSettingsData()
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const mounted = useRef(true)
  const busyRef = useRef(false)
  const jobRef = useRef<Job | null>(null)
  const statusRequest = useRef(0)
  const proposalRequest = useRef(0)
  const sessionOwnedJobs = useRef(new Set<number>())
  const incorporatedJobs = useRef(new Set<number>())
  const terminalNotifiedJobs = useRef(new Set<number>())

  useEffect(() => {
    mounted.current = true
    statusRequest.current++
    proposalRequest.current++
    return () => { mounted.current = false }
  }, [])

  const loadProposal = useCallback(async (jobId: number) => {
    if (incorporatedJobs.current.has(jobId)) return
    const request = ++proposalRequest.current
    try {
      const result = await api.getHeadMaskProposals(projectId, versionId, jobId)
      if (
        !mounted.current
        || request !== proposalRequest.current
        || jobRef.current?.id !== jobId
        || result.job_id !== jobId
        || incorporatedJobs.current.has(jobId)
      ) return
      incorporatedJobs.current.add(jobId)
      setError('')
      onResults(result)
    } catch (e) {
      if (mounted.current && request === proposalRequest.current && jobRef.current?.id === jobId) {
        setError(String(e))
        toast(String(e), 'error')
      }
    }
  }, [onResults, projectId, toast, versionId])

  const acceptJob = useCallback((next: Job | null) => {
    if (!mounted.current) return
    if (next && !ownsPreprocessJob(next, projectId, versionId, 'head_mask')) return
    const previous = jobRef.current
    if (
      next
      && previous?.id === next.id
      && (previous.status === 'pending' || previous.status === 'running')
      && (next.status === 'done' || next.status === 'failed' || next.status === 'canceled')
    ) {
      sessionOwnedJobs.current.add(next.id)
    }
    if (previous?.id !== next?.id) proposalRequest.current++
    jobRef.current = next
    setJob(next)
    if (next?.status === 'done' && sessionOwnedJobs.current.has(next.id)) {
      void loadProposal(next.id)
    }
    if (
      next
      && (next.status === 'failed' || next.status === 'canceled')
      && sessionOwnedJobs.current.has(next.id)
      && !terminalNotifiedJobs.current.has(next.id)
    ) {
      terminalNotifiedJobs.current.add(next.id)
      toast(t(next.status === 'failed'
        ? 'preprocessInpaint.headMask.detectFailed'
        : 'preprocessInpaint.headMask.detectCanceled'), next.status === 'failed' ? 'error' : 'info')
    }
  }, [loadProposal, projectId, t, toast, versionId])

  const refreshStatus = useCallback(async () => {
    if (busyRef.current) return
    const request = ++statusRequest.current
    try {
      const result = await api.getPreprocessStatusTrain(projectId, versionId, 'head_mask')
      if (!mounted.current || request !== statusRequest.current) return
      acceptJob(ownsPreprocessJob(result.job, projectId, versionId, 'head_mask') ? result.job : null)
    } catch (e) {
      if (mounted.current && request === statusRequest.current) setError(String(e))
    }
  }, [acceptJob, projectId, versionId])

  useEffect(() => { void refreshStatus() }, [refreshStatus])

  const running = job?.status === 'pending' || job?.status === 'running'
  const activeJobId = job?.id
  useEffect(() => {
    if (!running || !activeJobId) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const jobId = activeJobId
    const poll = async () => {
      try {
        const latest = await api.getJob(jobId)
        if (!active || !mounted.current || jobRef.current?.id !== jobId) return
        if (ownsPreprocessJob(latest, projectId, versionId, 'head_mask')) acceptJob(latest)
      } catch { /* Keep the owned snapshot until the next poll/SSE update. */ }
      if (active) timer = setTimeout(() => void poll(), 3000)
    }
    timer = setTimeout(() => void poll(), 1500)
    return () => { active = false; clearTimeout(timer) }
  }, [acceptJob, activeJobId, projectId, running, versionId])

  useEventStream((event) => {
    const current = jobRef.current
    if (!mounted.current || !current || event.job_id !== current.id) return
    if (event.project_id != null && event.project_id !== projectId) return
    if (event.version_id != null && event.version_id !== versionId) return
    if (event.type === 'job_state_changed') {
      statusRequest.current++
      acceptJob({ ...current, status: String(event.status) as Job['status'] })
    }
  }, { onOpen: () => { void refreshStatus() } })

  useEffect(() => { onBusyChange?.(busy || running) }, [busy, onBusyChange, running])

  const installedModels = useMemo(
    () => (catalog?.model_sources?.head_detector ?? []).filter((row) => row.exists),
    [catalog],
  )
  const defaultModel = installedModels.some((row) => row.value === catalog?.head_detector?.current)
    ? catalog?.head_detector?.current ?? ''
    : installedModels[0]?.value ?? ''
  const modelReady = installedModels.length > 0 && !catalogError
    && !downloadBusy.has('head_detector')

  const startDetection = async (
    scope: 'all' | 'selected',
    model: string,
    params: HeadMaskProposals['parameters'],
  ) => {
    if (
      busyRef.current || running || !modelReady || !installedModels.some((row) => row.value === model)
      || (scope === 'selected' && !activeName)
    ) return
    if (unsavedCount > 0) {
      setError(t('preprocessInpaint.headMask.saveFirst', { n: unsavedCount }))
      return
    }
    busyRef.current = true
    setBusy(true)
    setError('')
    statusRequest.current++
    proposalRequest.current++
    try {
      const next = await api.startHeadMaskDetection(projectId, versionId, {
        scope,
        ...(scope === 'selected' && activeName ? { filenames: [activeName] } : {}),
        model,
        ...params,
      })
      if (!mounted.current) return
      if (ownsPreprocessJob(next, projectId, versionId, 'head_mask')) {
        sessionOwnedJobs.current.add(next.id)
      }
      acceptJob(next)
      onCloseSetup()
    } catch (e) {
      if (mounted.current) setError(String(e))
    } finally {
      busyRef.current = false
      if (mounted.current) setBusy(false)
    }
  }

  if (!setupOpen) return null
  return (
    <HeadMaskSetupModal
      activeName={activeName}
      busy={busy}
      running={running}
      models={installedModels}
      defaultModel={defaultModel}
      unsavedCount={unsavedCount}
      error={error}
      onClose={onCloseSetup}
      onStart={startDetection}
    />
  )
}
