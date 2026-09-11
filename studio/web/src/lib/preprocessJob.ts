import type { Job } from '../api/client'

/** Missing historical stage denotes upscale, never crop or head-mask. */
export function ownsPreprocessJob(
  job: Job | null, projectId: number, versionId: number,
  stage: 'upscale' | 'crop' | 'head_mask',
): job is Job {
  if (!job || job.kind !== 'preprocess' || job.project_id !== projectId || job.version_id !== versionId) return false
  let params = job.params_decoded
  if (!params && typeof job.params === 'string') {
    try { params = JSON.parse(job.params) as Record<string, unknown> } catch { return false }
  }
  return (params?.stage ?? 'upscale') === stage
}
