import { useEffect, useId, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { HeadMaskProposals, ModelSourceRow } from '../../api/client'
import ActionGroup from '../ActionGroup'
import Alert from '../Alert'
import Button from '../Button'
import { Input, Select } from '../FormControl'
import { InfoButton } from '../InfoButton'
import Modal from '../Modal'

type Parameters = HeadMaskProposals['parameters']
const DEFAULTS = { confidence: '0.413', padding_ratio: '0.10', feather_ratio: '0.03' }
const DEFAULT_IOU = 0.7
const BOUNDS = { confidence: [0.01, 0.99], padding_ratio: [0, 1], feather_ratio: [0, 0.5] } as const

export default function HeadMaskSetupModal({
  activeName,
  busy,
  running,
  models,
  defaultModel,
  unsavedCount,
  error,
  onClose,
  onStart,
}: {
  activeName: string | null
  busy: boolean
  running: boolean
  models: ModelSourceRow[]
  defaultModel: string
  unsavedCount: number
  error: string
  onClose: () => void
  onStart: (
    scope: 'all' | 'selected',
    model: string,
    params: Parameters,
  ) => Promise<void>
}) {
  const { t } = useTranslation()
  const id = useId()
  const [scope, setScope] = useState<'all' | 'selected'>('all')
  const [model, setModel] = useState(defaultModel)
  const [draft, setDraft] = useState(DEFAULTS)

  useEffect(() => {
    if (!models.some((row) => row.value === model)) {
      setModel(defaultModel || models[0]?.value || '')
    }
  }, [defaultModel, model, models])

  const errors = useMemo(() => Object.fromEntries(
    (Object.keys(draft) as (keyof typeof draft)[]).map((key) => {
      const value = Number(draft[key])
      const [min, max] = BOUNDS[key]
      return [key, draft[key].trim() === '' || !Number.isFinite(value) || value < min || value > max
        ? t('preprocessInpaint.headMask.invalidRange', { min, max }) : '']
    }),
  ), [draft, t])

  const disabled = busy || running || !model || unsavedCount > 0
    || Object.values(errors).some(Boolean) || (scope === 'selected' && !activeName)

  const field = (key: keyof typeof draft) => {
    const label = t(`preprocessInpaint.headMask.${key}`)
    return (
    <div className="flex min-w-0 flex-col gap-related" key={key}>
      <div className="flex items-center gap-related">
        <label className="type-field-label" htmlFor={`${id}-${key}`}>
          {label}
        </label>
        <InfoButton ariaLabel={t('preprocessInpaint.headMask.parameterHelp', { name: label })}>
          {t(`preprocessInpaint.headMask.${key}Help`)}
        </InfoButton>
      </div>
      <Input
        id={`${id}-${key}`}
        type="number"
        mono
        step="any"
        min={BOUNDS[key][0]}
        max={BOUNDS[key][1]}
        value={draft[key]}
        disabled={busy || running}
        invalid={!!errors[key]}
        aria-describedby={errors[key] ? `${id}-${key}-error` : undefined}
        onChange={(event) => setDraft((prev) => ({ ...prev, [key]: event.target.value }))}
      />
      {errors[key] && (
        <span id={`${id}-${key}-error`} className="text-sm text-err">{errors[key]}</span>
      )}
    </div>
    )
  }

  return (
    <Modal
      as="form"
      size="md"
      title={t('preprocessInpaint.headMask.openSetup')}
      onClose={() => { if (!busy) onClose() }}
      closeOnBackdrop={!busy}
      closeOnEscape={!busy}
      onSubmit={(event) => {
        event.preventDefault()
        if (!disabled) {
          void onStart(scope, model, {
            confidence: Number(draft.confidence),
            iou_threshold: DEFAULT_IOU,
            padding_ratio: Number(draft.padding_ratio),
            feather_ratio: Number(draft.feather_ratio),
          })
        }
      }}
      footer={(
        <ActionGroup
          secondary={<Button variant="secondary" disabled={busy} onClick={onClose}>{t('common.cancel')}</Button>}
          primary={<Button type="submit" variant="primary" disabled={disabled} loading={busy}>{t('preprocessInpaint.headMask.start')}</Button>}
        />
      )}
    >
      <div className="flex flex-col gap-section">
        <div className="flex flex-col gap-related">
          <label className="type-field-label" htmlFor={`${id}-model`}>
            {t('preprocessInpaint.headMask.recognitionModel')}
          </label>
          <Select
            id={`${id}-model`}
            value={model}
            disabled={busy || running || models.length === 0}
            onChange={(event) => setModel(event.target.value)}
          >
            {models.map((row) => <option key={`${row.kind}:${row.value}`} value={row.value}>{row.label}</option>)}
          </Select>
        </div>
        {models.length === 0 && (
          <Alert tone="warning" size="sm">{t('preprocessInpaint.headMask.modelRecovery')}</Alert>
        )}
        <div className="flex flex-col gap-related">
          <label className="type-field-label" htmlFor={`${id}-scope`}>{t('preprocessInpaint.headMask.scope')}</label>
          <Select
            id={`${id}-scope`}
            value={scope}
            disabled={busy || running}
            onChange={(event) => setScope(event.target.value as 'all' | 'selected')}
          >
            <option value="all">{t('preprocessInpaint.headMask.scopeAll')}</option>
            <option value="selected" disabled={!activeName}>{t('preprocessInpaint.headMask.scopeCurrent')}</option>
          </Select>
          {scope === 'selected' && activeName && (
            <span className="truncate text-xs font-mono text-fg-tertiary" title={activeName}>{activeName}</span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-field">
          {field('confidence')}
          {field('padding_ratio')}
          {field('feather_ratio')}
        </div>
        {unsavedCount > 0 && (
          <Alert tone="warning" size="sm">{t('preprocessInpaint.headMask.saveFirst', { n: unsavedCount })}</Alert>
        )}
        {error && <Alert tone="danger" size="sm">{error}</Alert>}
      </div>
    </Modal>
  )
}
