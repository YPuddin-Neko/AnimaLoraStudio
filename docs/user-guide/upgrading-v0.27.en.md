# Upgrading to v0.27

[中文](upgrading-v0.27.md)

v0.27 changes configuration storage, the LyCORIS dependency, and workbench interactions. See the [v0.27.0 announcement](../announcements/2026-09-11-v0.27.0.en.md) for features; this page covers upgrade steps and compatibility.

## Before upgrading

1. Let running training / data jobs finish, or stop them normally through the queue, then close Studio. Do not replace dependencies underneath a running training process.
2. Back up the complete `studio_data/` directory to a secure local location. Separately back up any data or training outputs configured outside that directory. These backups contain credentials: do not attach them to issues or upload them publicly.
3. Follow the normal update workflow and restart with `studio.bat` / `studio.sh`. Wait for dependency synchronization before training. Custom environments must also install the current `requirements.txt`, not just pull code.

## Settings and LLM preset migration

First startup automatically migrates legacy `studio_data/secrets.json`; do not split it manually. The new layout is:

| Path | Contents |
|---|---|
| `studio_data/settings.json` | Non-sensitive global settings |
| `studio_data/credentials.json` | API keys / tokens; local plaintext, with no secret readback through the API |
| `studio_data/llm_presets/{id}.json` | Individual LLM presets or local overrides of built-in templates; references credentials by ID only |
| `studio_data/cache/llm_models/` | Refreshable model-discovery cache, not preset recipes |
| `studio_data/storage-layout.json` | Migration state; the new layout becomes authoritative only at `complete` |
| `studio_data/backups/legacy/` | Pre-migration backups, potentially containing plaintext credentials |

Continue using **Settings** to edit configuration after migration. Legacy `secrets.json` may serve as a compatibility projection for older releases, but **is no longer the new version's configuration source**. Editing it does not update the new settings. LLM presets, ordinary settings, and credentials each use their own save API; saving ordinary settings no longer replaces the entire preset list.

- **Conflict or corruption warnings**: preserve the current files and backups, and stop related edits first. Do not “repair” by deleting files, emptying them, or editing the migration marker. The system refuses to overwrite unrecoverable data. Reload the latest preset before resolving an edit conflict; restore corrupted data from a known-good backup.
- **Queued jobs**: LLM tagging freezes a non-secret preset snapshot at enqueue time. Later preset edits or deletion do not change that recipe. Credentials are resolved at execution; a missing reference fails explicitly. Repair the credential and retry rather than expecting an automatic key substitution.
- **Sharing is not backup**: use preset export for sharing, rather than copying all of `studio_data/`. Portable exports clear the local service address and credential reference, but you must still check prompts, model names, and other recipe text for information you entered yourself. Reconfigure the service address and credential after import.
- **Rollback**: close Studio and back up the full upgraded directory first, then restore a matching pre-upgrade environment and data snapshot together. Do not mix an old version's updated `secrets.json` into an already-migrated layout, or assume the compatibility projection contains every new setting.

See [ADR 0017](../adr/0017-split-llm-preset-settings-credentials.md) for the design boundaries.

## LyCORIS v4

The project pins `lycoris-lora==4.0.0` and uses the **eager** kernel baseline. The v0.26.1 hotfix constrained it to `<4`; do not retain that constraint for this release, or remove the project pin to install a newer upstream version.

The compatibility layer preserves legacy LoRA / LoHa / LoKr weight-scaling semantics, with existing save, load, and resume entry points. Triton, custom backward, and low-precision kernels are not enabled by default, and no training-speed improvement is promised. Third-party training scripts or custom kernels are outside this compatibility scope. See [ADR 0016](../adr/0016-adopt-lycoris-v4-with-safe-kernel-rollout.md).

## Workflow reminders

- **Tagging** skips existing captions by default; confirm the scope when choosing overwrite. The running Current task is a submitted snapshot, and Next-run settings do not modify it live.
- **Training** saves the latest draft before enqueueing. If saving fails, stay on the page and retry. An active task for the same version lets you prepare the next configuration but prevents duplicate submission.
- **Automatic head masks** require a recognition model installed under Settings → Preprocess. Results enter the unsaved mask layer and persist only through Save current / Save all. Enable masked loss for training, and disable incompatible Leap / NaViT Packing options. Character names, hair colors, and other caption tags are not removed automatically. See the [usage guide](auto-head-mask.en.md).
