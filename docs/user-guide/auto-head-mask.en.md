# Automatic head masks

Automatic head masks are intended for clothing, pose, and style LoRAs where cartoon faces, hair, ears, and head accessories should not contribute to the loss. The feature adds unsaved `.mask` edits only; source images and captions are never modified.

## Set up recognition models

Open **Settings → Preprocess → Head detector**. Head detectors use the same model-source controls as upscalers:

- Choose HuggingFace or ModelScope as the download source.
- Select the globally preferred installed detector with its radio button.
- Use the pinned built-in detector, add a download by repository plus `.onnx` filename, or register an existing absolute `.onnx` file.
- Downloaded models are managed in the detector model directory. Registered local files stay in place and are never copied or deleted.
- Removing or deleting the selected custom detector falls back to the built-in detector.

The built-in weight is revision-, size-, and SHA-256-pinned. Custom detectors must implement the same YOLO ONNX input/output layout; incompatible or missing files fail safely without changing an image or mask.

## Workflow

1. Open **Preprocess → Retouch** for a project version. Choose **Auto mask** in the header, immediately before **Save all**.
2. Select an installed **Recognition model**, then choose **All images** or **Current image** and set confidence, padding, and feather. If no compatible model is installed, the modal directs you to Settings → Preprocess. Model download and file management are not embedded in this modal.
3. Choose **Start**. The established NMS IoU default of 0.7 is used internally. Successful enqueue closes the modal while the page owns the background job.
4. When detection finishes, every successful, non-stale region is added directly to the corresponding mask layer as an unsaved edit. The editor switches to **Training mask** and **Brush** so the result can be corrected immediately. Failed, skipped, stale, and successful no-head images remain clean.
5. Each affected image receives one grouped history operation, regardless of how many heads it contains. Ctrl+Z removes that image's whole automatic addition; redo restores it. Manual mask brush and eraser edits continue on the same history.
6. Use the existing **Save current** or **Save all** actions to persist the mask. Until then, Filmstrip dirty state includes the automatic edits. Unsaved edits block starting another Auto mask run.
7. On the Train page, enable masked loss when required. Leap and NaViT Packing remain incompatible with masked loss and must be disabled first.

There is no proposal review panel, selectable box overlay, Apply action, or separate automatic-mask undo action. Proposal data is only an internal transport from the completed job to the existing mask editor.

## Parameters

- **Confidence**: higher values reduce false positives but may miss small heads. Default: 0.413.
- **Padding**: expands each detected rectangle by a percentage of its width and height. Default: 0.10 (10%).
- **Feather**: adds the exact configured transition outside the expanded rectangle. Default: 0.03 (3%).

Allowed ranges are confidence 0.01–0.99, padding 0–1, and feather 0–0.5. Empty or out-of-range values show field-local errors. NMS IoU is fixed at the established 0.7 default and is not shown in the modal.

## Safety

- Detection and result retrieval do not persist image or mask files. Save current / Save all are the only persistence boundaries.
- The result endpoint checks the source image identity. Results marked stale because dimensions, file size, or modification time changed are not incorporated.
- Project, version, and job identity guard asynchronous responses, so a late response cannot modify another workspace.
- Legacy proposal/apply/undo endpoints remain compatible for older clients, but this UI does not call apply or undo.
- This feature does not edit captions. Character names, hair colors, eye descriptions, and other identity tags still condition training; remove them manually on the Tagging page when required.

The built-in detector comes from `deepghs/anime_head_detection`. See [ADR 0018](../adr/0018-proposal-based-auto-head-mask.md) for its pinned revision and integrity contract, and `THIRD_PARTY_NOTICES.md` for licensing notes.
