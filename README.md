# MLR exercise 08 — live leaf classifier in the browser

**Open it here: <https://ncechova.github.io/mlr-ex08/>**

Point an iPhone (or any phone/laptop) rear camera at a leaf and the page classifies it
live, in the browser. Nothing is uploaded — the model runs locally on the device via
ONNX Runtime Web (WebAssembly).

<table>
  <tr>
    <td>
      <a href="https://ncechova.github.io/mlr-ex08/">
        <img src="qr.svg" width="222" height="222"
             alt="QR code that opens https://ncechova.github.io/mlr-ex08/ on a phone">
      </a>
    </td>
    <td>
      <strong><a href="https://ncechova.github.io/mlr-ex08/">ncechova.github.io/mlr-ex08</a></strong><br>
      Scan with the phone's camera app, or follow the link.<br>
      The page draws this same code at its own foot.
    </td>
  </tr>
</table>

| | |
|---|---|
| Live page | <https://ncechova.github.io/mlr-ex08/> |
| Model | `leaf_model.onnx` (4 classes, 96×96 input, ONNX opset 20) |
| Classes | apple · cherry · chestnut · maple |
| Runtime | [onnxruntime-web 1.31.1](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/) from the jsDelivr CDN, wasm EP, single thread |
| Build step | none — one `index.html`, plain JavaScript |

## Using it

1. Open the link on the phone (or scan the QR code at the bottom of the page).
2. Press **Start camera** and allow camera access.
3. Hold a leaf inside the square viewfinder — the centre square of the camera frame is
   what gets scored, and the viewfinder shows exactly that region.
4. All four class probabilities update live under the viewfinder, together with the
   processing rate.
5. No camera, or permission denied? **Choose photo** runs the same model over an image
   from the library, through identical code.
6. **Model self-check** (expandable panel) verifies this page against
   `check_expected.json` — see below.

## Preprocessing contract

Taken verbatim from `model_info.json` and applied identically to camera frames, uploaded
photos and the self-check image:

```
camera frame (video frame / photo)
  └─ central square            min(width, height), centred, floor offsets
     └─ resize → 96 × 96       canvas bilinear (imageSmoothingQuality: high)
        └─ RGB bytes → 0..1    byte / 255
           └─ per channel      (x − mean) / std
              mean = [0.485, 0.456, 0.406]
              std  = [0.229, 0.224, 0.225]
              └─ NCHW float32 tensor, shape [1, 3, 96, 96]  (input name "image")
                 └─ model output "logits" [1, 4] → softmax → 4 probabilities
```

The model outputs logits, so the softmax lives in the page (`index.html`, `softmax()`).

## Self-check against `check_expected.json`

The page runs `check_image.jpg` through its own pipeline and prints a per-class
comparison. Measured on the reference run:

| class | this page | expected | difference |
|---|---|---|---|
| apple | 5.39×10⁻⁸ | 5.394148×10⁻⁸ | ≈ 1×10⁻¹² |
| cherry | 99.9964 % | 99.996448 % | ≈ 8×10⁻⁸ |
| chestnut | 0.0035 % | 3.50894×10⁻⁵ | ≈ 9.6×10⁻¹⁰ |
| maple | 4.61×10⁻⁷ | 4.611038×10⁻⁷ | ≈ 1.5×10⁻¹¹ |

Largest per-class deviation **8.1×10⁻⁸**, i.e. agreement to about seven decimal places;
the check passes when the top class matches and every deviation stays under 1×10⁻³. The
residual difference comes from the browser's JPEG decoder and float32 rounding order, not
from the pipeline — `check_image.jpg` is already 96×96, so the crop and the resize are
both identities for this image.

The same contract was verified against an independent reference implementation
(Python + Pillow + onnxruntime 1.31.1 CPU): max deviation 7.4×10⁻¹⁰.

## Requirements

* **HTTPS** — `getUserMedia` is only available in a secure context, which is what GitHub
  Pages provides. (`http://localhost` also works for development.)
* **WebAssembly SIMD** — the runtime build needs it: iOS/iPadOS 16.4+, Safari 16.4+,
  Chrome/Edge 91+, Firefox 90+.
* First load downloads 3.2 MB of wasm runtime (3,196,527 B of wasm plus a 37 KB loader)
  and 5.7 MB of model — 8.9 MB in total, all measured in the browser's network log. After
  that the browser cache serves both.

iPhone notes: the video element uses `playsinline` and the camera is requested with
`facingMode: ideal "environment"`, falling back to plainer constraints on devices that do
not report a rear lens. Processing is capped at 8 frames/s and pauses when the tab is
hidden. GitHub Pages cannot send COOP/COEP headers, so the session is pinned to
`numThreads = 1` — no SharedArrayBuffer, no cross-origin isolation needed.

## QR code

The QR code at the bottom of the page is generated **inside the browser tab** by
`qrcode.js` (QR Code Generator for JavaScript, © 2009 Kazuhiko Arase, MIT licence, vendored
unmodified in this repository — see its header comment). No QR-web-service is involved:
the page never sends the URL anywhere. It encodes the canonical Pages URL
`https://ncechova.github.io/mlr-ex08/` and sits next to the same URL as a clickable link.

`qr.svg` at the top of this README is that same symbol, exported from what the deployed
page renders and committed as a static file, because GitHub renders README markup without
running any JavaScript. It is module-for-module identical to the on-page code — same payload,
same 29×29 modules, same 4-module quiet zone.

## Files

| file | role |
|---|---|
| `index.html` | the whole app: UI, preprocessing, inference, self-check, QR |
| `qrcode.js` | vendored QR encoder (unmodified upstream, MIT) |
| `qr.svg` | the same QR code as a static file, for this README |
| `leaf_model.onnx` | the trained model |
| `model_info.json` | class names, input size, mean/std — the preprocessing contract |
| `check_image.jpg` | verification image (96×96) |
| `check_expected.json` | reference probabilities for that image |
| `favicon.svg` | icon |
| `.nojekyll` | tells GitHub Pages to serve the files as they are (no Jekyll step) |
| `REPORT.md` | exercise Task 4 — which change brought what, and which model goes on the phone |
| `task4.py` | the experiment runner behind that report (15 variants, three validation sets) |
| `task4_results.json` | every measured run behind the report, per epoch |

## Running it locally

Any static server works; `getUserMedia` needs `localhost`, not a `file://` path:

```bash
python -m http.server 8000
# then open http://localhost:8000/
```

## Repository settings

Public repository, served by GitHub Pages from the `main` branch (root of the branch).
`MLRteacher` is invited as a collaborator.
