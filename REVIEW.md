# Review of the browser leaf classifier

I reviewed the files in this repository, especially `index.html`, `model_info.json`, `check_image.jpg` (via the self-check setup), `check_expected.json` and `README.md`. This is a source-code review. I could not independently operate a physical phone, so I have separated what the code shows from what still needs a real-device test.

## 1. Does it run on a phone and classify the live camera?

**The app has been tested on Android; iPhone/Safari has not been tested.**

The Android app works. However, the committed screenshot `Screenshot_leaf_id.jpg` shows the photo-picker fallback, not a live camera stream: the camera button is idle and the source label shows a selected image filename and its dimensions. The screenshot therefore documents the app running on Android and classifying an uploaded photo, but it is not evidence of the live-camera stream itself.

The live-camera implementation is present in `index.html`: `openCamera()` calls `navigator.mediaDevices.getUserMedia()` and prefers the rear-facing camera with `facingMode: "environment"`; the video element uses `playsinline`; and `startCamera()` starts the inference loop. The separate photo-picker fallback uses the same `classify()` and `preprocess()` functions.

I do not have an iPhone, so Safari/iOS remains untested. A separate screenshot with the camera actually running would make the live-camera evidence clearer.

## 2. Does the page give the same result as the notebook?

The repository contains `check_image.jpg` and `check_expected.json`. The page's self-check sends the image through the same `classify()` and `preprocess()` functions used for camera frames and uploaded photos, then compares the resulting probabilities with the expected values.

The self-check runs `check_image.jpg` through the page's normal `classify()` and `preprocess()` path, then compares the result with `check_expected.json`. The committed Android screenshot shows the self-check passing with a maximum absolute difference of approximately `8.1 × 10⁻⁸`, below the page's `1 × 10⁻³` tolerance.

The values displayed by the page and the reference values are:

| Class | This page (probability) | `check_expected.json` | Absolute difference |
|---|---:|---:|---:|
| apple | `5.39 × 10⁻⁸` | `5.394148 × 10⁻⁸` | `≈ 1 × 10⁻¹²` |
| cherry | `0.999964` | `0.9999644756` | `≈ 8.1 × 10⁻⁸` |
| chestnut | `3.5 × 10⁻⁵` | `3.508940 × 10⁻⁵` | `≈ 9.6 × 10⁻¹⁰` |
| maple | `4.61 × 10⁻⁷` | `4.611038 × 10⁻⁷` | `≈ 1.5 × 10⁻¹¹` |

The page formats very small probabilities in scientific notation and the cherry probability as a percentage; the table above writes all four as probabilities so they can be compared directly. The screenshot is evidence that the deployed page's self-check passed on Android. The remaining caveat is that this screenshot was taken with a selected photo, not with a live camera stream.

## 3. Does preprocessing match the model?

**Yes. The implementation in `index.html` matches the preprocessing values in `model_info.json` and the notebook's normalization transform.**

The model description specifies:

- Input size: 96 × 96
- Classes: apple, cherry, chestnut and maple
- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

In `preprocess()`, the page:

1. Takes the largest centred square from the source frame.
2. Resizes that square to 96 × 96 using a canvas.
3. Reads the RGB pixel values and divides each channel by 255.
4. Standardizes each channel using its corresponding mean and standard deviation.
5. Stores the channels in RGB order in a `Float32Array`.
6. Creates a float32 tensor with shape `[1, 3, 96, 96]`, i.e. NCHW layout.

This is consistent with the notebook's `ToTensor()` followed by `Normalize(mean, std)`. The notebook images are already prepared at the model's input size; the browser has to crop and resize live camera frames before applying the same scaling and normalization.

I did not find a channel-order mismatch or a missing normalization step in the inspected code.

## 4. Does image data leave the phone?

**I found no code that uploads camera frames or selected photos to a server. Inference is performed locally in the browser.**

The page uses the camera stream locally, draws the selected frame to a canvas and passes the resulting tensor to ONNX Runtime Web. The photo-upload fallback follows the same local path. I found no image POST request, remote inference endpoint or code that sends frame pixels to another service.

The browser does still make normal requests to download the page's resources:

- GitHub Pages serves the page itself and same-origin assets: `leaf_model.onnx`, `model_info.json`, `check_image.jpg` (loaded as an image by the self-check), `check_expected.json`, `qrcode.js` and `favicon.svg`.
- The ONNX Runtime Web JavaScript module and WebAssembly runtime are loaded from the jsDelivr CDN at `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/`.
- The QR encoder is loaded from the local `qrcode.js` file. The QR code is generated in the browser; the page does not use an external QR-image service.

The inspected page code contains no `XMLHttpRequest`, `sendBeacon`, WebSocket, or POST request for camera frames or selected photos. The browser must still download the app and model assets, and the hosting/CDN services can see normal request metadata.

Therefore, the image itself stays on the device during classification, but the hosting and CDN services receive ordinary asset requests and may see request metadata such as the IP address and requested file. It would be more accurate to say **"images are not uploaded; inference runs locally"** than to say that the phone makes no network requests at all.

## Fixes and remaining checks

The preprocessing contract matches the notebook and `model_info.json`, and the self-check code uses the same inference path as the app. The screenshot documents Android use and a passing self-check, but it shows photo upload rather than a live camera stream.

The main outstanding requirement is model deployment. The current `leaf_model.onnx` is the Task 2 export, not a demonstrated export of `group_plus_archive_all`. The Task 4 results file stores metrics and per-epoch histories, not trained weights, and `task4.py` does not save a checkpoint. Therefore, the selected Task 4 model is not yet running on the live page.

Before the hand-in, the remaining work is:

- Re-run `group_plus_archive_all` with the original group and archive training images, saving the final model weights (the repository does not contain those training datasets or a saved checkpoint).
- Export those weights to ONNX using the Task 2e export procedure, replace `leaf_model.onnx`, and regenerate `check_expected.json` for the newly exported model.
- Run the page's self-check again and confirm the maximum per-class difference is below `1 × 10⁻³`.
- If possible, add a screenshot with the live camera stream running. The existing screenshot is a photo-upload example.
- Test on iPhone/Safari only if an iPhone becomes available; Android is the device that has been tested.
