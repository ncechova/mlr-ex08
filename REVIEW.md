# Review of the browser leaf classifier

I reviewed the files in this repository, especially `index.html`, `model_info.json`, `check_image.jpg` (via the self-check setup), `check_expected.json` and `README.md`. This is a source-code review. I could not independently operate a physical phone, so I have separated what the code shows from what still needs a real-device test.

## 1. Does it run on a phone and classify the live camera?

**Verified on Android. iPhone has not been tested.**

The app was opened on an Android phone and successfully classified a leaf using the live camera. The screenshot `Screenshot_leaf_id.jpg` shows the app running and its classification output, so there is now visual evidence for the mobile-camera requirement.

The implementation requests camera access through `navigator.mediaDevices.getUserMedia()` and prefers the rear-facing camera with `facingMode: "environment"`. It uses a video element with `playsinline`, and includes a photo-upload fallback that follows the same classification pipeline.

I do not have an iPhone, so I could not test Safari/iOS. The Android result confirms that the live camera workflow works on at least one real mobile device, but it should not be described as verified on both Android and iPhone.

## 2. Does the page give the same result as the notebook?

The repository contains `check_image.jpg` and `check_expected.json`. The page's self-check sends the image through the same `classify()` and `preprocess()` functions used for camera frames and uploaded photos, then compares the resulting probabilities with the expected values.

The expected values in `check_expected.json` are:

| Class | Expected probability |
|---|---:|
| Apple | 0.0000000539415 |
| Cherry | 0.9999644756 |
| Chestnut | 0.0000350894 |
| Maple | 0.0000004611 |

The README records a reference browser run with a maximum difference of approximately `8.1 × 10⁻⁸`, which is comfortably below the stated `1 × 10⁻³` tolerance. That is consistent with agreement to at least three decimal places.

I verified that the self-check is wired to the same inference path and that the expected-value file is present. I did not rerun the deployed page in a browser during this review, so the README's measured browser result should be confirmed once more after the final model export.

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

- GitHub Pages serves the page and the repository files, including `leaf_model.onnx`, `model_info.json`, `check_image.jpg` and `check_expected.json`.
- The ONNX Runtime Web JavaScript module and WebAssembly runtime are loaded from the jsDelivr CDN at `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/`.
- The QR encoder is loaded from the local `qrcode.js` file in the repository. The QR code is generated in the browser; the page does not use an external QR-image service.

Therefore, the image itself stays on the device during classification, but the hosting and CDN services receive ordinary asset requests and may see request metadata such as the IP address and requested file. It would be more accurate to say **"images are not uploaded; inference runs locally"** than to say that the phone makes no network requests at all.

## Fixes and remaining checks

I did not find a preprocessing or image-upload issue that required a code change during this review. The main outstanding item is evidence from a real phone.

Before submitting, I would:

- Test the deployed page on at least one phone, preferably both Android and iPhone.
- Add a screenshot showing a live leaf classification.
- Run the model self-check on the deployed page after the final ONNX export and confirm that every probability differs from the expected value by less than 0.001.
- Confirm that `leaf_model.onnx` contains the weights from the selected `group_plus_archive_all` checkpoint. The architecture is compatible, but the report's best training result only applies to the browser app if those weights were actually exported and committed.
