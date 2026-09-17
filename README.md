This is an implementation of an encryption algorithm for images using compressive sensing, this is meant to be more of a proof of concept where we use compressive sensing in a novel way to encrypt images. CS is typically used to sample signals in an already compressed format but a by-product of this process is that the signal also becomes uniquely encoded upon measurement, this can be used for encryption .

The overall algorthm for this is as follows : 

```
Inputs : 𝜓 measurement matrix, 𝑥 vectorized image to be encrypted, 𝑁 signal length of 𝑥, 𝑘number of iterations for ADM, 𝜏, 𝛽 parameters for ADM, 𝑒𝑘, θ auxiliary vectors

// encryption step
𝑦 ← 𝜓𝑥 // extracting the measurements, this effectively encrypts the signal 𝑥 into y

//decryption step
For 𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // construct dictionary 𝐴
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝐴 : , 𝑖 = 𝜓 ∗ θ
End For

𝑠 = 𝐴𝐷𝑀(𝐴, 𝑦, 𝜏,𝛽, 𝑘) // solves the 𝐴 * 𝑥 = 𝑦 equation, where 𝑥 is the unknown encrypted signal using LMBFGS

For  𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // reconstruct signal 𝑥 which now represents the decrypted signal 𝑥′
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝑥′ = 𝑥′ + θ ∗ 𝑠(𝑖)
Sf - End

Outputs : 𝑥′ vectorized decrypted image
```

This works only if the  measurement matrix ```𝜓``` is identical upon encryption and decryption. ```𝜓``` is meant to be a random matrix but by using a deterministic number generator which is seeded using a passphrase we can encrypt and decrypt an arbitrary signal. Keep in mind that this method is not lossless, the reconstructed signal will not be 100% identical, this is why you'd only want to use something like this for things like images. ```𝑦``` represents the encrypted image, since it's obtained by multiplying the original image with a random matrix it will contain a bunch of seemeingly random numbers.

Because ```𝑥``` is a vectorized image which means it can have millions of elements the dictionary ```𝐴``` is going to be a matrix with potentially billions of elements (so dozens of GB in size). The challenge in doing something like this comes from the fact that the matrices involved occupy so much memory that it's impossible to solve this problem on a regular computer as is, however, we can divide the original image in smaller chunks that can fit in the memory of a typical computer. 

GPU acceleration no longer needed since switching to Limited-memory BFGS using [this](https://github.com/chokkan/liblbfgs) library. This brought unpon a huge speed increase and lower memory consumption. 

This method processes the image in tiles, it should be noted that this is technically not equivalent to solving this problem for one single large image, however for something like images it works quite well and can even improve quality in some ways (lower noise) when the compression ratio is higher.

## Security design (v2 container format)

The passphrase is the single root secret. It is stretched with **PBKDF2-HMAC-SHA256** (Windows CNG, 200,000 iterations, 16-byte random per-image salt) into a 256-bit key. The old `std::hash`-based `generate_seeds` chain (hash-then-pick-then-reseed funnel) was removed; the derived key is the only seeding input, so there is no accidental entropy bottleneck and offline passphrase guessing is deliberately expensive.

The 256-bit key is used for three things:

- **Header encryption** — the metadata header (`m|rows|cols|height|width`) is AES-256-CTR encrypted, so image dimensions are no longer recoverable from the ciphertext without the passphrase.
- **Authentication** — an HMAC-SHA256 tag (truncated to 128 bits), computed encrypt-then-MAC over the encrypted header *and* the entire measurement body, is stored in the header. Decryption verifies it first: a wrong password or any tampering with the container is rejected before the expensive L-BFGS solve starts.
- **Index regeneration** — MT19937 is seeded via `std::seed_seq` directly from the key bytes; the pixel shuffle seed is the first 4 key bytes. Both sides derive identical sampling indices from passphrase + salt alone.

Header layout (v2, 32 pixels = 96 bytes at the start of the container):

```
[0]      version byte (2)
[1..16]  PBKDF2 salt (random, per image)
[17..32] AES-CTR IV (random, per image)
[33..64] metadata (32 bytes) XOR AES-256-CTR keystream
[65..80] HMAC-SHA256 tag, truncated to 16 bytes, over [0..64] + measurement body
[81..95] reserved (zeroed)
```

Security caveats (this is still a proof of concept):

- **MT19937 is not cryptographically secure.** With 624 consecutive known outputs an attacker can recover the generator state. The PBKDF2 KDF keeps guessing the passphrase expensive, but the sampling stream itself must not be trusted against an adversary who can infer measurement values. A cryptographically secure stream (ChaCha20 / AES-CTR) driving the index selection would close this.
- Key material (`cs_key`) is wiped with a volatile write loop after sealing (encrypt) and in the `CSencryption` destructor (decrypt), but passphrase copies held in `std::string` by callers are outside this module's control.
- **Format change**: images encrypted with the previous format (11-pixel plaintext header, no version byte, `std::hash` seeding) can no longer be decrypted and are rejected with a legacy-format error — re-encrypt them.

Performance note : the PBKDF2 stretch adds ~0.2–0.4s per encrypt/decrypt operation (once per image, not per tile), which is negligible next to the solve time.

## Building

Requirements: C++17 compiler, OpenCV (developed against 4.13.0; any 4.x with core, imgproc, imgcodecs, highgui, photo), OpenMP, and a crypto backend (Windows CNG is built in; Linux/macOS use OpenSSL). x86-64 with AVX2+FMA is assumed for the solver kernels.

### Visual Studio (Windows)

1. Open `ImgReconstruct_backend.sln` (MSVC v143 / VS 2022 or newer).
2. The project expects OpenCV headers/libs at `C:\opencv\include` and `C:\opencv\lib` (see the `IncludePath`/`LibraryPath` entries in `ImgReconstruct_backend.vcxproj`); adjust if yours differs.
3. Build `Release | x64`. `bcrypt.lib` (CNG) is linked automatically.

### CMake + vcpkg (Windows/Linux)

A vcpkg manifest is included (`vcpkg.json`: `opencv4`, `openssl`).

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build            # runs the cs_tests suite
```

Alternatively point CMake at any OpenCV with CMake config files via `-DOpenCV_DIR=<path>`; on Windows the CNG backend means OpenSSL is not required.

### Test suite

`cs_tests` covers: PBKDF2/seal/verify unit tests, wrong-password and tamper rejection, seed determinism (same password+salt ⇒ same key/indices), shuffle determinism, tile-grid helpers, a synthetic encrypt→decrypt roundtrip scored by PSNR (~19.2 dB baseline), and an **opt-in photo roundtrip** that runs a real photo (auto-found: `$CS_TEST_PHOTO`, `IMG_3690.png`, or the project sample) for a second data point on natural image statistics (~16.7 dB baseline) and skips cleanly when no photo is available. Run with `ctest` or directly (`build\cs_tests.exe`).

### CLI usage

```
ImgReconstruct_backend encrypt  <input.png> <output.png> [--password <pw>] [--ratio R]
ImgReconstruct_backend decrypt  <input.png> <output.png> [--password <pw>] [--tiles N] [--overlap N] [--iterations N] [--threads N] [--coef F] [--manual] [--no-preview] [--denoise]
ImgReconstruct_backend roundtrip <input.png> <output.png> [options]
```

Without `--password` the passphrase is read from `CS_ENCRYPTION_PASSWORD` or prompted. Parameter overrides (`--tiles` etc.) switch the decrypt to manual mode; otherwise parameters are derived automatically from the container. `--denoise` adds an opt-in final non-local-means pass (`cv::fastNlMeansDenoisingColored`) that smooths solver noise at the cost of fine detail.

Quality notes: tiles are composited with a **cosine-feathered** weight ramp (width = tile overlap) instead of a fixed alpha blend — overlap zones sum to a smooth transition and the composite is order-independent. Tiles are solved in **wavefront (anti-diagonal) order**: every tile is warm-started from the already-solved west/north neighbors' overlap strips, which turned out to be by far the largest quality lever — roundtrip baselines went from ~19.2/16.7 dB (independent solves) to **~26.8/30.4 dB** (synthetic/photo) with no wall-time penalty, since the better warm-starts converge faster than the wave barriers cost. A YCrCb-domain solve was also tried (smoother chroma, softened chroma l1, neutral chroma warm-start) but measured ~15.6 dB in the old configuration — see the experiment notes in `decrypt_image::decrypt` for the details before re-attempting it (worth revisiting on top of the wavefront warm-starts).

Performance : 

~1.0 seconds to decompress and decrypt a 4032 X 3024 image on a Ryzen 7900 after adding upscaling using [this library](https://github.com/avaneev/avir).

TODO List : 

- [ ] Improve quality of decrypted image, right now there are fair amount of artifacts upon closer inspection.

Example :

Original image :

![IMG_36902](https://github.com/user-attachments/assets/52e80e7a-c58d-4466-9c72-11882fc827a3)

Encrypted image : 

![encrypted_img_g2](https://github.com/user-attachments/assets/639e7ad4-c9aa-4c1d-ab6e-74e2bdd701a4)

Decrypting the image :

![unknown_2025 04 04-22 37_1-ezgif com-cut](https://github.com/user-attachments/assets/37cdb048-2e94-4be1-9b65-555813523fde)

