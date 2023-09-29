# DeepFaceLab - Optimized version

[**Russian version / Русская версия**](README-ru-RU.md)

---

# Installation

#### Method 1
- Run the module installation `python -m pip install numba==0.53.1` via `_internal\python_console.bat`.
- Copy the files from this repository inside `_internal\DeepFaceLab`, replacing the existing files.
- You're amazing! 🎉

#### Method 2

- Download the ready version via [torrent](DFL.torrent) (based on `DeepFaceLab_NVIDIA_up_to_RTX2080Ti_build_11_20_2021.exe`)

---

## New features ✨
- Now in step 7 (merge), saving in jpg format with quality setting 100 is available.
- Added video codec selection for merging images into video at step 8


## Performance changes 🚀


### Extraction (step 4 and 5)
Estimated speedup: **1.52х**  

Parameters:
- 2000 pictures (HD 1280x720, 1k with faces, 1k without faces)
- detector s3fd
- image-size 320
- jpeg-quality 100
- output-debug


### Sorting (step 4.2 and 5.2)
**Estimated speedup:**
- Blur: **9.72x** (less for a small number of images)
- Motion blur: **1.90x**
- Face yaw direction: **8.09x**
- Face pitch direction: **8.09x**
- Face rect size in source image: **9.15x**
- Histogram similarity: **1.32x** (less for a small number of images)
- Histogram dissimilarity: **3.00x** (more for a small number of images)
- Brightness: **2.29x**
- Hue: **2.29x**
- Amount of black pixels: **2.47x**
- Original filename: **9.58x**
- One face in image: **1.00x**
- Absolute pixel difference: **1.00x**
- Best faces: **9.88x**
- Best faces faster: **4.01x**

Parameters:
- 10000 images 320x320


### Training (step 6)
Small decrease in iteration time. I got this: -10ms (~4%) on the DF 160 model.


### Merging (step 7)
**Estimated speedup:**
- Prepare: **8.22x**
- Merge: **1.13x**

Parameters:
- 2000 pictures (HD 1280x720, 1k with faces, 1k skip without faces)
- Saving results in jpg format (in my version)
- Number of streams = number of virtual streams + 1

### Joining (step 8)
Depends on codec: h264, h265 and its versions accelerated with video card