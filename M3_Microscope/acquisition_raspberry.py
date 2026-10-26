from picamera2 import Picamera2
import numpy as np, cv2, time
from concurrent.futures import ThreadPoolExecutor

# ---------- Paramètres ----------
EXPOSURE_US = 90_000          # temps d'exposition (µs)
GAIN = 1.0
N = 30                        # nombre d'images (voir note mémoire)
W, H = 4056, 3040
ROI = None                    # ex. (y0, y1, x0, x1) pour ne garder qu'une région

# Durée d'image : 100 ms minimum en pleine résolution (10 fps max)
FRAME_US = max(100_000, EXPOSURE_US + 2_000)
print(f"fps visé = {1e6/FRAME_US:.1f}")

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},   # petit flux traité (aperçu)
    raw={"size": (W, H), "format": "SBGGR12"},       # flux raw pleine résolution
    buffer_count=4,
    controls={
        "FrameDurationLimits": (FRAME_US, FRAME_US),
        "ExposureTime": EXPOSURE_US,
        "AnalogueGain": GAIN,
    },
)
picam2.configure(config)
picam2.start()
time.sleep(1)                 # laisse les réglages se stabiliser

frames, stamps, expos = [], [], []
t0 = time.time()
for i in range(N):
    req = picam2.capture_request()
    raw = req.make_array("raw").view(np.uint16)[:H, :W]
    if ROI is not None:
        y0, y1, x0, x1 = ROI
        raw = raw[y0:y1, x0:x1]
    frames.append(raw.copy())
    md = req.get_metadata()
    stamps.append(md["SensorTimestamp"])
    expos.append(md["ExposureTime"])
    req.release()
picam2.stop()
print(f"Acquisition terminée en {time.time()-t0:.1f} s", flush=True)
print("forme:", frames[0].shape, frames[0].dtype, "max:", frames[0].max())

def save(i):
    cv2.imwrite(f"frame_{i:04d}.png", frames[i], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"sauvegardé {i+1}/{N}", flush=True)

with ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(save, range(N)))

np.save("timestamps_ns.npy", np.array(stamps))
np.save("exposure_us.npy", np.array(expos))