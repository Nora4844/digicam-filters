"""
digicam_filters
===============

Small, readable reference implementations of the retro "old camera" photo
looks behind https://digicamfilter.online .

Each function applies one ingredient of the look (grain, bloom, vignette,
colour shift, fade, chromatic aberration, date stamp). They operate on a
NumPy float image in the range [0, 1] with shape (H, W, 3), and they are
written for clarity rather than speed, so the code doubles as an
explanation of how the effects actually work. Chain them to build a full
preset.

The browser version at digicamfilter.online does the same operations in
WebGL, with no install and no upload.

Dependencies: numpy, Pillow.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# --- I/O ----------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read an image as a float array in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: str) -> None:
    """Write a float array in [0, 1] back to disk."""
    out = np.clip(arr, 0.0, 1.0)
    Image.fromarray((out * 255).astype(np.uint8)).save(path)


# --- building blocks ----------------------------------------------------

def add_grain(img: np.ndarray, amount: float = 0.06, mono: bool = True) -> np.ndarray:
    """Film and sensor grain is, in essence, additive noise.

    Film grain and CCD sensor noise both add small random variations on top
    of the true signal. Modelling that as zero-mean Gaussian noise is crude
    but reads as convincingly analog, especially in flat areas like skies
    and shadows.

    mono=True uses one noise value per pixel across all channels (luminance
    grain, closer to film). mono=False adds independent noise per channel
    (chromatic noise, closer to a cheap digital sensor).
    """
    h, w, _ = img.shape
    if mono:
        noise = np.random.normal(0.0, amount, (h, w, 1))
    else:
        noise = np.random.normal(0.0, amount, (h, w, 3))
    return np.clip(img + noise, 0.0, 1.0)


def vignette(img: np.ndarray, strength: float = 0.4, radius: float = 0.7) -> np.ndarray:
    """Darken the corners with a smooth radial falloff.

    Cheap lenses let less light reach the edges of the frame, so corners
    come out darker. We build a mask from each pixel's distance to the
    centre and multiply the image by it. Nothing inside `radius` is touched;
    past it the image fades by up to `strength`.
    """
    h, w, _ = img.shape
    yy, xx = np.linspace(-1, 1, h), np.linspace(-1, 1, w)
    gx, gy = np.meshgrid(xx, yy)
    dist = np.sqrt(gx ** 2 + gy ** 2) / np.sqrt(2.0)
    t = np.clip((dist - radius) / (1.0 - radius), 0.0, 1.0)
    mask = 1.0 - strength * t ** 2
    return img * mask[:, :, None]


def white_balance(img: np.ndarray, warmth: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """Shift the colour temperature.

    Old cameras rarely nailed white balance. warmth > 0 pushes toward orange
    (more red, less blue); tint > 0 leans green, tint < 0 leans magenta.
    Useful range is roughly [-0.3, 0.3].
    """
    out = img.copy()
    out[..., 0] *= (1.0 + warmth)   # red
    out[..., 2] *= (1.0 - warmth)   # blue
    out[..., 1] *= (1.0 - tint)     # green
    return np.clip(out, 0.0, 1.0)


def fade(img: np.ndarray, amount: float = 0.12) -> np.ndarray:
    """Lift the blacks for a washed, faded-print look.

    A faded photo never reaches true black. We compress the range upward so
    the darkest pixel becomes `amount` instead of 0.
    """
    return img * (1.0 - amount) + amount


def contrast(img: np.ndarray, amount: float = 1.1) -> np.ndarray:
    """Scale contrast around mid-grey (0.5)."""
    return np.clip((img - 0.5) * amount + 0.5, 0.0, 1.0)


def saturation(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Scale saturation by mixing each pixel toward its luminance."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = (img * weights).sum(axis=2, keepdims=True)
    return np.clip(lum + (img - lum) * amount, 0.0, 1.0)


def bloom(img: np.ndarray, threshold: float = 0.7, intensity: float = 0.5,
          radius: float = 8.0) -> np.ndarray:
    """Make bright areas glow.

    CCD sensors and simple lenses bleed light around highlights. We isolate
    the bright pixels, blur that layer, and screen-blend the glow back on
    top. The screen blend, 1 - (1 - a)(1 - b), only ever brightens.
    """
    bright = np.clip(img - threshold, 0.0, 1.0) / max(1e-6, 1.0 - threshold)
    glow_img = Image.fromarray((bright * 255).astype(np.uint8))
    glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius))
    glow = np.asarray(glow_img, dtype=np.float32) / 255.0 * intensity
    return 1.0 - (1.0 - img) * (1.0 - glow)


def chromatic_aberration(img: np.ndarray, shift: int = 2) -> np.ndarray:
    """Offset the red and blue channels for old-lens colour fringing.

    A real lens focuses different wavelengths at slightly different points,
    so edges pick up a faint red/cyan fringe. Sliding the red and blue
    channels a couple of pixels apart fakes it well.
    """
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], shift, axis=1)    # red shifts right
    out[..., 2] = np.roll(img[..., 2], -shift, axis=1)   # blue shifts left
    return out


def date_stamp(img: np.ndarray, text: str = "'05 08 14") -> np.ndarray:
    """Burn the classic orange corner date stamp into the photo.

    The dot-matrix date in the bottom-right corner is the single most
    recognisable marker of a 2000s point-and-shoot. We draw it with PIL.
    For the authentic dot-matrix glyphs, drop a font like VT323 next to this
    file; otherwise it falls back to PIL's built-in font.
    """
    pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    size = max(14, int(min(w, h) * 0.05))
    try:
        font = ImageFont.truetype("VT323-Regular.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    margin = int(min(w, h) * 0.04)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = (w - margin - tw, h - margin - th)
    draw.text((pos[0] + 1, pos[1] + 1), text, fill=(0, 0, 0), font=font)        # shadow
    draw.text(pos, text, fill=(255, 159, 28), font=font)                       # camera orange
    return np.asarray(pil, dtype=np.float32) / 255.0


# --- preset recipes -----------------------------------------------------
# Each preset is just a chain of the building blocks above. Tweak the
# numbers to taste; that is exactly what the sliders on the website do.

def warm_digicam(img: np.ndarray) -> np.ndarray:
    """Warm, golden-hour point-and-shoot."""
    x = white_balance(img, warmth=0.10)
    x = saturation(x, 1.12)
    x = contrast(x, 1.06)
    x = bloom(x, threshold=0.72, intensity=0.35)
    x = add_grain(x, 0.04)
    x = vignette(x, strength=0.25)
    return x


def cool_ccd(img: np.ndarray) -> np.ndarray:
    """Cooler, glowing early-CCD compact."""
    x = white_balance(img, warmth=-0.06)
    x = saturation(x, 1.08)
    x = contrast(x, 1.10)
    x = bloom(x, threshold=0.65, intensity=0.55, radius=10)
    x = chromatic_aberration(x, shift=2)
    x = add_grain(x, 0.03)
    return x


def disposable(img: np.ndarray) -> np.ndarray:
    """Hard-flash disposable-camera snapshot."""
    x = white_balance(img, warmth=0.08)
    x = contrast(x, 1.18)
    x = bloom(x, threshold=0.60, intensity=0.70, radius=12)
    x = add_grain(x, 0.07, mono=False)
    x = vignette(x, strength=0.35)
    x = date_stamp(x, "'99 12 31")
    return x


def iphone4(img: np.ndarray) -> np.ndarray:
    """Soft, gently warm early-smartphone look."""
    x = white_balance(img, warmth=0.05)
    x = saturation(x, 0.95)
    x = fade(x, 0.06)
    x = bloom(x, threshold=0.78, intensity=0.25)
    x = add_grain(x, 0.035)
    return x


PRESETS = {
    "warm_digicam": warm_digicam,
    "cool_ccd": cool_ccd,
    "disposable": disposable,
    "iphone4": iphone4,
}


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    preset = sys.argv[2] if len(sys.argv) > 2 else "warm_digicam"
    if preset not in PRESETS:
        raise SystemExit(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")

    image = load_image(src)
    result = PRESETS[preset](image)
    out_path = f"out_{preset}.jpg"
    save_image(result, out_path)
    print(f"Wrote {out_path}")
