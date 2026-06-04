# digicam-filters

Reference Python implementations of the retro "old camera" photo looks: grain, bloom, vignette, colour shift, fade, chromatic aberration, and the classic orange date stamp.

These are the same effects that power [digicamfilter.online](https://digicamfilter.online), a free in-browser tool that needs no install and never uploads your photos. This repo is the explained, hackable version: each effect is a short NumPy function you can read, run, and remix.

## Why this exists

Modern phone photos are sharp, evenly lit, and a little lifeless. The looks people miss come from older cameras getting things "wrong": grain in the shadows, highlights that bloom instead of clipping flat, colour that drifts warm or cool, and a date burned into the corner. None of it is complicated once you see it as a few small operations stacked together. That is what this code shows.

## Quick start

```bash
pip install -r requirements.txt
python digicam_filters.py your_photo.jpg disposable
```

That writes `out_disposable.jpg`. Swap in any preset: `warm_digicam`, `cool_ccd`, `disposable`, `iphone4`.

Or use the pieces directly:

```python
from digicam_filters import load_image, save_image, white_balance, bloom, add_grain, vignette

img = load_image("your_photo.jpg")
img = white_balance(img, warmth=0.1)
img = bloom(img, threshold=0.7, intensity=0.4)
img = add_grain(img, 0.05)
img = vignette(img, strength=0.3)
save_image(img, "out.jpg")
```

## How the effects work

Every function takes a float image in `[0, 1]` and returns one. The short version of each:

- **Grain** is additive Gaussian noise. Luminance noise (one value per pixel) reads like film; per-channel noise reads like a cheap sensor.
- **Vignette** multiplies the image by a radial mask so the corners fall off. Cheap lenses do this for real.
- **White balance** scales the red and blue channels in opposite directions for warmth, and the green channel for tint. Old cameras rarely got this right, which is half the charm.
- **Fade** lifts the blacks so the darkest pixel is no longer true black, the way an old print looks.
- **Bloom** isolates the bright pixels, blurs them, and screen-blends the glow back on top. That is why CCD highlights seem to leak light.
- **Chromatic aberration** slides the red and blue channels a couple of pixels apart, faking the colour fringe a real lens leaves at high-contrast edges.
- **Date stamp** draws the orange date in the corner. Drop a font like VT323 next to the file for the authentic dot-matrix glyphs.

A preset is just a chain of these with specific numbers. `disposable`, for instance, is warm balance, high contrast, strong bloom, chromatic noise, a vignette, and a `'99 12 31` stamp. Adjusting those numbers is exactly what the sliders on the website do.

## The no-install version

If you just want the look without writing any code, [digicamfilter.online](https://digicamfilter.online) runs all of this in your browser with presets, sliders, a before/after compare, and the date stamp. Nothing is uploaded; the editing happens on your device.

## License

MIT. Use it, learn from it, build on it.
