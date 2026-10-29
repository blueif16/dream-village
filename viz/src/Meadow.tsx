// The meadow behind the site: a field of warm colour in pixel blocks, ordered-dithered between small steps of a
// meadow palette and drifting on slow periodic paths (warm light above, green ground below). The town's window
// lights the field round its edge, a few sparks run its rim and motes drift in its light. Vivid on the
// landing page; soft behind the page about the town; calm (pale and still) round the town itself, so nothing pulls
// the eye from the simulation.
import { useEffect, useRef } from 'react'

export type MeadowMode = 'vivid' | 'soft' | 'calm'

const CELL = 4 // css px per pixel block
const FPS = 14

type RGB = [number, number, number]
const rgb = (h: number): RGB => [(h >> 16) & 255, (h >> 8) & 255, h & 255]
const mix = (a: RGB, b: RGB, f: number): RGB => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
const pack = ([r, g, b]: RGB) => (255 << 24) | (Math.round(b) << 16) | (Math.round(g) << 8) | Math.round(r) // RGBA bytes, little-endian

// warmth, from the ground up: moss, forest, leaf, lime, sun, marigold, ember
const RAMP = [0x173d2a, 0x2c6a3c, 0x5c9c3e, 0xb9d65a, 0xf6cd4c, 0xf1913a, 0xdf5f2e].map(rgb)
const BLOSSOM = rgb(0xf3b5c2)
const LIGHT = rgb(0xfff4d2)

// Every block takes one tone of a small table: the warmth in 13 steps (the ramp and the halfway steps between its
// colours), each with two blossom tints and four steps of the window's light. Each axis is dithered on its own.
const STEPS = 12
const HAZE = 0.2 // a little of the light over the whole field, so it reads airy rather than dense
const PINKS_LV = 2
const LIGHT_LV = 4
const PINK_MIX = [0, 0.4, 0.72]
const LIGHT_MIX = [0, 0.3, 0.55, 0.8, 1]
function rampAt(s: number): RGB {
  const x = s * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(x))
  return mix(RAMP[i], RAMP[i + 1], x - i)
}
const TABLE = new Uint32Array((STEPS + 1) * (PINKS_LV + 1) * (LIGHT_LV + 1))
for (let r = 0; r <= STEPS; r++)
  for (let p = 0; p <= PINKS_LV; p++)
    for (let g = 0; g <= LIGHT_LV; g++)
      TABLE[(r * (PINKS_LV + 1) + p) * (LIGHT_LV + 1) + g] = pack(mix(mix(rampAt(r / STEPS), BLOSSOM, PINK_MIX[p]), LIGHT, LIGHT_MIX[g]))
const tone = (r: number, p: number, g: number) => TABLE[(r * (PINKS_LV + 1) + p) * (LIGHT_LV + 1) + g]
const BRIGHT = tone(STEPS - 8, 0, LIGHT_LV) // the light itself
const WARM = tone(STEPS - 4, 0, 2) // a sun-warm spark tail

// a 4×4 ordered dither; each axis reads it at its own offset so their patterns don't line up
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16)
const th = (x: number, y: number) => BAYER[(y & 3) * 4 + (x & 3)]

const TAU = Math.PI * 2
const fract = (x: number) => x - Math.floor(x)
const hash = (i: number) => fract(Math.sin(i * 127.1 + 311.7) * 43758.5453)

/** Slow drifting bodies of warmth (+) and green (−); each loops on its own period, in seconds. */
const BLOBS = [
  { u: 0.18, v: 0.2, ru: 0.16, rv: 0.08, r: 0.3, k: 0.22, p: 71 },
  { u: 0.8, v: 0.26, ru: 0.12, rv: 0.1, r: 0.26, k: 0.16, p: 89 },
  { u: 0.55, v: 0.55, ru: 0.3, rv: 0.06, r: 0.3, k: -0.12, p: 103 },
  { u: 0.14, v: 0.86, ru: 0.1, rv: 0.04, r: 0.2, k: 0.1, p: 61 },
]
const PINK_BODIES = [
  { u: 0.88, v: 0.12, ru: 0.08, rv: 0.06, r: 0.16, p: 97 },
  { u: 0.04, v: 0.42, ru: 0.03, rv: 0.1, r: 0.1, p: 83 },
]
const SPARKS = [0, 0.27, 0.58, 0.81].map((o, i) => ({ o, speed: [74, 58, 90, 66][i], len: 9 }))
// motes: single blocks that drift slowly round the window in its light, nearer and further, twinkling
const MOTES = Array.from({ length: 34 }, (_, i) => ({ o: hash(i), drift: (hash(i + 50) - 0.5) * 3, ph: hash(i + 130) * TAU, hot: i % 3 === 0 }))

/** One frame of the field into `px` (cols × rows), at time `t` seconds, lit by `win` (in cells) if given. */
function paint(px: Uint32Array, cols: number, rows: number, t: number, win: [number, number, number, number] | null) {
  const blobs = BLOBS.map((b) => {
    const a = (TAU * t) / b.p
    return [b.u + b.ru * Math.cos(a), b.v + b.rv * Math.sin(a * 1.3), 1 / (b.r * b.r), b.k]
  })
  const pinks = PINK_BODIES.map((b) => {
    const a = (TAU * t) / b.p
    return [b.u + b.ru * Math.sin(a), b.v + b.rv * Math.cos(a), 1 / (b.r * b.r)]
  })
  const breathe = 0.86 + 0.14 * Math.sin((TAU * t) / 7)
  const reach = 40 / CELL // the light's fall-off, in cells
  const aspect = cols / rows
  // Each body is a gaussian, which is separable: its fall-off along x and along y, once a frame. So is the wind,
  // sin(A + B) = sin A cos B + cos A sin B. The light reads a table by squared distance.
  const along = (n: number, f: (i: number) => number) => Float32Array.from({ length: n }, (_, i) => f(i))
  const bx = blobs.map(([bu, , ir]) => along(cols, (x) => Math.exp(-(((x / cols - bu) * aspect) ** 2) * ir)))
  const by = blobs.map(([, bv, ir]) => along(rows, (y) => Math.exp(-((y / rows - bv) ** 2) * ir)))
  const px2 = pinks.map(([bu, , ir]) => along(cols, (x) => Math.exp(-(((x / cols - bu) * aspect) ** 2) * ir)))
  const py2 = pinks.map(([, bv, ir]) => along(rows, (y) => Math.exp(-((y / rows - bv) ** 2) * ir)))
  const ws = along(cols, (x) => Math.sin((x / cols) * 7.5 - t * 0.35))
  const wc = along(cols, (x) => Math.cos((x / cols) * 7.5 - t * 0.35))
  const lean = along(cols, (x) => (x / cols - 0.5) * 0.06)
  const far = Math.ceil(reach * 6)
  const lit = along(far * far * 2 + 1, (d2) => Math.exp(-Math.sqrt(d2) / reach) * breathe)
  const K = blobs.map((b) => b[3])
  const nb = blobs.length
  const np = pinks.length
  for (let y = 0; y < rows; y++) {
    const v = y / rows
    // warm light above, green ground below, a slight lean, and a wind of long bands crossing it
    const base = 0.92 - v * 0.56 // marigold at the top down to leaf and lime: a light meadow, never a deep one
    const vs = 0.04 * Math.sin(v * 4)
    const vc = 0.04 * Math.cos(v * 4)
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x
      let g = 0
      if (win) {
        const dx = Math.max(win[0] - x, 0, x - win[2])
        const dy = Math.max(win[1] - y, 0, y - win[3])
        if (dx === 0 && dy === 0) {
          px[i] = BRIGHT // under the town
          continue
        }
        if (dx < far && dy < far) g = lit[dx * dx + dy * dy]
      }
      g = HAZE + g * (1 - HAZE)
      let s = base + lean[x] + ws[x] * vc + wc[x] * vs
      for (let k = 0; k < nb; k++) s += K[k] * bx[k][x] * by[k][y]
      let pk = 0
      for (let k = 0; k < np; k++) pk += px2[k][x] * py2[k][y]
      const r = Math.min(STEPS, Math.floor(Math.min(1, Math.max(0, s)) * STEPS + th(x, y)))
      const p = Math.min(PINKS_LV, Math.floor(Math.min(0.9, pk) * PINKS_LV + th(x + 2, y + 1)))
      const l = Math.min(LIGHT_LV, Math.floor(g * LIGHT_LV + th(x + 1, y + 2)))
      px[i] = tone(r, p, l)
    }
  }
  if (!win) return
  // sparks running round the window's rim, just outside it
  const [x0, y0, x1, y1] = [win[0] - 1, win[1] - 1, win[2] + 1, win[3] + 1]
  const w = x1 - x0
  const h = y1 - y0
  const per = 2 * (w + h)
  // a point on the rim, `d` cells round it from the top-left corner, and the way out from there
  const at = (d: number): [number, number, number, number] => {
    d = ((d % per) + per) % per
    if (d < w) return [x0 + d, y0, 0, -1]
    if (d < w + h) return [x1, y0 + d - w, 1, 0]
    if (d < 2 * w + h) return [x1 - (d - w - h), y1, 0, 1]
    return [x0, y1 - (d - 2 * w - h), -1, 0]
  }
  for (const sp of SPARKS) {
    const head = sp.o * per + (t * sp.speed) / CELL
    for (let k = 0; k < sp.len; k++) {
      const [sx, sy] = at(head - k)
      const xi = Math.round(sx)
      const yi = Math.round(sy)
      if (xi < 0 || yi < 0 || xi >= cols || yi >= rows) continue
      if (k < 2) px[yi * cols + xi] = BRIGHT
      else if (th(xi, yi) < 1 - k / sp.len) px[yi * cols + xi] = WARM
    }
  }
  for (const m of MOTES) {
    if (Math.sin(t * 1.7 + m.ph) < 0.3) continue
    const [sx, sy, nx, ny] = at(m.o * per + t * m.drift)
    const out = 3 + 6 * (0.5 + 0.5 * Math.sin(t * 0.5 + m.ph * 3)) // 12 to 36 px out from the rim
    const xi = Math.round(sx + nx * out)
    const yi = Math.round(sy + ny * out)
    if (xi >= 0 && xi < cols && yi >= 0 && yi < rows) px[yi * cols + xi] = m.hot ? WARM : BRIGHT
  }
}

/** The field, fixed behind the page. `light` is the town's window, in viewport px, when there is one to glow. */
export function Meadow({ mode, light }: { mode: MeadowMode; light: DOMRect | null }) {
  const cv = useRef<HTMLCanvasElement>(null)
  const lightRef = useRef(light)
  lightRef.current = light
  const redraw = useRef<() => void>(() => {})
  const moving = mode !== 'calm'

  useEffect(() => {
    const canvas = cv.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    let img: ImageData | null = null
    let px: Uint32Array | null = null
    let cols = 0
    let rows = 0
    let raf = 0
    let last = -Infinity
    const t0 = performance.now() - 20_000 // start mid-drift, not at every loop's origin
    const size = () => {
      cols = Math.ceil(window.innerWidth / CELL)
      rows = Math.ceil(window.innerHeight / CELL)
      canvas.width = cols
      canvas.height = rows
      canvas.style.width = `${cols * CELL}px`
      canvas.style.height = `${rows * CELL}px`
      img = ctx.createImageData(cols, rows)
      px = new Uint32Array(img.data.buffer)
    }
    const draw = (now: number) => {
      if (!img || !px) return
      const r = lightRef.current
      const win: [number, number, number, number] | null = r && r.width > 0
        ? [Math.round(r.left / CELL), Math.round(r.top / CELL), Math.round(r.right / CELL) - 1, Math.round(r.bottom / CELL) - 1]
        : null
      paint(px, cols, rows, (now - t0) / 1000, win)
      ctx.putImageData(img, 0, 0)
    }
    const still = !moving || window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (document.hidden || now - last < 1000 / FPS) return
      last = now
      draw(now)
    }
    const onResize = () => {
      size()
      draw(performance.now())
    }
    size()
    draw(performance.now())
    redraw.current = () => draw(performance.now())
    if (!still) raf = requestAnimationFrame(loop)
    window.addEventListener('resize', onResize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
    }
  }, [moving])

  // a still field still has to follow its window
  useEffect(() => redraw.current(), [light])

  return (
    <div className={`meadow ${mode}`} aria-hidden="true">
      <canvas ref={cv} />
      <i className="veil" />
    </div>
  )
}
