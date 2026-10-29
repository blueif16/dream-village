// One colour per resident, used for their figure, their phone's glow, their route and their swatch everywhere.
// Today's twelve keep the colours they had in the pixel town; anyone new gets the next colour, from a
// golden-angle walk round OKLCH, that is clearly different from every colour already taken.

const KNOWN: Record<string, number> = {
  arjun: 0xc0573e, dulce: 0xe3a33b, elliot: 0x3d5a80, grace: 0x6f9e76, isabel: 0x9a6fb0, jun: 0xd1495b,
  linh: 0x2a9d8f, marcus: 0x5b7a99, noemi: 0xee8959, rafael: 0x7d9a3a, rochelle: 0xb5838d, thao: 0x277da1,
}

const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
const gam = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)

/** sRGB (0xRRGGBB) to OKLab [L, a, b]. */
export function oklab(rgb: number): [number, number, number] {
  const r = lin(((rgb >> 16) & 255) / 255)
  const g = lin(((rgb >> 8) & 255) / 255)
  const b = lin((rgb & 255) / 255)
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

/** OKLCH to sRGB, lowering chroma until the colour fits the gamut. */
export function oklch(L: number, C: number, hDeg: number): number {
  const h = (hDeg * Math.PI) / 180
  for (let c = C; c >= 0; c -= 0.005) {
    const a = c * Math.cos(h)
    const b = c * Math.sin(h)
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
    const rgb = [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ]
    if (rgb.every((v) => v >= -0.0005 && v <= 1.0005)) {
      const [r, g, bl] = rgb.map((v) => Math.round(Math.min(1, Math.max(0, gam(Math.min(1, Math.max(0, v))))) * 255))
      return (r << 16) | (g << 8) | bl
    }
  }
  return 0x808080
}

const distance = (a: [number, number, number], b: [number, number, number]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

export function residentColors(ids: string[]): Map<string, number> {
  const out = new Map<string, number>()
  const taken: [number, number, number][] = []
  const sorted = [...new Set(ids)].sort()
  for (const id of sorted) {
    if (KNOWN[id] === undefined) continue
    out.set(id, KNOWN[id])
    taken.push(oklab(KNOWN[id]))
  }
  const L = [0.66, 0.56, 0.74, 0.6]
  let k = 0
  for (const id of sorted) {
    if (out.has(id)) continue
    // the first candidate far enough from every colour taken; failing that, the farthest of the next 600
    let best = 0
    let bestD = -1
    for (let tries = 0; tries < 600; tries++, k++) {
      const c = oklch(L[k % L.length], 0.15, (k * 137.508 + 25) % 360)
      const d = Math.min(Infinity, ...taken.map((t) => distance(t, oklab(c))))
      if (d > bestD) {
        best = c
        bestD = d
      }
      if (d >= 0.075) break
    }
    k++
    out.set(id, best)
    taken.push(oklab(best))
  }
  return out
}

export const cssHex = (n: number) => `#${n.toString(16).padStart(6, '0')}`
