// The pieces of a resident's badge that are only drawing: the cord it hangs from, the art panel's dithered ground,
// the barcode on the stub. What the badge says lives in PersonDetail (Detail.tsx); every word of it is from the data.
import { useId } from 'react'
import { barcode } from './avatar.ts'
import './card.css'

/** The cord from the top edge of the map, looped through the slot punched in the badge (the slot is at y 13–23). */
export function Cord() {
  return (
    <svg className="idcard-cord" viewBox="0 -64 60 92" aria-hidden="true">
      <path d="M30 -64V-16" strokeWidth="3.4" />
      <path d="M28.4 -8C22 -1 18.6 8 18.4 16.6" />
      <path d="M31.6 -8C38 -1 41.4 8 41.6 16.6" />
      <path className="through" d="M18.4 16.6Q30 21.6 41.6 16.6" />
      <rect className="bead" x="25.6" y="-18" width="8.8" height="12" rx="3.4" />
    </svg>
  )
}

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]

/** The art panel's ground: the resident's colour, a deeper shade of it dithered in toward the bottom. */
export function Dither({ height, px = 3 }: { height: number; px?: number }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const levels = [1, 2, 3, 5, 7, 9, 11] // of 16, top band to bottom
  const band = Math.round((height * 0.56) / levels.length / (px * 4)) * px * 4
  const top = height - band * levels.length
  return (
    <svg className="dither" width="100%" height={height} aria-hidden="true">
      <defs>
        {levels.map((k) => (
          <pattern key={k} id={`${uid}d${k}`} width={4 * px} height={4 * px} patternUnits="userSpaceOnUse" y={top}>
            {BAYER.map((v, i) => v < k && <rect key={i} x={(i % 4) * px} y={Math.floor(i / 4) * px} width={px} height={px} />)}
          </pattern>
        ))}
      </defs>
      {levels.map((k, i) => <rect key={k} y={top + i * band} width="100%" height={band} fill={`url(#${uid}d${k})`} />)}
    </svg>
  )
}

/** The stub's barcode, from the id: decorative, the same every time. */
export function Barcode({ id }: { id: string }) {
  const w = barcode(id)
  const at = w.map((_, i) => w.slice(0, i).reduce((a, b) => a + b, 0))
  return (
    <svg className="barcode" viewBox={`0 0 ${at[at.length - 1] + w[w.length - 1]} 1`} preserveAspectRatio="none" aria-hidden="true">
      {w.map((m, i) => i % 2 === 0 && <rect key={i} x={at[i]} width={m} height={1} />)}
    </svg>
  )
}

/** The role split where it turns from what they are to where and what else: the first part is set bold. */
export function roleSplit(role: string): [string, string] {
  const m = /,|\s(?:at|on|in|for|with|from)\s|\s[a-z]+ing\s/.exec(role)
  return m && m.index > 0 ? [role.slice(0, m.index), role.slice(m.index)] : [role, '']
}

/** Ink or paper, whichever reads better on the colour (WCAG contrast). */
export function onColor(rgb: number): 'ink' | 'paper' {
  const ch = (v: number) => {
    const c = v / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  const L = 0.2126 * ch((rgb >> 16) & 255) + 0.7152 * ch((rgb >> 8) & 255) + 0.0722 * ch(rgb & 255)
  const inkL = 0.0109 // #1a1c1a
  return (L + 0.05) / (inkL + 0.05) >= 1.05 / (L + 0.05) ? 'ink' : 'paper'
}
