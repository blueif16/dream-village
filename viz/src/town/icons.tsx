// The console's glyphs, drawn the way the town is: true 30° isometric, hairline, white faces over the ones behind.
// Controls (play, step, zoom) are plain strokes of the same weight.
import { Children, isValidElement } from 'react'

type V = [number, number, number]
const C = Math.cos(Math.PI / 6)

/** Project a point of a 28-unit box: x runs down-right, y down-left, z up. */
const P = ([x, y, z]: V) => `${(14 + (x - y) * C).toFixed(2)},${(16.5 + (x + y) * 0.5 - z).toFixed(2)}`
const face = (pts: V[], cls?: string) => <polygon className={cls} points={pts.map(P).join(' ')} />
const line = (pts: V[]) => <polyline points={pts.map(P).join(' ')} />

/** A box's three visible faces, back to front. */
function box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, key: string) {
  return (
    <g key={key}>
      {face([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]])}
      {face([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]])}
      {face([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], 'top')}
    </g>
  )
}

/** Every point a glyph draws: its polygons' and polylines' corners, through any groups. */
function corners(node: React.ReactNode, out: number[][] = []): number[][] {
  Children.forEach(node, (c) => {
    if (!isValidElement<{ points?: string; children?: React.ReactNode }>(c)) return
    if (typeof c.props.points === 'string') for (const p of c.props.points.split(' ')) out.push(p.split(',').map(Number))
    if (c.props.children) corners(c.props.children, out)
  })
  return out
}

// how much of its box a glyph's drawing fills, along its longer side
const FILL = 0.74

/** The glyph's box: square, centred on its own drawing, and sized so every glyph fills it by the same share. A pin
 *  and a house then sit level beside each other, though one is tall and thin and the other squat. */
function fit(node: React.ReactNode): string {
  const pts = corners(node)
  if (!pts.length) return '0 0 28 28'
  const xs = pts.map((q) => q[0])
  const ys = pts.map((q) => q[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const side = Math.max(x1 - x0, y1 - y0) / FILL
  return [(x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side, side].map((v) => +v.toFixed(3)).join(' ')
}

function Glyph({ children, label }: { children: React.ReactNode; label?: string }) {
  return (
    <svg className="iso" viewBox={fit(children)} aria-hidden={label ? undefined : true} role={label ? 'img' : undefined} aria-label={label}>
      {children}
    </svg>
  )
}

/** A resident's pin, as it hangs over the town: an octahedron on a stem over its ground ring. */
export function PeopleGlyph() {
  const r = 5.4
  const h = 6.2
  const z = 10.8
  const top: V = [0, 0, z + h]
  const bot: V = [0, 0, z - h]
  const e: V[] = [[r, 0, z], [0, r, z], [-r, 0, z], [0, -r, z]]
  return (
    <Glyph>
      {face([[-4, 0, 0], [0, -4, 0], [4, 0, 0], [0, 4, 0]], 'ground')}
      {line([[0, 0, 0], bot])}
      {face([top, e[3], e[0]], 'top')}
      {face([top, e[1], e[2]], 'top')}
      {face([bot, e[0], e[1]])}
      {face([top, e[0], e[1]], 'accent')}
    </Glyph>
  )
}

/** A gabled house. */
export function PlacesGlyph() {
  const [x0, x1, y0, y1, h, ridge] = [-4.5, 4.5, -3.5, 3.5, 6, 9.5]
  return (
    <Glyph>
      {face([[x0, y1, 0], [x1, y1, 0], [x1, y1, h], [x0, y1, h]])}
      {face([[x1, y0, 0], [x1, y1, 0], [x1, y1, h], [x1, 0, ridge], [x1, y0, h]])}
      {face([[x0, y1, h], [x1, y1, h], [x1, 0, ridge], [x0, 0, ridge]], 'top')}
      {face([[-1.2, y1, 0], [1.2, y1, 0], [1.2, y1, 3.6], [-1.2, y1, 3.6]], 'top')}
    </Glyph>
  )
}

/** Dream Village's mark: the house, with a moon over it. */
export function MarkGlyph() {
  const [x0, x1, y0, y1, h, ridge] = [-4.5, 4.5, -3.5, 3.5, 6, 9.5]
  return (
    <Glyph>
      {face([[x0, y1, 0], [x1, y1, 0], [x1, y1, h], [x0, y1, h]])}
      {face([[x1, y0, 0], [x1, y1, 0], [x1, y1, h], [x1, 0, ridge], [x1, y0, h]])}
      {face([[x0, y1, h], [x1, y1, h], [x1, 0, ridge], [x0, 0, ridge]], 'top')}
      <path className="moon" d="M22.6 1.6a3.6 3.6 0 1 0 3.9 5 3 3 0 1 1-3.9-5z" />
    </Glyph>
  )
}

/** A pennant on a pole, planted on a paving square. */
export function EventsGlyph() {
  return (
    <Glyph>
      {face([[-4.5, -4.5, 0], [4.5, -4.5, 0], [4.5, 4.5, 0], [-4.5, 4.5, 0]], 'ground')}
      {line([[0, 0, 0], [0, 0, 15.5]])}
      {face([[0, 0, 15.5], [8, 0, 15.5], [6, 0, 13], [8, 0, 10.5], [0, 0, 10.5]], 'accent')}
    </Glyph>
  )
}

/** Memory as stacked sheets, the top one written on. */
export function MemoryGlyph() {
  const s = 5
  return (
    <Glyph>
      {box(-s, s, -s, s, 0, 1.4, 'a')}
      {box(-s, s, -s, s, 3.4, 4.8, 'b')}
      {box(-s, s, -s, s, 6.8, 8.2, 'c')}
      {line([[-3.2, -1, 8.2], [2.4, -1, 8.2]])}
      {line([[-3.2, 1.4, 8.2], [0.8, 1.4, 8.2]])}
    </Glyph>
  )
}

/** Three bars on a slab: the data view. */
export function DataGlyph() {
  return (
    <Glyph>
      {box(-5.5, 5.5, -5.5, 5.5, 0, 1, 'base')}
      {box(-4.5, -1.5, -1.5, 1.5, 1, 6, 'a')}
      {box(-1.5, 1.5, -1.5, 1.5, 1, 11, 'b')}
      {box(1.5, 4.5, -1.5, 1.5, 1, 8, 'c')}
    </Glyph>
  )
}

const stroke = (d: string) => (
  <svg className="ico" viewBox="0 0 16 16" aria-hidden="true">
    <path d={d} />
  </svg>
)
export const Play = () => (
  <svg className="ico" viewBox="0 0 16 16" aria-hidden="true"><path className="solid" d="M5.2 3.4v9.2L12.6 8z" /></svg>
)
export const Pause = () => stroke('M5.5 3.5v9M10.5 3.5v9')
export const Prev = () => stroke('M10 3.5 5.5 8l4.5 4.5')
export const Next = () => stroke('M6 3.5 10.5 8 6 12.5')
export const Up = () => stroke('M8 13V3.5M3.8 7.7 8 3.5l4.2 4.2')
export const Plus = () => stroke('M8 3v10M3 8h10')
export const Minus = () => stroke('M3 8h10')
export const Fit = () => stroke('M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10')
export const Close = () => stroke('M4 4l8 8M12 4l-8 8')
export const Out = () => stroke('M6 3.5h6.5V10M12.5 3.5 4 12')
export const Follow = () => stroke('M8 1.8v2.4M8 11.8v2.4M1.8 8h2.4M11.8 8h2.4M8 5.2a2.8 2.8 0 1 0 0 5.6 2.8 2.8 0 0 0 0-5.6z')
export const Sun = () => stroke('M8 5.4a2.6 2.6 0 1 0 0 5.2 2.6 2.6 0 0 0 0-5.2zM8 1.6v1.6M8 12.8v1.6M1.6 8h1.6M12.8 8h1.6M3.5 3.5l1.1 1.1M11.4 11.4l1.1 1.1M3.5 12.5l1.1-1.1M11.4 4.6l1.1-1.1')
export const GitHub = () => (
  <svg className="ico" viewBox="0 0 16 16" aria-hidden="true">
    <path className="solid" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
  </svg>
)
export const Moon = () => stroke('M12.6 9.8A5 5 0 0 1 6.2 3.4a5 5 0 1 0 6.4 6.4z')
