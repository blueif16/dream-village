import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DataGlyph, EventsGlyph, MemoryGlyph, PeopleGlyph, PlacesGlyph } from './icons'

// Glyphs sit side by side (in a button, in the dock), so each must be centred on its own drawing and fill its box by
// the same share: otherwise two glyphs of the same size look out of line with each other.
const GLYPHS = { PeopleGlyph, PlacesGlyph, EventsGlyph, MemoryGlyph, DataGlyph }

function measure(G: () => React.ReactNode) {
  const svg = renderToStaticMarkup(<G />)
  const [vx, vy, vw, vh] = /viewBox="([^"]+)"/.exec(svg)![1].split(' ').map(Number)
  const pts = [...svg.matchAll(/points="([^"]+)"/g)].flatMap((m) => m[1].split(' ').map((p) => p.split(',').map(Number)))
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const box = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  return { vx, vy, vw, vh, ...box }
}

describe('iso glyphs', () => {
  it.each(Object.entries(GLYPHS))('%s is centred on its own drawing', (_, G) => {
    const m = measure(G)
    expect(m.vw).toBeCloseTo(m.vh, 6)
    expect((m.x0 + m.x1) / 2).toBeCloseTo(m.vx + m.vw / 2, 2)
    expect((m.y0 + m.y1) / 2).toBeCloseTo(m.vy + m.vh / 2, 2)
  })

  it('every glyph fills its box by the same share', () => {
    const fills = Object.values(GLYPHS).map((G) => {
      const m = measure(G)
      return Math.max(m.x1 - m.x0, m.y1 - m.y0) / m.vw
    })
    for (const f of fills) expect(f).toBeCloseTo(fills[0], 3)
  })
})
