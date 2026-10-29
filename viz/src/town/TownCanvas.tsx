import { Application, Container, Graphics, Rectangle } from 'pixi.js'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { drawBubble, drawLot, drawMarker, drawPerson, outline, pennant } from './draw'
import { H, MARGIN, ROAD, BLOCK_H, BLOCK_W, W, layout, slots, type Block, type Lot } from './layout'
import { look } from './people'
import type { Selection, Town } from './types'

const SELECT = 0xc2410c
const EVENT = 0xe0a526

export interface Hover {
  kind: 'person' | 'place'
  id: string
  x: number // px within the canvas
  y: number
}

interface Props {
  town: Town
  day: number
  selection: Selection
  onSelect: (s: Selection) => void
  onHover: (h: Hover | null) => void
}

interface Scene {
  update: (day: number, selection: Selection) => void
}

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function onDay(ev: { date: string; end_date: string | null }, d: string) {
  return ev.date <= d && d <= (ev.end_date ?? ev.date)
}

function buildScene(app: Application, town: Town, props: { current: Props }): Scene {
  const { blocks, lots } = layout(town.places)
  const ground = new Graphics()
  ground.rect(0, 0, W, H).fill(0xcfc7b1)
  // roads between the blocks, with a dashed centre line
  const roadsY = MARGIN + BLOCK_H
  ground.rect(0, roadsY, W, ROAD).fill(0x8f8a80)
  for (let x = 2; x < W; x += 8) ground.rect(x, roadsY + 4, 4, 1).fill(0xe2dac6)
  for (const c of [1, 2]) {
    const x = MARGIN + c * BLOCK_W + (c - 1) * ROAD
    ground.rect(x, 0, ROAD, H).fill(0x8f8a80)
    for (let y = 2; y < H; y += 8) ground.rect(x + 4, y, 1, 4).fill(0xe2dac6)
  }
  blocks.forEach((b: Block) => {
    ground.rect(b.x, b.y, b.w, b.h).fill(b.ground)
    ground.rect(b.x, b.y + b.h - 1, b.w, 1).fill({ color: 0x000000, alpha: 0.08 })
  })
  app.stage.addChild(ground)

  const buildings = new Container()
  let i = 0
  for (const lot of lots.values()) {
    const g = new Graphics()
    drawLot(g, lot, i++)
    g.eventMode = 'static'
    g.cursor = 'pointer'
    g.hitArea = new Rectangle(lot.x - 2, lot.y - 8, lot.w + 4, lot.h + 10)
    g.on('pointertap', () => props.current.onSelect({ kind: 'place', id: lot.place.id }))
    g.on('pointerover', (e) => props.current.onHover({ kind: 'place', id: lot.place.id, x: e.global.x, y: e.global.y }))
    g.on('pointerout', () => props.current.onHover(null))
    buildings.addChild(g)
  }
  app.stage.addChild(buildings)
  const marks = new Graphics()
  app.stage.addChild(marks)

  const fallback = [...lots.values()].find((l) => l.place.kind === 'main_street') ?? [...lots.values()][0]
  const people = new Container()
  app.stage.addChild(people)
  const actors = town.principals.map((p, k) => {
    const c = new Container()
    const body = new Graphics()
    drawPerson(body, look(p.id))
    const bubble = new Graphics()
    drawBubble(bubble)
    const marker = new Graphics()
    drawMarker(marker, SELECT)
    c.addChild(body, bubble, marker)
    c.eventMode = 'static'
    c.cursor = 'pointer'
    c.hitArea = new Rectangle(-3, -11, 7, 12)
    c.on('pointertap', (e) => {
      e.stopPropagation()
      props.current.onSelect({ kind: 'person', id: p.id })
    })
    c.on('pointerover', (e) => props.current.onHover({ kind: 'person', id: p.id, x: e.global.x, y: e.global.y }))
    c.on('pointerout', () => props.current.onHover(null))
    people.addChild(c)
    return { p, c, bubble, marker, x: 0, y: 0, tx: 0, ty: 0, phase: (k * 397) % 1600, placed: false }
  })

  app.ticker.add(() => {
    const still = reduced()
    const now = performance.now()
    for (const a of actors) {
      if (still || !a.placed) {
        a.x = a.tx
        a.y = a.ty
        a.placed = true
      } else {
        a.x += (a.tx - a.x) * 0.12
        a.y += (a.ty - a.y) * 0.12
        if (Math.abs(a.tx - a.x) < 0.5) a.x = a.tx
        if (Math.abs(a.ty - a.y) < 0.5) a.y = a.ty
      }
      const bob = !still && (now + a.phase) % 1600 < 160 ? -1 : 0 // a breath, not a bounce
      a.c.position.set(Math.round(a.x), Math.round(a.y) + bob)
    }
  })

  return {
    update(day, selection) {
      const d = town.days[day]
      marks.clear()
      const today = town.events.filter((ev) => onDay(ev, d))
      for (const ev of today) {
        for (const pid of ev.places) {
          const l = lots.get(pid)
          if (l) {
            outline(marks, l, EVENT)
            pennant(marks, l)
          }
        }
      }
      if (selection?.kind === 'place') {
        const l = lots.get(selection.id)
        if (l) outline(marks, l, SELECT, 3)
      }
      const groups = new Map<Lot, typeof actors>()
      for (const a of actors) {
        const place = a.p.days[day]?.place
        const lot = (place && lots.get(place)) || fallback
        groups.set(lot, [...(groups.get(lot) ?? []), a])
      }
      for (const [lot, group] of groups) {
        slots(lot, group.length).forEach(([x, y], k) => {
          group[k].tx = x
          group[k].ty = y
        })
      }
      for (const a of actors) {
        const talked = (a.p.sessions[d] ?? []).length > 0
        a.bubble.visible = talked
        a.marker.visible = selection?.kind === 'person' && selection.id === a.p.id
        a.marker.y = talked ? -7 : 0
        a.c.zIndex = a.ty
      }
      people.sortableChildren = true
    },
  }
}

/** The town as a pixel map: integer zoom, nearest-neighbour, drawn in code. `overlay` renders over the canvas. */
export function TownCanvas({ town, day, selection, onSelect, onHover, overlay }: Props & { overlay?: ReactNode }) {
  const host = useRef<HTMLDivElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  const scene = useRef<Scene | null>(null)
  const props = useRef<Props>({ town, day, selection, onSelect, onHover })
  props.current = { town, day, selection, onSelect, onHover }
  const [zoom, setZoom] = useState(0)

  useEffect(() => {
    const el = host.current
    const fr = frame.current
    if (!el || !fr) return
    const app = new Application()
    let alive = true
    let ready = false
    let ro: ResizeObserver | null = null
    const fit = () => {
      const cs = getComputedStyle(el) // fit the content box, so the padding stays clear
      const w = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)
      const h = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)
      const z = Math.max(1, Math.floor(Math.min(w / W, h / H)))
      app.renderer.resize(W * z, H * z)
      app.stage.scale.set(z)
      setZoom(z)
    }
    app
      .init({ width: W, height: H, background: 0xcfc7b1, antialias: false, resolution: 1, roundPixels: true })
      .then(() => {
        if (!alive) {
          app.destroy(true, { children: true })
          return
        }
        ready = true
        app.canvas.style.imageRendering = 'pixelated'
        app.canvas.style.display = 'block'
        fr.prepend(app.canvas)
        scene.current = buildScene(app, town, props)
        scene.current.update(props.current.day, props.current.selection)
        fit()
        ro = new ResizeObserver(fit)
        ro.observe(el)
      })
    return () => {
      alive = false
      ro?.disconnect()
      scene.current = null
      if (ready) app.destroy(true, { children: true })
    }
  }, [town])

  useEffect(() => {
    scene.current?.update(day, selection)
  }, [day, selection])

  const { blocks } = layout(town.places)
  return (
    <div className="town-host" ref={host}>
      <div className="town-frame" ref={frame} style={zoom ? { width: W * zoom, height: H * zoom } : undefined}>
        {zoom > 0 &&
          blocks.map((b) => (
            <span key={b.id} className="town-district" style={{ left: (b.x + 5) * zoom, top: (b.y + 2) * zoom }}>
              {b.label}
            </span>
          ))}
        {overlay}
      </div>
    </div>
  )
}

