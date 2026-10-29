// The live town. React draws the plate once and talks to this through a few methods and two callbacks; the engine
// owns the scene and the frame loop. Every motion is either periodic (townsfolk, cars, the bus, the train, the day
// itself) or follows town.json (where each resident is, when they talked to the assistant, when their memory dreamed).
import * as THREE from 'three'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { buildBoat, buildBus, buildCar, buildFigure, buildPin, buildRing, buildTrain, pose, type Figure } from './actors'
import { W } from './iso'
import { buildEventDressing, buildLand, buildLot } from './models'
import { DAY_GLASS, DETAIL, INK, NIGHT, SOFT_LINE, districtName } from './palette'
import { Part, makeMaterials, type Materials } from './part'
import { planTown, type Lot, type Plan } from './plan'
import { hash, rng } from './rand'
import { DAY_MIN, dreamTimes, lastAtOrBefore, talkTimes, talkingAt, targetAt } from './schedule'
import type { Town, TownPerson } from './types'

export type Sel = { kind: 'person' | 'place' | 'train' | 'bus'; id: string } | null

export interface Status {
  mode: 'asleep' | 'here' | 'walking' | 'talking' | 'dreaming'
  at: string | null // the place they are at, or left
  to: string | null // where they are walking to
}

export interface EngineOpts {
  canvas: HTMLCanvasElement
  stage: HTMLElement
  tagHover: HTMLElement
  tagSel: HTMLElement
  labels: HTMLElement
  town: Town
  colors: Map<string, number>
  onSelect: (s: Sel) => void
  onTick: (t: number) => void
  onNight: (night: boolean) => void
}

export const MIN_PER_SEC = 8 // at 1×, a day passes in three minutes
const PIN_PX = 22 // a resident's pin is this tall on screen however far out the view is
const REDUCED = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const ISO = new THREE.Vector3(1, 1, 1).normalize()
const ease = (t: number) => t * t * (3 - 2 * t)
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v))

/** 0 by day, 1 by night: dusk 18:30–20:30, dawn 05:30–07:00. */
export function nightAt(t: number): number {
  const m = ((t % DAY_MIN) + DAY_MIN) % DAY_MIN
  if (m >= 1230 || m < 330) return 1
  if (m >= 1110) return ease((m - 1110) / 120)
  if (m < 420) return 1 - ease((m - 330) / 90)
  return 0
}

class PathWalk {
  pts: [number, number][]
  cum: number[]
  length: number
  constructor(pts: [number, number][]) {
    this.pts = pts
    this.cum = [0]
    for (let i = 1; i < pts.length; i++) this.cum.push(this.cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    this.length = this.cum.at(-1) ?? 0
  }
  at(s: number): { x: number; y: number; h: number } {
    const P = this.pts
    if (P.length === 1 || this.length === 0) return { x: P[0][0], y: P[0][1], h: 0 }
    s = clamp(s, 0, this.length)
    let i = 1
    while (i < P.length - 1 && this.cum[i] < s) i++
    const seg = this.cum[i] - this.cum[i - 1] || 1
    const f = (s - this.cum[i - 1]) / seg
    const [a, b] = [P[i - 1], P[i]]
    return { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, h: Math.atan2(b[1] - a[1], b[0] - a[0]) }
  }
}

/** A closed loop walked round and round by arc length. */
function loopAt(walk: PathWalk, s: number) {
  return walk.at(((s % walk.length) + walk.length) % walk.length)
}

interface Resident {
  p: TownPerson
  color: number
  fig: Figure
  pin: ReturnType<typeof buildPin>
  ring: THREE.Mesh // the pulse while they talk
  hit: THREE.Mesh
  talks: number[]
  extraTalks: number[] // exact session starts, once loaded for the selected resident
  dreams: number[]
  dest: string | null
  walk: PathWalk | null
  s: number
  speed: number // m per real second
  x: number
  y: number
  h: number
  phase: number
  snap: boolean
  fan: number // where its pin sits in a row of pins over a shared place
  stem: LineSegments2
}

interface Mover {
  group: THREE.Object3D
  loop: PathWalk
  s: number
  speed: number
  fig?: Figure
  shy: number // the hour-density below which it stays in
}

export class TownEngine {
  private o: EngineOpts
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 6000)
  private controls: OrbitControls
  private m: Materials
  private plan: Plan
  private lotTop = new Map<string, number>()
  private buildings: THREE.Group
  private residents: Resident[] = []
  private folk: Mover[] = []
  private cars: Mover[] = []
  private bus: { group: THREE.Group; loop: PathWalk; s: number; wait: number; stops: number[] }
  private train: { group: THREE.Group; length: number; x: number; dir: 1 | -1; wait: number; stops: number[]; x0: number; x1: number; rest: number }
  private eventPart: THREE.Group | null = null
  private eventDay = -1
  private crowd: Figure[] = []
  private dreamCols = new Map<string, THREE.Group>()
  private selRing: THREE.Mesh
  private reticle: LineSegments2
  private routeLine: LineSegments2
  private routeMat: LineMaterial
  private raycaster = new THREE.Raycaster()
  private ro: ResizeObserver
  private raf = 0
  private last = performance.now()
  private tickAcc = 0
  private fitZoom = 1
  private home = new THREE.Vector3()
  private goal: { zoom: number; target?: THREE.Vector3 } | null = null
  private sized = false
  private nightOn = false
  private down: { x: number; y: number; t: number } | null = null
  private pointer: [number, number] | null = null
  private hovered: Sel = null
  t = 7.5 * 60
  playing = true
  speed = 1
  sel: Sel = null
  follow = false
  views: { id: string; label: string; box: [number, number, number, number] }[] = []

  constructor(o: EngineOpts) {
    this.o = o
    this.renderer = new THREE.WebGLRenderer({ canvas: o.canvas, antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.renderer.setClearColor(0x000000, 0)
    this.m = makeMaterials()
    this.plan = planTown(o.town.places)

    // ---- the land and the buildings
    const { land, hills, extent, light } = buildLand(this.plan)
    const landG = land.build(this.m, 'land')
    const hillsG = hills.build(this.m, 'hills')
    landG.traverse((c) => (c.raycast = () => {}))
    hillsG.traverse((c) => (c.raycast = () => {}))
    this.scene.add(landG, hillsG)
    const bp = new Part()
    for (const lot of this.plan.lots.values()) this.lotTop.set(lot.place.id, buildLot(bp, lot))
    this.buildings = bp.build(this.m, 'buildings')
    this.scene.add(this.buildings)
    this.addLabels()

    // ---- residents
    for (const p of o.town.principals) {
      const color = o.colors.get(p.id) ?? 0x888888
      const fig = buildFigure(this.m, p.id, color, 1.7)
      const pin = buildPin(color)
      const ring = buildRing(color, 1.2, 1.7, 0.8)
      ring.visible = false
      const hit = new THREE.Mesh(new THREE.BoxGeometry(2.4, 4.4, 2.4).translate(0, 2.2, 0), new THREE.MeshBasicMaterial({ visible: false }))
      hit.userData.sel = { kind: 'person', id: p.id }
      pin.group.children[0].userData.sel = { kind: 'person', id: p.id }
      const stem = new LineSegments2(new LineSegmentsGeometry().setPositions([0, 0, 0, 0, 1, 0]), new LineMaterial({ color, linewidth: 1, worldUnits: false, transparent: true, opacity: 0.9 }))
      stem.raycast = () => {}
      this.scene.add(fig.group, pin.group, ring, hit, stem)
      this.residents.push({
        p, color, fig, pin, ring, hit, talks: talkTimes(p, o.town.days), extraTalks: [], dreams: dreamTimes(p, o.town.days),
        dest: null, walk: null, s: 0, speed: 3, x: 0, y: 0, h: 0, phase: (hash(p.id) % 1000) / 1000, snap: true, fan: 0, stem,
      })
    }

    // ---- periodic life: townsfolk round the blocks, a car round each block, the bus round the town, the train
    const S = this.plan.sidewalk
    for (const b of this.plan.blocks) {
      const ring = new PathWalk([[b.x + S / 2, b.y + S / 2], [b.x + b.w - S / 2, b.y + S / 2], [b.x + b.w - S / 2, b.y + b.d - S / 2], [b.x + S / 2, b.y + b.d - S / 2], [b.x + S / 2, b.y + S / 2]])
      const r = rng(`folk:${b.id}`)
      for (let k = 0; k < 3; k++) {
        const id = `${b.id}:f${k}`
        const fig = buildFigure(this.m, id, null, 1.15)
        this.scene.add(fig.group)
        const dir = k % 2 ? -1 : 1
        this.folk.push({ group: fig.group, fig, loop: ring, s: r() * ring.length, speed: dir * (1 + r() * 0.5), shy: r() })
      }
      const off = 4.2
      const lane = new PathWalk([[b.x - off, b.y - off], [b.x + b.w + off, b.y - off], [b.x + b.w + off, b.y + b.d + off], [b.x - off, b.y + b.d + off], [b.x - off, b.y - off]])
      const car = buildCar(this.m, `${b.id}:car`)
      this.scene.add(car)
      this.cars.push({ group: car, loop: lane, s: rng(`car:${b.id}`)() * lane.length, speed: 7 + (hash(b.id) % 3), shy: (hash(b.id) % 100) / 100 })
    }
    const g = this.plan.grid
    const busLoop = new PathWalk([[g.x0 + 3.6, g.y0 + 3.6], [g.x1 - 3.6, g.y0 + 3.6], [g.x1 - 3.6, g.y1 - 3.6], [g.x0 + 3.6, g.y1 - 3.6], [g.x0 + 3.6, g.y0 + 3.6]])
    const busG = buildBus(this.m)
    busG.traverse((c) => (c.userData.sel = { kind: 'bus', id: 'bus' }))
    this.scene.add(busG)
    this.bus = { group: busG, loop: busLoop, s: 0, wait: 0, stops: [0.12, 0.37, 0.62, 0.87].map((f) => f * busLoop.length) }
    const tr = buildTrain(this.m)
    tr.group.traverse((c) => (c.userData.sel = { kind: 'train', id: 'train' }))
    this.scene.add(tr.group)
    const stops = [...this.plan.lots.values()].filter((l) => l.arche === 'station').map((l) => l.x + l.w / 2)
    this.train = { group: tr.group, length: tr.length, x: extent.x0 - 80, dir: 1, wait: 0, stops, x0: extent.x0 - 80, x1: extent.x1 + 80, rest: 6 }

    const br = rng('boats')
    for (let i = 0; i < 4; i++) {
      const boat = buildBoat(this.m, `boat:${i}`)
      boat.traverse((c) => (c.raycast = () => {}))
      this.scene.add(boat)
      const span = g.x1 - g.x0
      const x0 = g.x0 + br() * span * 0.5
      this.boats.push({ group: boat, x0, x1: x0 + 40 + br() * span * 0.4, y: g.y1 + 34 + i * 16 + br() * 8, period: 70 + br() * 60, phase: br() })
    }

    // the lighthouse beam, turning after dusk
    // (bright at the lamp, fading out along its length: additive black is nothing)
    const beamGeo = new THREE.BufferGeometry()
      .setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 90, 0, -10, 90, 0, 10], 3))
      .setAttribute('color', new THREE.Float32BufferAttribute([1, 0.94, 0.78, 0, 0, 0, 0, 0, 0], 3))
    this.beam = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }))
    this.beam.position.copy(W(...light))
    this.beam.raycast = () => {}
    this.scene.add(this.beam)

    // ---- selection marks
    this.selRing = buildRing(0x1c1d22, 2.1, 2.45, 0.9)
    this.selRing.visible = false
    const ret = new LineMaterial({ color: 0x1c1d22, linewidth: 1.5, worldUnits: false, depthTest: false, transparent: true })
    this.reticle = new LineSegments2(new LineSegmentsGeometry(), ret)
    this.reticle.renderOrder = 18
    this.reticle.visible = false
    this.routeMat = new LineMaterial({ color: 0x000000, linewidth: 2, worldUnits: false, depthTest: false, transparent: true })
    this.routeLine = new LineSegments2(new LineSegmentsGeometry(), this.routeMat)
    this.routeLine.renderOrder = 17
    this.routeLine.visible = false
    this.reticle.raycast = () => {}
    this.routeLine.raycast = () => {}
    this.scene.add(this.selRing, this.reticle, this.routeLine)

    // ---- the places the caption links jump to
    const byDistrict = new Map<string, [number, number, number, number]>()
    for (const b of this.plan.blocks) {
      const r = byDistrict.get(b.district)
      byDistrict.set(b.district, r ? [Math.min(r[0], b.x), Math.max(r[1], b.x + b.w), Math.min(r[2], b.y), Math.max(r[3], b.y + b.d)] : [b.x, b.x + b.w, b.y, b.y + b.d])
    }
    this.views = [...byDistrict].map(([id, box]) => ({ id, label: districtName(id), box }))
    const st = [...this.plan.lots.values()].filter((l) => l.arche === 'station')
    if (st.length) this.views.push({ id: 'rail', label: 'Rail', box: [Math.min(...st.map((l) => l.x)) - 30, Math.max(...st.map((l) => l.x + l.w)) + 30, this.plan.rail.y - 20, 10] })

    // ---- camera and controls
    this.camera.position.copy(W(0, 0, 0)).addScaledVector(ISO, 3000)
    this.camera.lookAt(W(0, 0, 0))
    this.controls = new OrbitControls(this.camera, o.canvas)
    Object.assign(this.controls, { enableRotate: false, screenSpacePanning: true, zoomToCursor: true, enableDamping: !REDUCED(), dampingFactor: 0.12 })
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
    this.controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN }
    this.ro = new ResizeObserver(() => this.resize())
    this.ro.observe(o.stage)
    this.resize()

    o.canvas.addEventListener('pointerdown', this.onDown)
    o.canvas.addEventListener('pointerup', this.onUp)
    o.canvas.addEventListener('pointermove', this.onMove)
    o.canvas.addEventListener('pointerleave', this.onLeave)
    o.canvas.addEventListener('wheel', this.onWheel, { passive: true })
    this.applyNight(true)
    if (new URLSearchParams(location.search).has('debug')) (window as unknown as { town: TownEngine }).town = this
    this.raf = requestAnimationFrame(this.loop)
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    this.ro.disconnect()
    const c = this.o.canvas
    c.removeEventListener('pointerdown', this.onDown)
    c.removeEventListener('pointerup', this.onUp)
    c.removeEventListener('pointermove', this.onMove)
    c.removeEventListener('pointerleave', this.onLeave)
    c.removeEventListener('wheel', this.onWheel)
    this.controls.dispose()
    this.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      mesh.geometry?.dispose?.()
    })
    this.renderer.dispose()
  }

  // ---------------------------------------------------------------- public controls

  setTime(t: number) {
    this.t = clamp(t, 0, this.o.town.days.length * DAY_MIN - 1)
    this.o.onTick(this.t)
  }
  setSpeed(k: number) {
    this.speed = k
  }
  setPlaying(on: boolean) {
    this.playing = on
  }
  setFollow(on: boolean) {
    this.follow = on && this.sel?.kind === 'person'
  }
  select(s: Sel) {
    this.sel = s
    if (s?.kind !== 'person') this.follow = false
    this.drawSelection()
    this.o.onSelect(s)
  }
  /** Exact session start times for a resident (from their person file), to glow for sessions with no write. */
  setSessions(pid: string, starts: string[]) {
    const r = this.residents.find((x) => x.p.id === pid)
    if (!r) return
    const d0 = Date.parse(`${this.o.town.days[0]}T00:00:00Z`)
    r.extraTalks = starts.map((s) => Math.round((Date.parse(`${s}Z`) - d0) / 60000)).sort((a, b) => a - b)
  }
  zoomBy(k: number) {
    this.goal = { zoom: clamp((this.goal?.zoom ?? this.camera.zoom) * k, this.controls.minZoom, this.controls.maxZoom), target: this.goal?.target }
  }
  resetView() {
    this.follow = false
    this.goal = { zoom: this.fitZoom, target: this.home.clone() }
  }
  goTo(view: string) {
    const v = this.views.find((x) => x.id === view)
    if (!v) return
    this.follow = false
    this.goal = this.frame(v.box)
  }
  /** Bring a place or resident into view, close up. */
  focus(s: Sel) {
    this.pendingFocus = s
  }
  private pendingFocus: Sel = null
  private applyFocus() {
    const s = this.pendingFocus
    this.pendingFocus = null
    const at = this.worldOf(s)
    if (at) this.goal = { zoom: Math.max(this.camera.zoom, this.fitZoom * 3), target: this.ground(at) }
  }
  statusOf(pid: string): Status {
    const r = this.residents.find((x) => x.p.id === pid)
    if (!r) return { mode: 'here', at: null, to: null }
    return this.status(r)
  }
  placeOf(pid: string): string | null {
    return this.statusOf(pid).at
  }
  lotOf(id: string): Lot | undefined {
    return this.plan.lots.get(id)
  }
  get dayCount() {
    return this.o.town.days.length
  }

  // ---------------------------------------------------------------- camera

  private frame([x0, x1, y0, y1]: [number, number, number, number], [z0, z1] = [0, 14]) {
    const w = this.o.stage.clientWidth
    const h = this.o.stage.clientHeight
    this.camera.updateMatrixWorld()
    const inv = this.camera.matrixWorldInverse
    const b = new THREE.Box3()
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) b.expandByPoint(W(x, y, z).applyMatrix4(inv))
    const c = b.getCenter(new THREE.Vector3()).applyMatrix4(this.camera.matrixWorld)
    return { zoom: Math.min(w / (b.max.x - b.min.x), h / (b.max.y - b.min.y)) * 0.92, target: this.ground(c) }
  }
  /** Slide a point down the view ray onto the ground. */
  private ground(c: THREE.Vector3) {
    return c.clone().addScaledVector(ISO, -c.y / ISO.y)
  }
  private moveTarget(to: THREE.Vector3) {
    const d = to.clone().sub(this.controls.target)
    this.controls.target.add(d)
    this.camera.position.add(d)
  }
  private resize() {
    const w = this.o.stage.clientWidth
    const h = this.o.stage.clientHeight
    if (!w || !h) return
    this.renderer.setSize(w, h, false)
    Object.assign(this.camera, { left: -w / 2, right: w / 2, top: h / 2, bottom: -h / 2 })
    const g = this.plan.grid
    const f = this.frame([g.x0 - 10, g.x1 + 10, this.plan.rail.y - 30, g.y1 + 24])
    this.fitZoom = f.zoom
    this.home = f.target
    this.controls.minZoom = f.zoom * 0.5
    this.controls.maxZoom = f.zoom * 14
    if (!this.sized) {
      this.sized = true
      this.camera.zoom = f.zoom
      this.moveTarget(f.target)
    }
    this.camera.zoom = clamp(this.camera.zoom, this.controls.minZoom, this.controls.maxZoom)
    this.camera.updateProjectionMatrix()
  }

  // ---------------------------------------------------------------- labels painted on the ground

  private labels: { el: HTMLElement; at: THREE.Vector3 }[] = []
  private addLabels() {
    this.o.labels.replaceChildren()
    for (const b of this.plan.blocks.filter((k) => k.first)) {
      const el = document.createElement('span')
      el.className = 'dlabel'
      el.textContent = districtName(b.district)
      this.o.labels.append(el)
      this.labels.push({ el, at: W(b.x, b.y, 0) })
    }
  }
  private drawLabels() {
    const far = this.camera.zoom < this.fitZoom * 2.2 // names show from afar, and step aside up close
    for (const l of this.labels) {
      const [x, y] = this.screenOf(l.at)
      l.el.style.left = `${x.toFixed(1)}px`
      l.el.style.top = `${y.toFixed(1)}px`
      l.el.classList.toggle('hide', !far)
    }
  }

  // ---------------------------------------------------------------- the frame loop

  private loop = (now: number) => {
    this.raf = requestAnimationFrame(this.loop)
    const dt = Math.min(0.1, Math.max(0, (now - this.last) / 1000))
    this.last = now
    const tMax = this.o.town.days.length * DAY_MIN - 1
    if (this.playing) {
      this.t = Math.min(tMax, this.t + dt * MIN_PER_SEC * this.speed)
      if (this.t >= tMax) this.playing = false
    }
    this.update(dt, now / 1000)
    this.camStep(dt)
    if ((this.tickAcc -= dt) <= 0) {
      this.tickAcc = 0.15
      this.o.onTick(this.t)
    }
    this.renderer.render(this.scene, this.camera)
  }

  private camStep(dt: number) {
    const k = REDUCED() ? 1 : 1 - Math.exp(-dt * 6)
    if (this.follow && this.sel?.kind === 'person') {
      const r = this.residents.find((x) => x.p.id === this.sel!.id)
      if (r) this.moveTarget(this.controls.target.clone().lerp(W(r.x, r.y, 0), k))
    }
    if (this.goal) {
      if (this.goal.target) this.moveTarget(this.controls.target.clone().lerp(this.goal.target, k))
      this.camera.zoom += (this.goal.zoom - this.camera.zoom) * k
      this.camera.updateProjectionMatrix()
      if (Math.abs(this.goal.zoom - this.camera.zoom) < 1e-3 * this.goal.zoom && (!this.goal.target || this.controls.target.distanceTo(this.goal.target) < 0.05)) this.goal = null
    }
    this.controls.update(dt)
    this.moveTarget(this.ground(this.controls.target))
  }

  private applyNight(force = false) {
    const n = nightAt(this.t)
    const lit = Math.min(1, n * 1.8)
    this.m.fill.color.setRGB(1, 1, 1).lerp(new THREE.Color(NIGHT), n * 0.8)
    this.m.glow.color.copy(new THREE.Color(DAY_GLASS)).lerp(new THREE.Color(0xffffff), lit)
    this.m.line.ink.color.set(INK.day).lerp(new THREE.Color(INK.night), n)
    this.m.line.detail.color.set(DETAIL.day).lerp(new THREE.Color(DETAIL.night), n)
    this.m.line.soft.color.set(SOFT_LINE.day).lerp(new THREE.Color(SOFT_LINE.night), n)
    const on = this.nightOn ? n > 0.45 : n > 0.55
    if (on !== this.nightOn || force) {
      this.nightOn = on
      this.o.onNight(on)
    }
  }

  private status(r: Resident): Status {
    if (r.walk) return { mode: 'walking', at: null, to: r.dest }
    const t = this.t
    const home = !!r.dest && r.p.homes.includes(r.dest)
    if (talkingAt(r.talks, t) || talkingAt(r.extraTalks, t)) return { mode: 'talking', at: r.dest, to: null }
    const di = lastAtOrBefore(r.dreams, t + 30)
    if (di >= 0 && t >= r.dreams[di] - 30 && t < r.dreams[di] + 60) return { mode: 'dreaming', at: r.dest, to: null }
    const m = ((t % DAY_MIN) + DAY_MIN) % DAY_MIN
    if (home && (m >= 23 * 60 + 30 || m < 6 * 60 + 30)) return { mode: 'asleep', at: r.dest, to: null }
    return { mode: 'here', at: r.dest, to: null }
  }

  private fallback(): string | null {
    const l = [...this.plan.lots.values()].find((x) => x.arche === 'arcade') ?? [...this.plan.lots.values()][0]
    return l?.place.id ?? null
  }

  private update(dt: number, secs: number) {
    const t = this.t
    const still = REDUCED()
    this.applyNight()

    // ---- residents: head for where the data says they are
    const jump = !this.playing || this.speed >= 16
    for (const r of this.residents) {
      const tgt = targetAt(r.p, t) ?? this.fallback()
      if (tgt !== r.dest) {
        const from = r.dest
        r.dest = tgt
        const lot = tgt ? this.plan.lots.get(tgt) : undefined
        if (!from || !lot || jump || still) {
          r.walk = null
          r.snap = true
        } else {
          const pts = this.plan.walk(from, tgt!)
          const start: [number, number] = [r.x, r.y]
          r.walk = new PathWalk([start, ...pts])
          r.s = 0
          r.speed = clamp(r.walk.length / 9, 2.2, 14) * Math.max(1, this.speed / 2)
        }
      }
      if (r.walk) {
        r.s += r.speed * dt
        const q = r.walk.at(r.s)
        r.x = q.x
        r.y = q.y
        r.h = q.h
        if (r.s >= r.walk.length) r.walk = null
      }
    }
    // standing residents line up along the sidewalk by their place's door
    const groups = new Map<string, Resident[]>()
    for (const r of this.residents) if (!r.walk && r.dest) groups.set(r.dest, [...(groups.get(r.dest) ?? []), r])
    for (const [pid, rs] of groups) {
      const lot = this.plan.lots.get(pid)
      if (!lot) continue
      rs.sort((a, b) => a.p.id.localeCompare(b.p.id))
      rs.forEach((r, k) => {
        r.fan = k - (rs.length - 1) / 2
        const tx = lot.door[0] + (k - (rs.length - 1) / 2) * 1.8
        const ty = lot.door[1] + (lot.side === 'n' ? -0.4 : 0.4)
        const f = still || r.snap ? 1 : 1 - Math.exp(-dt * 5)
        r.snap = false
        r.x += (tx - r.x) * f
        r.y += (ty - r.y) * f
        if (Math.abs(tx - r.x) + Math.abs(ty - r.y) < 0.05) r.h = lot.side === 'n' ? -Math.PI / 2 : Math.PI / 2
      })
    }
    for (const r of this.residents) if (r.walk) r.fan = 0
    for (const r of this.residents) {
      const st = this.status(r)
      const indoors = st.mode === 'asleep' || (st.mode === 'dreaming' && !r.walk)
      const moving = !!r.walk
      const swing = moving && !still ? Math.sin(secs * 9 + r.phase * 6) * 0.55 : 0
      r.fig.legs[0].rotation.z = swing
      r.fig.legs[1].rotation.z = -swing
      pose(r.fig.group, r.x, r.y, r.h, moving && !still ? Math.abs(Math.sin(secs * 9 + r.phase * 6)) * 0.08 : 0)
      r.fig.group.visible = !indoors
      r.hit.visible = !indoors
      pose(r.hit, r.x, r.y)
      // the pin: over their head, or over the roof when they're indoors; a slow turn, never a bounce
      const talking = st.mode === 'talking'
      const k = Math.max(1, PIN_PX / (2.7 * this.camera.zoom)) * (talking ? 1.25 : 1)
      const top = (indoors && r.dest ? (this.lotTop.get(r.dest) ?? 6) + 1.5 : 3.6) + 1.6 * k
      const bob = still ? 0 : Math.sin(secs * 1.6 + r.phase * 6) * 0.12 * k
      // a row of pins over a shared place runs straight across the screen: along x and against y
      const fx = r.fan * 1.5 * k * Math.SQRT1_2
      r.pin.group.position.copy(W(r.x + fx, r.y - fx, top + bob))
      r.pin.group.rotation.y = still ? 0 : secs * 0.8 + r.phase * 6
      r.pin.group.scale.setScalar(k)
      const foot = indoors ? (this.lotTop.get(r.dest ?? '') ?? 6) * 0.6 : 3.3
      const tip = r.pin.group.position.clone().add(new THREE.Vector3(0, -1.3 * k, 0))
      const base = W(r.x, r.y, foot)
      r.stem.position.copy(base)
      r.stem.scale.set(1, 1, 1)
      r.stem.geometry.setPositions([0, 0, 0, tip.x - base.x, tip.y - base.y, tip.z - base.z])
      r.stem.visible = tip.y - base.y > 0.5
      r.pin.fill.opacity = st.mode === 'asleep' ? 0.55 : 1
      r.ring.visible = talking
      if (talking) {
        const f = still ? 0.5 : (secs * 0.9 + r.phase) % 1
        r.ring.scale.setScalar(1 + f * 2.2)
        ;(r.ring.material as THREE.MeshBasicMaterial).opacity = 0.85 * (1 - f)
        r.ring.position.copy(W(r.x, r.y, 0.1))
      }
      this.dream(r, st.mode === 'dreaming', secs)
    }

    // ---- townsfolk, fewer out at night
    const n = nightAt(t)
    const m = ((t % DAY_MIN) + DAY_MIN) % DAY_MIN
    const density = m < 6 * 60 || m > 22 * 60 ? 0.15 : 1 - n * 0.6
    for (const f of this.folk) {
      f.s += f.speed * dt * (still ? 0 : 1)
      const q = loopAt(f.loop, f.s)
      const swing = still ? 0 : Math.sin(secs * 7 + f.shy * 20) * 0.5
      f.fig!.legs[0].rotation.z = swing
      f.fig!.legs[1].rotation.z = -swing
      pose(f.group, q.x, q.y, f.speed < 0 ? q.h + Math.PI : q.h)
      f.group.visible = f.shy < density
    }
    for (const c of this.cars) {
      c.s += c.speed * dt * (still ? 0 : 1)
      const q = loopAt(c.loop, c.s)
      pose(c.group, q.x, q.y, q.h)
      c.group.visible = c.shy < Math.max(0.3, density)
    }
    this.moveBus(dt, still)
    this.moveTrain(dt, still)
    this.dressEvents()
    if (this.pendingFocus) this.applyFocus()
    this.drawTags()
    this.drawLabels()
    this.moveBoats(secs, still)
    ;(this.beam.material as THREE.MeshBasicMaterial).opacity = 0.55 * n
    this.beam.visible = n > 0.02
    this.beam.rotation.y = still ? 0.6 : secs * 0.9
    if (this.sel) this.drawSelection()
  }

  private dream(r: Resident, on: boolean, secs: number) {
    const home = r.p.homes[0]
    if (!home) return
    let d = this.dreamCols.get(r.p.id)
    if (on && !d) {
      const lot = this.plan.lots.get(home)
      if (!lot) return
      const top = this.lotTop.get(home) ?? 6
      const glow = new THREE.Color(r.color).lerp(new THREE.Color(0xffffff), 0.35)
      const geo = new THREE.CylinderGeometry(1.4, 3.2, 22, 20, 1, true).translate(0, 11, 0)
      const beam = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: glow, transparent: true, opacity: 0.3, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }))
      const N = 26
      const motes = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(N * 3), 3))
      const pts = new THREE.Points(motes, new THREE.PointsMaterial({ color: glow, size: 3, sizeAttenuation: false, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending }))
      const g = new THREE.Group()
      g.add(beam, pts)
      g.traverse((c) => (c.raycast = () => {}))
      g.position.copy(W(lot.door[0] + r.fan * 2, lot.y + lot.d / 2, top * 0.5))
      d = g
      this.dreamCols.set(r.p.id, g)
      this.scene.add(g)
    }
    if (!d) return
    d.visible = on
    if (!on) return
    const [beam, pts] = d.children as [THREE.Mesh, THREE.Points]
    ;(beam.material as THREE.MeshBasicMaterial).opacity = 0.22 + 0.12 * (0.5 + 0.5 * Math.sin(secs * 1.4 + r.phase * 5))
    const pos = pts.geometry.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < pos.count; i++) {
      const f = (secs * 0.12 + i / pos.count + r.phase) % 1 // each mote rises through the beam and starts again
      const a = i * 2.4 + secs * 0.5
      const rad = 0.6 + 2.2 * (1 - f)
      pos.setXYZ(i, Math.cos(a) * rad, f * 24, Math.sin(a) * rad)
    }
    pos.needsUpdate = true
  }

  private beam: THREE.Mesh
  private boats: { group: THREE.Group; x0: number; x1: number; y: number; period: number; phase: number }[] = []
  private moveBoats(secs: number, still: boolean) {
    for (const b of this.boats) {
      const f = still ? 0.5 : 0.5 - 0.5 * Math.cos(((secs / b.period) + b.phase) * Math.PI * 2)
      const dir = Math.sin(((secs / b.period) + b.phase) * Math.PI * 2) >= 0 ? 0 : Math.PI
      pose(b.group, b.x0 + (b.x1 - b.x0) * f, b.y, dir, still ? 0 : Math.sin(secs * 1.3 + b.phase * 9) * 0.08)
    }
  }

  private moveBus(dt: number, still: boolean) {
    const b = this.bus
    if (b.wait > 0) b.wait -= dt
    else {
      const prev = b.s
      b.s = (b.s + (still ? 0 : 6) * dt) % b.loop.length
      for (const st of b.stops) if (prev < st && b.s >= st) b.wait = 3
    }
    const q = loopAt(b.loop, b.s)
    pose(b.group, q.x, q.y, q.h)
  }

  private moveTrain(dt: number, still: boolean) {
    const tr = this.train
    if (tr.wait > 0) tr.wait -= dt
    else if (tr.rest > 0) {
      tr.rest -= dt
      tr.group.visible = false
      return
    } else {
      const prev = tr.x
      tr.x += tr.dir * (still ? 0 : 16) * dt
      for (const s of tr.stops) if ((prev - s) * (tr.x - s) <= 0 && prev !== s) {
        tr.x = s
        tr.wait = 4
      }
      if (tr.x > tr.x1 || tr.x < tr.x0) {
        tr.dir = tr.dir === 1 ? -1 : 1
        tr.x = tr.dir === 1 ? tr.x0 : tr.x1
        tr.rest = 14
      }
    }
    tr.group.visible = true
    pose(tr.group, tr.x, this.plan.rail.y, tr.dir === 1 ? 0 : Math.PI)
  }

  private dressEvents() {
    const day = Math.floor(this.t / DAY_MIN)
    if (day === this.eventDay) return
    this.eventDay = day
    if (this.eventPart) {
      this.scene.remove(this.eventPart)
      this.eventPart = null
    }
    for (const f of this.crowd) this.scene.remove(f.group)
    this.crowd = []
    const d = this.o.town.days[day]
    const p = new Part()
    for (const ev of this.o.town.events.filter((e) => e.date <= d && d <= (e.end_date ?? e.date))) {
      for (const pid of ev.places) {
        const lot = this.plan.lots.get(pid)
        if (!lot) continue
        buildEventDressing(p, lot, this.lotTop.get(pid) ?? 6)
        // townsfolk gather at an event's door
        const r = rng(`crowd:${ev.id}:${pid}`)
        for (let k = 0; k < 7; k++) {
          const f = buildFigure(this.m, `${ev.id}:${pid}:${k}`, null, 1.15)
          const x = lot.door[0] + (r() - 0.5) * Math.min(lot.w, 14)
          const y = lot.door[1] + (lot.side === 'n' ? -1 : 1) * (0.6 + r() * 1.6)
          pose(f.group, x, y, lot.side === 'n' ? Math.PI / 2 : -Math.PI / 2)
          this.scene.add(f.group)
          this.crowd.push(f)
        }
      }
    }
    if (!p.empty) {
      this.eventPart = p.build(this.m, 'events')
      this.eventPart.traverse((c) => (c.raycast = () => {}))
      this.scene.add(this.eventPart)
    }
  }

  // ---------------------------------------------------------------- selection, tags, picking

  private worldOf(s: Sel): THREE.Vector3 | null {
    if (!s) return null
    if (s.kind === 'person') {
      const r = this.residents.find((x) => x.p.id === s.id)
      return r ? W(r.x, r.y, 0) : null
    }
    if (s.kind === 'place') {
      const l = this.plan.lots.get(s.id)
      return l ? W(l.x + l.w / 2, l.y + l.d / 2, 0) : null
    }
    if (s.kind === 'bus') return this.bus.group.position.clone()
    return this.train.group.position.clone()
  }

  private screenOf(v: THREE.Vector3): [number, number] {
    const p = v.clone().project(this.camera)
    return [((p.x + 1) / 2) * this.o.stage.clientWidth, ((1 - p.y) / 2) * this.o.stage.clientHeight]
  }

  private labelOf(s: Sel): string {
    if (!s) return ''
    if (s.kind === 'person') {
      const r = this.residents.find((x) => x.p.id === s.id)!
      const st = this.status(r)
      const where = (id: string | null) => (id ? this.plan.lots.get(id)?.place.name ?? '' : '')
      const word = { asleep: 'asleep', here: `at ${where(st.at)}`, walking: `to ${where(st.to)}`, talking: 'talking to the assistant', dreaming: 'memory dreaming' }[st.mode]
      return `${r.p.name.split(' ')[0]} · ${word}`
    }
    if (s.kind === 'place') return this.plan.lots.get(s.id)?.place.name ?? s.id
    return s.kind === 'bus' ? 'Bus · town loop' : 'Tule Line'
  }

  private anchor(s: Sel): THREE.Vector3 | null {
    if (!s) return null
    if (s.kind === 'person') {
      const r = this.residents.find((x) => x.p.id === s.id)
      return r ? r.pin.group.position.clone().add(W(0, 0, 1.6)) : null
    }
    if (s.kind === 'place') {
      const l = this.plan.lots.get(s.id)
      return l ? W(l.x + l.w / 2, l.y + l.d / 2, (this.lotTop.get(s.id) ?? 6) + 1) : null
    }
    return this.worldOf(s)?.add(W(0, 0, 5)) ?? null
  }

  private placeTag(el: HTMLElement, s: Sel) {
    const a = this.anchor(s)
    if (!s || !a) {
      el.hidden = true
      return
    }
    const [x, y] = this.screenOf(a)
    const text = this.labelOf(s)
    if (el.textContent !== text) el.textContent = text
    el.hidden = false
    const hw = el.offsetWidth / 2 + 6
    el.style.left = `${clamp(x, hw, this.o.stage.clientWidth - hw)}px`
    el.style.top = `${clamp(y, el.offsetHeight + 12, this.o.stage.clientHeight)}px`
  }

  private drawTags() {
    this.placeTag(this.o.tagSel, this.sel)
    const h = this.hovered && (this.hovered.kind !== this.sel?.kind || this.hovered.id !== this.sel?.id) ? this.hovered : null
    this.placeTag(this.o.tagHover, h)
  }

  private drawSelection() {
    const s = this.sel
    this.selRing.visible = false
    this.reticle.visible = false
    this.routeLine.visible = false
    if (!s) return
    if (s.kind === 'person') {
      const r = this.residents.find((x) => x.p.id === s.id)
      if (!r) return
      if (r.fig.group.visible) {
        this.selRing.visible = true
        this.selRing.position.copy(W(r.x, r.y, 0.12))
      }
      // the way they are walking, or the day's round trip from home to the day's place
      let pts: [number, number][] = []
      if (r.walk) pts = r.walk.pts.slice(0)
      else {
        const day = Math.floor(this.t / DAY_MIN)
        const away = r.p.days[day]?.place
        const home = r.p.homes[0]
        if (away && home && away !== home) pts = this.plan.walk(home, away)
      }
      if (pts.length > 1) {
        const seg: number[] = []
        const w = new PathWalk(pts)
        for (let d = r.walk ? r.s : 0; d < w.length; d += 2.2) {
          const a = w.at(d)
          const b = w.at(Math.min(w.length, d + 1.2))
          seg.push(a.x, 0.3, a.y, b.x, 0.3, b.y)
        }
        if (seg.length) {
          this.routeLine.geometry.dispose()
          this.routeLine.geometry = new LineSegmentsGeometry().setPositions(seg)
          this.routeMat.color.set(r.color)
          this.routeLine.visible = true
        }
      }
      return
    }
    if (s.kind === 'place') {
      const l = this.plan.lots.get(s.id)
      if (!l) return
      const L = Math.min(4, l.w / 3, l.d / 3)
      const x0 = l.x - 0.8
      const y0 = l.y - 0.8
      const x1 = l.x + l.w + 0.8
      const y1 = l.y + l.d + 0.8
      const p: number[] = []
      for (const [x, dx] of [[x0, 1], [x1, -1]] as const)
        for (const [y, dy] of [[y0, 1], [y1, -1]] as const) p.push(x, 0.2, y, x + dx * L, 0.2, y, x, 0.2, y, x, 0.2, y + dy * L)
      this.reticle.geometry.dispose()
      this.reticle.geometry = new LineSegmentsGeometry().setPositions(p)
      this.reticle.visible = true
    }
  }

  private hit(cx: number, cy: number): Sel {
    const r = this.o.canvas.getBoundingClientRect()
    const ndc = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1)
    this.raycaster.setFromCamera(ndc, this.camera)
    // people and pins first: they are small and drawn over everything
    const people = this.residents.flatMap((x) => (x.fig.group.visible ? [x.hit, x.pin.group.children[0]] : [x.pin.group.children[0]]))
    const hp = this.raycaster.intersectObjects(people, false)[0]
    if (hp) return hp.object.userData.sel as Sel
    const moving = [this.bus.group, ...(this.train.group.visible ? [this.train.group] : [])]
    const hv = this.raycaster.intersectObjects(moving, true)[0]
    const hb = this.raycaster.intersectObject(this.buildings, true)[0]
    if (hv && (!hb || hv.distance < hb.distance)) return hv.object.userData.sel as Sel
    if (hb) {
      const x = hb.point.x
      const y = hb.point.z
      let best: Lot | null = null
      for (const l of this.plan.lots.values()) {
        if (x >= l.x - 0.9 && x <= l.x + l.w + 0.9 && y >= l.y - 0.9 && y <= l.y + l.d + 0.9 && (!best || l.w * l.d < best.w * best.d)) best = l
      }
      if (best) return { kind: 'place', id: best.place.id }
    }
    return null
  }

  private onDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY, t: performance.now() }
  }
  private onUp = (e: PointerEvent) => {
    if (this.down && Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) < 6 && performance.now() - this.down.t < 700) {
      this.select(this.hit(e.clientX, e.clientY))
    }
    this.down = null
  }
  private onMove = (e: PointerEvent) => {
    if (this.down && Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) >= 6) {
      this.goal = null
      this.follow = false
    }
    if (e.pointerType === 'mouse' && !e.buttons) {
      this.pointer = [e.clientX, e.clientY]
      this.hovered = this.hit(...this.pointer)
      this.o.canvas.classList.toggle('over', !!this.hovered)
    }
  }
  private onLeave = () => {
    this.pointer = null
    this.hovered = null
    this.o.canvas.classList.remove('over')
  }
  private onWheel = () => {
    this.goal = null
  }
}
