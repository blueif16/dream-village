// Pixel buildings and people, drawn as integer rects (no assets). One restrained palette per kind of place.
import { Graphics } from 'pixi.js'
import type { Lot } from './layout'

const INK = 0x2a2e38
const WIN = 0x5b6c7c
const DOOR = 0x4a3b33
const SHADE = 0x000000

function r(g: Graphics, x: number, y: number, w: number, h: number, color: number, alpha = 1) {
  if (w > 0 && h > 0) g.rect(x, y, w, h).fill({ color, alpha })
}

function windows(g: Graphics, x: number, y: number, w: number, h: number, color = WIN, step = 4) {
  for (let yy = y; yy + 2 <= y + h; yy += step) for (let xx = x; xx + 2 <= x + w; xx += step) r(g, xx, yy, 2, 2, color)
}

/** A flat-roofed box: roof band, eave, walls, a window grid and a door. */
function box(g: Graphics, l: Lot, wall: number, roof: number, opts: { win?: number; door?: boolean } = {}) {
  const { x, y, w, h } = l
  r(g, x + 1, y + h, w, 1, SHADE, 0.12) // ground shadow
  r(g, x, y, w, h, wall)
  r(g, x, y, w, 3, roof)
  r(g, x, y + 3, w, 1, SHADE, 0.15)
  windows(g, x + 3, y + 6, w - 5, h - 12, opts.win ?? WIN)
  if (opts.door !== false) r(g, x + Math.floor(w / 2) - 1, y + h - 5, 3, 5, DOOR)
}

/** A stepped (pixel) gable roof over a wall. */
function gable(g: Graphics, x: number, y: number, w: number, color: number) {
  const steps = Math.ceil(w / 4)
  for (let i = 0; i < steps; i++) r(g, x + i * 2, y - i, w - i * 4, 1, color)
}

const AWNING: Record<string, number> = {
  cafe: 0x2c6b66, taqueria: 0xc0573e, pho_restaurant: 0xd1a03a, bakery: 0xd98c8c, gym: 0x50565f,
  laundromat: 0x7f98ab, grocery_store: 0x5e8f63,
}
const SHOP_WALL = [0xeadbc3, 0xe3cdb4, 0xd9dcc8, 0xe8d6b8]

function shop(g: Graphics, l: Lot, i: number) {
  const { x, y, w, h } = l
  const aw = AWNING[l.place.kind] ?? 0x9b8a74
  r(g, x + 1, y + h, w, 1, SHADE, 0.12)
  r(g, x, y, w, h, SHOP_WALL[i % SHOP_WALL.length])
  r(g, x, y, w, 2, 0x8c7b68) // parapet
  windows(g, x + 3, y + 3, w - 5, 3, WIN, 4)
  for (let k = 0; k < w; k += 2) r(g, x + k, y + 7, 1, 3, aw) // striped awning
  for (let k = 1; k < w; k += 2) r(g, x + k, y + 7, 1, 3, 0xf3ece0)
  r(g, x, y + 10, w, 1, SHADE, 0.15)
  r(g, x + 2, y + 11, w - 8, h - 11, 0x8fa9b6) // display window
  r(g, x + w - 5, y + 11, 3, h - 11, DOOR)
}

export function drawLot(g: Graphics, l: Lot, i: number) {
  const { x, y, w, h } = l
  switch (l.place.kind) {
    case 'high_school':
    case 'middle_school':
      box(g, l, 0xe6d6b4, 0xb0573f)
      r(g, x + 2, y - 7, 1, 7, INK) // flagpole
      r(g, x + 3, y - 7, 3, 2, 0x3d5a80)
      break
    case 'community_college':
      box(g, l, 0xd9d1c3, 0x5f7686)
      break
    case 'state_university':
      box(g, l, 0xd3cabb, 0x3d5a80, { door: false })
      gable(g, x + Math.floor(w / 2) - 9, y - 1, 18, 0x3d5a80)
      for (let k = x + Math.floor(w / 2) - 7; k <= x + Math.floor(w / 2) + 6; k += 3) r(g, k, y + h - 9, 1, 9, 0xf2ede3)
      r(g, x + Math.floor(w / 2) - 1, y + h - 5, 3, 5, DOOR)
      break
    case 'library':
      box(g, l, 0xa65e46, 0x4f3529, { win: 0xe9d9a8 })
      break
    case 'hospital':
      box(g, l, 0xeef0ec, 0x8fa3ad, { win: 0x7d9db0 })
      r(g, x + w - 9, y + 5, 6, 2, 0xc0392b)
      r(g, x + w - 7, y + 3, 2, 6, 0xc0392b)
      break
    case 'community_center':
      box(g, l, 0xd8c8a6, 0x8a7560)
      break
    case 'church':
      r(g, x + 1, y + h, w, 1, SHADE, 0.12)
      r(g, x, y + 8, w, h - 8, 0xf3efe5)
      gable(g, x, y + 8, w, 0x9a6b52)
      r(g, x + Math.floor(w / 2) - 2, y - 2, 5, 10, 0xf3efe5) // steeple
      r(g, x + Math.floor(w / 2) - 2, y - 4, 5, 2, 0x9a6b52)
      r(g, x + Math.floor(w / 2), y - 8, 1, 4, INK)
      r(g, x + Math.floor(w / 2) - 1, y - 7, 3, 1, INK)
      r(g, x + Math.floor(w / 2) - 1, y + h - 6, 3, 6, DOOR)
      r(g, x + 3, y + 12, 2, 4, 0x8fa9b6)
      r(g, x + w - 5, y + 12, 2, 4, 0x8fa9b6)
      break
    case 'temple':
      r(g, x + 1, y + h, w, 1, SHADE, 0.12)
      r(g, x, y + 8, w, h - 8, 0xf1e6c8)
      r(g, x + 6, y + 4, w - 12, 4, 0xd9a441) // dome, stepped
      r(g, x + 8, y + 2, w - 16, 2, 0xd9a441)
      r(g, x + Math.floor(w / 2), y - 1, 1, 3, 0xb98a2e)
      windows(g, x + 3, y + 11, w - 5, 3, 0xb98a2e)
      r(g, x + Math.floor(w / 2) - 1, y + h - 5, 3, 5, DOOR)
      break
    case 'theater':
      box(g, l, 0x9e5240, 0x5c3128, { win: 0xe9d9a8 })
      r(g, x + 2, y + h - 10, w - 4, 3, 0xe8c25a) // marquee
      for (let k = x + 3; k < x + w - 3; k += 2) r(g, k, y + h - 9, 1, 1, 0xfff4d0)
      break
    case 'warehouse_employer':
      r(g, x + 1, y + h, w, 1, SHADE, 0.12)
      r(g, x, y, w, h, 0xa9aeb3)
      r(g, x, y, w, 4, 0x8a9096)
      r(g, x, y + 4, w, 1, SHADE, 0.12)
      r(g, x + 4, y + 7, 16, 3, 0x3d5a80) // sign band
      for (let k = x + 3; k + 5 < x + w; k += 7) r(g, k, y + h - 8, 5, 8, 0x5d6268) // truck doors
      break
    case 'industrial_area':
      r(g, x, y + 4, w, h - 4, 0x8e8b84)
      r(g, x + 3, y + 7, 10, 5, 0xb55d3f) // containers
      r(g, x + 14, y + 7, 10, 5, 0x3f6f8f)
      r(g, x + 9, y + 3, 10, 4, 0x6f8f4f)
      r(g, x + w - 12, y + 6, 9, 8, 0xbfb8aa)
      r(g, x + w - 12, y + 6, 9, 2, 0x7d766b)
      break
    case 'apartment_complex': {
      const modern = /verdant/i.test(l.place.name)
      box(g, l, modern ? 0xc9cdd1 : 0xe2d5bf, modern ? 0x6f7b85 : 0x8a6f5a, { win: modern ? 0x9cb6c3 : WIN })
      if (!modern) r(g, x, y + Math.floor(h / 2), w, 1, 0x8a6f5a) // balcony line
      break
    }
    case 'house': {
      const walls = [0xe7d3b5, 0xd9c5a8, 0xcfd3c2, 0xe4c9b0]
      r(g, x + 1, y + h, w, 1, SHADE, 0.12)
      r(g, x + 1, y + 5, w - 2, h - 5, walls[i % walls.length])
      gable(g, x, y + 5, w, 0x7a4b3a)
      r(g, x + 3, y + 8, 2, 2, WIN)
      r(g, x + w - 6, y + h - 5, 3, 5, DOOR)
      break
    }
    case 'park': {
      const marsh = /marsh/i.test(l.place.name)
      r(g, x, y, w, h, marsh ? 0xa9c49a : 0x9cbf7a)
      if (marsh) {
        r(g, x + 2, y + 10, w - 4, h - 13, 0x8fb8c2)
        for (let k = x + 4; k < x + w - 4; k += 5) r(g, k, y + 8, 1, 4, 0x6f8f4f) // reeds
        r(g, x + 2, y + 17, w - 4, 2, 0xa68a64) // boardwalk
        r(g, x + w - 12, y + 3, 2, 1, 0xf6f4ee) // an egret
        r(g, x + w - 11, y + 2, 1, 1, 0xf6f4ee)
      } else {
        r(g, x, y + Math.floor(h / 2), w, 2, 0xe6d7b0)
        r(g, x + Math.floor(w / 2), y, 2, h, 0xe6d7b0)
        for (const [tx, ty] of [[4, 3], [12, 18], [36, 4], [42, 20], [6, 21], [30, 22]]) {
          r(g, x + tx, y + ty, 5, 4, 0x5f8f4e)
          r(g, x + tx + 2, y + ty + 4, 1, 2, 0x6b4f3a)
        }
        r(g, x + Math.floor(w / 2) - 3, y + Math.floor(h / 2) - 3, 8, 7, 0xefe7d6) // bandstand
        r(g, x + Math.floor(w / 2) - 4, y + Math.floor(h / 2) - 4, 10, 2, 0xb0573f)
      }
      break
    }
    case 'transit_station':
      r(g, x - 2, y + h - 3, w + 4, 1, 0x6b6258) // rails
      r(g, x - 2, y + h - 1, w + 4, 1, 0x6b6258)
      for (let k = x - 2; k < x + w + 2; k += 3) r(g, k, y + h - 3, 1, 3, 0x8b7d6b)
      r(g, x, y + 4, w, 6, 0xbdb6a8) // platform
      r(g, x + 4, y, w - 8, 2, 0x4d6b7a) // canopy
      r(g, x + 6, y + 2, 1, 2, INK)
      r(g, x + w - 7, y + 2, 1, 2, INK)
      break
    case 'main_street':
      r(g, x, y + 2, w, h - 2, 0xcbbca2)
      for (let k = x; k < x + w; k += 4) r(g, k, y + 2, 2, h - 2, 0xd6c8ae, 0.9) // pavers
      for (let k = x + 1; k < x + w; k += 3) r(g, k, y + (k % 2), 1, 1, [0xe0a526, 0xc0573e, 0x2c6b66][k % 3]) // string lights
      r(g, x + 2, y + 3, 1, h - 3, INK) // lamp posts
      r(g, x + w - 3, y + 3, 1, h - 3, INK)
      break
    case 'strip_mall':
      r(g, x + 1, y + h, w, 1, SHADE, 0.12)
      r(g, x, y, w, h, 0xd3c7b2)
      r(g, x, y, w, 2, 0x8c7b68)
      for (let k = 0; k < 4; k++) {
        r(g, x + 1 + k * 8, y + 3, 7, 2, [0xc0573e, 0x3f6f8f, 0xd1a03a, 0x5e8f63][k])
        r(g, x + 2 + k * 8, y + 6, 5, h - 6, 0x8fa9b6)
      }
      break
    default:
      shop(g, l, i)
  }
}

/** Outline a lot (event highlight or selection), one pixel outside it. */
export function outline(g: Graphics, l: Lot, color: number, pad = 2) {
  const top = l.place.kind === 'church' ? l.y - 9 : l.y - 2
  g.rect(l.x - pad, top - pad + 2, l.w + pad * 2, l.y + l.h - top + pad * 2 - 1).stroke({ width: 1, color, alignment: 1 })
}

export function pennant(g: Graphics, l: Lot) {
  const px = l.x + l.w - 3
  r(g, px, l.y - 8, 1, 8, INK)
  r(g, px + 1, l.y - 8, 4, 1, 0xe0a526)
  r(g, px + 1, l.y - 7, 3, 1, 0xe0a526)
  r(g, px + 1, l.y - 6, 2, 1, 0xe0a526)
}

export interface Look {
  body: number
  hair: number
  skin: number
}

/** A 5x9 person standing with feet at (0, 0). */
export function drawPerson(g: Graphics, look: Look) {
  r(g, -2, 0, 5, 1, SHADE, 0.18) // shadow
  r(g, -1, -9, 3, 1, look.hair)
  r(g, -2, -8, 1, 2, look.hair)
  r(g, -1, -8, 3, 2, look.skin)
  r(g, 2, -8, 1, 1, look.hair)
  r(g, -2, -6, 5, 4, look.body)
  r(g, -1, -2, 1, 2, INK)
  r(g, 1, -2, 1, 2, INK)
}

export function drawBubble(g: Graphics) {
  r(g, 0, -17, 7, 5, 0xfbfaf6)
  r(g, 1, -12, 1, 1, 0xfbfaf6)
  g.rect(0, -17, 7, 5).stroke({ width: 1, color: 0x8a8577, alignment: 1, alpha: 0.6 })
  r(g, 1, -15, 1, 1, INK)
  r(g, 3, -15, 1, 1, INK)
  r(g, 5, -15, 1, 1, INK)
}

export function drawMarker(g: Graphics, color: number) {
  r(g, -2, -14, 5, 1, color)
  r(g, -1, -13, 3, 1, color)
  r(g, 0, -12, 1, 1, color)
}
