import type { Look } from './draw'

// One colour per principal, used for their pixel character and their swatch everywhere in the Town view.
const BODY: Record<string, number> = {
  arjun: 0xc0573e, dulce: 0xe3a33b, elliot: 0x3d5a80, grace: 0x6f9e76, isabel: 0x9a6fb0, jun: 0xd1495b,
  linh: 0x2a9d8f, marcus: 0x5b7a99, noemi: 0xee8959, rafael: 0x7d9a3a, rochelle: 0xb5838d, thao: 0x277da1,
}
const HAIR = [0x2b2118, 0x3b2a1e, 0x1d1a17, 0x5a3b24, 0x6b4a2f]
const SKIN = [0xe8b98f, 0xc98e62, 0xa86f45, 0xf0c8a0, 0x8d5a3b]

function hash(s: string): number {
  let h = 7
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return h
}

export function look(id: string): Look {
  const h = hash(id)
  return { body: BODY[id] ?? 0x777777, hair: HAIR[h % HAIR.length], skin: SKIN[(h >>> 3) % SKIN.length] }
}

export const css = (n: number) => `#${n.toString(16).padStart(6, '0')}`
