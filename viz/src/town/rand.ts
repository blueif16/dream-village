// Seeded randomness keyed by ids, never by build order: adding a place or a resident never reshuffles the others.

/** FNV-1a: a stable 32-bit hash of a string. */
export function hash(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** mulberry32: a small fast generator in [0, 1), seeded from a key. */
export function rng(key: string): () => number {
  let s = hash(key)
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
