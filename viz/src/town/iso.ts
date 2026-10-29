// The iso kernel, after Factory Yard (MIT © 2026 Khalid Abdi) and its source ai-iso-skill (MIT © 2026 Tolga Cohce);
// see LICENSE.factory-yard. Scene coordinates are the skill's: +x runs down-right, +y down-left, +z up. Three.js is
// y-up, so W swaps y and z, and an orthographic camera looking down (-1,-1,-1) draws the skill's projection exactly.
import * as THREE from 'three'

export type P3 = [number, number, number]

export const W = (x: number, y: number, z = 0) => new THREE.Vector3(x, z, y)

/** A plane through O with in-plane axes U, V: anything drawn in (u, v) through this matrix lies flat on it. */
export function plane(O: P3, U: P3, V: P3): THREE.Matrix4 {
  const u = W(...U)
  const v = W(...V)
  const n = new THREE.Vector3().crossVectors(u, v).normalize()
  return new THREE.Matrix4().makeBasis(u, v, n).setPosition(W(...O))
}

/** z = const, u along +x and v along +y. */
export const TOP = (x: number, y: number, z: number) => plane([x, y, z], [1, 0, 0], [0, 1, 0])
/** y = const, the face toward the lower left; pass its top-left corner. */
export const FRONT = (x: number, y: number, z: number) => plane([x, y, z], [1, 0, 0], [0, 0, -1])
/** x = const, the face toward the lower right; pass its top-left corner. */
export const SIDE = (x: number, y: number, z: number) => plane([x, y, z], [0, -1, 0], [0, 0, -1])
