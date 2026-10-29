# Dream Village inspector

A read-only browser for the Dream Village (full13-v1), its v0 memory replay and the scorer's outputs. Static: the app
only fetches the JSON bundle that `export.py` writes to `public/data/`.

From the repo root:

```bash
.venv/bin/python viz/export.py          # write viz/public/data/ (re-run as scorer files appear)
cd viz && npm install
npm run dev                             # http://localhost:5173: the landing page; Explore the town opens #/town
```

`export.py --village --replay --scorer --out` override the default paths (the main checkout's `data/villages/full13-v1`,
`runs/replays/full13-v1-v0`, `runs/scorer/full13-v1-v0`).

## Town view

The landing page (`#/`) shows the live town through a wide, flat window; scrolling down or Explore the town opens that
window out into the whole console (`#/town`), the same engine, by easing its clip-path, and Back closes it again.
`#/behind` explains how the town is made and how its memory is questioned, with the season's numbers and one real probe
from the bundle. Every page's nav has Explore the town; the landing page's also shows the town's clock as it runs
(`src/clock.ts`).

Behind the site pages is the meadow (`src/Meadow.tsx`): a canvas of 4 px blocks, ordered-dithered over a small meadow
palette and drifting on periodic paths. It glows round the window (or the page's column) with sparks running the rim.
It's vivid on the landing page, soft behind `#/behind`, and calm and still round the town.

The console is the live map inset in one rounded panel, with
the season strip, play controls and a dock below. Only the map has night; the panels stay light. The dock opens
Residents, Places, Events and Memory; clicking anyone or anything opens its panel on the right. A resident's panel shows
their day, their talks and what their assistant remembers as of the moment shown, with the lines the latest write added
marked. The village's data pages (`#/data`, `#/p/<pid>`) have no header, only a button back to the town.

The map is a live isometric plate of the town in three.js, drawn in the hairline style of [Factory Yard](https://github.com/Khalidabdi1/factory)
(its iso kernel is adapted under MIT; the notice is in `src/town/LICENSE.factory-yard`). Everything on it comes from
`town.json`, so more places, districts and residents need no code:

- **The plan** (`src/town/plan.ts`) packs each district's places into blocks on a street grid, with stations on the rail.
- **The buildings** (`src/town/models.ts`) are drawn per kind and varied by place id. A kind it doesn't know gets a
  generic building.
- **Residents follow the data** (`src/town/schedule.ts`): home at night, at the day's place between a seeded leave and
  return, glowing when a write commit says they talked to the assistant, their memory dreaming over home at 03:00.
- **Townsfolk, cars, the bus, the train and the boats** are periodic scenery.

`npm test` runs the plan, schedule, colour and console (`src/town/insight.ts`) tests. `?debug` (e.g. `http://localhost:5173/?debug#/town`) exposes the
engine as `window.town` for headless checks: `setTime(t)` (minutes since the first day's midnight), `setPlaying`,
`select({ kind, id })`, `focus`, `goTo(district)`.
