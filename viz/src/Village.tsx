import { DAY, fmtDay, fmtPct, ms } from './data'
import type { Index, Round1, Round1Cell, Round1Dim, Round1Run, ScorecardSystem } from './types'

const FREQ: Record<string, string> = { daily: 'daily', several_per_week: 'several / week', weekly: 'weekly' }
const DIMS = ['remembering', 'understanding', 'restraint', 'task'] as const
const SYSTEM_ORDER = ['v0', 'no_memory', 'recent_window', 'oracle', 'full_context']

export function systemOrder(names: string[]): string[] {
  const rank = (s: string) => (SYSTEM_ORDER.includes(s) ? SYSTEM_ORDER.indexOf(s) : 99)
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
}

function Pass({ ok }: { ok: boolean | null | undefined }) {
  if (ok === null || ok === undefined) return <span className="badge none">— n/a</span>
  return ok ? <span className="badge pass">✓ PASS</span> : <span className="badge fail">✗ FAIL</span>
}

function Scorecard({ index }: { index: Index }) {
  const card = index.scorecard
  const s = index.scorer
  if (!card) {
    const rows: [string, string][] = [
      ['probes.jsonl', s.probes === null ? 'not yet generated' : `${s.probes} probes`],
      ['answers.jsonl', s.answers === null ? 'not yet generated'
        : systemOrder(Object.keys(s.answers)).map((k) => `${k} ${s.answers![k]}`).join(' · ')],
      ['verdicts.jsonl', s.verdicts === null ? 'not yet generated' : `${s.verdicts} verdicts`],
      ['scorecard.json', 'not yet generated'],
    ]
    return (
      <>
        <p className="section-note">
          No scorecard yet. The scorer's files as of this export (re-run the export as they appear):
        </p>
        <ul className="status-list">
          {rows.map(([f, v]) => (
            <li key={f}>
              <span className="mono">{f}</span>
              <span className={v === 'not yet generated' ? 'muted' : undefined}>{v}</span>
            </li>
          ))}
        </ul>
      </>
    )
  }
  const systems = systemOrder(Object.keys(card.systems))
  const gateNames = [...new Set(systems.flatMap((n) => Object.keys(card.systems[n].gates)))]
  return (
    <>
      <p className="section-note">
        {card.probes} probes{card.provisional ? ' · provisional' : ''}
        {card.rubric_version ? ` · rubric ${card.rubric_version}` : ''}. Share of decided checks passing (n checks).
      </p>
      <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>system</th>
            {DIMS.map((d) => <th key={d} className="num">{d}</th>)}
            <th className="num">integrity</th>
            <th className="num">ctx tokens</th>
            {gateNames.map((g) => <th key={g}>{g.replace(/_/g, ' ')}</th>)}
          </tr>
        </thead>
        <tbody>
          {systems.map((name) => {
            const row: ScorecardSystem = card.systems[name]
            return (
              <tr key={name}>
                <td className="mono">{name}</td>
                {DIMS.map((d) => (
                  <td key={d} className="num" title={row[d].ci_probes ? `CI ${row[d].ci_probes!.map(fmtPct).join('–')}` : undefined}>
                    {fmtPct(row[d].score)} <span className="muted">({row[d].n ?? 0})</span>
                  </td>
                ))}
                <td className="num">{row.integrity ? fmtPct(row.integrity.score) : '—'}</td>
                <td className="num">{row.cost.context_tokens_mean ? Math.round(row.cost.context_tokens_mean).toLocaleString() : '—'}</td>
                {gateNames.map((g) => <td key={g}><Pass ok={row.gates[g]} /></td>)}
              </tr>
            )
          })}
        </tbody>
      </table>
      </div>
    </>
  )
}

// ------------------------------------------------------------------ optimization round 1

const R1_DIMS: [Round1Dim, string, string][] = [
  ['remembering', 'Remembering', 'Rem.'],
  ['understanding', 'Understanding', 'Und.'],
  ['restraint', 'Restraint', 'Restr.'],
  ['integrity', 'Integrity', 'Integ.'],
]

/** Points as the exporter wrote them (already rounded), with a typographic minus. */
function pts(x: number | null | undefined, plus = true): string {
  if (x === null || x === undefined) return '—'
  const s = Math.abs(x).toFixed(1)
  return x < 0 ? `−${s}` : x > 0 && plus ? `+${s}` : s
}

const ciText = (ci: [number, number]) => `[${pts(ci[0], false)}, ${pts(ci[1], false)}]`

/** Text with `backticked` spans set as code. */
function Ticks({ text }: { text: string }) {
  return <>{text.split('`').map((part, i) => (i % 2 ? <code key={i} className="mono">{part}</code> : part))}</>
}

/** The principal count shared by these runs, for a column header; nothing if they differ or none is known. */
function nOf(runs: (Round1Run | null)[]): string {
  const ns = [...new Set(runs.flatMap((r) => (r?.principals ? [r.principals.length] : [])))]
  return ns.length === 1 ? ` (n=${ns[0]})` : ''
}

function Delta({ cell }: { cell: Round1Cell | null }) {
  if (!cell) return <>—</>
  return (
    <>
      {pts(cell.delta)}
      {cell.source === 'deltas' && '†'}
      {cell.ci && <span className="ci">{ciText(cell.ci)}</span>}
    </>
  )
}

function RunCells({ run }: { run: Round1Run | null }) {
  if (!run) return <td colSpan={R1_DIMS.length} className="gstart muted">not run</td>
  if (!run.present || !run.dims) {
    return (
      <td colSpan={R1_DIMS.length} className="gstart muted" title={run.scorecard}>
        {run.present ? 'no comparison in the scorecard' : 'scorecard absent'}
      </td>
    )
  }
  return (
    <>
      {R1_DIMS.map(([d], i) => (
        <td key={d} className={i === 0 ? 'num gstart' : 'num'} title={`${run.scorecard}/scorecard.json`}>
          <Delta cell={run.dims![d]} />
        </td>
      ))}
    </>
  )
}

function Round1Panel({ r }: { r: Round1 }) {
  const nf = r.noise_floor
  const runs = [nf, ...r.candidates.flatMap((c) => [c.screen, c.confirm])]
  const provisional = [...new Set(runs.flatMap((x) => (x?.provisional ? [x.provisional] : [])))]
  const dagger = runs.some((x) => x?.dims && Object.values(x.dims).some((c) => c?.source === 'deltas'))
  return (
    <section className="round1">
      <h2 className="section-title">Optimization round 1</h2>
      <p className="round1-lesson">{r.lesson}</p>
      <p className="section-note">
        Each candidate changes v0. It is screened on a few principals and, if it passes, confirmed on all of them.
        Δ is candidate minus v0 in points, with the CI over principals in brackets.
      </p>
      <div className="table-scroll">
        <table className="data round1">
          <thead>
            <tr>
              <th rowSpan={2}>candidate</th>
              <th rowSpan={2}>change</th>
              <th colSpan={R1_DIMS.length} className="group gstart">screen Δ{nOf(r.candidates.map((c) => c.screen))}</th>
              <th colSpan={R1_DIMS.length} className="group gstart">confirm Δ{nOf(r.candidates.map((c) => c.confirm))}</th>
              <th rowSpan={2} className="gstart">verdict</th>
            </tr>
            <tr>
              {['screen', 'confirm'].flatMap((g) =>
                R1_DIMS.map(([d, label, abbr], i) => (
                  <th key={`${g}-${d}`} className={i === 0 ? 'num gstart' : 'num'}>
                    <abbr title={label}>{abbr}</abbr>
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {r.candidates.map((c) => (
              <tr key={c.id}>
                <td className="mono">{c.id}</td>
                <td className="change"><Ticks text={c.change} /></td>
                <RunCells run={c.screen} />
                <RunCells run={c.confirm} />
                <td className="gstart">
                  <span className="verdict">{c.verdict}</span>
                  {c.reason && <span className="reason">{c.reason}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="round1-notes">
        <li>{r.rule}</li>
        <li title={`${nf.scorecard}/scorecard.json`}>
          A/A noise floor{nf.label ? ` (${nf.label})` : ''}:{' '}
          {nf.dims
            ? R1_DIMS.map(([d, label]) => {
                const c = nf.dims![d]
                return `${label} ${c ? pts(c.delta) + (c.source === 'deltas' ? '†' : '') + (c.ci ? ' ' + ciText(c.ci) : '') : '—'}`
              }).join(' · ')
            : nf.present ? 'no comparison in the scorecard' : 'scorecard absent'}
        </li>
        {dagger && (
          <li>
            † Integrity has no CI: the scorer pairs only the check dimensions, so this is its unpaired delta. On a screen
            it compares the screened principals with v0 on the whole village.
          </li>
        )}
        {provisional.map((p) => <li key={p}>{p}</li>)}
      </ul>
    </section>
  )
}

export function Village({ index, onOpen }: { index: Index; onOpen: (pid: string) => void }) {
  const weeks = Math.round((ms(index.range.end) - ms(index.range.start) + DAY) / (7 * DAY))
  const total = index.principals.reduce(
    (a, p) => ({ sessions: a.sessions + p.sessions, commits: a.commits + p.commits, probes: a.probes + p.probes }),
    { sessions: 0, commits: 0, probes: 0 },
  )
  const people = [...index.principals].sort((a, b) => (a.age ?? 0) - (b.age ?? 0) || a.name.localeCompare(b.name))
  return (
    <div className="village">
      <section className="hero">
        <div>
          <h1>{index.town.name}</h1>
          <div className="sub">{index.town.region}</div>
          <p className="summary">{index.town.summary}</p>
        </div>
        <dl className="hero-facts">
          <dt>village</dt>
          <dd className="mono">{index.village}</dd>
          <dt>span</dt>
          <dd>
            {weeks} weeks · {fmtDay(ms(index.range.start))} – {fmtDay(ms(index.range.end))}, {index.range.end.slice(0, 4)}
          </dd>
          <dt>principals</dt>
          <dd>{index.principals.length}</dd>
          <dt>sessions</dt>
          <dd>{total.sessions}</dd>
          <dt>memory commits</dt>
          <dd>{total.commits} (v0 replay)</dd>
          <dt>probes</dt>
          <dd>{index.scorer.probes ?? 'not yet generated'}</dd>
          <dt>redacted</dt>
          <dd>{index.redactions} never-store spans</dd>
        </dl>
      </section>

      <h2 className="section-title">Principals</h2>
      <p className="section-note">
        The twelve residents with an assistant. Memory lines are counted at each one's last v0 commit. Open a card for the
        timeline.
      </p>
      <div className="cards">
        {people.map((p) => (
          <button key={p.id} className="card" onClick={() => onOpen(p.id)}>
            <div className="card-head">
              <span className="name">{p.name}</span>
              <span className="age">{p.minor ? 'minor' : `${p.age}`}</span>
            </div>
            <div className="role">{p.role}</div>
            <div className="card-stats">
              <div className="stat">
                <b>{p.sessions}</b>
                <span>sessions</span>
              </div>
              <div className="stat mem">
                <b>{p.memory_lines}</b>
                <span>memory lines</span>
              </div>
              <div className="stat">
                <b>{p.probes}</b>
                <span>probes</span>
              </div>
            </div>
            <div className="card-foot">
              <span>
                {FREQ[p.usage.frequency] ?? p.usage.frequency} · {p.verify_passed}/{p.sessions} verified
              </span>
              <span>
                {p.writes} writes · {p.dreams} dreams · {p.memory_files} files
              </span>
            </div>
          </button>
        ))}
      </div>

      <div className="two-col">
        <div>
          <h2 className="section-title">Generation gate</h2>
          <p className="section-note">From the village manifest: what the generator checked before the data was used.</p>
          {index.gate ? (
            <table className="data">
              <thead>
                <tr>
                  <th>check</th>
                  <th>value</th>
                  <th>threshold</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {index.gate.map((g) => (
                  <tr key={g.check}>
                    <td>{g.check}</td>
                    <td className="mono">{g.value}</td>
                    <td className="mono">{g.threshold}</td>
                    <td><Pass ok={g.passed} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="empty">The manifest has no gate.</p>
          )}
        </div>
        <div>
          <h2 className="section-title">Scorecard</h2>
          <Scorecard index={index} />
        </div>
      </div>

      {index.round1 && <Round1Panel r={index.round1} />}
    </div>
  )
}
