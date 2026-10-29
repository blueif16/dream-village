// What's behind the town: how its lives are made, how the assistant's memory works, and how we question it.
// Every number is read from the exported bundle; the worked example is a real probe from it.
import { useEffect, useMemo, useState } from 'react'
import { ms, usePrincipal } from './data'
import { SiteNav } from './SiteNav'
import { Txt } from './text'
import { DataGlyph, EventsGlyph, MemoryGlyph, PeopleGlyph, PlacesGlyph } from './town/icons'
import type { Index, PrincipalBundle, Probe, TruthEntry } from './types'
import { systemOrder } from './Village'

const n = (x: number) => x.toLocaleString('en-US')
const pct = (x: number | null | undefined) => (x === null || x === undefined ? '—' : `${Math.round(x * 100)}`)
const LAG: Record<string, string> = {
  same_session: 'in the same talk', same_day: 'later that day', after_consolidation: 'after a night’s dream',
  out_of_window: 'beyond the last ten talks', weeks_later: 'weeks later', months_later: 'months later',
}
const SYSTEM: Record<string, string> = {
  v0: 'The memory under test: its files, plus the last ten talks.',
  no_memory: 'Only the date and the request.',
  recent_window: 'The last ten talks, no memory.',
  oracle: 'The truth known by then, plus the last ten talks: the ceiling.',
  full_context: 'Every earlier talk, word for word.',
}

function sum(index: Index | null, k: 'sessions' | 'writes' | 'dreams' | 'verify_passed' | 'probes') {
  return index ? index.principals.reduce((a, p) => a + p[k], 0) : null
}

function Step({ id, no, tone, glyph, title, children, value, unit }: {
  id: string
  no: string
  tone: string
  glyph: React.ReactNode
  title: string
  children: React.ReactNode
  value: string | null
  unit: string
}) {
  return (
    <li className="step" id={id} style={{ '--tone': `var(--${tone})` } as React.CSSProperties}>
      <div className="step-head"><span className="no">{no}</span>{glyph}</div>
      <div className="step-body">
        <h3>{title}</h3>
        <p>{children}</p>
      </div>
      <div className="step-foot"><b>{value ?? '—'}</b> {unit}</div>
    </li>
  )
}

/** A probe's plan in plain words: item ids dropped, lag names spelled out. */
const plain = (s: string) =>
  s.replace(/\b(?:[a-z]+\.)?[fir]\d{2}\b/g, '')
    .replace(/\((same_session|same_day|after_consolidation|out_of_window|weeks_later|months_later)\)/g, (_, k: string) => `(told ${LAG[k]})`)
    .replace(/^Tests\s*:\s*/, '')
    .replace(/\s+([,.;:)])/g, '$1')
    .replace(/\s{2,}/g, ' ')
    .trim()

/** The worked example: a probe the memory passed and no memory failed, told long before if possible. */
function pickProbe(b: PrincipalBundle): Probe | null {
  const passed = (p: Probe, s: string) => {
    const v = p.verdicts?.[s]
    return v && v.length ? v.every((x) => x.pass) : null
  }
  const shows = b.probes.filter((p) => passed(p, 'v0') === true && passed(p, 'no_memory') === false && p.kind !== 'restraint')
  const far = shows.filter((p) => p.lag === 'months_later' || p.lag === 'weeks_later' || p.lag === 'out_of_window')
  return far[0] ?? shows[0] ?? b.probes.find((p) => p.verdicts) ?? null
}

function Example({ index }: { index: Index }) {
  const who = useMemo(
    () => [...index.principals].filter((p) => !p.minor).sort((a, b) => a.bundle_bytes - b.bundle_bytes)[0]?.id ?? null,
    [index],
  )
  const { data: b } = usePrincipal(who)
  const probe = useMemo(() => (b ? pickProbe(b) : null), [b])
  if (!b || !probe) return <div className="ex-card loading">Reading a real question from the data…</div>
  const truth = new Map(b.truth.map((t) => [t.id, t]))
  const items = (ids: string[]) => ids.flatMap((id) => (truth.get(id) ? [truth.get(id)!] : []))
  const starts = new Map(b.sessions.map((s) => [s.id, s.start]))
  const toldOn = (t: TruthEntry) => t.disclosed_in.flatMap((id) => (starts.has(id) ? [starts.get(id)!] : [])).sort()[0] ?? null
  const day = (iso: string) => new Date(`${iso}Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
  const told = items(probe.must_use).flatMap((t) => (toldOn(t) ? [toldOn(t)!] : [])).sort()
  const [s0, s1] = [ms(index.range.start), ms(index.range.end)]
  const along = (iso: string) => `${((ms(iso) - s0) / (s1 - s0)) * 100}%`
  const name = b.principal.name.split(' ')[0]
  const groups: [string, string, ReturnType<typeof items>][] = [
    ['must', 'It must use', items(probe.must_use)],
    ['not', 'It must not bring up', items(probe.must_not)],
    ['stale', 'No longer true', items(probe.stale)],
  ]
  const when = new Date(`${probe.t}Z`).toLocaleString('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' })
  const systems = systemOrder(Object.keys(probe.answers))
  return (
    <div className="ex-card">
      <div className="ex-ask">
        <p className="kicker">{name} · {when} · {probe.kind}{probe.lag ? `, told ${LAG[probe.lag] ?? probe.lag}` : ''}</p>
        <blockquote><Txt text={probe.request} /></blockquote>
        {groups.filter(([, , xs]) => xs.length).map(([k, label, xs]) => (
          <div key={k} className={`ex-targets ${k}`}>
            <h4>{label}</h4>
            <ul>{xs.map((t) => <li key={t.id}><Txt text={t.text} />{k === 'must' && toldOn(t) && <span className="told">told {day(toldOn(t)!)}</span>}</li>)}</ul>
          </div>
        ))}
        {told.length > 0 && (
          <div className="ex-line" aria-label="When it was told, and when it was asked">
            <div className="track">
              {told.map((iso) => <i key={iso} className="told" style={{ left: along(iso) }} />)}
              <i className="asked" style={{ left: along(probe.t) }} />
            </div>
            <p>
              <i className="told" /> told {day(told[0])} <i className="asked" /> asked {day(probe.t)}
              <span> · {Math.round((ms(probe.t) - ms(told[0])) / 86_400_000)} days later</span>
            </p>
          </div>
        )}
        {probe.criteria.length > 0 && (
          <div className="ex-targets must">
            <h4>A good answer</h4>
            <ul>{probe.criteria.map((c) => <li key={c}>{c}</li>)}</ul>
          </div>
        )}
        {probe.plan && (
          <div className="ex-why">
            <h4>Why this question</h4>
            <p><Txt text={plain(probe.plan)} /></p>
          </div>
        )}
      </div>
      <div className="ex-answers">
        <h4>Asked five ways</h4>
        <ul>
          {systems.map((s) => {
            const v = probe.verdicts?.[s] ?? []
            const ok = v.length ? v.every((x) => x.pass) : null
            const failed = v.filter((x) => x.pass === false).length
            return (
              <li key={s} className={ok === true ? 'pass' : ok === false ? 'fail' : undefined}>
                <span className="sys">{s.replace(/_/g, ' ')}</span>
                <span className="verdict">{ok === null ? 'not judged' : ok ? `passed ${v.length} of ${v.length}` : `failed ${failed} of ${v.length}`}</span>
                <span className="what">{SYSTEM[s] ?? ''}</span>
              </li>
            )
          })}
        </ul>
        {probe.answers.v0?.text && (
          <figure className="said">
            <figcaption>What the memory under test answered</figcaption>
            <p><Txt text={probe.answers.v0.text.length > 300 ? `${probe.answers.v0.text.slice(0, 300).trimEnd()}…` : probe.answers.v0.text} /></p>
          </figure>
        )}
      </div>
    </div>
  )
}

function Score({ index }: { index: Index }) {
  const sc = index.scorecard
  if (!sc?.systems.v0) return null
  const dims: [string, 'remembering' | 'understanding' | 'restraint' | 'task', string][] = [
    ['Remembering', 'remembering', 'uses the facts and shared references it was told'],
    ['Understanding', 'understanding', 'acts on impressions that were shown, never stated'],
    ['Restraint', 'restraint', 'keeps out what must not be said, and what has gone stale'],
    ['Task', 'task', 'still gets the job done'],
  ]
  const at = (s: string, d: (typeof dims)[number][1]) => sc.systems[s]?.[d]?.score ?? null
  return (
    <div className="score-card">
      <ul className="gauges">
        {dims.map(([label, d, what]) => {
          const v = at('v0', d)
          const lo = at('no_memory', d)
          const hi = at('oracle', d)
          return (
            <li key={d}>
              <div className="g-head"><span>{label}</span><b>{pct(v)}</b></div>
              <div className="g-track" aria-hidden="true">
                <i className="g-fill" style={{ width: `${(v ?? 0) * 100}%` }} />
                {lo !== null && <i className="g-tick lo" style={{ left: `${lo * 100}%` }} title={`no memory ${pct(lo)}`} />}
                {hi !== null && <i className="g-tick hi" style={{ left: `${hi * 100}%` }} title={`oracle ${pct(hi)}`} />}
              </div>
              <p>{what}</p>
            </li>
          )
        })}
      </ul>
      <p className="score-note">
        Scores for the memory under test, out of 100, with ticks for <i className="g-key lo" /> no memory and <i className="g-key hi" /> the
        oracle. Code decides first (an empty answer, a forbidden word); the rest goes to a judge model, and a second judge
        rechecks what must stay out. Every memory file is also audited claim by claim: {pct(sc.systems.v0.integrity?.score)}% of
        its claims are supported.{sc.provisional ? ' Provisional: both judges are Claude models.' : ''}
      </p>
    </div>
  )
}

// the page's chapters, for the rail's index: the five steps, then the worked question and the scores
const CHAPTERS: [string, string, string | null][] = [
  ['b-town', 'The town', 'leaf'],
  ['b-talks', 'The talks', 'sun'],
  ['b-memory', 'The memory', 'marigold'],
  ['b-questions', 'The questions', 'blossom'],
  ['b-score', 'The score', 'lime'],
  ['b-example', 'One real question', null],
  ['b-scores', 'How it scores', null],
]

/** Which chapter is being read: the last one whose top has passed a line a third of the way down the window. */
function useChapter(ids: string[]) {
  const [at, setAt] = useState(ids[0])
  useEffect(() => {
    const on = () => {
      const line = window.innerHeight / 3
      let cur = ids[0]
      for (const id of ids) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= line) cur = id
      }
      // at the foot of the page the last chapter is the one being read, even if its top never reaches the line
      if (window.scrollY > 0 && window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) cur = ids[ids.length - 1]
      setAt(cur)
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => {
      window.removeEventListener('scroll', on)
      window.removeEventListener('resize', on)
    }
  }, [ids])
  return at
}

const IDS = CHAPTERS.map(([id]) => id)

export function Behind({ index }: { index: Index | null }) {
  const weeks = index ? Math.round((Date.parse(index.range.end) - Date.parse(index.range.start)) / (7 * 86_400_000)) : null
  const at = useChapter(IDS)
  const go = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  return (
    <div className="site behind">
      <header className="masthead">
        <SiteNav here="behind" />
      </header>
      <main className="b-grid">
        <aside className="b-rail">
          <h1>Invented lives, <span className="hl">a real memory</span>, and questions at any moment.</h1>
          <p className="lede">
            We write every life in the town with a known truth, replay its residents’ talks through an assistant’s memory,
            then ask that memory questions at chosen moments and score the answers and the memory files.
          </p>
          <nav className="b-index" aria-label="On this page">
            {CHAPTERS.map(([id, title, tone], i) => (
              <button key={id} aria-current={at === id ? 'true' : undefined} onClick={() => go(id)}
                style={tone ? ({ '--tone': `var(--${tone})` } as React.CSSProperties) : undefined}>
                <span className="no">{tone ? `0${i + 1}` : ''}</span>
                {title}
              </button>
            ))}
          </nav>
        </aside>

        <div className="b-flow">
          <ol className="steps">
            <Step id="b-town" no="01" tone="leaf" glyph={<PlacesGlyph />} title="The town" value={index ? String(index.principals.length) : null} unit={`residents · ${weeks ?? '—'} weeks`}>
              An invented town on the real calendar, so no model knows it from training. Each resident has a bible of dated
              facts, impressions that are shown but never said, and a daily life log the assistant never sees.
            </Step>
            <Step id="b-talks" no="02" tone="sun" glyph={<PeopleGlyph />} title="The talks" value={index ? n(sum(index, 'sessions')!) : null} unit="talks with the assistant">
              Each talk is a fixed script: the resident asks for help with a real task, and a plan decides what slips out,
              said outright or only hinted. Some things must never be stored; some they ask it to forget.
            </Step>
            <Step id="b-memory" no="03" tone="marigold" glyph={<MemoryGlyph />} title="The memory" value={index ? `${n(sum(index, 'writes')!)} · ${n(sum(index, 'dreams')!)}` : null} unit="writes · dreams">
              After each talk the assistant files what matters as markdown, each line tagged stated or inferred. At 03:00 it
              dreams: merges, resolves, prunes, and adds nothing new. Every state is a git commit.
            </Step>
            <Step id="b-questions" no="04" tone="blossom" glyph={<EventsGlyph />} title="The questions" value={index ? n(sum(index, 'probes')!) : null} unit="probes at checkpoints">
              At checkpoints we ask a new request, answered from a read-only copy of the memory as it stood then. Each probe
              names what it must use, what it must not bring up, and what has gone stale.
            </Step>
            <Step id="b-score" no="05" tone="lime" glyph={<DataGlyph />} title="The score" value={index?.scorecard ? pct(index.scorecard.systems.v0?.remembering.score) : null} unit="remembering, out of 100">
              Every answer is judged on remembering, understanding, restraint and the task, against four other ways of
              answering the same question, from no memory at all to the truth known by then.
            </Step>
          </ol>

          <section className="b-section" id="b-example">
            <div className="b-head">
              <h2>One real question</h2>
              <p>How we query the memory: a request from the season, what a good answer has to draw on, and how each way of answering did.</p>
            </div>
            {index ? <Example index={index} /> : <div className="ex-card loading">Reading the data…</div>}
          </section>

          {index?.scorecard && (
            <section className="b-section" id="b-scores">
              <div className="b-head">
                <h2>How it scores</h2>
                <p>Across all {n(index.scorecard.probes ?? 0)} probes of the season.</p>
              </div>
              <Score index={index} />
            </section>
          )}
        </div>
      </main>
    </div>
  )
}
