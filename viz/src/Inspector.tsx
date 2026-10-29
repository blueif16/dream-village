import { useMemo, useState } from 'react'
import { commitAt, fmtWhen, ms, parseDiff, short, weekOf } from './data'
import { MemLine, MemoryFile, Txt } from './text'
import { systemOrder } from './Village'
import type { Checkpoint, Commit, PlanItem, PrincipalBundle, Probe, Session, TruthEntry, Verdict } from './types'

interface Ctx {
  bundle: PrincipalBundle
  start: string // the village's first day, for week numbers
  truth: Map<string, TruthEntry>
  go: (key: string) => void
  // kept across commits, so stepping through history keeps the same tab and file open
  commitTab: 'diff' | 'snapshot'
  setCommitTab: (t: 'diff' | 'snapshot') => void
  memFile: string | null
  setMemFile: (path: string) => void
}

function VerifyBadge({ s }: { s: Session }) {
  if (!s.verify) return <span className="badge none">— not verified</span>
  return s.verify.passed ? (
    <span className="badge pass">✓ VERIFIED</span>
  ) : (
    <span className="badge fail">✗ VERIFY FAILED</span>
  )
}

function ModeChip({ it }: { it: PlanItem }) {
  if (it.never_store) return <span className="chip never" title="never-store: the memory must not keep it">never-store</span>
  if (it.forget_request) return <span className="chip forget" title="the user asks the assistant to forget this">forget</span>
  return <span className={`chip ${it.mode}`}>{it.mode}</span>
}

function SessionView({ s, ctx }: { s: Session; ctx: Ctx }) {
  const write = ctx.bundle.commits.find((c) => c.session_id === s.id)
  return (
    <>
      <header className="insp-head">
        <div className="insp-kind">Session</div>
        <h2>
          {s.id} <VerifyBadge s={s} />
        </h2>
        <div className="insp-meta">
          <span>{fmtWhen(s.start)}</span>
          <span>week {weekOf(s.start, ctx.start)}</span>
          <span className="mono">{s.task_type}</span>
          <span>{s.turns.length} turns</span>
          {write ? (
            <span>
              memory write <button className="linkish" onClick={() => ctx.go(`c:${write.sha}`)}>{short(write.sha)}</button>
            </span>
          ) : (
            <span className="muted">no memory commit (nothing changed)</span>
          )}
        </div>
      </header>

      <section className="insp-section truth-zone">
        <h3>Planned disclosure · ground truth <span className="count">{s.items.length} items</span></h3>
        {s.task && <p className="task" style={{ marginBottom: 10 }}><Txt text={s.task} /></p>}
        <ul className="items">
          {s.items.map((it, i) => (
            <li key={`${it.item_id}-${i}`}>
              <ModeChip it={it} />
              <span className="note"><Txt text={it.note} /></span>
              {it.truth && (
                <span className="gt">
                  <span className="mono">{it.item_id}</span> {it.truth.kind}
                  {it.truth.attribute && it.truth.kind === 'fact' ? ` · ${it.truth.attribute}` : ''} ({it.truth.cls}):{' '}
                  <Txt text={it.truth.text} />
                </span>
              )}
              {!it.truth && <span className="gt"><span className="mono">{it.item_id}</span> not in the truth files</span>}
            </li>
          ))}
        </ul>
      </section>

      {s.verify && s.verify.flags.length > 0 && (
        <section className="insp-section">
          <h3>Verify flags <span className="count">{s.verify.flags.length}</span> <span className="count">attempt {s.verify.attempt}</span></h3>
          <ul className="flags">
            {s.verify.flags.map((f, i) => (
              <li key={i}>
                <span className="chip">{f.kind}</span>
                <span>
                  {f.item_id && <span className="mono">{f.item_id} </span>}
                  <Txt text={f.detail} />
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="insp-section">
        <h3>Dialogue</h3>
        <div className="chat">
          {s.turns.map((t, i) => (
            <div key={i} className={`bubble ${t.role}`}>
              <div className="bubble-role">{t.role === 'user' ? ctx.bundle.principal.name.split(' ')[0] : 'assistant'}</div>
              <Txt text={t.text} />
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

function DiffView({ c }: { c: Commit }) {
  const files = useMemo(() => parseDiff(c.diff), [c.diff])
  if (files.length === 0) return <p className="empty">No changes.</p>
  return (
    <>
      {files.map((f) => {
        const st = c.files.find((x) => x.path === f.path)?.status ?? 'M'
        return (
          <div key={f.path} className="diff-file">
            <div className="diff-file-head">
              <span className={`st ${st}`}>{st}</span>
              <span>{f.path}</span>
              <span className="stats">
                <span className="plus">+{f.add}</span> <span className="minus">−{f.del}</span>
              </span>
            </div>
            <table className="diff-table">
              <tbody>
                {f.lines.map((l, i) =>
                  l.kind === 'hunk' ? (
                    <tr key={i} className="hunk">
                      <td colSpan={4}>{l.text}</td>
                    </tr>
                  ) : (
                    <tr key={i} className={l.kind}>
                      <td className="ln">{l.old ?? ''}</td>
                      <td className="ln">{l.new ?? ''}</td>
                      <td className="sign">{l.kind === 'add' ? '+' : l.kind === 'del' ? '−' : ' '}</td>
                      <td className="code"><MemLine text={l.text} /></td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        )
      })}
    </>
  )
}

function SnapshotView({ c, ctx }: { c: Commit; ctx: Ctx }) {
  const paths = Object.keys(c.snapshot).sort((a, b) => {
    const da = a.includes('/') ? 1 : 0
    const db = b.includes('/') ? 1 : 0
    return da - db || a.localeCompare(b)
  })
  const changed = new Map(c.files.map((f) => [f.path, f.status]))
  const first = c.files.find((f) => f.status !== 'D' && c.snapshot[f.path] !== undefined)?.path ?? paths[0]
  const pick = ctx.memFile
  const open = pick && c.snapshot[pick] !== undefined ? pick : first
  const groups = new Map<string, string[]>()
  for (const p of paths) {
    const dir = p.includes('/') ? p.slice(0, p.lastIndexOf('/') + 1) : ''
    groups.set(dir, [...(groups.get(dir) ?? []), p])
  }
  const deleted = c.files.filter((f) => f.status === 'D')
  if (paths.length === 0) return <p className="empty" style={{ padding: 20 }}>The memory is empty at this commit.</p>
  return (
    <div className="snapshot">
      <nav className="tree" aria-label="memory files">
        {[...groups.entries()].map(([dir, ps]) => (
          <div key={dir}>
            {dir && <div className="dir">{dir}</div>}
            {ps.map((p) => {
              const st = changed.get(p)
              const n = c.snapshot[p].split('\n').filter((l) => l.trim()).length
              return (
                <button key={p} className={dir ? undefined : 'root'} aria-current={p === open} onClick={() => ctx.setMemFile(p)}
                  title={st ? `${st === 'A' ? 'added' : 'modified'} at this commit` : undefined}>
                  <span className={`st ${st ?? ''}`}>{st ?? ''}</span>
                  {p.slice(dir.length)}
                  <span className="n">{n}</span>
                </button>
              )
            })}
          </div>
        ))}
        {deleted.length > 0 && (
          <div>
            <div className="dir">deleted here</div>
            {deleted.map((f) => (
              <div key={f.path} style={{ padding: '2px 12px 2px 22px' }} className="muted">
                <span className="st D">D</span>
                <s>{f.path}</s>
              </div>
            ))}
          </div>
        )}
      </nav>
      <div className="file-view">
        <div className="file-view-head">/{open}</div>
        {open && <MemoryFile text={c.snapshot[open]} />}
      </div>
    </div>
  )
}

function CommitView({ c, ctx }: { c: Commit; ctx: Ctx }) {
  const tab = ctx.commitTab
  const setTab = ctx.setCommitTab
  const i = ctx.bundle.commits.indexOf(c)
  const prev = i > 0 ? ctx.bundle.commits[i - 1] : null
  const delta = prev ? c.lines - prev.lines : c.lines
  return (
    <>
      <header className="insp-head">
        <div className="insp-kind">
          Memory commit ·{' '}
          <span style={{ color: c.event === 'dream' ? 'var(--dream)' : 'var(--write)', fontWeight: 700 }}>
            {c.event === 'dream' ? '◆ dream' : '■ write'}
          </span>
        </div>
        <h2>{short(c.sha)}</h2>
        <div className="insp-meta">
          <span>{fmtWhen(c.sim_time)}</span>
          <span>week {weekOf(c.sim_time, ctx.start)}</span>
          {c.event === 'write' && c.session_id && (
            <span>
              from <button className="linkish" onClick={() => ctx.go(`s:${c.session_id}`)}>{c.session_id}</button>
            </span>
          )}
          <span>
            {c.lines} memory lines ({delta >= 0 ? '+' : '−'}
            {Math.abs(delta)})
          </span>
          {c.cost_usd !== null && <span className="mono">${c.cost_usd.toFixed(4)}</span>}
          {c.tokens?.output_tokens !== undefined && <span className="mono">{c.tokens.output_tokens} out tok</span>}
          {prev && (
            <span>
              parent <button className="linkish" onClick={() => ctx.go(`c:${prev.sha}`)}>{short(prev.sha)}</button>
            </span>
          )}
        </div>
      </header>
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'diff'} onClick={() => setTab('diff')}>
          Diff <span className="mono">{c.files.length} files</span>
        </button>
        <button role="tab" aria-selected={tab === 'snapshot'} onClick={() => setTab('snapshot')}>
          Memory at this commit <span className="mono">{Object.keys(c.snapshot).length} files</span>
        </button>
      </div>
      {tab === 'diff' ? (
        <section className="insp-section">
          <div className="files-changed">
            {c.files.map((f) => (
              <span key={f.path}>
                <span className={`st ${f.status}`}>{f.status}</span>
                {f.path}
              </span>
            ))}
          </div>
          <DiffView c={c} />
        </section>
      ) : (
        <SnapshotView c={c} ctx={ctx} />
      )}
    </>
  )
}

function TargetList({ ids, probe, ctx }: { ids: string[]; probe: Probe; ctx: Ctx }) {
  return (
    <dd>
      {ids.map((id) => {
        const e = ctx.truth.get(id)
        return (
          <span key={id} className="target">
            <span className="mono">{id}</span>
            {e ? (
              <>
                <span className="muted">{e.kind}{e.kind === 'fact' && e.attribute ? ` · ${e.attribute}` : ''} ({e.cls}): </span>
                <Txt text={e.text} />
              </>
            ) : (
              <span className="muted">not in the truth files</span>
            )}
            {probe.item_lags[id] && <span className="lag">{probe.item_lags[id].replace(/_/g, ' ')}</span>}
          </span>
        )
      })}
    </dd>
  )
}

function Glyph({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="g none" title="undecided">?</span>
  return ok ? <span className="g pass" title="pass">✓</span> : <span className="g fail" title="fail">✗</span>
}

function Checks({ vs }: { vs: Verdict[] }) {
  return (
    <ul className="checks">
      {vs.map((v) => {
        const ev = v.primary?.evidence ?? v.second?.evidence
        return (
          <li key={v.check}>
            <Glyph ok={v.pass} />
            <span>
              <span className="mono">{v.check}</span> <span className="muted">· {v.decided_by}</span>
              {v.code_rule && <span className="muted"> · {v.code_rule}</span>}
              {ev && <div className="ev">“<Txt text={ev} />”</div>}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function ProbeDetail({ p, ctx }: { p: Probe; ctx: Ctx }) {
  const systems = systemOrder([...new Set([...Object.keys(p.answers), ...Object.keys(p.verdicts ?? {})])])
  const targets: [string, string[]][] = [
    ['must use', p.must_use],
    ['may use', p.may_use],
    ['must not', p.must_not],
    ['stale', p.stale],
  ]
  return (
    <>
      <section className="insp-section">
        <h3>Request <span className="count">{p.id}</span></h3>
        <div className="probe-request chat">
          <div className="bubble user">
            <div className="bubble-role">{ctx.bundle.principal.name.split(' ')[0]} · {fmtWhen(p.t)}</div>
            <Txt text={p.request} />
          </div>
        </div>
        <div className="insp-meta" style={{ marginTop: 8 }}>
          <span className="chip kind">{p.kind}</span>
          {p.lag && <span>lag: {p.lag.replace(/_/g, ' ')}</span>}
          {p.shortcut && <span className="badge fail">shortcut-flagged</span>}
        </div>
      </section>
      <section className="insp-section truth-zone">
        <h3>Targets · ground truth</h3>
        <dl className="targets">
          {targets.filter(([, ids]) => ids.length > 0).map(([label, ids]) => (
            <div key={label} style={{ display: 'contents' }}>
              <dt>{label}</dt>
              <TargetList ids={ids} probe={p} ctx={ctx} />
            </div>
          ))}
          {p.criteria.length > 0 && (
            <>
              <dt>criteria</dt>
              <dd>
                <ol className="criteria">
                  {p.criteria.map((c, i) => <li key={i}>{c}</li>)}
                </ol>
              </dd>
            </>
          )}
        </dl>
        {p.plan && (
          <details className="plan" style={{ marginTop: 10 }}>
            <summary>probe designer's note</summary>
            <p><Txt text={p.plan} /></p>
          </details>
        )}
      </section>
      <section className="insp-section">
        <h3>
          Answers <span className="count">{Object.keys(p.answers).length} systems</span>
          {p.verdicts === null && <span className="count">· verdicts not yet generated</span>}
        </h3>
        {systems.length === 0 ? (
          <p className="empty">No answers yet.</p>
        ) : (
          <div className="answers">
            {systems.map((sys) => {
              const a = p.answers[sys]
              const vs = p.verdicts?.[sys] ?? null
              const passed = vs ? vs.filter((v) => v.pass === true).length : 0
              return (
                <article key={sys} className={`answer ${sys}`}>
                  <div className="answer-head">
                    <span className="sys">{sys}</span>
                    {vs && (
                      <span className={`badge ${passed === vs.length ? 'pass' : 'fail'}`}>
                        {passed}/{vs.length} checks
                      </span>
                    )}
                    {a?.context_tokens ? <span className="tok">{a.context_tokens.toLocaleString()} ctx tok</span> : null}
                  </div>
                  {vs ? (
                    <Checks vs={vs} />
                  ) : (
                    <div className="checks empty">{p.verdicts === null ? 'verdicts not yet generated' : 'no verdicts for this system'}</div>
                  )}
                  <div className="answer-text">
                    {!a && <span className="empty">no answer yet</span>}
                    {a?.error && <span className="badge fail">error: {a.error}</span>}
                    {a?.refused && <span className="badge fail">refused</span>}
                    {a && <Txt text={a.text} />}
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </section>
    </>
  )
}

function CheckpointView({ cp, ctx }: { cp: Checkpoint; ctx: Ctx }) {
  const [pick, setPick] = useState(0)
  const p = cp.probes[Math.min(pick, cp.probes.length - 1)]
  const at = commitAt(ctx.bundle, ms(cp.t))
  const mem = at >= 0 ? ctx.bundle.commits[at] : null
  return (
    <>
      <header className="insp-head">
        <div className="insp-kind">Probe checkpoint</div>
        <h2>{cp.id}</h2>
        <div className="insp-meta">
          <span>{fmtWhen(cp.t)}</span>
          <span>week {weekOf(cp.t, ctx.start)}</span>
          <span>{cp.probes.length} probes</span>
          {mem ? (
            <span>
              v0 reads memory at{' '}
              <button className="linkish" onClick={() => ctx.go(`c:${mem.sha}`)}>{short(mem.sha)}</button> ({mem.lines} lines)
            </span>
          ) : (
            <span className="muted">no memory commit before this time</span>
          )}
        </div>
      </header>
      <div className="tabs" role="tablist">
        {cp.probes.map((q, i) => (
          <button key={q.id} role="tab" aria-selected={q === p} onClick={() => setPick(i)} title={q.request}>
            {q.id.split('.').pop()} <span className="mono">{q.kind}</span>
          </button>
        ))}
      </div>
      {p && <ProbeDetail key={p.id} p={p} ctx={ctx} />}
    </>
  )
}

export function Inspector({ selected, checkpoints, ctx }: { selected: string | null; checkpoints: Checkpoint[]; ctx: Ctx }) {
  if (!selected) return <div className="placeholder">Select a session, a memory commit or a probe checkpoint on the timeline.</div>
  const kind = selected[0]
  const id = selected.slice(2)
  if (kind === 's') {
    const s = ctx.bundle.sessions.find((x) => x.id === id)
    if (s) return <SessionView key={s.id} s={s} ctx={ctx} />
  } else if (kind === 'c') {
    const c = ctx.bundle.commits.find((x) => x.sha === id)
    if (c) return <CommitView key={c.sha} c={c} ctx={ctx} />
  } else if (kind === 'k') {
    const cp = checkpoints.find((x) => x.id === id)
    if (cp) return <CheckpointView key={cp.id} cp={cp} ctx={ctx} />
  }
  return <div className="placeholder">Nothing named {selected} for this principal.</div>
}

export type { Ctx }
