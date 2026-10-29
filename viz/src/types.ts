// The bundle viz/export.py writes. Field names follow the exporter one to one.

export interface Usage {
  frequency: string
  active_hours: number[]
}

export interface PrincipalInfo {
  id: string
  name: string
  age: number | null // null for a minor: their age is never shown
  minor: boolean
  role: string
  usage: Usage
}

export interface PrincipalSummary extends PrincipalInfo {
  sessions: number
  commits: number
  writes: number
  dreams: number
  memory_lines: number
  memory_files: number
  verify_passed: number
  probes: number
  replay_usd: number
  bundle_bytes: number
}

export interface GateRow {
  check: string
  value: string
  threshold: string
  passed: boolean
}

export interface Score {
  score: number | null
  n: number | null
  ci_probes?: [number, number] | null
}

export interface ScorecardSystem {
  remembering: Score
  understanding: Score
  restraint: Score
  task: Score
  integrity: Score | null
  gain: Record<string, number | null> | null
  gates: Record<string, boolean | null>
  cost: { answer_usd_mean: number | null; context_tokens_mean: number | null; write_usd_total: number | null }
}

export interface Index {
  generated_at: string
  village: string
  town: { name: string; region: string; summary: string; places: number; events: { date: string; end_date: string | null; title: string }[] }
  range: { start: string; end: string }
  principals: PrincipalSummary[]
  gate: GateRow[] | null
  scorer: {
    dir: string
    files: Record<string, boolean>
    probes: number | null
    answers: Record<string, number> | null
    verdicts: number | null
  }
  scorecard: { probes: number | null; provisional: boolean | null; rubric_version: string | null; systems: Record<string, ScorecardSystem> } | null
  redactions: number
  sources: Record<string, string>
}

export interface Turn {
  role: 'user' | 'assistant'
  text: string
}

export interface PlanItem {
  item_id: string
  mode: 'stated' | 'hinted'
  never_store: boolean
  forget_request: boolean
  note: string
  truth: { kind: string; attribute: string; cls: string; text: string } | null
}

export interface Flag {
  kind: string
  item_id: string | null
  detail: string
}

export interface Session {
  id: string
  start: string
  task: string | null
  task_type: string | null
  turns: Turn[]
  items: PlanItem[]
  verify: { passed: boolean; attempt: number; flags: Flag[] } | null
}

export interface Commit {
  sha: string
  parent: string | null
  sim_time: string
  event: 'write' | 'dream'
  session_id: string | null
  cost_usd: number | null
  tokens: Record<string, number> | null
  files: { path: string; status: string }[]
  diff: string
  snapshot: Record<string, string>
  lines: number
}

export interface TruthEntry {
  id: string
  kind: 'fact' | 'impression' | 'reference'
  subject: string
  attribute: string
  cls: string
  text: string
  valid_from: string | null
  valid_to: string | null
  superseded_by: string | null
  disclosed_in: string[]
  forget_at: string | null
}

export interface Judged {
  model: string
  pass: boolean | null
  evidence: string | null
}

export interface Verdict {
  check: string
  type: string
  item_id: string | null
  pass: boolean | null
  decided_by: string | null
  code_rule: string | null
  primary: Judged | null
  second: Judged | null
}

export interface Answer {
  text: string | null
  refused: boolean | null
  error: string | null
  context_tokens: number | null
  tool_calls: unknown[] | null
}

export interface Probe {
  id: string
  checkpoint: string
  t: string
  kind: string
  lag: string | null
  request: string
  plan: string | null
  must_use: string[]
  may_use: string[]
  must_not: string[]
  stale: string[]
  criteria: string[]
  forbidden: string[]
  item_lags: Record<string, string>
  shortcut: boolean | null
  answers: Record<string, Answer>
  verdicts: Record<string, Verdict[]> | null // null: verdicts.jsonl not written yet
}

export interface PrincipalBundle {
  principal: PrincipalInfo
  sessions: Session[]
  commits: Commit[]
  truth: TruthEntry[]
  probes: Probe[]
}

// One selectable thing on the timeline.
export type EventKind = 'session' | 'commit' | 'checkpoint'

export interface TimelineEvent {
  key: string // "s:<session id>" | "c:<sha>" | "k:<checkpoint>"
  kind: EventKind
  t: number // ms since epoch, sim time read as UTC
  index: number // into sessions, commits or checkpoints
}

export interface Checkpoint {
  id: string
  t: string
  probes: Probe[]
}
