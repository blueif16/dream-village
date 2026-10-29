import { Fragment, type ReactNode } from 'react'

// viz/export.py replaces never-store content with this fixed-width run; it renders as a redaction bar.
const BAR = '█'.repeat(8)

function bars(text: string, keyBase: string): ReactNode[] {
  return text.split(BAR).map((part, i) => (
    <Fragment key={`${keyBase}-${i}`}>
      {i > 0 && <span className="redact" title="never-store content: redacted" aria-label="redacted" role="img" />}
      {part}
    </Fragment>
  ))
}

/** Plain text with redaction bars. */
export function Txt({ text }: { text: string | null | undefined }) {
  if (!text) return null
  return <>{bars(text, 't')}</>
}

const TAG = /(\[stated\]|\[inferred\])/

/** One line of a memory file: [stated] / [inferred] tags marked, redactions as bars. */
export function MemLine({ text }: { text: string }) {
  return (
    <>
      {text.split(TAG).map((part, i) =>
        part === '[stated]' ? (
          <span key={i} className="tag-stated">{part}</span>
        ) : part === '[inferred]' ? (
          <span key={i} className="tag-inferred">{part}</span>
        ) : (
          <Fragment key={i}>{bars(part, String(i))}</Fragment>
        ),
      )}
    </>
  )
}

/** A whole memory file: frontmatter dimmed, headings set, tags marked. */
export function MemoryFile({ text }: { text: string }) {
  const lines = text.split('\n')
  let fm = lines[0]?.trim() === '---'
  return (
    <pre className="md">
      {lines.map((line, i) => {
        const inFm = fm
        if (fm && i > 0 && line.trim() === '---') fm = false
        const cls = inFm ? 'fm' : line.startsWith('#') ? 'h' : undefined
        return (
          <div key={i} className={cls}>
            {line === '' ? ' ' : inFm ? line : <MemLine text={line} />}
          </div>
        )
      })}
    </pre>
  )
}
