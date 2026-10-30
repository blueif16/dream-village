// An assistant's message, rendered the way a chat app renders it: its markdown set as paragraphs, lists, tables and
// emphasis. The text is the model's own, so nothing in it becomes live markup: raw HTML shows as text, a link keeps its
// words and goes nowhere. Never-store redactions (viz/export.py writes them as a run of █) are drawn as bars.
import { Marked } from 'marked'

const BAR = '█'.repeat(8)
const REDACT = '<span class="redact" title="never-store content: redacted" aria-label="redacted" role="img"></span>'

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const md = new Marked({
  gfm: true,
  breaks: true, // a chat keeps its line breaks: an address, a sign-off
  async: false,
  renderer: {
    html({ text }) {
      return esc(text)
    },
    link({ tokens }) {
      return this.parser.parseInline(tokens)
    },
    image({ text }) {
      return esc(text)
    },
  },
})

/** A chat message's markdown as HTML, safe to set as innerHTML. */
export function messageHtml(text: string): string {
  return (md.parse(text) as string).split(BAR).join(REDACT)
}
