import { describe, expect, it } from 'vitest'
import { messageHtml } from './chat'

const BAR = '█'.repeat(8)

describe('an assistant message rendered as chat', () => {
  it('sets **bold** as strong', () => {
    expect(messageHtml('Open **Settings** first')).toContain('<strong>Settings</strong>')
  })

  it('turns numbered lines into an ordered list', () => {
    const html = messageHtml('Steps:\n\n1. Log in\n2. Open payroll\n3. Save')
    expect(html).toContain('<ol>')
    expect(html.match(/<li>/g)?.length).toBe(3)
  })

  it('draws a pipe table as a table', () => {
    const html = messageHtml('| Tray | Left over |\n|---|---|\n| A | 2 |')
    expect(html).toContain('<table>')
    expect(html).toContain('<th>Tray</th>')
    expect(html).toContain('<td>2</td>')
  })

  it('keeps a single line break, as a sign-off needs', () => {
    expect(messageHtml('Thank you,\nMs. Tolentino')).toMatch(/Thank you,<br\s*\/?>\s*Ms\. Tolentino/)
  })

  it('shows raw HTML in a message as text, never as markup', () => {
    const html = messageHtml('<script>alert(1)</script>\n\nand <b>bold</b> inline')
    expect(html).not.toContain('<script')
    expect(html).not.toContain('<b>')
    expect(html).toContain('&lt;script&gt;')
  })

  it('keeps a link’s words but never makes a link', () => {
    const html = messageHtml('see [the portal](javascript:alert(1)) now')
    expect(html).not.toContain('<a')
    expect(html).not.toContain('javascript:')
    expect(html).toContain('the portal')
  })

  it('draws never-store redactions as bars', () => {
    const html = messageHtml(`Your SSN is ${BAR}.`)
    expect(html).not.toContain(BAR)
    expect(html).toContain('class="redact"')
  })
})
