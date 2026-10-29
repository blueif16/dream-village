// The front page. The live town sits behind it, seen through a wide window; scrolling down, clicking the window or
// "Explore the town" opens that window out to the whole console.
import { useEffect, useLayoutEffect, useRef } from 'react'
import { SiteNav } from './SiteNav'
import { MemoryGlyph, Next, PlacesGlyph } from './town/icons'

export function Landing({ onEnter, onWindow, leaving, back }: {
  onEnter: () => void
  onWindow: (r: DOMRect) => void
  leaving: boolean
  back: boolean
}) {
  const hero = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const measure = () => hero.current && onWindow(hero.current.getBoundingClientRect())
    measure()
    const ro = new ResizeObserver(measure)
    if (hero.current) ro.observe(hero.current)
    window.addEventListener('resize', measure)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [onWindow])

  // scrolling down (wheel, a swipe up, or the keys that scroll) enters the town
  useEffect(() => {
    let pulled = 0
    let y0: number | null = null
    const wheel = (e: WheelEvent) => {
      pulled = Math.max(0, pulled + e.deltaY)
      if (pulled > 40) onEnter()
    }
    const touchStart = (e: TouchEvent) => (y0 = e.touches[0]?.clientY ?? null)
    const touchMove = (e: TouchEvent) => {
      if (y0 !== null && y0 - (e.touches[0]?.clientY ?? y0) > 48) onEnter()
    }
    const key = (e: KeyboardEvent) => {
      const onPage = e.target === document.body || e.target === hero.current
      if (['ArrowDown', 'PageDown', ' '].includes(e.key) && onPage) {
        e.preventDefault()
        onEnter()
      }
    }
    window.addEventListener('wheel', wheel, { passive: true })
    window.addEventListener('touchstart', touchStart, { passive: true })
    window.addEventListener('touchmove', touchMove, { passive: true })
    window.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('wheel', wheel)
      window.removeEventListener('touchstart', touchStart)
      window.removeEventListener('touchmove', touchMove)
      window.removeEventListener('keydown', key)
    }
  }, [onEnter])

  return (
    <div className={`site landing${leaving ? ' leaving' : ''}${back ? ' back' : ''}`}>
      <header className="masthead">
        <SiteNav here="home" live />
      </header>
      <div className="hero-window" ref={hero} role="button" tabIndex={0} aria-label="Explore the town"
        onClick={onEnter} onKeyDown={(e) => e.key === 'Enter' && onEnter()}>
        <span className="cue">Scroll to enter <Next /></span>
      </div>
      <section className="intro">
        <h1 className="lede">A simulated town: our playground for a <span className="nb">self-optimizing</span> memory harness for AI agents.</h1>
        <div className="ctas">
          <button className="btn-sm primary" onClick={onEnter}><PlacesGlyph /> Explore the town</button>
          <a className="btn-sm" href="#/behind"><MemoryGlyph /> What's behind the town</a>
        </div>
      </section>
    </div>
  )
}
