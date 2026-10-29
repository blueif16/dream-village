import { useTownTime } from './clock'
import { useTown } from './data'
import { clock, monthDay, weekday } from './town/common'
import { GitHub, Moon, Sun, Up } from './town/icons'
import { DAY_MIN } from './town/schedule'

export type Page = 'home' | 'behind' | 'data'

const REPO = 'https://github.com/blueif16/dream-village'

/** The town's clock as it runs, for the landing page's nav: where, what time, which day, each its own pill. */
function TownClock() {
  const now = useTownTime()
  const { data: town } = useTown()
  if (!now || !town) return <span className="clock" aria-hidden="true" />
  const day = Math.floor(now.t / DAY_MIN)
  const d = town.days[day]
  return (
    <span className="clock" role="timer" aria-live="off" aria-label={`In ${town.town} it is ${clock(now.t)}, ${weekday(d)} ${monthDay(d)}, day ${day + 1} of ${town.days.length}`}>
      <span className="pill where"><i className="live" />{town.town}</span>
      <span className="pill time"><b>{clock(now.t)}</b>{now.night ? <Moon /> : <Sun />}{weekday(d)} {monthDay(d)}</span>
      <span className="pill day">Day {day + 1} of {town.days.length}</span>
    </span>
  )
}

/** The one nav, as separate pills. On the landing page it carries the town's clock and the code, and leaves the ways in
 *  (Explore the town, Behind the town) to the page's own buttons; everywhere else it is only the way back up into the
 *  town. The data pages are still there by their address, but nothing links to them: the town holds their data now. */
export function SiteNav({ here, compact, live }: { here?: Page; compact?: boolean; live?: boolean }) {
  const cur = (p: Page) => (here === p ? 'page' : undefined)
  return (
    <nav className={compact ? 'site-nav compact' : 'site-nav'} aria-label="Dream Village">
      <a className="pill mark" href="#/" aria-current={cur('home')}>Dream Village</a>
      {live ? <TownClock /> : <span />}
      <span className="pages">
        {live ? (
          <a className="pill gh" href={REPO} target="_blank" rel="noreferrer" aria-label="Dream Village on GitHub" title="Dream Village on GitHub">
            <GitHub />
          </a>
        ) : (
          <a className="pill cta" href="#/town">Explore the town <Up /></a>
        )}
      </span>
    </nav>
  )
}
