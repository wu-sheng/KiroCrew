/**
 * Screenshot harness for the crewmate switcher's unread column (issue #19045).
 *
 * The closed chip is the other crewmates' faces and a chevron — no "Crewmates"
 * word, no roster count. Under it hangs one pill per crewmate with an unread
 * message (not the open one), capped at four plus a "+N" pill that opens the
 * list. The identity pill in the middle keeps face + name + activity at rest.
 *
 *   node scripts/capture-switcher-unread-column.mjs <outDir> <distDir>
 *
 * Runs the REAL built SPA behind `serveDist` with every `/api/**` answered from
 * fixtures (`stubDashboardApi`): no gateway, no auth, no kiro-cli. Unread is
 * seeded the way the websocket marker leaves it (`mc-unread-slots`). Every
 * frame asserts what it photographs before writing the PNG.
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { json } from './lib/boot-api.mjs'
import { serveDist, DEFAULT_DIST } from './lib/serve-dist.mjs'
import { stubDashboardApi, logPageProblems } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || join(process.env.KIROCREW_SCRATCH || '/tmp', 'switcher-unread-column')
const DIST = process.argv[3] ? resolve(process.argv[3]) : DEFAULT_DIST
mkdirSync(OUT, { recursive: true })

const now = Math.floor(Date.now() / 1000)
const member = (name, display, secsAgo, last, extra = {}) => ({
  name, slug: name, display_name: display, bound: true, slot_key: `member-${name}`, running: false,
  kiro_agent: 'kirocrew', workspace: '/srv/repos/kirocrew', memory_store: `member-${name}`,
  memory_version: 2, memory_owner: name, model: '', source: 'kirocrew',
  has_dm_message: true, dashboard_created: true, last_active_ts: now - secsAgo, last_message: last, ...extra,
})
const MEMBERS = [
  member('mate', 'Mate', 45, 'Shipping the folder fix now.', { running: true }),
  member('build-watcher', 'Build Watcher', 600, 'CI on the release branch is green.'),
  member('researcher', 'Researcher', 1800, 'The brief is in the vault.'),
  member('oncall', 'Oncall', 3600, 'No pages overnight.'),
  member('pipeline', 'Pipeline', 7200, 'Release 0.4.2 is staged.'),
  member('scribe', 'Scribe', 9000, 'Notes from standup are filed.'),
  member('radar', 'Radar', 12000, 'Two new issues match your filters.'),
]
const OPEN_MEMBER = 'mate'
const AGENTS = MEMBERS.map((m) => ({
  name: m.name, display_name: m.display_name, kiro_agent: m.kiro_agent, workspace: m.workspace,
  memory_store: m.memory_store, model: m.model, description: '', source: m.source,
}))
const iso = (secsAgo) => new Date((now - secsAgo) * 1000).toISOString()
const SLOTS = MEMBERS.map((m) => ({
  key: m.slot_key, title: m.display_name, mode: 'member',
  created: iso(86400), last_ts: iso(now - m.last_active_ts), running: !!m.running,
  project: '/srv/repos/kirocrew', agent: m.name,
}))

const extra = async (path, route) => {
  if (path === '/api/members') { await json(route, { members: MEMBERS, default_agent: OPEN_MEMBER }); return true }
  if (path === '/api/agents') { await json(route, { agents: AGENTS, default_agent: OPEN_MEMBER }); return true }
  if (path === '/api/default-agent') { await json(route, { default_agent: OPEN_MEMBER }); return true }
  if (path === '/api/crons') { await json(route, { jobs: [] }); return true }
  if (path === '/api/cron-folders') { await json(route, []); return true }
  if (path === '/api/teams') { await json(route, { teams: [] }); return true }
  if (path === '/api/autonudge') { await json(route, { enabled: true, loops: [] }); return true }
  // What the route answers for a thread that never ran a goal; the resume
  // greeting probes it on every crewmate open and treats it as "no board".
  if (path === '/api/crew-board') { await json(route, { error: 'no work ledger for this session', code: 'no_ledger' }, 404); return true }
  const thread = path.match(/^\/api\/members\/([^/]+)\/thread$/)
  if (thread) {
    const slug = decodeURIComponent(thread[1])
    await json(route, { slot_key: `member-${slug}`, slug, member: slug, created: false })
    return true
  }
  if (/^\/api\/members\/[^/]+\/activity$/.test(path)) { await json(route, { slug: '', member: '', capped: false, entries: [] }); return true }
  if (/^\/api\/members\/[^/]+\/briefing$/.test(path)) { await json(route, { slug: '', member: '', supported: true, text: '', updated_ts: now, redacted: false, truncated: false }); return true }
  if (/^\/api\/members\/[^/]+\/panel$/.test(path)) { await json(route, { panel: null, html: null }); return true }
  return false
}

const { srv, base } = await serveDist(DIST)
const browser = await chromium.launch()
let failed = false
function check(name, ok, detail = '') {
  console.log(`${name}: ${ok ? 'OK' : 'MISMATCH'} ${detail}`)
  if (!ok) failed = true
}

/** The header band: chip, pill, column — cropped so the faces are readable. */
const CLIP = { x: 0, y: 0, width: 1100, height: 260 }

/** One scene: `unread` names the crewmates whose slots are unread. */
async function scene(tag, theme, unread, { frames = false } = {}) {
  const context = await browser.newContext({ viewport: { width: 1500, height: 940 }, deviceScaleFactor: 2, colorScheme: theme })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    theme, extra, slots: SLOTS,
    localStorageEntries: {
      'mc-lang': 'en', 'mc-crewmates-onboarded': '1', 'mc-crewmates-page-entered': '1', 'mc-nav': '1',
      'mc-members-panel-open': '0',
      'mc-unread-slots': JSON.stringify(unread.map((n) => `member-${n}`)),
    },
  })
  await page.goto(`${base}/members?member=${encodeURIComponent(OPEN_MEMBER)}`, { waitUntil: 'domcontentloaded' })
  const pill = page.getByTestId('member-identity-pill')
  await pill.waitFor({ state: 'visible', timeout: 30000 })
  await page.waitForTimeout(900)
  await page.mouse.move(5, 600)
  await page.waitForTimeout(300)

  // The identity pill names the crewmate at rest — no fold.
  const title = page.getByTestId('member-title-row')
  check(`[${tag}/${theme}] identity pill shows the name at rest`, await title.isVisible() && (await title.boundingBox()).width > 20)
  check(`[${tag}/${theme}] identity pill carries no fold column`, (await page.getByTestId('member-pill-text').count()) === 0)
  // The chip: faces + chevron, no count, roster word folded.
  check(`[${tag}/${theme}] chip has no count`, (await page.getByTestId('crewmate-switcher-count').count()) === 0)
  const labelW = (await page.getByTestId('crewmate-switcher-label').boundingBox())?.width ?? 0
  check(`[${tag}/${theme}] roster word folded at rest`, labelW < 2, `width=${labelW}`)
  // The column.
  const pills = page.getByTestId('crewmate-switcher-unread-pill')
  const expectPills = Math.min(unread.filter((n) => n !== OPEN_MEMBER).length, 4)
  check(`[${tag}/${theme}] ${expectPills} unread pill(s)`, (await pills.count()) === expectPills, `got ${await pills.count()}`)
  const more = page.getByTestId('crewmate-switcher-unread-more')
  const rest = unread.filter((n) => n !== OPEN_MEMBER).length - expectPills
  check(`[${tag}/${theme}] "+N" pill ${rest > 0 ? `says +${rest} more` : 'absent'}`,
    rest > 0 ? (await more.innerText()).trim() === `+${rest} more` : (await more.count()) === 0)
  await page.screenshot({ path: join(OUT, `${tag}-${theme}-rest.png`), clip: CLIP })

  // Hover the chip: the roster word slides in.
  const chip = page.getByTestId('crewmate-switcher')
  if (frames) {
    // The slide, as frames: rest, mid-transition, open. The GIF built from
    // these is the evidence that the word grows out of the chip rather than
    // popping in beside it.
    await chip.hover()
    for (const [i, wait] of [[0, 0], [1, 60], [2, 60], [3, 120]]) {
      await page.waitForTimeout(wait)
      await page.screenshot({ path: join(OUT, `frame-${theme}-${i}.png`), clip: { x: 0, y: 0, width: 420, height: 130 } })
    }
  } else {
    await chip.hover()
  }
  await page.waitForTimeout(400)
  const openW = (await page.getByTestId('crewmate-switcher-label').boundingBox())?.width ?? 0
  check(`[${tag}/${theme}] roster word open under hover`, openW > 40, `width=${openW}`)
  await page.screenshot({ path: join(OUT, `${tag}-${theme}-hover-chip.png`), clip: CLIP })

  if (expectPills > 0) {
    // A pill names who and why, and opens that thread.
    const first = pills.first()
    const who = await first.getAttribute('data-member')
    check(`[${tag}/${theme}] first pill named "${who}, Unread message"`, (await first.getAttribute('aria-label')) === `${MEMBERS.find((m) => m.name === who).display_name}, Unread message`)
    await first.hover()
    await page.waitForTimeout(300)
    await page.screenshot({ path: join(OUT, `${tag}-${theme}-hover-pill.png`), clip: CLIP })
    await first.click()
    await page.waitForURL((u) => u.searchParams.get('member') === who, { timeout: 10000 })
    await page.waitForTimeout(700)
    check(`[${tag}/${theme}] clicking the pill opens ${who}`, (await page.getByTestId('member-title-row').innerText()).includes(MEMBERS.find((m) => m.name === who).display_name))
    await page.screenshot({ path: join(OUT, `${tag}-${theme}-after-pick.png`), clip: CLIP })
  }
  if (rest > 0) {
    // Back on the first thread: "+N" opens the list, whose rows carry the dot.
    await page.goto(`${base}/members?member=${encodeURIComponent(OPEN_MEMBER)}`, { waitUntil: 'domcontentloaded' })
    await pill.waitFor({ state: 'visible', timeout: 30000 })
    await page.waitForTimeout(700)
    await more.click()
    const list = page.getByTestId('crewmate-switcher-list')
    await list.waitFor({ state: 'visible', timeout: 10000 })
    await page.waitForTimeout(400)
    check(`[${tag}/${theme}] "+N" opens the list with every unread dot`, (await list.getByTestId('crewmate-switcher-unread-dot').count()) >= unread.length - 1)
    await page.screenshot({ path: join(OUT, `${tag}-${theme}-more-open.png`), clip: { x: 0, y: 0, width: 1100, height: 560 } })
  }
  await context.close()
}

for (const theme of ['light', 'dark']) {
  await scene('none', theme, [])
  await scene('two', theme, ['build-watcher', 'oncall'], { frames: theme === 'light' })
  await scene('six', theme, ['build-watcher', 'researcher', 'oncall', 'pipeline', 'scribe', 'radar'])
}

console.log(`\nshots in ${OUT}`)
await browser.close()
srv.close()
process.exit(failed ? 1 : 0)
