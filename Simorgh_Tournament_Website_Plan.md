# Simorgh Tournament Website Plan

## 1. Cleaned-up Product Summary
Simorgh needs a bilingual-friendly (Persian-first, English-capable) tournament platform that works under real event pressure: fast registration, reliable team formation, clear schedules, live score visibility, and low-friction staff operations on mobile. The product should combine:
- A polished public-facing sports site (branding, teams, schedule, standings, gallery, sponsors, rules).
- A controlled tournament operations backend (approvals, score entry, scheduling, corrections, audit logs, exports).
- A structured registration and team lifecycle system (captain/team creation, players joining, free agents, referee assignments, status transitions).

Primary success criteria:
1. Participants can register correctly without admin hand-holding.
2. Staff can run match operations from phones with minimal error.
3. Public pages remain accurate, timely, and trusted.
4. Sponsors see clear value and credible metrics.
5. Persian identity is visually strong but uncluttered.

## 2. Main Users and Their Goals
1. **Captains**
   - Create/manage team draft.
   - Reach minimum 6 players.
   - Finalize team identity (name/color/logo).
   - Assign referees and submit for approval.

2. **Players (joining team)**
   - Register quickly.
   - Upload photo once.
   - Select team/captain.
   - Track approval status.

3. **Free Agents**
   - Register profile with position/skill/city.
   - Be discoverable by captains/admin.
   - Receive invitations.

4. **Sponsors / Sponsor Prospects**
   - Understand audience and exposure.
   - Compare sponsorship tiers.
   - Contact organizers with clear next steps.

5. **Spectators / Community**
   - View schedule, live scores, standings, team pages, gallery, winners.

6. **Staff (score desk / court ops)**
   - Enter scores fast on mobile.
   - Avoid accidental errors.
   - See only relevant matches.

7. **Admins / Organizers**
   - Approve participants, teams, and sponsors.
   - Build and adjust schedule.
   - Resolve disputes and corrections.
   - Export operational data.

## 3. Key Product Decisions
1. **Mobile-first operations**: staff score-entry UX is prioritized over decorative UI.
2. **Two-tier backend roles**:
   - Admin (full control)
   - Staff (score entry + limited match operations)
3. **Explicit lifecycle states** for team/player/match/sponsor to avoid ambiguity.
4. **Team completeness gate**:
   - Team reaches “complete” at >=6 approved players.
   - Only then can captain finalize team name/color/logo and referee assignments.
5. **Approval workflow**:
   - Player registration is pending until captain/admin approval.
   - Team publication requires admin approval.
6. **Standings calculation is deterministic and documented** (public tie-break rules).
7. **Rules section is hybrid**:
   - Structured summaries by topic + downloadable/embeddable PDFs.
8. **Live updates default to polling** (e.g., 10–20 seconds) for reliability and cost simplicity; realtime optional later.
9. **Public privacy constraints**:
   - No public phone/email for players unless explicit opt-in.
10. **Sponsor page is sales-oriented** with tier matrix + CTA + credibility metrics.

## 4. Ambiguities and Questions for Amin
Only implementation-critical questions are listed.

1. **Final tournament format?**
   - Options: round robin, groups + playoffs, single elimination, double elimination, hybrid.
   - **Default recommendation:** group stage + playoffs (best balance for 8–24 teams).

2. **Are minors allowed?**
   - Impacts consent/legal flows and photo policy.
   - **Default:** assume adults only until policy confirmed.

3. **Bilingual scope at launch?**
   - Full Persian + English UI vs Persian primary with selected English pages.
   - **Default:** Persian primary with English fallback for sponsors/rules metadata.

4. **Public visibility of referee names?**
   - Show role + team only, or full names.
   - **Default:** show names publicly unless safety/privacy concern.

5. **Score publication mode:** immediate or admin-confirmed?
   - **Default:** immediate publish after staff submit, with admin correction window.

6. **Sponsor metrics availability** (attendance history, city distribution, exposure stats):
   - **Default:** launch with known historical numbers + mark estimates clearly.

7. **Discord integration needed at launch?**
   - **Default:** not required for MVP; add webhook notifications in Phase 3+.

## 5. Recommended Site Map
1. Home
2. Registration Hub
   - Captain Registration
   - Player Registration
   - Free Agent Registration
3. Current Teams
4. Team Detail
5. Free Agents (public-safe listing or captains-only listing depending policy)
6. Schedule
7. Live Match (single match live view)
8. Standings/Rankings
9. Rules
10. Sponsors
11. Gallery
12. About Simorgh
13. Winners / Previous Champions
14. Player Profile (optional public-lite profile)
15. Admin Portal
16. Staff Score Entry
17. Auth pages (login/reset)
18. Legal/Consent pages (terms/privacy/photo consent)

## 6. Public Page Plans
### 6.1 Home
- **Purpose:** quick orientation + strongest actions.
- **Main sections:** hero, key CTAs, tournament metadata, featured sponsor, live match snippet, standings preview, gallery preview.
- **Mobile:** stacked cards; sticky CTA strip.
- **Desktop:** hero split layout + 2-column widgets.
- **Data:** date, location, registration deadline, teams count, players count, live status.
- **CTAs:** Register Team, Join Team, Free Agent, Become Sponsor, View Schedule.

### 6.2 Registration Hub
- **Purpose:** route users into correct flow.
- **Main sections:** three role cards + requirements checklist + FAQ.
- **Mobile:** step indicators, minimal text.
- **Desktop:** side-by-side flow comparison.
- **Data:** open/close status, remaining spots (optional).
- **CTAs:** Start as Captain / Player / Free Agent.

### 6.3 Current Teams
- **Purpose:** show approved complete teams.
- **Sections:** filters (status/group), team cards, completion badges.
- **Mobile:** card list with quick stats.
- **Desktop:** grid + filter sidebar.
- **Data:** team name/color/logo, captain, player count, status.
- **CTAs:** View Team.

### 6.4 Team Detail
- **Purpose:** full team profile.
- **Sections:** header, roster with photos, referee assignments, match summary, status timeline.
- **Mobile:** collapsible tabs.
- **Desktop:** roster + metadata columns.
- **Data:** captain, players, referees, record, status.
- **CTAs:** (admin-only hidden publicly) edit/report issue.

### 6.5 Free Agents
- **Purpose:** help incomplete teams recruit.
- **Sections:** searchable cards by position/skill/city.
- **Mobile:** compact cards.
- **Desktop:** table + filters.
- **Data:** display name, city, position, skill, availability.
- **CTAs:** Invite Free Agent (captain/admin authenticated).

### 6.6 Schedule
- **Purpose:** discover all matches.
- **Sections:** date tabs, filters (team/court/status), match cards.
- **Mobile:** timeline cards.
- **Desktop:** day table + side filters.
- **Data:** teams, court, time, status, score if available, referees (if public).
- **CTAs:** View Live Match, Add to Calendar.

### 6.7 Live Match Page
- **Purpose:** single source of truth for current match.
- **Sections:** live status badge, set-by-set score, serving indicator (optional), officials, notes.
- **Mobile:** large score blocks.
- **Desktop:** score center + side info.
- **Data:** current set, set history, winner (if finished).
- **CTAs:** Back to Schedule, View Standings.

### 6.8 Standings/Rankings
- **Purpose:** transparent ranking.
- **Sections:** ranking table/cards, tie-breaker explainer.
- **Mobile:** horizontally scrollable compact table or card rows.
- **Desktop:** full sortable table.
- **Data:** rank, team, MP, W, L, sets +/-, points +/-, diff, qualification.
- **CTAs:** View team, view tie-break rules.

### 6.9 Rules
- **Purpose:** usable rules, not just files.
- **Sections:** quick rule summaries by category, searchable index, PDF viewer, downloads.
- **Mobile:** accordion sections + open PDF in full screen.
- **Desktop:** left nav + content pane + viewer.
- **Data:** rule sections, source docs, revision date.
- **CTAs:** Download Persian PDF, Download FIVB PDF.

### 6.10 Sponsors
- **Purpose:** convert prospects into leads.
- **Sections:** value proposition, tier comparison, proof metrics, sponsor logos, CTA form.
- **Mobile:** stacked tier cards.
- **Desktop:** comparison matrix + charts.
- **Data:** tiers, benefits, attendance, city map/chart, estimated reach.
- **CTAs:** Become a Sponsor, Download Sponsor Kit.

### 6.11 Gallery
- **Purpose:** social proof and community identity.
- **Sections:** album filters by year/event, masonry media grid.
- **Mobile:** 2-column masonry.
- **Desktop:** 4+ column grid with lightbox.
- **Data:** captions, event tags, sponsor-tagged items.
- **CTAs:** View full album.

### 6.12 About Simorgh
- **Purpose:** mission, story, brand identity.
- **Sections:** tournament story, Simorgh symbolism, organizers, values.
- **CTAs:** Register / Sponsor.

### 6.13 Winners / Champions
- **Purpose:** legacy + recurring engagement.
- **Sections:** yearly champions, runner-up, MVP/awards (optional), recap links.
- **Data:** final standings snapshots, sponsor mentions.

## 7. Registration and Team Formation Flow
### A. Captain Flow
1. Choose “Captain / Create Team”.
2. Create account (email OTP or password) + consent.
3. Enter profile + upload photo.
4. Create team draft (temporary team code).
5. Invite players via join link/code.
6. Track roster progress.
7. At >=6 approved players: unlock finalization form (team name/color/logo + referee assignments).
8. Submit for admin approval.

### B. Player Joining Team
1. Choose “Join Existing Team”.
2. Create account + profile + photo.
3. Select team/captain from searchable list.
4. Submit; status = pending captain/admin approval.
5. On approval, status = active roster member.

### C. Free Agent Flow
1. Choose “Register as Free Agent”.
2. Profile + photo + position + skill + city + contact preference.
3. Status = available.
4. Captains/admin can invite.
5. If accepted and approved, convert to team membership.

### D. Referee Assignment
- Required roles per complete team: main, second, line1, line2.
- Validation before team can be marked “approved for tournament.”
- Allow temporary “pending referee” only if admin override flag is set.

### E. Team Completion & Publication Rules
- Completion threshold: >=6 approved players.
- Final team identity locked at approval (admins may unlock edits).
- Public “Current Teams” shows only approved complete teams.

## 8. Admin and Staff Portal Plan
### 8.1 Full Admin Dashboard
Modules:
1. Registrations queue (players/captains/free agents).
2. Team management (status edits, roster changes, lock/unlock identity).
3. Referee validation board.
4. Schedule builder (manual + template assist).
5. Match control (start, delay, cancel, finish).
6. Score corrections (with reason + audit).
7. Standings recalculation view.
8. Sponsors CMS (tiers, logos, approvals, featured priority).
9. Rules CMS (PDF upload, section tagging).
10. Gallery CMS (album/image upload and moderation).
11. Exports (CSV for teams, players, referees, matches, standings).

### 8.2 Staff Score-Entry Portal (Mobile-first)
- Login with staff role.
- “Today’s matches” large tappable cards.
- Enter set scores in sequence (Set 1..N).
- Winner auto-detected, manual confirm required.
- Optional match notes.
- Submit confirmation modal.
- Post-submit: status badge + “edit request” if outside correction window.

### 8.3 Permissions Model
- **Admin:** full CRUD + override.
- **Staff:** match score input for assigned matches, no sponsor/rules/gallery config.
- **Captain:** manage own team draft and referee proposal.
- **Public user:** read-only public pages.

### 8.4 Audit/Edit Behavior
- Every critical action logged (before/after payload, actor, timestamp, reason).
- Correction window default: 15 minutes for staff edits.
- Beyond window: admin-only edits.

## 9. Schedule, Live Match, and Standings System
### Match Statuses
Draft -> Scheduled -> Check-in -> Live -> Finished
Branches: Delayed, Canceled, Forfeit

### Score Entry
- Set-level input, validate volleyball score logic (configurable match format: best-of-3 or best-of-5).
- Winner calculated after valid set completion.

### Ranking Algorithm (recommended)
1. Match wins (or competition points if adopted).
2. Set differential (sets won - sets lost).
3. Point differential (points won - points lost).
4. Head-to-head.
5. Admin tie-break decision with mandatory note.

### Tie-break Transparency
- Display tie-break order on standings page.
- Show “tie-break applied” tag for impacted teams.

### Public Display
- Schedule page for all matches.
- Live match page for active matches.
- Standings auto-refresh by polling.

### Edge Cases
- No-show -> forfeit with predefined score policy.
- Duplicate or wrong score -> correction workflow.
- Schedule shifts -> notify and mark updated timestamp.

## 10. Sponsor System
### Sponsor Page Layout
1. Hero + value proposition.
2. Proof metrics (attendance, city diversity, reach estimates).
3. Tier comparison table.
4. Current/past sponsor showcase.
5. Sponsor lead form + direct contact CTA.

### Sponsor Tiers
- Bronze, Silver, Gold, Main Sponsor.
- Include clear deliverables (logo placement, booth option, merchandising visibility, recap mentions, website profile card).

### Sponsor Benefits (structured)
- Website logo + link.
- Featured banner/card placement.
- On-site branding opportunities.
- Winner ceremony/merch branding.
- Mention in recap/gallery/highlights.

### Sponsor Contact Flow
- Form submit -> admin review queue -> follow-up SLA target (e.g., within 48h).
- Optional auto-email acknowledgment.

### Charts/Graphs
- Attendance trend (last 3 tournaments).
- Participant city distribution.
- Estimated digital reach (website/social).

### Sponsor Data Fields
- Business name, contact person, email, phone, website/social, logo upload, preferred tier, custom message, budget range (optional), consent checkbox.

## 11. Rules System
1. **Two-source model**: Persian tournament PDF + official FIVB PDF.
2. **Readable structure**:
   - Gameplay basics
   - Tournament-specific adjustments
   - Conduct/discipline
   - Referee responsibilities
   - Tie-break and protest process
3. **Navigation**: search box + anchored section list.
4. **PDF behavior**: embed viewer + guaranteed download buttons.
5. **Persian/RTL support**:
   - Rule summaries available in Persian first.
   - Proper RTL rendering for Persian sections.
6. **Versioning**:
   - Show “effective date” and revision notes.
7. **Future PDF parsing plan**:
   - Admin can map uploaded PDF to structured RuleSection entries manually at first; optional OCR/semantic splitting later.

## 12. Gallery and Previous Tournaments
- Gallery organized by year and event stage (opening, matches, finals, awards).
- Album metadata: location, date, photographer/credit, sponsor tags.
- Champions page links to corresponding gallery/recap.
- Attendance and city stats reused across Winners and Sponsors pages.
- Sponsor visibility: tagged sponsor badges in recap sections (not intrusive).

## 13. Data Model
1. **User**
   - id, email, phone (optional), password_hash/otp_identifier, role, status, created_at, last_login.
2. **PlayerProfile**
   - user_id, first_name, last_name, display_name, photo_url, city, birth_date(optional), preferred_position, skill_level, bio(optional), consent_flags.
3. **Team**
   - id, captain_user_id, draft_code, name, color_hex, logo_url, status, group_id(optional), created_at, approved_at.
4. **TeamMembership**
   - id, team_id, user_id, role_in_team(player/captain), join_status, joined_at, approved_by.
5. **FreeAgentProfile**
   - user_id, availability_status, preferred_position, skill_level, city, contact_preference, notes.
6. **FreeAgentInvite** (additional)
   - id, team_id, free_agent_user_id, invited_by, status, created_at, responded_at.
7. **RefereeAssignment**
   - id, team_id, role(main/second/line1/line2), person_type(member/external), member_user_id(optional), external_name(optional), external_contact(optional), status.
8. **Sponsor**
   - id, business_name, contact_name, email, phone, website, social_link, logo_url, tier_id, status, notes.
9. **SponsorTier**
   - id, name, rank, benefits_json, price_range(optional), active.
10. **Match**
   - id, tournament_id, stage, group_id(optional), team_a_id, team_b_id, court_no, scheduled_at, started_at, finished_at, status, winner_team_id, notes.
11. **SetScore**
   - id, match_id, set_number, team_a_points, team_b_points, completed.
12. **Standing**
   - id, tournament_id, team_id, played, wins, losses, sets_won, sets_lost, points_won, points_lost, set_diff, point_diff, rank, qualification_status, updated_at.
13. **RuleDocument**
   - id, language, title, source_type(persian/fivb), file_url, version, effective_date, published.
14. **RuleSection**
   - id, document_id, section_key, title, content_richtext, order_no, language, searchable_text.
15. **GalleryAlbum** (additional)
   - id, title, year, event_stage, cover_image_url, published.
16. **GalleryImage**
   - id, album_id, image_url, thumb_url, caption, taken_at, sponsor_tag(optional), sort_order.
17. **Tournament** (additional)
   - id, name, year, location, start_date, end_date, registration_deadline, status.
18. **AdminActionLog / AuditLog**
   - id, actor_user_id, action_type, entity_type, entity_id, before_json, after_json, reason, created_at.
19. **NotificationEvent** (optional)
   - id, event_type, payload_json, channel(web/discord/email), sent_at, status.

## 14. Status Machines and Validation Rules
### Team Status
Draft -> Recruiting -> Complete -> PendingApproval -> Approved -> Active -> Eliminated -> Champion
Alternate: Withdrawn, Rejected

### Player Membership Status
Pending -> Approved -> Active
Alternate: Rejected, Removed, Withdrawn

### Match Status
Draft -> Scheduled -> Live -> Finished
Alternate: Delayed, Canceled, Forfeit

### Sponsor Status
Lead -> InReview -> Approved -> Published
Alternate: Rejected, Archived

### Validation Rules
- Unique email per user.
- Prevent duplicate active membership in multiple teams (unless special admin override tournament format).
- Team complete requires >=6 approved players.
- Referee roles all required before Approved (unless admin override).
- Photo upload required for captain/player; enforce mime/type/size.
- Logo/image virus scan or content moderation basic checks.
- Strong form validation for phone/email formats.
- Score validation per match format and volleyball rules.

## 15. UI/UX Design System
### Color Palette
- Navy #101828 (layout/frame)
- Crimson #B42318 (primary CTA)
- Turquoise #1CA7A8 (interactive highlights/live)
- Gold #D6A84F (premium accents/champion)
- Ivory #FFF8EA (soft backgrounds)
- White #FFFFFF (cards)
- Text #111827, muted #667085
- Success #12B76A, Warning #F79009, Error #F04438

### Typography
- Persian: Vazirmatn (or licensed IRANSans alternative)
- Latin: Inter or Geist
- Clear hierarchy with generous line-height for bilingual readability

### Core Components
- Role selection cards
- Match status badges
- Team chips (color-coded)
- Standings table/card hybrid
- Stepper forms for registration
- Mobile bottom action bar for key CTAs
- Audit-friendly confirmation modals for score submit/edit

### Navigation
- Mobile: compact top nav + optional sticky bottom quick actions on event days.
- Desktop: full header nav + contextual subnav for tournament pages.

### Accessibility
- WCAG contrast targets.
- Keyboard navigability on forms/tables/modals.
- Alt text on all images.
- Form labels + error summaries.
- Proper RTL support for Persian content.

### Simorgh Identity
- Subtle abstract feather motifs and geometric Persian accents in section dividers/backgrounds.
- Avoid busy ornamental overlays on functional pages (schedule, score entry).

## 16. Technical Architecture Recommendation
- **Frontend:** Next.js + TypeScript + Tailwind.
- **Backend:** Next.js API routes/server actions or dedicated backend service.
- **DB:** PostgreSQL.
- **ORM:** Prisma (or Drizzle).
- **Auth:** NextAuth/Supabase Auth with role claims.
- **Storage:** Cloudinary or Supabase Storage for photos/logos/gallery/PDFs.
- **Live updates:** polling (10–20s) for schedule/live/standings in MVP; websocket/realtime later.
- **Hosting:** Vercel for app + managed Postgres.
- **Security:** RBAC middleware, admin routes protected, rate limits, CSRF protections where relevant.
- **Backup/export:** automated DB backups + on-demand CSV export from admin portal.

## 17. Risk Analysis and Fixes
1. **Registration confusion** -> use role-based wizard with progress indicators.
2. **Incomplete teams near deadline** -> admin alerts + captain reminders + free-agent matching tools.
3. **Score entry errors under pressure** -> constrained inputs + confirmation modal + correction window + audit log.
4. **Standings disputes** -> publish algorithm and tie-break trace.
5. **Photo/privacy concerns** -> consent fields + private contact data separation.
6. **Schedule volatility** -> visible “last updated” + delayed/canceled statuses.
7. **Sponsor page weak credibility** -> require source-backed metrics with timestamp.
8. **Over-designed branding harming usability** -> keep ornament density low on high-utility pages.

## 18. Phase-by-Phase Implementation Plan
### Phase 1: Foundation + Core Public Presence
- **Objective:** launch branded public site shell and essential content.
- **Features:** home, about, sponsors (static draft), rules placeholder, gallery placeholder, auth skeleton.
- **Pages:** Home, About, Sponsors, Rules, Gallery, basic nav/footer.
- **Backend/data:** base schema (User, Tournament, Team, PlayerProfile minimal), storage setup.
- **Admin tools:** admin login + minimal content publish toggles.
- **Acceptance criteria:** responsive public pages, branding system established, deploy pipeline ready.
- **Tests:** responsive checks, basic auth flow, content rendering.

### Phase 2: Registration + Team Formation
- **Objective:** operational participant onboarding.
- **Features:** captain flow, player join flow, free-agent flow, photo upload, team completion gate.
- **Pages:** Registration Hub + 3 flows + Current Teams + Team Detail basic.
- **Backend/data:** TeamMembership, FreeAgentProfile, RefereeAssignment (initial), validations.
- **Admin tools:** approvals queue, team edit/approve.
- **Acceptance criteria:** end-to-end registration to approved team path works.
- **Tests:** duplicate prevention, status transitions, upload validations.

### Phase 3: Schedule, Scoring, Standings
- **Objective:** event-day readiness.
- **Features:** schedule management, staff score-entry mobile UI, live match page, standings auto-update.
- **Pages:** Schedule, Live Match, Standings, Staff Portal.
- **Backend/data:** Match, SetScore, Standing recalculation engine, audit logging.
- **Admin tools:** schedule builder, score correction controls.
- **Acceptance criteria:** match result submission updates public standings reliably.
- **Tests:** ranking algorithm correctness, correction window logic, load/polling behavior.

### Phase 4: Sponsor + Gallery + Rules Maturity
- **Objective:** elevate professionalism and stakeholder value.
- **Features:** sponsor lead form + tiers + analytics charts, full gallery albums, structured rules sections with bilingual/RTL support.
- **Pages:** Sponsors advanced, Gallery advanced, Rules advanced, Winners page.
- **Backend/data:** SponsorTier, RuleDocument/Section, GalleryAlbum/Image, champion history fields.
- **Admin tools:** sponsor CMS, gallery uploader, rules document manager.
- **Acceptance criteria:** sponsor conversion-ready page and structured rules experience live.
- **Tests:** form submissions, media optimization, RTL rendering.

### Phase 5: Hardening + Nice-to-Haves
- **Objective:** reliability, observability, and polish.
- **Features:** exports, enhanced analytics, optional Discord webhook notifications, performance tuning.
- **Pages:** admin analytics panels, optional notification settings.
- **Backend/data:** NotificationEvent, expanded logs, backup verification jobs.
- **Admin tools:** CSV export center, operational dashboards.
- **Acceptance criteria:** stable under tournament traffic; admins can recover from common errors quickly.
- **Tests:** backup restore drill, audit completeness, security checks.

## 19. MVP vs Later Features
### Must Have (Launch)
- Registration flows (captain/player/free agent)
- Team approval/completion logic
- Schedule + live statuses + score entry
- Standings with clear tie-breakers
- Core public pages (Home, Teams, Schedule, Standings, Rules basic, Sponsors basic)
- Admin/staff auth and permissions

### Should Have
- Structured rules summaries + embedded PDF
- Sponsor tiers with lead form
- Gallery with albums
- Winners page
- CSV exports

### Nice to Have
- Free-agent invitation workflow automation
- Advanced sponsor analytics visualizations
- Player public profiles
- Court-specific dashboards

### Later Advanced
- Discord/webhook automations
- Real-time websocket updates
- OCR-assisted Persian rule section extraction
- Predictive schedule conflict detection

## 20. Final Implementation Checklist
1. Confirm tournament format and scoring model.
2. Confirm bilingual/RTL scope and content ownership.
3. Finalize registration required fields and consent/legal text.
4. Finalize role/permission matrix.
5. Approve status machines and validation rules.
6. Approve standings algorithm and tie-break policy text.
7. Define sponsor tier deliverables and pricing ranges.
8. Gather historical metrics (attendance, cities, exposure) with sources.
9. Prepare initial rules documents (Persian + FIVB) and summary outline.
10. Prepare brand assets (logo, motifs, color usage examples).
11. Set SLA for score corrections and sponsor lead follow-up.
12. Define backup/export operational schedule.
13. Run event-day simulation (staff score entry + public live updates).
14. Lock MVP scope before coding sprint starts.
