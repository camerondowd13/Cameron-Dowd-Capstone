# AE Prospecting Copilot — Product Requirements Document

**Author:** Cameron Dowd
**Status:** v1 scoped

---

## 1. The Problem

Cameron Dowd, AE at Stampli, spends 1–2 hours/day (~5–10 hrs/week) on manual account and contact research before he can begin outreach. This time is unpredictable and low-leverage: yield ranges from 10–15 qualified accounts/hour to as few as 2/hour depending on method and source availability that day.

The process has two failure-prone steps before outreach: (1) finding the right company, ideally with an active buying trigger, falling back to ICP-fit industry search when none exists; (2) finding a reachable contact — one with no working email or phone is a dead end, wasting the time spent finding them.

Because this research competes with money-generating activity (customer calls) during the workday, it frequently bleeds into personal time Cameron doesn't want to sustain.

The underlying problem is **time allocation**, not a lack of separation between list-building and outreach — those are already distinct steps. List-building consumes a disproportionate, unpredictable share of limited prospecting time, leaving too little for the step that generates pipeline. Cameron wants list-building to shrink to near-zero effort so 80–100% of prospecting time goes to contacting people with intent to book a discovery call.

## 2. Who Has It

**Primary persona:** AEs and SDRs who self-source their own outbound target lists — companies that don't hand reps a pre-built list. Excludes inbound-only reps.

**Evidence:** Cameron's own experience across AE and SDR roles, corroborated informally by coworkers reporting the same friction. Anecdotal, not survey-validated.

**Noted, out of scope for v1:** could let inbound-only reps layer on outbound activity — a future expansion, not core v1.

## 3. Current Workflow

1. Trigger search — ZoomInfo, Apollo, 6sense, LinkedIn, plus lost opportunities in CRM.
2. ICP confirmation — cross-check firmographic fit.
3. Fallback (no trigger) — industry search within the same tools, plus Indeed and Growjo.
4. Contact discovery — decision-makers via LinkedIn/Sales Navigator.
5. Contact info lookup — phone/email via ZoomInfo/Apollo, website, last-resort RealPeopleFinder.
6. CRM entry — manually check/create the contact in NetSuite or HubSpot.
7. List assembly — compile validated contacts into a working list.
8. Outreach — bulk call/email, separately.

Only step 8 is revenue-facing.

## 4. The AI's Role

The AI fully owns steps 1–7: trigger detection, binary ICP match/no-match against **Appendix A: ICP Definition** (no confidence scoring in v1), industry fallback search, contact discovery, phone/email resolution (both required or the output is invalid), CRM write-back to HubSpot only, and list assembly. Cameron's only manual step in the ideal case is outreach. Calling is always human — never automated.

**List inclusion rule:** a company is only added to the output list if it has at least one contact that fully meets ICP + contact-validity criteria. Companies with no qualifying contact are excluded entirely — never shown as a partial or empty result.

**Multi-user lead claiming (Cameron + his SDR):** once a contact is delivered to either user's list, it is marked *claimed* and excluded from being served to the other user for a defined cooldown period (proposed: 90 days), unless manually released. This prevents both users from independently generating overlapping lists for the same contact.

## 5. What I'm NOT Building

**Deferred to v2:** confidence-scored ICP matching, multi-CRM support, general multi-tenant accounts (v1 = Cameron + his one SDR only), automated email sending, cost/budget modeling for data provider usage, formal pre-launch time-tracking baseline, and a documented compliance review of personal-data lookup tools (e.g., RealPeopleFinder) beyond the ToS-scraping rule below. Also deferred: final selection of the underlying data provider (Apollo/ZoomInfo/PDL/etc.) and verification of Cameron's HubSpot API/admin access at Stampli — **both remain open risks that should be resolved before implementation begins**, not silently dropped.

**Hard constraints:** no scraping platforms whose ToS prohibits it (LinkedIn); no automated/robo-calling, ever.

**CRM write-back rule (dedup/non-destructive):**
- Company doesn't exist in HubSpot → create it.
- Company exists, contact doesn't → add contact to existing company.
- Contact already exists → never create a duplicate, never overwrite an existing field; only fill fields currently blank.

**Design philosophy:** feeds existing CRMs — not a CRM, sales engagement platform, or pipeline tool.

## 6. Success Metrics

1. 80–90% reduction in time spent on list-building, vs. baseline of 5–10 hrs/week.
2. Qualified lead volume: the tool delivers exactly as many fully-qualified contacts as requested (territory + industry + title + verified phone + verified email, all five, no partial credit) — benchmarked against manual baseline of 2–15 accounts/hr.

## Appendix A: ICP Definition (v1)

- **Territory:** New York, Florida, Virginia.
- **Company size:** 20–399 employees. Exception: companies below 20 employees still qualify *if* a specific buying trigger is present *and* an appropriate contact is reachable.
- **Ability to pay (guiding principle, not a hard filter field):** should plausibly afford a $1,000–$6,000/month product.
- **Invoice volume (target, not directly queryable):** 200+ invoices/month. Not available as raw data — approximated via industry proxy: construction, manufacturing, and healthcare typically run high invoice volume; SaaS typically runs low. *(Open item: finalize this as a concrete, enumerable industry list rather than illustrative examples.)*
- **Target contacts:** CEO, CFO, CTO, AP Manager, Director of Finance, or others involved in purchasing AP automation software. *(Open item: the trailing clause is open-ended; needs a concrete title list for matching logic eventually.)*
