-- Run in Supabase SQL Editor. Safe to re-run (IF NOT EXISTS guards).
-- Adds real verification fields to the accounts table so "verified"
-- is a stored fact, not just implied by the UI.

alter table accounts add column if not exists email_verified boolean not null default false;
alter table accounts add column if not exists phone_verified boolean not null default false;
alter table accounts add column if not exists email_verified_at timestamptz;
alter table accounts add column if not exists phone_verified_at timestamptz;

-- Optional: if you want to mark all existing sample rows as verified
-- (since they represent the "fully-contactable" promise), run:
-- update accounts set email_verified = true, phone_verified = true,
--   email_verified_at = now(), phone_verified_at = now();


-- ---------------------------------------------------------------
-- Lists: turns the board from one implicit table into named lists.
-- Run this block once.
-- ---------------------------------------------------------------

create table if not exists lists (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  created_at timestamptz not null default now()
);

alter table accounts add column if not exists list_id uuid references lists(id);

-- Enable the same open (anon-key) access pattern the rest of the
-- app already uses, so the board can read/write lists client-side.
alter table lists enable row level security;
drop policy if exists "public read lists" on lists;
create policy "public read lists" on lists for select using (true);
drop policy if exists "public write lists" on lists;
create policy "public write lists" on lists for insert with check (true);
drop policy if exists "public update lists" on lists;
create policy "public update lists" on lists for update using (true);

-- Seed a default list and assign all existing accounts to it so
-- nothing "disappears" once list_id becomes the filter key.
insert into lists (name) values ('Sample Pipeline')
  on conflict do nothing;

update accounts set list_id = (select id from lists order by created_at asc limit 1)
  where list_id is null;


-- ---------------------------------------------------------------
-- ProspectingAgent: contact_name/contact_title/contact_phone/contact_email
-- were NOT NULL, which only fits a fully-named + fully-verified contact.
-- AccountFinder's redefined "qualified" bar (see git log: "Redefine
-- 'qualified' to what's actually achievable without a paid provider")
-- also accepts a general_office fallback (real phone/email, no named
-- person) and a contacts_seen-only match (real named person, no confirmed
-- phone/email) -- both legitimately leave some of these 4 columns null.
-- Empirically, general_office is the COMMON case, not the exception (0/3
-- test candidates had a fully-verified named contact), so the schema
-- needs to support it, not just the rare full-verification case.
-- Run this block once.
-- ---------------------------------------------------------------

alter table accounts alter column contact_name drop not null;
alter table accounts alter column contact_title drop not null;
alter table accounts alter column contact_phone drop not null;
alter table accounts alter column contact_email drop not null;


-- ---------------------------------------------------------------
-- ProspectingAgent: status has a CHECK constraint limiting it to the 6
-- original board.html values (new/researching/contacted/in_conversation/
-- opportunity/closed) -- 'partial' (used when a company fails partway
-- through processing, so whatever was gathered still gets saved instead
-- of silently lost) was rejected by the DB even though board.html's
-- dropdown already lists it. Run this block once.
-- ---------------------------------------------------------------

alter table accounts drop constraint if exists accounts_status_check;
alter table accounts add constraint accounts_status_check
  check (status in ('new', 'partial', 'researching', 'contacted', 'in_conversation', 'opportunity', 'closed'));


-- ---------------------------------------------------------------
-- Apollo phone reveal: Apollo's People API only delivers a revealed
-- mobile/direct-dial number asynchronously, via a webhook POST minutes
-- after the request (see api/apollo-phone-webhook.js) -- there's no
-- synchronous response field for it. This table is that webhook's
-- landing spot; apollo_client.py's poll_phone_reveal() polls it by
-- apollo_person_id after kicking off a reveal, since the calling script
-- has no long-running listener of its own to receive the callback
-- directly. Run this block once.
-- ---------------------------------------------------------------

create table if not exists apollo_phone_reveals (
  apollo_person_id text primary key,
  phone text,
  raw_payload jsonb,
  received_at timestamptz not null default now()
);

-- Same open (anon-key) access pattern as `lists` -- the webhook writes
-- with the anon key (already public in config.js, nothing new exposed),
-- and apollo_client.py's poller reads with it too.
alter table apollo_phone_reveals enable row level security;
drop policy if exists "public read apollo_phone_reveals" on apollo_phone_reveals;
create policy "public read apollo_phone_reveals" on apollo_phone_reveals for select using (true);
drop policy if exists "public write apollo_phone_reveals" on apollo_phone_reveals;
create policy "public write apollo_phone_reveals" on apollo_phone_reveals for insert with check (true);
drop policy if exists "public upsert apollo_phone_reveals" on apollo_phone_reveals;
create policy "public upsert apollo_phone_reveals" on apollo_phone_reveals for update using (true);

-- ---------------------------------------------------------------
-- demo_rate_limits: caps how many times the public "try it free" page
-- (contact.html) can trigger a live account_finder.find_accounts() run per
-- visitor -- each run is a real Anthropic (Opus) + Exa cost. Unlike every
-- other table above, this one is NOT open to the anon key -- api/find-
-- accounts.js reads/writes it with the Supabase service-role key, so no
-- client-side script can see or tamper with the rate-limit counts. RLS is
-- enabled with zero policies, which blocks all anon/authenticated access
-- by default and leaves only the service role able to touch it.
-- ---------------------------------------------------------------

create table if not exists demo_rate_limits (
  id bigint generated always as identity primary key,
  ip text not null,
  requested_at timestamptz not null default now()
);

alter table demo_rate_limits enable row level security;

-- BYPASSRLS lets service_role skip the policies above, but it's a separate
-- thing from ordinary SQL table privileges -- without this grant, service_role
-- gets a plain "permission denied for table" error, not an RLS-related one.
grant select, insert on public.demo_rate_limits to service_role;
