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
