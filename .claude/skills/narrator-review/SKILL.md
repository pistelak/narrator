---
name: narrator-review
description: Run narrator's escalated (frontier) independent review — a model from a different family, such as Codex, reviews a diff or plan read-only, and every finding is verified against the code. Use before pushing anything beyond a trivial fix in this repo, when the /local-review gate exits 2, or when the user asks to "ask codex", for a second opinion, or for an outside review of narrator changes.
---

# Frontier review for narrator

The obligations — when a review is required, the different-model-family rule,
verifying every finding — are in the **Independent review** section of
`AGENTS.md`. This is the mechanics.

## Pick the reviewer

A capable model from a **different family than the author**: the Codex CLI
below fits changes authored outside the GPT family. For Codex-authored changes
use a non-GPT equivalent; if none is available, say plainly that the
independent-review requirement is unmet.

## Run it

```bash
P=$(mktemp); O=$(mktemp); cat >"$P" <<'PROMPT'
<goal, exact paths in scope, constraints, non-goals,
 proof expected per claim, output shape>
PROMPT
codex exec -s read-only -C . \
  -m gpt-6-astra -c model_reasoning_effort="medium" \
  -o "$O" - <"$P"
```

- `-s read-only`: a review reads code; it never needs to write.
- Effort scales with the change: medium by default, `xhigh` for verifier
  semantics or anything that could turn a refusal into a false accept.
- When the /local-review gate prints its own escalation command, run that — its
  prompt and tree are frozen — substituting only the effort flag.

## The prompt contract does the work

State the goal, the exact scope, what is out of scope, and demand file:line
evidence for every claim. Ask two questions, not one: is it correct, **and is
it worth it** — license the reviewer to say "ship part, cut X".

## After

Read the output file and verify every finding against the code; a reviewer's
silence is never evidence of safety. In review-only tasks, report verified
findings without editing; when implementation is in scope, fix verified
findings and name any left unresolved.
