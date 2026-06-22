# Final-Review Fix Report — Branch g1-deep-curriculum

**Status:** COMPLETE
**Branch:** `g1-deep-curriculum`
**Files changed:** 11 chapter .md files

---

## Finding 1 — broken cross-chapter promise (ch07→ch08)

**File:** `docs/reports/07-proving-its-real.md` line 25

**Old:** `…Using a *different* seed would test robustness to initialization; we will do that kind of experiment in Chapter 08. Here, the goal…`

**New:** `…Using a *different* seed would test robustness to initialization; that is a different question from the one we are answering here. Here, the goal…`

The false forward-promise to ch08 is removed. The two remaining "Chapter 08" references in ch07 are correct (they introduce ch08 as the knob-turning chapter at the closing navigation).

---

## Finding 2 — wrong Part label in ch05

**File:** `docs/reports/05-reading-the-training.md` lines 98 and 117

- Line 98: `That is Part III's job.` → `That is Part IV's job.`
- Line 117: `That is most of Part III and beyond.` → `That is most of Part IV and beyond.`

The remaining "Part III" at line 128 (`In Part III you will watch it happen in real time`) is correct and was left untouched — Part III covers the walking chapters (06–08).

---

## Finding 3 — git commit trailers in chapter bodies

**Files affected (10 total):**
- `docs/reports/05-reading-the-training.md`
- `docs/reports/06-watching-it-walk.md`
- `docs/reports/07-proving-its-real.md`
- `docs/reports/09-the-running-dive.md`
- `docs/reports/10-more-gaits-and-commands.md`
- `docs/reports/11-the-reward-hacking-gallery.md`
- `docs/reports/12-imitation-and-the-cartwheel.md`
- `docs/reports/13-the-backflip-in-three-attempts.md`
- `docs/reports/14-building-get-up-from-scratch.md`
- `docs/reports/15-reward-engineering-as-craft.md`

Pattern removed from each: trailing `\n---\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN` (some files had italic `*` markers, some did not).

**Verify grep result:**
```
CLEAN - no trailer found in any reports/*.md
docs/index.md is clean
```

---

## Finding 4 — ch15 tier-spec links 404 on Pages

**File:** `docs/reports/15-reward-engineering-as-craft.md` lines 131–134

Converted four relative `../superpowers/specs/` links to absolute GitHub blob URLs:

| Tier | Filename |
|------|----------|
| 1 | `2026-06-19-tier1-gait-tweaks-compact.md` |
| 2 | `2026-06-19-tier2-acrobatics-compact.md` |
| 3 | `2026-06-19-tier3-tasks-compact.md` |
| 4 | `2026-06-19-tier4-objects-compact.md` |

Base URL: `https://github.com/chaotic-curiosity-io/g1-humanoid-rl/blob/main/docs/superpowers/specs/`

Link text unchanged.

---

## Concerns

None. All changes are documentation-only, correct the four specific issues from the final whole-branch review, and introduce no new content, draft banners, or LaTeX.
