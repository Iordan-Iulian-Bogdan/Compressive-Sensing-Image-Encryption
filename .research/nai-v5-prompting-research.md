# NovelAI Diffusion V5 — NSFW Prompting Research

Researched 2026-10-02. Companion file: `nsfw_tags_reference.md` (tag list), `danbooru-tags.csv` (full ~40k-tag dataset with usage counts).

## Sources
| Source | URL | What it covers |
|---|---|---|
| NAI official docs — Tags | https://docs.novelai.net/en/image/tags/ | Quality/aesthetic/year/dataset tags |
| NAI official docs — Steps & Guidance | https://docs.novelai.net/en/image/stepsguidance | Steps, guidance 5–6 sweet spot, Decrisper |
| NAI official docs — Strengthening & Weakening | https://docs.novelai.net/en/image/strengthening-weakening | `::` numerical emphasis (V4+), negative values |
| NAI official docs — Undesired Content | https://docs.novelai.net/en/image/undesiredcontent | Negative field, V5 presets |
| MAKIBISHI — V5 Practical Tips (note.com) | https://note.com/makibishi_1619/n/nee6d19908213?hl=en | Recommended starting settings, Opus quota limits |
| ero-prompt.com — V5 features guide | https://ero-prompt.com/?p=5984&lang=en | Japanese prompts, positioning, transparency, manga |
| ero-prompt.com — NSFW basics | https://ero-prompt.com/?p=5989&lang=en | Quality tags, background prompting |
| airmore.ai — V5 review | https://airmore.ai/ai-review/novelai-v5-review | Multilingual support, prompt structure advice |
| NAI Tags+ userscript (gist) | https://gist.github.com/catboxanon/cfcd89f2838b410d3c5b10ca46e799da | Adds "de-aligned"/artist tags to NAI tag suggestions; alias map |
| HDiffusion danbooru tag counts (HF) | https://huggingface.co/datasets/HDiffusion/historical-danbooru-tag-counts | Source of the local CSV |

## V5 facts
- Released 2026-08-21. Models: **V5 Full** and **V5 Curated**.
- Natural-language prompting in English + Japanese (also Traditional Chinese, German, Spanish, Portuguese). Long prompts work — a ~1,000-char Japanese passage functions as a prompt.
- New vs V4.5: better multi-character positioning control, transparent output / partial transparency, text rendering, manga generation with dialogue.
- V5 Curated-specific inpainting and Vibe Transfer not yet available at time of writing.

## Recommended starting settings (V5)
1. **Steps ≤ 28** — standard resolution up to 28 steps stays inside the Opus flat-rate quota; beyond that it consumes Anlas.
2. **Sampler: Euler Ancestral** — only recommended sampler for V5.
3. **Prompt Guidance ≈ 5.0** (official docs say 5–6 is the sweet spot). Lower = softer/painterly, higher = sharper/more saturated. Enable **Decrisper** if colors get "deepfried" at high guidance.
4. **Mode: Anime** (or Furry) — don't leave it in the wrong mode; furry mode auto-prefixes `fur dataset`.
5. Don't carry V4.5 settings over unchanged — considered the most disadvantageous approach.

## Prompt structure that works
1. **Dataset tags first** in the base prompt: `fur dataset` (V4+), `background dataset` (V4.5+, for landscapes/still lifes, photographic style).
2. **Quality tag**: `best quality` / `amazing quality` / `great quality`. Negatives exist too (`bad quality`, `worst quality`).
3. **Aesthetic tags** (optional): `masterpiece` (V4.5+), `top aesthetic`, `very aesthetic` / `aesthetic`; negatives: `displeasing`, `very displeasing`.
4. **Content in a sensible order**: characters → position → action → expression → background → short text elements.
5. Danbooru-style tags are the native vocabulary; use exact tag names (see reference file).

## Emphasis syntax (V4+)
- Numerical emphasis with `::`: `1girl, 1.5::rain, night ::, 0.5::coat ::` — number before `::` sets weight for everything until a closing `::`.
- Weaken: weights between 0.0 and 1.0. Classic brackets `{}` / `[]` still work; a bare `::` closes any open brackets.
- **Negative values** (`-1::hat ::`, `-3::hat ::`) remove concepts or prompt for the *opposite* — e.g. `-1::monochrome ::` restores color, `-6::simple illustration ::` adds detail, `-1::simple background ::, location` fixes white-void backgrounds. Better than generic negative prompting when you want targeted removal.

## Undesired Content (negative prompt)
- Separate tab of the same input field; can be detached below the prompt box.
- Built-in presets: **Heavy V5 Full**, **Light V5 Full** (`lowres, bad hands, bad anatomy, artistic error, sepia, white haze, worst quality, very displeasing, jpeg artifacts, 0::ai-generated::`), **Furry Focus V5 Full**. Recommended ones are on by default.
- Trick: to kill an artifact caused by a tag, negate a *related* concept (e.g. `freckles` artifacts → add `tattoo` to undesired).

## NSFW-specific notes
- Nothing special is needed — NSFW output comes from explicit danbooru tags; the model has no content filter in the prompt language. Use precise tag names, not prose adjectives.
- NovelAI's built-in tag suggestions hide "de-aligned" (explicit) and artist tags; the **NAI Tags+** userscript re-adds them to autocomplete. Its alias map also shows NAI-specific renames (e.g. `pubic_tattoo` → `womb_tattoo`, `eyepatch_bikini` → `square_bikini`).
- For heavy explicit scenes, combine: quality tag + rating/act tags from the reference list + emphasis on the key elements + a Light V5 Full undesired preset.
