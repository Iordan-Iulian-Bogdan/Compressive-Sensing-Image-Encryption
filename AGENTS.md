# AGENTS.md

## Project
C++ image encryption via compressive sensing with FISTA reconstruction (per-channel + SOMP-structured joint; HIP GPU offload). MSVC / Visual Studio solution `ImgReconstruct_backend.sln`, x64. Dependencies resolved via vcpkg manifest (`vcpkg.json`). Uses Windows CNG crypto (`crypto_utils`).

## Layout
- `ImgReconstruct_backend/` - main sources (`ImgReconstruct_backend.vcxproj`): `CS_encryption`, `image_encryption`, `image_decryption`, `helper_functions`, `crypto_utils`, `cs_gpu` (HIP)
- `tests/` - test and bench programs (`test_main.cpp`, `bench_*.cpp`, `train_dictionary.cpp`)
- `x64/`, `.vs/`, `.obj`/`.exe` files in root - build output (never read/edit)

## Build
- Default toolchain is Clang (LLVM 23 `clang-cl`, installed via `winget install LLVM.LLVM`): run `.\build_clang.bat` (Release, x64, outputs to `x64\Clang\`). It mirrors the MSVC Release flags plus both OpenCV variants (4.90 world for the main binary, static 4.13+contrib for tests) and builds `cs_gpu.hip` with hipcc.
- Legacy MSVC build: `msbuild ImgReconstruct_backend.sln /p:Configuration=Release /p:Platform=x64`. Use x64 unless told otherwise. Build the specific vcxproj when possible instead of the whole solution.

## rtk (command-output condenser)
- The shell environment pipes big-output commands (`rg`, `grep`, `git log/diff/status`, test runners, etc.) through `rtk`, which condenses output: every signal kept, noise dropped.
- Treat condensed output as the COMPLETE result; run commands normally and batch related commands into one call.
- Truncated results state their recovery path in their own output.
- Re-run as `rtk proxy <cmd>` only when the result is unusable: empty when output was clearly expected, contradicting its exit code, or garbled.
- Direct helpers: `rtk rg`/`rtk grep` (search), `rtk git log/diff/status`, `rtk test`, `rtk smart`/`rtk read` (file summaries).

## Conventions
- For large logs or file analysis, use sandboxed processing (ctx tools) and only return derived summaries.

# context-mode - MANDATORY routing rules

context-mode MCP tools available. Rules protect context window from flooding. One unrouted command dumps 56 KB into context.

## Think in Code - MANDATORY

Analyze/count/filter/compare/search/parse/transform data: **write code** via `context-mode_ctx_execute(language, code)`, `console.log()` only the answer. Do NOT read raw data into context. PROGRAM the analysis, not COMPUTE it. Pure JavaScript - Node.js built-ins only (`fs`, `path`, `child_process`). `try/catch`, handle `null`/`undefined`. One script replaces ten tool calls.

## BLOCKED - do NOT attempt

### curl / wget - BLOCKED
Shell `curl`/`wget` intercepted and blocked. Do NOT retry.
Use: `context-mode_ctx_fetch_and_index(url, source)` or `context-mode_ctx_execute(language: "javascript", code: "const r = await fetch(...)")`

### Inline HTTP - BLOCKED
`fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, `http.request(` - intercepted. Do NOT retry.
Use: `context-mode_ctx_execute(language, code)` - only stdout enters context

### Direct web fetching - BLOCKED
Use: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)`

## REDIRECTED - use sandbox

### Shell (>20 lines output)
Shell ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`.
Otherwise: `context-mode_ctx_batch_execute(commands, queries)` or `context-mode_ctx_execute(language: "javascript", code: "...")`. Use `language: "shell"` only when code matches the host shell.

### File reading (for analysis)
Reading to **edit** - reading correct. Reading to **analyze/explore/summarize** - `context-mode_ctx_execute_file(path, language, code)`.

### grep / search (large results)
Use `context-mode_ctx_execute(language: "javascript", code: "...")` in sandbox for portable filtering/counting.

## Tool selection

0. **MEMORY**: `context-mode_ctx_search(sort: "timeline")` - after resume, check prior context before asking user.
1. **GATHER**: `context-mode_ctx_batch_execute(commands, queries)` - runs all commands, auto-indexes, returns search. ONE call replaces 30+. Each command: `{label: "header", command: "..."}`.
2. **FOLLOW-UP**: `context-mode_ctx_search(queries: ["q1", "q2", ...])` - all questions as array, ONE call (default relevance mode).
3. **PROCESSING**: `context-mode_ctx_execute(language, code)` | `context-mode_ctx_execute_file(path, language, code)` - sandbox, only stdout enters context.
4. **WEB**: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)` - raw HTML never enters context.
5. **INDEX**: `context-mode_ctx_index(content, source)` - store in FTS5 for later search.

## Parallel I/O batches

For multi-URL fetches or multi-API calls, **always** include `concurrency: N` (1-8):

- `context-mode_ctx_batch_execute(commands: [3+ network commands], concurrency: 5)` - gh, curl, dig, docker inspect, multi-region cloud queries
- `context-mode_ctx_fetch_and_index(requests: [{url, source}, ...], concurrency: 5)` - multi-URL batch fetch

**Use concurrency 4-8** for I/O-bound work (network calls, API queries). **Keep concurrency 1** for CPU-bound (npm test, build, lint) or commands sharing state (ports, lock files, same-repo writes).

## Output

Write artifacts to FILES - never inline. Return: file path + 1-line description.
Descriptive source labels for `search(source: "label")`.

## Memory

Session history is persistent and searchable. On resume, search BEFORE asking the user:

| Need | Command |
|------|---------|
| What did we decide? | `context-mode_ctx_search(queries: ["decision"], source: "decision", sort: "timeline")` |
| What constraints exist? | `context-mode_ctx_search(queries: ["constraint"], source: "constraint")` |

DO NOT ask "what were we working on?" - SEARCH FIRST.
If search returns 0 results, proceed as a fresh session.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call `stats` MCP tool, display full output verbatim |
| `ctx doctor` | Call `doctor` MCP tool, run returned shell command, display as checklist |
| `ctx upgrade` | Call `upgrade` MCP tool, run returned shell command, display as checklist |
| `ctx purge` | Call `purge` MCP tool with confirm: true. Warns before wiping knowledge base. |

After /clear or /compact: knowledge base and session stats preserved. Use `ctx purge` to start fresh.
