import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { ProviderPrompts } from "../../types/prompts"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { memoryDecayPrompts } from "./prompts"
import { execFile } from "child_process"
import { promisify } from "util"
import { readFileSync } from "fs"
import { resolve, dirname } from "path"
import { fileURLToPath } from "url"

const execFileAsync = promisify(execFile)

const ONE_DAY_MS = 24 * 60 * 60 * 1000

interface CachedMessage {
  text: string
  importance: number
  created_tick: number
  sessionDateMs: number
  speaker: string
}

interface CachedQuestion {
  messages: CachedMessage[]
  sessionDatesMs: number[]
}

/**
 * Memory Decay Provider
 *
 * Bridges MemoryBench to the memory-decay FastAPI server via HTTP.
 * The server implements decay-based memory retrieval using a NetworkX
 * graph with activation scores that decay over simulated time.
 *
 * MemoryBench batches phases (all ingests, then all searches), but our
 * server has a single graph. To handle per-question isolation, we cache
 * ingest data locally during ingest() and replay reset → store → tick
 * at search() time. The server's embedding cache ensures re-ingestion
 * of previously seen texts is fast (no duplicate API calls).
 */
export class MemoryDecayProvider implements Provider {
  name = "memory-decay"
  prompts: ProviderPrompts = memoryDecayPrompts

  private baseUrl = "http://localhost:8100"
  private cache = new Map<string, CachedQuestion>()
  private useAgentMode = false

  // Agent-based answer function: invokes Claude Code CLI
  answerFunction?: (question: string, context: unknown[], questionDate?: string) => Promise<string>

  async initialize(config: ProviderConfig): Promise<void> {
    if (config.baseUrl) {
      this.baseUrl = config.baseUrl as string
    }

    // Enable agent mode with config flag: { agentMode: true }
    if (config.agentMode) {
      this.useAgentMode = true
      this.answerFunction = this.agentAnswer.bind(this)
      logger.info("[memory-decay] Agent mode enabled — answers will use Claude Code CLI")
    }

    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      })
      if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
      const data = (await res.json()) as { status: string; current_tick: number }
      logger.info(
        `Connected to memory-decay server at ${this.baseUrl} (tick=${data.current_tick})`
      )
    } catch (e) {
      throw new Error(
        `Cannot reach memory-decay server at ${this.baseUrl}. ` +
          `Start it with: python -m memory_decay.server --port 8100 ` +
          `--experiment-dir experiments/exp_lme_0292`
      )
    }
  }

  async ingest(
    sessions: UnifiedSession[],
    options: IngestOptions
  ): Promise<IngestResult> {
    // Cache locally — actual server ingest happens in search() to handle
    // MemoryBench's batched phase execution.
    // NOTE: The orchestrator calls ingest() once per session (not per question),
    // so we ACCUMULATE messages across calls with the same containerTag.

    const existing = this.cache.get(options.containerTag) || {
      messages: [],
      sessionDatesMs: [] as number[],
    }
    const startIndex = existing.messages.length

    const sessionDates = this.extractSessionDates(sessions)

    // Accumulate messages with their session dates
    for (let si = 0; si < sessions.length; si++) {
      const session = sessions[si]
      const sessionDateMs = sessionDates[si].getTime()
      existing.sessionDatesMs.push(sessionDateMs)

      for (const msg of session.messages) {
        const importance = msg.role === "user" ? 0.7 : 0.4
        const speaker = msg.speaker || (msg.role === "user" ? "User" : "Assistant")

        // Keep original [User]/[Assistant] prefix to preserve embedding cache;
        // speaker name and date travel as metadata for the answer LLM
        const prefix = msg.role === "user" ? "[User]" : "[Assistant]"

        existing.messages.push({
          text: `${prefix} ${msg.content}`,
          importance,
          created_tick: 0, // will be recomputed in finalizeCache()
          sessionDateMs,
          speaker,
        })
      }
    }

    this.cache.set(options.containerTag, existing)

    const documentIds = existing.messages
      .slice(startIndex)
      .map((_, i) => `${options.containerTag}_${startIndex + i}`)

    logger.debug(
      `Cached ${existing.messages.length} messages (${sessions.length} new sessions) for ${options.containerTag}`
    )

    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // No-op — actual simulation happens in search()
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const cached = this.cache.get(options.containerTag)
    if (!cached || cached.messages.length === 0) return []

    // Compute tick mapping from all accumulated session dates
    const allDatesMs = cached.sessionDatesMs.filter((d) => d > 0)
    const earliestMs =
      allDatesMs.length > 0 ? Math.min(...allDatesMs) : 0
    const latestMs =
      allDatesMs.length > 0 ? Math.max(...allDatesMs) : 0

    // Assign created_tick based on relative dates
    for (const msg of cached.messages) {
      if (msg.sessionDateMs > 0 && earliestMs > 0) {
        msg.created_tick = Math.floor(
          (msg.sessionDateMs - earliestMs) / ONE_DAY_MS
        )
      } else {
        msg.created_tick = 0
      }
    }

    // Simulate ticks: latest session date + 30 days
    const simulateTicks =
      earliestMs > 0
        ? Math.max(1, Math.floor((latestMs - earliestMs) / ONE_DAY_MS) + 30)
        : 30

    // Reset → ingest → simulate → search (per-question isolation)
    await fetch(`${this.baseUrl}/reset`, { method: "POST" })

    // Hybrid storage: individual messages + session chunks (≥15 messages only).
    // Chunks help LoCoMo (long dialogues) without flooding LongMemEval (shorter sessions).
    const individualItems = cached.messages.map((msg) => ({
      text: msg.text,
      importance: msg.importance,
      mtype: "episode" as const,
      created_tick: msg.created_tick,
      speaker: msg.speaker,
    }))

    const sessionChunks = this.buildSessionChunks(cached.messages)
    const meaningfulChunks = sessionChunks.filter((c) => c.text.includes("\n"))
    const batchPayload = [...individualItems, ...meaningfulChunks]

    logger.debug(
      `[memory-decay] Hybrid: ${individualItems.length} individual + ${meaningfulChunks.length} chunks = ${batchPayload.length} total`
    )

    try {
      await fetch(`${this.baseUrl}/store-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sessionChunks),
        signal: AbortSignal.timeout(120000),
      })
    } catch (e) {
      logger.warn(`[memory-decay] /store-batch error: ${e}, falling back to individual stores`)
      for (const item of sessionChunks) {
        try {
          await fetch(`${this.baseUrl}/store`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(item),
            signal: AbortSignal.timeout(30000),
          })
        } catch (e2) {
          logger.warn(`[memory-decay] /store fallback error: ${e2}`)
        }
      }
    }

    // Run decay simulation
    let remaining = simulateTicks
    while (remaining > 0) {
      const batch = Math.min(remaining, 1000)
      await fetch(`${this.baseUrl}/tick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count: batch }),
        signal: AbortSignal.timeout(30000),
      })
      remaining -= batch
    }

    // Search
    try {
      const res = await fetch(`${this.baseUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: options.limit || 30 }),
        signal: AbortSignal.timeout(30000),
      })
      if (!res.ok) return []
      const data = (await res.json()) as {
        results: Array<{ text: string; score: number; created_tick: number; speaker?: string; [k: string]: unknown }>
      }
      // Ignore framework threshold — score ranges vary by embedding model,
      // and the server already returns top_k ranked results
      const filteredResults = data.results

      // Enrich results with computed calendar date from tick mapping
      const enriched = filteredResults.map((r) => {
        const dateMs = earliestMs > 0 ? earliestMs + r.created_tick * ONE_DAY_MS : 0
        const date = dateMs > 0 ? new Date(dateMs).toISOString().slice(0, 10) : ""
        return { ...r, date }
      })

      // Log token efficiency
      const totalChars = enriched.reduce((sum, r) => sum + (r.text?.length || 0), 0)
      const estTokens = Math.ceil(totalChars / 4)
      logger.debug(
        `[memory-decay] Search for "${query.substring(0, 40)}...": ` +
          `${enriched.length} results, ~${estTokens} context tokens ` +
          `(from ${cached.messages.length} total memories)`
      )

      return enriched
    } catch {
      return []
    }
  }

  async clear(containerTag: string): Promise<void> {
    this.cache.delete(containerTag)
  }

  // --- Agent-based answer ---

  private async agentAnswer(
    question: string,
    context: unknown[],
    questionDate?: string
  ): Promise<string> {
    const results = context as Array<{
      text: string
      score: number
      date?: string
      speaker?: string
      created_tick?: number
    }>

    // Format initial search results as conversation context
    const formatted = results
      .map((r, i) => {
        const speaker = r.speaker || "Unknown"
        const date = r.date || "unknown date"
        return `[Memory ${i + 1}] (speaker: ${speaker}, date: ${date}, score: ${r.score?.toFixed(3) || "?"})\n${r.text}`
      })
      .join("\n\n")

    const todayStr = questionDate || "unknown"

    // Load the skill file
    const skillPath = resolve(process.cwd(), "../memory-decay/skills/memory-retrieval/skill.md")
    let skillContent = ""
    try {
      skillContent = readFileSync(skillPath, "utf8")
    } catch {
      // Try alternative paths
      const altPaths = [
        resolve(process.env.HOME || "~", "memory-decay/skills/memory-retrieval/skill.md"),
        resolve(dirname(fileURLToPath(import.meta.url)), "../../../../memory-decay/skills/memory-retrieval/skill.md"),
      ]
      for (const p of altPaths) {
        try {
          skillContent = readFileSync(p, "utf8")
          break
        } catch {
          continue
        }
      }
    }

    const prompt = `${skillContent ? skillContent + "\n\n---\n\n" : ""}Today's date: ${todayStr}
Server URL: ${this.baseUrl}

<previous_conversations>
${formatted}
</previous_conversations>

Question: ${question}

Answer the question using the memories above. If you need more context, search the memory-decay server at ${this.baseUrl} using curl. Give a concise, direct answer.`

    try {
      const { stdout } = await execFileAsync("claude", [
        "-p", prompt,
        "--output-format", "json",
        "--allowedTools", "Bash",
        "--max-turns", "10",
        "--model", "sonnet",
      ], {
        timeout: 120_000,
        env: { ...process.env },
        maxBuffer: 10 * 1024 * 1024,
      })

      const response = JSON.parse(stdout)
      const answer = response.result || response.text || ""

      if (!answer) {
        logger.warn(`[memory-decay] Agent returned empty answer for question: ${question.substring(0, 50)}`)
        return "I don't know."
      }

      return answer.trim()
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e)
      logger.error(`[memory-decay] Agent answer failed: ${error}`)
      throw new Error(`Agent answer failed: ${error}`)
    }
  }

  // --- Private helpers ---

  private buildSessionChunks(
    messages: CachedMessage[]
  ): Array<{ text: string; importance: number; mtype: string; created_tick: number; speaker: string }> {
    // Adaptive chunking: group by session, but split large sessions
    // and keep short sessions as individual messages.
    //
    // - Sessions with ≥ 5 messages: chunk into session transcript
    // - Sessions with < 5 messages: keep as individual messages
    //   (short sessions lack enough context for meaningful chunks,
    //    and LongMemEval has many short sessions where chunking hurts)

    const MIN_CHUNK_SIZE = 15

    const sessionMap = new Map<number, CachedMessage[]>()
    for (const msg of messages) {
      const key = msg.sessionDateMs
      if (!sessionMap.has(key)) sessionMap.set(key, [])
      sessionMap.get(key)!.push(msg)
    }

    type ChunkItem = { text: string; importance: number; mtype: string; created_tick: number; speaker: string }
    const chunks: ChunkItem[] = []

    for (const [, sessionMsgs] of sessionMap) {
      // Skip chunking for undated sessions (sessionDateMs=0, e.g., ConvoMem)
      // — all messages share the same key, creating one giant unusable chunk
      const sessionDateMs = sessionMsgs[0].sessionDateMs
      if (sessionMsgs.length < MIN_CHUNK_SIZE || sessionDateMs === 0) {
        // Keep as individual messages (original format)
        for (const m of sessionMsgs) {
          chunks.push({
            text: m.text,
            importance: m.importance,
            mtype: "episode",
            created_tick: m.created_tick,
            speaker: m.speaker,
          })
        }
      } else {
        // Chunk into session transcript
        const lines = sessionMsgs.map((m) => `${m.speaker}: ${m.text.replace(/^\[(User|Assistant)\]\s*/, "")}`)
        const chunkText = lines.join("\n")

        const maxImportance = Math.max(...sessionMsgs.map((m) => m.importance))
        const tick = sessionMsgs[0].created_tick

        const speakerCounts = new Map<string, number>()
        for (const m of sessionMsgs) {
          speakerCounts.set(m.speaker, (speakerCounts.get(m.speaker) || 0) + 1)
        }
        const primarySpeaker = [...speakerCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || ""

        chunks.push({
          text: chunkText,
          importance: maxImportance,
          mtype: "episode",
          created_tick: tick,
          speaker: primarySpeaker,
        })
      }
    }

    return chunks
  }

  private extractSessionDates(sessions: UnifiedSession[]): Date[] {
    return sessions.map((session) => {
      const meta = session.metadata || {}

      const dateStr = (meta.date || meta.formattedDate || meta.iso_date) as
        | string
        | undefined
      if (dateStr) {
        const parsed = new Date(dateStr)
        if (!isNaN(parsed.getTime())) return parsed
      }

      if (session.messages.length > 0 && session.messages[0].timestamp) {
        const parsed = new Date(session.messages[0].timestamp)
        if (!isNaN(parsed.getTime())) return parsed
      }

      return new Date(0)
    })
  }

}

export default MemoryDecayProvider
