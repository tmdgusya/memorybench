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

const ONE_DAY_MS = 24 * 60 * 60 * 1000

interface CachedMessage {
  text: string
  importance: number
  created_tick: number
  sessionDateMs: number
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

  async initialize(config: ProviderConfig): Promise<void> {
    if (config.baseUrl) {
      this.baseUrl = config.baseUrl as string
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

    const sessionDates = this.extractSessionDates(sessions)

    // Accumulate messages with their session dates
    for (let si = 0; si < sessions.length; si++) {
      const session = sessions[si]
      const sessionDateMs = sessionDates[si].getTime()
      existing.sessionDatesMs.push(sessionDateMs)

      for (const msg of session.messages) {
        const importance = msg.role === "user" ? 0.7 : 0.4
        const prefix = msg.role === "user" ? "[User]" : "[Assistant]"
        existing.messages.push({
          text: `${prefix} ${msg.content}`,
          importance,
          created_tick: 0, // will be recomputed in finalizeCache()
          sessionDateMs,
        })
      }
    }

    this.cache.set(options.containerTag, existing)

    const documentIds = existing.messages.map(
      (_, i) => `${options.containerTag}_${i}`
    )

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

    // Ingest all cached messages
    for (const msg of cached.messages) {
      try {
        await fetch(`${this.baseUrl}/store`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: msg.text,
            importance: msg.importance,
            mtype: "episode",
            created_tick: msg.created_tick,
          }),
          signal: AbortSignal.timeout(30000),
        })
      } catch (e) {
        logger.warn(`[memory-decay] /store error during search: ${e}`)
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
        body: JSON.stringify({ query, top_k: options.limit || 7 }),
        signal: AbortSignal.timeout(30000),
      })
      if (!res.ok) return []
      const data = (await res.json()) as { results: unknown[] }
      return data.results
    } catch {
      return []
    }
  }

  async clear(containerTag: string): Promise<void> {
    this.cache.delete(containerTag)
  }

  // --- Private helpers ---

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
