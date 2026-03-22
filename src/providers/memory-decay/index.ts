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

/**
 * Memory Decay Provider
 *
 * Bridges MemoryBench to the memory-decay FastAPI server via HTTP.
 * The server implements decay-based memory retrieval using a NetworkX
 * graph with activation scores that decay over simulated time.
 *
 * Each question runs in isolation: reset → ingest → simulate decay → search.
 * Session dates are mapped to ticks (1 tick = 1 day) so older memories
 * decay more than recent ones.
 */
export class MemoryDecayProvider implements Provider {
  name = "memory-decay"
  prompts: ProviderPrompts = memoryDecayPrompts

  private baseUrl = "http://localhost:8100"
  private simulateTicks = 0

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
    // Reset graph for per-question isolation
    await fetch(`${this.baseUrl}/reset`, { method: "POST" })

    // Compute date-to-tick mapping
    const sessionDates = this.extractSessionDates(sessions)
    const earliestMs = Math.min(...sessionDates.map((d) => d.getTime()))

    // Compute target ticks for simulation from question metadata
    const questionDate = this.extractQuestionDate(options)
    if (questionDate) {
      this.simulateTicks = Math.max(
        1,
        Math.floor((questionDate.getTime() - earliestMs) / ONE_DAY_MS)
      )
    } else {
      // Fallback: latest session date + 30 days
      const latestMs = Math.max(...sessionDates.map((d) => d.getTime()))
      this.simulateTicks = Math.max(
        1,
        Math.floor((latestMs - earliestMs) / ONE_DAY_MS) + 30
      )
    }

    // Ingest each message as a memory
    const documentIds: string[] = []
    let stored = 0
    let skipped = 0

    for (let si = 0; si < sessions.length; si++) {
      const session = sessions[si]
      const sessionDate = sessionDates[si]
      const createdTick = Math.floor(
        (sessionDate.getTime() - earliestMs) / ONE_DAY_MS
      )

      for (const msg of session.messages) {
        const importance = msg.role === "user" ? 0.7 : 0.4
        const prefix = msg.role === "user" ? "[User]" : "[Assistant]"
        const text = `${prefix} ${msg.content}`

        try {
          const res = await fetch(`${this.baseUrl}/store`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text,
              importance,
              mtype: "episode",
              created_tick: createdTick,
            }),
            signal: AbortSignal.timeout(30000),
          })
          if (res.ok) {
            const data = (await res.json()) as { id: string }
            documentIds.push(data.id)
            stored++
          } else {
            logger.warn(
              `[memory-decay] /store failed for session ${session.sessionId}: ${res.status}`
            )
            skipped++
          }
        } catch (e) {
          logger.warn(`[memory-decay] /store error: ${e}`)
          skipped++
        }
      }
    }

    logger.debug(
      `Ingested ${stored} memories (${skipped} skipped) across ${sessions.length} sessions, ` +
        `will simulate ${this.simulateTicks} ticks`
    )

    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // Run decay simulation to the question date.
    // Server caps /tick at 1000, so loop for large ranges.
    let remaining = this.simulateTicks
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

    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
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

  async clear(_containerTag: string): Promise<void> {
    await fetch(`${this.baseUrl}/reset`, { method: "POST" })
  }

  // --- Private helpers ---

  private extractSessionDates(sessions: UnifiedSession[]): Date[] {
    return sessions.map((session) => {
      const meta = session.metadata || {}

      // Try ISO date from metadata
      const dateStr = (meta.date || meta.formattedDate || meta.iso_date) as
        | string
        | undefined
      if (dateStr) {
        const parsed = new Date(dateStr)
        if (!isNaN(parsed.getTime())) return parsed
      }

      // Try first message timestamp
      if (session.messages.length > 0 && session.messages[0].timestamp) {
        const parsed = new Date(session.messages[0].timestamp)
        if (!isNaN(parsed.getTime())) return parsed
      }

      // Fallback to epoch
      return new Date(0)
    })
  }

  private extractQuestionDate(options: IngestOptions): Date | null {
    const meta = options.metadata || {}
    const dateStr = (meta.questionDate || meta.question_date) as
      | string
      | undefined
    if (dateStr) {
      const parsed = new Date(dateStr)
      if (!isNaN(parsed.getTime())) return parsed
    }
    return null
  }
}

export default MemoryDecayProvider
