import type { UnifiedSession } from "./unified"
import type { ProviderPrompts } from "./prompts"
import type { ConcurrencyConfig } from "./concurrency"

export interface ProviderConfig {
  apiKey: string
  baseUrl?: string
  [key: string]: unknown
}

export interface IngestOptions {
  containerTag: string
  metadata?: Record<string, unknown>
}

export interface SearchOptions {
  containerTag: string
  limit?: number
  threshold?: number
}

export interface IngestResult {
  documentIds: string[]
  taskIds?: string[]
}

export interface IndexingProgress {
  completedIds: string[]
  failedIds: string[]
  total: number
}

export type IndexingProgressCallback = (progress: IndexingProgress) => void

export interface AnswerResult {
  text: string
  agentMetrics?: {
    numTurns: number
    inputTokens: number
    outputTokens: number
    totalCostUsd: number
    durationMs: number
  }
}

export interface Provider {
  name: string
  prompts?: ProviderPrompts
  concurrency?: ConcurrencyConfig
  initialize(config: ProviderConfig): Promise<void>
  ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult>
  awaitIndexing(
    result: IngestResult,
    containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void>
  search(query: string, options: SearchOptions): Promise<unknown[]>
  clear(containerTag: string): Promise<void>

  /**
   * Optional custom answer function. When defined, answer.ts calls this
   * instead of generateText(). Enables agent-based answer generation
   * (e.g., Claude Code CLI) that can autonomously re-query and reason.
   */
  answerFunction?: (
    question: string,
    context: unknown[],
    questionDate?: string
  ) => Promise<AnswerResult>
}

export type ProviderName = "supermemory" | "mem0" | "zep" | "filesystem" | "rag" | "memory-decay"
