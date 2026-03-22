import type { ProviderPrompts } from "../../types/prompts"

interface MemoryDecayResult {
  id: string
  text: string
  score: number
  storage_score: number
  retrieval_score: number
  category: string
  created_tick: number
}

export const memoryDecayPrompts: ProviderPrompts = {
  answerPrompt: (question: string, context: unknown[], questionDate?: string) => {
    const results = context as MemoryDecayResult[]
    const formatted = results
      .map((r, i) => `[Memory ${i + 1}] (score: ${r.score.toFixed(3)}) ${r.text}`)
      .join("\n\n")

    // Rough token estimate (~4 chars per token)
    const contextTokens = Math.ceil(formatted.length / 4)
    const totalMemories = results.length

    return `You are answering a question using retrieved memories from past conversations.

Instructions:
- Answer based ONLY on the provided memories.
- If memories contain dates or temporal markers, pay attention to chronological order.
- If multiple memories are relevant, synthesize information across them.
- If memories conflict, prefer the most recent one (higher memory number = more recent).
- If the memories don't contain enough information, say "I don't know".
- Be concise and direct.

Memories (${totalMemories} retrieved, ~${contextTokens} tokens):
${formatted}

Question: ${question}
Answer:`
  },
}
