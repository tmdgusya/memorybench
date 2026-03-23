import type { ProviderPrompts } from "../../types/prompts"

interface MemoryDecayResult {
  id: string
  text: string
  score: number
  storage_score: number
  retrieval_score: number
  category: string
  created_tick: number
  speaker?: string
  date?: string
}

export const memoryDecayPrompts: ProviderPrompts = {
  answerPrompt: (question: string, context: unknown[], questionDate?: string) => {
    const results = context as MemoryDecayResult[]
    const formatted = results
      .map((r, i) => {
        const speaker = r.speaker || "Unknown"
        const date = r.date || "unknown date"
        return `[Memory ${i + 1}] (speaker: ${speaker}, date: ${date}) ${r.text}`
      })
      .join("\n\n")

    const contextTokens = Math.ceil(formatted.length / 4)
    const totalMemories = results.length

    return `You are answering a question using retrieved memories from past conversations.

Instructions:
- Answer based ONLY on the provided memories.
- Each memory has a speaker name and a calendar date in its header.
- TEMPORAL REASONING: Convert all relative time references to absolute dates.
  - "yesterday" from a memory dated 2023-05-08 = 2023-05-07
  - "last week" from 2023-06-09 = approximately 2023-06-02
  - "last year" from a memory in 2023 = 2022
  - "two weekends ago" from 2023-07-17 = approximately 2023-07-03
  - Always provide the computed date/year in your answer, not the relative reference.
- INFERENCE: Read between the lines. If someone says "I applied to adoption agencies" without a partner mentioned, they are likely single. If someone discusses "my transgender journey", their identity is transgender.
- Synthesize information across multiple memories when needed.
- If memories conflict, prefer the most recent one.
- If the memories truly don't contain enough information, say "I don't know".
- Be concise and direct. Give specific dates, names, and facts.

Memories (${totalMemories} retrieved, ~${contextTokens} tokens):
${formatted}

Question: ${question}
Answer:`
  },
}
