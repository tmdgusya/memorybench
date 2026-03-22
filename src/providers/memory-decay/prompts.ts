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

    return `Answer based only on the provided memories.
If the memories don't contain the answer, say "I don't know".

Memories:
${formatted}

Question: ${question}
Answer:`
  },
}
