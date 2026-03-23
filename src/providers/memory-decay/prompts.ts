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
    const todayStr = questionDate || "unknown"

    return `You are answering a question using retrieved memories from past conversations.

Today's date: ${todayStr}

Instructions:
- Answer based ONLY on the provided memories.
- Each memory has a speaker name and a calendar date in its header.
- TEMPORAL REASONING is critical:
  - Today's date is ${todayStr}. Use this to compute "how many days/months ago" questions.
  - Convert relative references within memories to absolute dates:
    "yesterday" from 2023-05-08 = 2023-05-07
    "last week" from 2023-06-09 ≈ 2023-06-02
    "10 days ago" from 2023-03-20 = 2023-03-10
  - For "which came first" questions: compare the dates of the memories, not their order in the list.
  - For "how many days ago" questions: compute (today - event_date) in days. Show your math.
- KNOWLEDGE UPDATES: When multiple memories mention the same fact with different values, the MOST RECENT memory (latest date) has the current value. Older values are outdated.
- COMPLETENESS: Scan ALL memories before answering. For "how many" or "list all" questions, gather evidence from every memory, not just the top few.
- INFERENCE: Read between the lines. If someone says "I applied to adoption agencies" without a partner mentioned, they are likely single. If someone discusses "my transgender journey", their identity is transgender.
- Synthesize information across multiple memories when needed.
- IMPORTANT: Prefer giving your best answer over saying "I don't know". If the memories contain relevant clues, use reasoning to provide an answer even if it requires inference. Only say "I don't know" if the memories contain absolutely no relevant information.
- Be concise and direct. Give specific dates, names, and facts.

Memories (${totalMemories} retrieved, ~${contextTokens} tokens):
${formatted}

Question: ${question}
Answer:`
  },
}
