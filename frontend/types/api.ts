export type RetrievalMethod = 'dense' | 'bm25' | 'hybrid'

export interface Citation {
  chunk_id: string
  text: string
  similarity_score: number
}

export interface SafetyInfo {
  is_safe: boolean
  reason?: string
}

export interface AnswerResponse {
  answer: string
  citations: Citation[]
  safety: SafetyInfo
  disclaimer: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  safety?: SafetyInfo
  timestamp: Date
}
