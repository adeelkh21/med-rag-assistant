import { AnswerResponse, RetrievalMethod } from '@/types/api'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function askQuestion(
  question: string,
  retrieval_method: RetrievalMethod = 'dense'
): Promise<AnswerResponse> {
  const response = await fetch(`${API_URL}/api/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, retrieval_method }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }))
    throw new Error(error.detail || 'Failed to get answer')
  }

  return response.json()
}
