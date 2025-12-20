'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { RetrievalMethod } from '@/types/api'

interface QuestionInputProps {
  onSubmit: (question: string) => void
  disabled?: boolean
  retrievalMethod: RetrievalMethod
  onRetrievalMethodChange: (method: RetrievalMethod) => void
  onInfoClick?: () => void
}

const RETRIEVAL_METHODS = [
  {
    value: 'dense' as RetrievalMethod,
    label: 'Dense (FAISS)',
    icon: '🧠',
  },
  {
    value: 'bm25' as RetrievalMethod,
    label: 'BM25',
    icon: '🔍',
  },
  {
    value: 'hybrid' as RetrievalMethod,
    label: 'Hybrid',
    icon: '⚡',
  },
]

export default function QuestionInput({ 
  onSubmit, 
  disabled,
  retrievalMethod,
  onRetrievalMethodChange,
  onInfoClick
}: QuestionInputProps) {
  const [question, setQuestion] = useState('')
  const [isMethodOpen, setIsMethodOpen] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim() && !disabled) {
      onSubmit(question)
      setQuestion('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value)
    
    // Auto-resize
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative flex items-end gap-3">
        {/* Retrieval Method Dropdown */}
        <div ref={dropdownRef} className="relative">
          <button
            type="button"
            onClick={() => setIsMethodOpen(!isMethodOpen)}
            className="flex items-center gap-2 px-3.5 py-3 bg-slate-900/40 border border-slate-800/60 rounded-xl text-sm text-slate-300 hover:text-slate-100 hover:border-emerald-500/40 transition-all backdrop-blur whitespace-nowrap h-full"
            title="Retrieval method"
          >
            <span className="text-lg">{RETRIEVAL_METHODS.find(m => m.value === retrievalMethod)?.icon}</span>
            <span className="font-medium hidden sm:inline">{RETRIEVAL_METHODS.find(m => m.value === retrievalMethod)?.label}</span>
            <svg
              className={`w-4 h-4 transition-transform ${isMethodOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Dropdown Menu */}
          {isMethodOpen && (
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.1 }}
              className="absolute bottom-full mb-2 left-0 bg-slate-900/95 border border-slate-800/60 rounded-xl shadow-2xl z-50 overflow-hidden backdrop-blur min-w-max"
            >
              {RETRIEVAL_METHODS.map((method) => (
                <button
                  key={method.value}
                  type="button"
                  onClick={() => {
                    onRetrievalMethodChange(method.value)
                    setIsMethodOpen(false)
                  }}
                  className={`w-full px-4 py-3 text-left text-sm font-medium transition-colors flex items-center gap-3 ${
                    retrievalMethod === method.value 
                      ? 'bg-emerald-500/10 text-slate-100 border-l-2 border-emerald-500' 
                      : 'text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <span className="text-lg">{method.icon}</span>
                  <span>{method.label}</span>
                  {retrievalMethod === method.value && (
                    <svg className="w-4 h-4 text-emerald-500 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              ))}
            </motion.div>
          )}
        </div>

        {/* Info Button */}
        {onInfoClick && (
          <button
            type="button"
            onClick={onInfoClick}
            className="p-3 text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 rounded-xl transition-all backdrop-blur h-full"
            title="Learn about retrieval methods"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        )}

        {/* Input Area */}
        <div className="flex-1 relative flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={question}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Ask a medical question..."
              disabled={disabled}
              rows={1}
              className="w-full px-5 py-3 pr-14 bg-slate-900/40 border border-slate-800/60 rounded-xl 
                       text-slate-100 placeholder-slate-500 
                       focus:outline-none focus:border-emerald-500/40 focus:ring-1 focus:ring-emerald-500/20
                       disabled:opacity-50 disabled:cursor-not-allowed
                       resize-none overflow-hidden
                       transition-all duration-200 backdrop-blur"
            />
            <motion.button
              type="submit"
              disabled={!question.trim() || disabled}
              whileHover={question.trim() && !disabled ? { scale: 1.05 } : {}}
              whileTap={question.trim() && !disabled ? { scale: 0.95 } : {}}
              className="absolute right-2 bottom-2.5 p-2.5 rounded-lg
                       bg-emerald-600/80 hover:bg-emerald-600
                       disabled:bg-slate-700/50 disabled:cursor-not-allowed
                       transition-all duration-200 backdrop-blur"
            >
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </motion.button>
          </div>
        </div>
      </div>
    </form>
  )
}
