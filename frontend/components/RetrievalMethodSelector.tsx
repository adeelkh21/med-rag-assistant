'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export type RetrievalMethod = 'dense' | 'bm25' | 'hybrid'

interface RetrievalMethodOption {
  value: RetrievalMethod
  label: string
  description: string
  icon: string
}

const RETRIEVAL_METHODS: RetrievalMethodOption[] = [
  {
    value: 'dense',
    label: 'Dense (FAISS)',
    description: 'Semantic similarity with embeddings',
    icon: '🧠',
  },
  {
    value: 'bm25',
    label: 'BM25',
    description: 'Keyword-based sparse retrieval',
    icon: '🔍',
  },
  {
    value: 'hybrid',
    label: 'Hybrid',
    description: 'Combined BM25 + Dense fusion',
    icon: '⚡',
  },
]

interface RetrievalMethodSelectorProps {
  value: RetrievalMethod
  onChange: (method: RetrievalMethod) => void
  onInfoClick?: () => void
}

export default function RetrievalMethodSelector({
  value,
  onChange,
  onInfoClick,
}: RetrievalMethodSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const selectedMethod = RETRIEVAL_METHODS.find((m) => m.value === value)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelect = (method: RetrievalMethod) => {
    onChange(method)
    setIsOpen(false)
  }

  return (
    <div ref={dropdownRef} className="relative flex items-center gap-2">
      {/* Info Button */}
      {onInfoClick && (
        <button
          onClick={onInfoClick}
          className="p-2.5 text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 rounded-lg transition-all backdrop-blur"
          title="Learn about retrieval methods"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </button>
      )}

      {/* Selector Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2.5 bg-slate-900/40 border border-slate-800/60 rounded-lg text-sm text-slate-300 hover:text-slate-100 hover:border-emerald-500/40 transition-all backdrop-blur"
      >
        <span>{selectedMethod?.icon}</span>
        <span className="font-medium">{selectedMethod?.label}</span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full mt-2 left-0 w-80 bg-slate-900/95 border border-slate-800/60 rounded-xl shadow-2xl z-50 overflow-hidden backdrop-blur"
          >
            {RETRIEVAL_METHODS.map((method) => (
              <button
                key={method.value}
                onClick={() => handleSelect(method.value)}
                className={`w-full px-4 py-3 text-left hover:bg-slate-800/50 transition-colors ${
                  value === method.value ? 'bg-emerald-500/10 border-l-2 border-emerald-500' : ''
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="text-xl">{method.icon}</span>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-200">{method.label}</span>
                      {value === method.value && (
                        <svg className="w-4 h-4 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <p className="text-xs text-slate-400 mt-1">{method.description}</p>
                  </div>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
