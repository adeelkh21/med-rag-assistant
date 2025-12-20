'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Citation } from '@/types/api'

interface CitationsListProps {
  citations: Citation[]
}

export default function CitationsList({ citations }: CitationsListProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="mt-4 border border-navy-700/50 rounded-xl overflow-hidden bg-navy-800/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-navy-800/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-sm font-medium">📚 Sources</span>
          <span className="text-gray-500 text-xs">({citations.length})</span>
        </div>
        <motion.svg
          animate={{ rotate: expanded ? 180 : 0 }}
          className="w-5 h-5 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </motion.svg>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-navy-700/50"
          >
            <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
              {citations.map((citation, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="p-3 bg-navy-900/50 rounded-lg border border-navy-700/30"
                >
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-xs font-bold text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded">
                      [{idx + 1}]
                    </span>
                    <span className="text-xs font-mono text-gray-400">
                      {citation.chunk_id}
                    </span>
                    <span className="text-xs text-gray-500 ml-auto">
                      Score: {(citation.similarity_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed">{citation.text}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
