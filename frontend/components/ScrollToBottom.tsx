'use client'

import { motion } from 'framer-motion'

interface ScrollToBottomProps {
  onClick: () => void
}

export default function ScrollToBottom({ onClick }: ScrollToBottomProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="fixed bottom-24 left-1/2 -translate-x-1/2 z-10"
    >
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={onClick}
        className="p-3 bg-navy-800 border border-navy-700 rounded-full shadow-lg hover:bg-navy-700 transition-colors glow-blue"
      >
        <svg
          className="w-5 h-5 text-cyan-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </motion.button>
    </motion.div>
  )
}
