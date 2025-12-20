'use client'

import { motion } from 'framer-motion'
import { Message } from '@/types/api'
import AnswerText from './AnswerText'
import CitationsList from './CitationsList'
import SafetyWarning from './SafetyWarning'

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 flex items-center justify-center backdrop-blur">
            <span className="text-lg">🏥</span>
          </div>
        </div>
      )}

      {/* Content */}
      <div className={`flex-1 max-w-2xl space-y-3 ${isUser ? 'text-right' : 'text-left'}`}>
        {isUser ? (
          <div className="inline-block bg-emerald-600/20 border border-emerald-500/30 rounded-xl px-5 py-3 text-slate-100 leading-relaxed backdrop-blur">
            {message.content}
          </div>
        ) : (
          <>
            {/* Safety Warning */}
            {message.safety?.is_safe === false && (
              <SafetyWarning reason={message.safety.reason} />
            )}

            {/* Answer with typewriter effect */}
            <div className="bg-slate-900/30 border border-slate-800/60 rounded-xl p-5 backdrop-blur">
              <AnswerText text={message.content} citations={message.citations} />
            </div>

            {/* Citations */}
            {message.citations && message.citations.length > 0 && (
              <CitationsList citations={message.citations} />
            )}
          </>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-600/40 to-teal-600/40 border border-emerald-500/30 flex items-center justify-center backdrop-blur">
            <span className="text-sm font-semibold text-slate-100">U</span>
          </div>
        </div>
      )}
    </motion.div>
  )
}
