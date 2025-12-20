'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Citation } from '@/types/api'

interface AnswerTextProps {
  text: string
  citations?: Citation[]
}

export default function AnswerText({ text, citations }: AnswerTextProps) {
  const [displayText, setDisplayText] = useState('')
  const [isComplete, setIsComplete] = useState(false)

  // Replace chunk IDs with numbered citations
  const getProcessedText = (rawText: string) => {
    if (!citations || citations.length === 0) return rawText

    let processed = rawText
    const citationMap = new Map<string, number>()
    
    // Build citation map
    citations.forEach((citation, idx) => {
      citationMap.set(citation.chunk_id, idx + 1)
    })

    // Replace (CHUNK_ID) with [number]
    citations.forEach((citation) => {
      const chunkId = citation.chunk_id
      const number = citationMap.get(chunkId)
      // Match both (CHUNK_ID) and [CHUNK_ID] formats
      const regex = new RegExp(`[\\(\\[]${chunkId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[\\)\\]]`, 'g')
      processed = processed.replace(regex, `[${number}]`)
    })

    return processed
  }

  const processedText = getProcessedText(text)

  useEffect(() => {
    setDisplayText('')
    setIsComplete(false)

    let index = 0
    const interval = setInterval(() => {
      if (index < processedText.length) {
        setDisplayText(processedText.slice(0, index + 1))
        index++
      } else {
        setIsComplete(true)
        clearInterval(interval)
      }
    }, 10) // Fast typewriter effect

    return () => clearInterval(interval)
  }, [processedText])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="text-gray-100 leading-relaxed whitespace-pre-wrap"
    >
      {displayText}
      {!isComplete && (
        <motion.span
          animate={{ opacity: [1, 0] }}
          transition={{ duration: 0.8, repeat: Infinity }}
          className="inline-block w-1 h-5 bg-cyan-500 ml-1"
        />
      )}
    </motion.div>
  )
}
