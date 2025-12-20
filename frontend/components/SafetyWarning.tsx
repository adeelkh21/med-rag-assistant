'use client'

import { motion } from 'framer-motion'

interface SafetyWarningProps {
  reason?: string
}

export default function SafetyWarning({ reason }: SafetyWarningProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-4 bg-red-900/20 border border-red-700/50 rounded-xl"
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">🚨</span>
        <div>
          <h4 className="text-red-400 font-semibold mb-1">Safety Notice</h4>
          <p className="text-sm text-gray-300">
            {reason || 'This question involves topics that require professional medical consultation.'}
          </p>
          <p className="text-xs text-gray-400 mt-2">
            Please consult with a healthcare provider for personalized medical advice.
          </p>
        </div>
      </div>
    </motion.div>
  )
}
