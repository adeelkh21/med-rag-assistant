'use client'

import { motion, AnimatePresence } from 'framer-motion'

interface RetrievalInfoModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function RetrievalInfoModal({ isOpen, onClose }: RetrievalInfoModalProps) {
  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-navy-800 border border-navy-700/50 rounded-2xl shadow-2xl max-w-3xl w-full max-h-[80vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="sticky top-0 bg-navy-800 border-b border-navy-700/50 p-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <span className="text-xl">💡</span>
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-100">Retrieval Methods Guide</h2>
                <p className="text-sm text-gray-400">Choose the right method for your query</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-200 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Dense */}
            <div className="bg-navy-900/50 rounded-xl p-5 border border-navy-700/30">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">🧠</span>
                <h3 className="text-lg font-bold text-cyan-400">Dense (FAISS)</h3>
                <span className="text-xs bg-cyan-900/30 text-cyan-400 px-2 py-1 rounded">Semantic</span>
              </div>
              <p className="text-sm text-gray-300 mb-4">
                Uses AI embeddings to understand meaning and context. Matches documents with similar concepts, not just keywords.
              </p>
              <div className="bg-navy-800/50 rounded-lg p-3 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Understands synonyms and medical terminology</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Handles paraphrasing and context</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-400 text-xs mt-0.5">✗</span>
                  <p className="text-xs text-gray-300">May miss exact keyword matches</p>
                </div>
              </div>
            </div>

            {/* BM25 */}
            <div className="bg-navy-900/50 rounded-xl p-5 border border-navy-700/30">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">🔍</span>
                <h3 className="text-lg font-bold text-cyan-400">BM25</h3>
                <span className="text-xs bg-purple-900/30 text-purple-400 px-2 py-1 rounded">Keyword</span>
              </div>
              <p className="text-sm text-gray-300 mb-4">
                Traditional keyword-based search. Fast and precise for exact term matches like drug names and medical codes.
              </p>
              <div className="bg-navy-800/50 rounded-lg p-3 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Fast and efficient</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Perfect for exact matches (drug names, codes)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-400 text-xs mt-0.5">✗</span>
                  <p className="text-xs text-gray-300">Misses synonyms and related concepts</p>
                </div>
              </div>
            </div>

            {/* Hybrid */}
            <div className="bg-navy-900/50 rounded-xl p-5 border border-navy-700/30">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">⚡</span>
                <h3 className="text-lg font-bold text-cyan-400">Hybrid</h3>
                <span className="text-xs bg-amber-900/30 text-amber-400 px-2 py-1 rounded">Recommended</span>
              </div>
              <p className="text-sm text-gray-300 mb-4">
                Combines both BM25 and Dense methods. Gets exact matches AND semantic understanding for comprehensive results.
              </p>
              <div className="bg-navy-800/50 rounded-lg p-3 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Best overall performance</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-0.5">✓</span>
                  <p className="text-xs text-gray-300">Catches both exact and semantic matches</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-400 text-xs mt-0.5">✗</span>
                  <p className="text-xs text-gray-300">Slightly slower (runs both methods)</p>
                </div>
              </div>
            </div>

            {/* Examples Section */}
            <div className="border-t border-navy-700/50 pt-6">
              <h3 className="text-lg font-bold text-gray-100 mb-4 flex items-center gap-2">
                <span>📋</span>
                Example Queries
              </h3>
              
              <div className="space-y-4">
                {/* Example 1 */}
                <div className="bg-navy-900/30 rounded-lg p-4">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-cyan-400 font-mono text-sm">"What is myocardial infarction?"</span>
                  </div>
                  <div className="space-y-1 ml-4 text-xs">
                    <div className="flex items-start gap-2">
                      <span className="text-cyan-400">🧠 Dense:</span>
                      <span className="text-gray-400">Will match "heart attack" documents</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-purple-400">🔍 BM25:</span>
                      <span className="text-gray-400">Only matches exact term</span>
                    </div>
                  </div>
                </div>

                {/* Example 2 */}
                <div className="bg-navy-900/30 rounded-lg p-4">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-cyan-400 font-mono text-sm">"tamoxifen"</span>
                    <span className="text-xs text-gray-500">(exact drug name)</span>
                  </div>
                  <div className="space-y-1 ml-4 text-xs">
                    <div className="flex items-start gap-2">
                      <span className="text-purple-400">🔍 BM25:</span>
                      <span className="text-gray-400">Strong exact match</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-cyan-400">🧠 Dense:</span>
                      <span className="text-gray-400">May match related concepts</span>
                    </div>
                  </div>
                </div>

                {/* Example 3 */}
                <div className="bg-navy-900/30 rounded-lg p-4">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-cyan-400 font-mono text-sm">"How do I prevent getting sick?"</span>
                    <span className="text-xs text-gray-500">(vague query)</span>
                  </div>
                  <div className="space-y-1 ml-4 text-xs">
                    <div className="flex items-start gap-2">
                      <span className="text-amber-400">⚡ Hybrid:</span>
                      <span className="text-gray-400">Best chance of finding relevant health info</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-purple-400">🔍 BM25:</span>
                      <span className="text-gray-400">May struggle with non-specific terms</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Stats */}
            <div className="border-t border-navy-700/50 pt-6">
              <h3 className="text-lg font-bold text-gray-100 mb-4 flex items-center gap-2">
                <span>📊</span>
                Performance Comparison
              </h3>
              
              <div className="bg-navy-900/30 rounded-lg p-4">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-navy-700/30">
                      <th className="text-left py-2 text-gray-400 font-medium">Method</th>
                      <th className="text-center py-2 text-gray-400 font-medium">Top-1</th>
                      <th className="text-center py-2 text-gray-400 font-medium">Top-5</th>
                      <th className="text-left py-2 text-gray-400 font-medium">Best For</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300">
                    <tr className="border-b border-navy-700/20">
                      <td className="py-2">🔍 BM25</td>
                      <td className="text-center font-bold text-green-400">7.5%</td>
                      <td className="text-center">21.0%</td>
                      <td>Exact matches</td>
                    </tr>
                    <tr className="border-b border-navy-700/20">
                      <td className="py-2">🧠 Dense</td>
                      <td className="text-center">0.0%</td>
                      <td className="text-center">23.2%</td>
                      <td>Semantic search</td>
                    </tr>
                    <tr>
                      <td className="py-2">⚡ Hybrid</td>
                      <td className="text-center">3.3%</td>
                      <td className="text-center font-bold text-green-400">26.2%</td>
                      <td>Overall quality</td>
                    </tr>
                  </tbody>
                </table>
                <p className="text-xs text-gray-500 mt-3">
                  * Based on 50-question medical benchmark. Top-K shows recall percentage.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
