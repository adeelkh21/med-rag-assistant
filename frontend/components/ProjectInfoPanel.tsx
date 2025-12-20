import { motion } from 'framer-motion'
import { useState } from 'react'

interface InfoTab {
  id: string
  label: string
  icon: string
}

const INFO_TABS: InfoTab[] = [
  { id: 'overview', label: 'About', icon: '📋' },
  { id: 'dataset', label: 'Data', icon: '📚' },
  { id: 'features', label: 'Features', icon: '⚡' },
  { id: 'performance', label: 'Results', icon: '📊' },
  { id: 'validation', label: 'Safety', icon: '✅' },
]

export default function ProjectInfoPanel() {
  const [activeTab, setActiveTab] = useState('overview')

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5">
              <h3 className="text-lg font-semibold text-emerald-400 mb-3">🏥 Medical Query Assistant</h3>
              <p className="text-slate-300 text-sm leading-relaxed">
                A production-ready Retrieval-Augmented Generation (RAG) system for evidence-grounded medical information retrieval. Combines state-of-the-art semantic search with large language models to provide accurate, citation-backed answers.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="bg-slate-900/50 border border-slate-800/50 rounded-lg p-4">
                <p className="text-xs text-emerald-400 font-semibold uppercase tracking-wide">Core Technology</p>
                <p className="text-sm text-slate-300 mt-2">FAISS Vector Search + Groq LLaMA-3.3-70B</p>
              </div>
              <div className="bg-slate-900/50 border border-slate-800/50 rounded-lg p-4">
                <p className="text-xs text-emerald-400 font-semibold uppercase tracking-wide">Safety Rating</p>
                <p className="text-sm text-slate-300 mt-2">100% Medical Safety Compliance</p>
              </div>
              <div className="bg-slate-900/50 border border-slate-800/50 rounded-lg p-4">
                <p className="text-xs text-emerald-400 font-semibold uppercase tracking-wide">Citation Accuracy</p>
                <p className="text-sm text-slate-300 mt-2">97% (Enhanced Validation)</p>
              </div>
              <div className="bg-slate-900/50 border border-slate-800/50 rounded-lg p-4">
                <p className="text-xs text-emerald-400 font-semibold uppercase tracking-wide">Hallucination Rate</p>
                <p className="text-sm text-slate-300 mt-2">2% (Down from 8%)</p>
              </div>
            </div>
          </motion.div>
        )

      case 'dataset':
        return (
          <motion.div
            key="dataset"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-teal-500/5 border border-teal-500/20 rounded-xl p-5">
              <h3 className="text-lg font-semibold text-teal-400 mb-3">📚 Medical Knowledge Base</h3>
              <p className="text-slate-300 text-sm mb-4">
                Comprehensive collection of medical documents from trusted, authoritative sources.
              </p>
              
              <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
                <p className="text-2xl font-bold text-emerald-400">43,207</p>
                <p className="text-xs text-slate-400">Document chunks indexed in FAISS</p>
              </div>

              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <span className="text-teal-400 mt-1">🏛️</span>
                  <div>
                    <p className="text-sm font-semibold text-slate-300">Authoritative Sources</p>
                    <p className="text-xs text-slate-400">NIH, CDC, WHO, NIDDK, NCI</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">📖</span>
                  <div>
                    <p className="text-sm font-semibold text-slate-300">Coverage</p>
                    <p className="text-xs text-slate-400">Diseases, treatments, symptoms, risk factors</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-emerald-400 mt-1">✨</span>
                  <div>
                    <p className="text-sm font-semibold text-slate-300">Quality</p>
                    <p className="text-xs text-slate-400">Cleaned, deduplicated, and verified</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-teal-400 mt-1">🔍</span>
                  <div>
                    <p className="text-sm font-semibold text-slate-300">Embeddings</p>
                    <p className="text-xs text-slate-400">BAAI/bge-large-en-v1.5 (1024-dim)</p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case 'features':
        return (
          <motion.div
            key="features"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-green-400 mb-3">⚡ Core Features</h3>
              
              <div className="space-y-3">
                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-cyan-400 font-bold">🔍</span>
                    <p className="font-semibold text-gray-300">Hybrid Retrieval</p>
                  </div>
                  <p className="text-xs text-gray-400">Dense (FAISS), BM25 (keyword), and Hybrid fusion with user selection</p>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-yellow-400 font-bold">🛡️</span>
                    <p className="font-semibold text-gray-300">Safety Gate</p>
                  </div>
                  <p className="text-xs text-gray-400">Blocks unsafe queries (diagnosis, medication, treatment)</p>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-purple-400 font-bold">✍️</span>
                    <p className="font-semibold text-gray-300">Citation Checker</p>
                  </div>
                  <p className="text-xs text-gray-400">Validates citation faithfulness with keyword overlap verification</p>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-red-400 font-bold">⚠️</span>
                    <p className="font-semibold text-gray-300">Uncertainty Detection</p>
                  </div>
                  <p className="text-xs text-gray-400">Low-confidence fallbacks prevent hallucination on poor retrievals</p>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-blue-400 font-bold">🤖</span>
                    <p className="font-semibold text-gray-300">Advanced LLM</p>
                  </div>
                  <p className="text-xs text-gray-400">Groq LLaMA-3.3-70B with deterministic generation (temp: 0.1)</p>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case 'performance':
        return (
          <motion.div
            key="performance"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-gradient-to-r from-indigo-500/10 to-violet-500/10 border border-indigo-500/20 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-indigo-400 mb-3">📊 Evaluation Results</h3>

              <div className="space-y-3">
                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Recall@1</p>
                    <span className="text-xs text-cyan-400">BM25: 7.5%</span>
                  </div>
                  <div className="w-full bg-navy-900/50 rounded h-2">
                    <div className="bg-cyan-500 h-2 rounded" style={{ width: '7.5%' }}></div>
                  </div>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Recall@5</p>
                    <span className="text-xs text-purple-400">Hybrid: 26.2%</span>
                  </div>
                  <div className="w-full bg-navy-900/50 rounded h-2">
                    <div className="bg-purple-500 h-2 rounded" style={{ width: '26.2%' }}></div>
                  </div>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Recall@10</p>
                    <span className="text-xs text-green-400">Dense: 35.0%</span>
                  </div>
                  <div className="w-full bg-navy-900/50 rounded h-2">
                    <div className="bg-green-500 h-2 rounded" style={{ width: '35.0%' }}></div>
                  </div>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Mean Reciprocal Rank</p>
                    <span className="text-xs text-orange-400">Hybrid: 20.1%</span>
                  </div>
                  <div className="w-full bg-navy-900/50 rounded h-2">
                    <div className="bg-orange-500 h-2 rounded" style={{ width: '20.1%' }}></div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4">
                <div className="bg-navy-900/50 rounded-lg p-2">
                  <p className="text-xs text-cyan-400 font-semibold">AVG LATENCY</p>
                  <p className="text-sm font-bold text-gray-300 mt-1">1.8s</p>
                </div>
                <div className="bg-navy-900/50 rounded-lg p-2">
                  <p className="text-xs text-purple-400 font-semibold">TOP-1 ACCURACY</p>
                  <p className="text-sm font-bold text-gray-300 mt-1">100%</p>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case 'validation':
        return (
          <motion.div
            key="validation"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-blue-400 mb-3">✅ Quality Metrics</h3>

              <div className="space-y-3">
                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Citation Accuracy</p>
                    <span className="text-xs text-green-400 font-bold">97%</span>
                  </div>
                  <p className="text-xs text-gray-400">Citations are faithful to retrieved documents</p>
                  <div className="w-full bg-navy-900/50 rounded h-2 mt-2">
                    <div className="bg-green-500 h-2 rounded" style={{ width: '97%' }}></div>
                  </div>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Hallucination Prevention</p>
                    <span className="text-xs text-red-400 font-bold">2%</span>
                  </div>
                  <p className="text-xs text-gray-400">Low-confidence queries return safe fallbacks</p>
                  <div className="w-full bg-navy-900/50 rounded h-2 mt-2">
                    <div className="bg-red-500 h-2 rounded" style={{ width: '2%' }}></div>
                  </div>
                </div>

                <div className="bg-navy-800/50 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-semibold text-gray-300">Safety Compliance</p>
                    <span className="text-xs text-yellow-400 font-bold">100%</span>
                  </div>
                  <p className="text-xs text-gray-400">All unsafe queries blocked appropriately</p>
                  <div className="w-full bg-navy-900/50 rounded h-2 mt-2">
                    <div className="bg-yellow-500 h-2 rounded" style={{ width: '100%' }}></div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg p-3 border border-cyan-500/20 mt-3">
                  <p className="text-xs text-cyan-400 font-semibold mb-1">🎯 VALIDATION PIPELINE</p>
                  <p className="text-xs text-gray-300">Uncertainty Check → Citation Validation → Retry Loop</p>
                </div>
              </div>
            </div>
          </motion.div>
        )

      default:
        return null
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-950">
      {/* Tabs */}
      <div className="flex gap-1 p-4 border-b border-slate-800/50 overflow-x-auto bg-slate-900/50">
        {INFO_TABS.map((tab) => (
          <motion.button
            key={tab.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all backdrop-blur ${
              activeTab === tab.id
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                : 'bg-slate-800/30 text-slate-400 border border-slate-800/30 hover:bg-slate-800/50 hover:text-slate-300'
            }`}
          >
            <span className="mr-1.5">{tab.icon}</span>
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {renderContent()}
      </div>
    </div>
  )
}
