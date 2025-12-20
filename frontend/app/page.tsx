'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { askQuestion } from '@/lib/api-client'
import { Message, RetrievalMethod } from '@/types/api'
import QuestionInput from '@/components/QuestionInput'
import ChatMessage from '@/components/ChatMessage'
import LoadingIndicator from '@/components/LoadingIndicator'
import ScrollToBottom from '@/components/ScrollToBottom'
import RetrievalMethodSelector from '@/components/RetrievalMethodSelector'
import RetrievalInfoModal from '@/components/RetrievalInfoModal'
import ProjectInfoPanel from '@/components/ProjectInfoPanel'

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [chatStarted, setChatStarted] = useState(false)
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>('dense')
  const [showInfoModal, setShowInfoModal] = useState(false)
  const [showProjectInfo, setShowProjectInfo] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // Clear chat function
  const handleClearChat = () => {
    setMessages([])
    setChatStarted(false)
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = 0
    }
  }

  // Check if user needs scroll button
  useEffect(() => {
    const container = chatContainerRef.current
    if (!container) return

    const checkScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      setShowScrollButton(!isNearBottom && messages.length > 0)
    }

    container.addEventListener('scroll', checkScroll)
    checkScroll()

    return () => container.removeEventListener('scroll', checkScroll)
  }, [messages])

  const handleSubmit = async (question: string) => {
    if (!question.trim() || isLoading) return

    // Start chat mode on first message
    if (!chatStarted) {
      setChatStarted(true)
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await askQuestion(question, retrievalMethod)
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        safety: response.safety,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      {/* Project Info Modal */}
      <AnimatePresence>
        {showProjectInfo && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-md z-50 flex items-center justify-center p-4"
            onClick={() => setShowProjectInfo(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-2xl max-h-[80vh] rounded-2xl shadow-2xl overflow-hidden border border-slate-800"
            >
              <ProjectInfoPanel />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="border-b border-slate-800/50 bg-slate-950/95 backdrop-blur-xl"
      >
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 flex items-center justify-center">
                <span className="text-2xl">🏥</span>
              </div>
              <div>
                <h1 className="text-2xl font-semibold text-slate-50 tracking-tight">MediQuery</h1>
                <p className="text-sm text-slate-400 font-light">Evidence-based medical information</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Clear Chat Button - Only show when chat started */}
              <AnimatePresence>
                {chatStarted && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleClearChat}
                    className="px-3.5 py-2.5 rounded-lg bg-slate-900/50 border border-slate-800 text-slate-400 hover:text-red-400 hover:border-red-500/30 hover:bg-red-500/5 transition-all text-sm font-medium flex items-center gap-2 backdrop-blur group"
                    title="Clear chat history"
                  >
                    <svg className="w-4 h-4 group-hover:rotate-12 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </motion.button>
                )}
              </AnimatePresence>
              
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: 'rgba(15, 23, 42, 0.8)' }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowProjectInfo(!showProjectInfo)}
                className="px-4 py-2.5 rounded-lg bg-slate-900/50 border border-slate-800 text-slate-300 hover:text-slate-100 transition-all text-sm font-medium flex items-center gap-2 backdrop-blur"
                title="About this project"
              >
                <span>ℹ️</span>
                <span>About</span>
              </motion.button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Retrieval Info Modal */}
      <RetrievalInfoModal
        isOpen={showInfoModal}
        onClose={() => setShowInfoModal(false)}
      />

      {/* Chat Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900"
      >
        <AnimatePresence mode="wait">
          {!chatStarted ? (
            // Welcome Screen
            <motion.div
              key="welcome"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="h-full flex flex-col items-center justify-center px-4 py-8"
            >
              {/* Logo */}
              <motion.div
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1, type: "spring", stiffness: 100 }}
                className="w-24 h-24 rounded-3xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/30 flex items-center justify-center mb-8 backdrop-blur"
              >
                <span className="text-6xl">🏥</span>
              </motion.div>

              {/* Title and Subtitle */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-center mb-10"
              >
                <h2 className="text-4xl font-semibold text-slate-50 mb-3 tracking-tight">
                  Medical Information Assistant
                </h2>
                <p className="text-slate-400 text-lg max-w-xl leading-relaxed">
                  Get evidence-based answers to your medical questions. 
                  <span className="block text-sm mt-2 text-slate-500">Powered by trusted medical sources and advanced AI</span>
                </p>
              </motion.div>

              {/* Example Questions */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="w-full max-w-3xl"
              >
                <p className="text-xs uppercase tracking-widest text-slate-500 mb-4 text-center">Try asking</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    { icon: '🩺', text: 'What are the symptoms of diabetes?' },
                    { icon: '🚭', text: 'How can lung cancer be prevented?' },
                    { icon: '❤️', text: 'What causes high blood pressure?' },
                    { icon: '📋', text: 'Tell me about heart disease risk factors' },
                  ].map((item, i) => (
                    <motion.button
                      key={i}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.35 + i * 0.08 }}
                      whileHover={{ scale: 1.02, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSubmit(item.text)}
                      className="p-4 bg-slate-900/40 border border-slate-800/60 rounded-xl text-left text-sm text-slate-300 hover:text-slate-100 hover:border-emerald-500/40 hover:bg-slate-900/60 transition-all group backdrop-blur"
                    >
                      <span className="text-lg mb-2 block group-hover:scale-110 transition-transform">{item.icon}</span>
                      <span className="font-light">{item.text}</span>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          ) : (
            // Chat Mode
            <motion.div
              key="chat"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="max-w-4xl mx-auto px-4 py-8"
            >
              <div className="space-y-6">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isLoading && <LoadingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Scroll to Bottom Button */}
      {showScrollButton && <ScrollToBottom onClick={scrollToBottom} />}

      {/* Input Area */}
      <motion.div
        initial={!chatStarted ? { y: 0 } : { y: 100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", damping: 20 }}
        className="border-t border-slate-800/50 bg-gradient-to-t from-slate-950 via-slate-950 to-slate-900 backdrop-blur-xl"
      >
        <div className="max-w-4xl mx-auto px-4 py-6">
          <QuestionInput 
            onSubmit={handleSubmit} 
            disabled={isLoading}
            retrievalMethod={retrievalMethod}
            onRetrievalMethodChange={setRetrievalMethod}
            onInfoClick={() => setShowInfoModal(true)}
          />
          <p className="text-xs text-center text-slate-500 mt-4 font-light">
            📋 Educational purposes only • Always consult healthcare professionals
          </p>
        </div>
      </motion.div>
    </div>
  )
}
