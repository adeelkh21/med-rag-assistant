'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="h-screen flex items-center justify-center bg-slate-950">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-6xl font-bold text-slate-100 mb-4">404</h1>
        <p className="text-xl text-slate-400 mb-8">Page not found</p>
        <Link
          href="/"
          className="px-6 py-3 bg-emerald-600/80 hover:bg-emerald-600 text-white rounded-lg transition-all"
        >
          Go back home
        </Link>
      </motion.div>
    </div>
  )
}
