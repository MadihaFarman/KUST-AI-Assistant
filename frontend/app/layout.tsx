import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'KUST AI Assistant',
  description: 'Bilingual AI assistant for Kohat University of Science & Technology',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ height: '100vh', overflow: 'hidden' }}>{children}</body>
    </html>
  )
}