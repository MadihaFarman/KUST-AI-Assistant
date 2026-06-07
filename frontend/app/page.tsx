'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Plus, Trash2, MessageSquare, GraduationCap, Menu, X } from 'lucide-react'
import { MessageBubble } from '@/components/MessageBubble'
import { VoiceButton } from '@/components/VoiceButton'

interface Citation { source: string; page: number; score: number }
interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  language?: string
  isStreaming?: boolean
}
interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: number
}

const SUGGESTED = [
  'What are the CGPA probation rules?',
  'Admission ki last date kya hai?',
  'What are the semester fee refund rules?',
  'Vice Chancellor ke powers kya hain?',
]

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2)
}

function getTitle(messages: Message[]): string {
  const first = messages.find(m => m.role === 'user')
  if (!first) return 'New conversation'
  return first.content.slice(0, 42) + (first.content.length > 42 ? '…' : '')
}

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Load from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('kust_conversations')
    if (saved) {
      const parsed: Conversation[] = JSON.parse(saved)
      setConversations(parsed)
      if (parsed.length > 0) setActiveId(parsed[0].id)
    }
  }, [])

  // Save to localStorage
  useEffect(() => {
    if (conversations.length > 0)
      localStorage.setItem('kust_conversations', JSON.stringify(conversations))
  }, [conversations])

  const active = conversations.find(c => c.id === activeId)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active?.messages])

  const newConversation = useCallback(() => {
    const conv: Conversation = {
      id: generateId(),
      title: 'New conversation',
      messages: [],
      createdAt: Date.now(),
    }
    setConversations(prev => [conv, ...prev])
    setActiveId(conv.id)
    setInput('')
    setTimeout(() => inputRef.current?.focus(), 50)
  }, [])

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id)
      localStorage.setItem('kust_conversations', JSON.stringify(next))
      return next
    })
    if (activeId === id) {
      setActiveId(prev => {
        const rest = conversations.filter(c => c.id !== id)
        return rest.length > 0 ? rest[0].id : null
      })
    }
  }, [activeId, conversations])

  const sendMessage = async (text?: string) => {
    const query = (text || input).trim()
    if (!query || loading) return

    setInput('')
    setLoading(true)

    let convId = activeId
    if (!convId) {
      const conv: Conversation = {
        id: generateId(), title: query.slice(0, 42),
        messages: [], createdAt: Date.now(),
      }
      setConversations(prev => [conv, ...prev])
      convId = conv.id
      setActiveId(convId)
    }

    const userMsg: Message = { role: 'user', content: query }
    const assistantMsg: Message = { role: 'assistant', content: '', isStreaming: true }

    setConversations(prev => prev.map(c =>
      c.id === convId
        ? {
          ...c, messages: [...c.messages, userMsg, assistantMsg],
          title: c.messages.length === 0 ? query.slice(0, 42) : c.title
        }
        : c
    ))

    try {
      const history = (active?.messages ?? []).map(m => ({
        role: m.role, content: m.content,
      }))

      const res = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, history }),
      })
      const data = await res.json()

      setConversations(prev => prev.map(c => {
        if (c.id !== convId) return c
        const msgs = [...c.messages.slice(0, -1), {
          role: 'assistant' as const,
          content: data.answer,
          citations: data.citations,
          language: data.detected_language,
          isStreaming: false,
        }]
        return { ...c, messages: msgs }
      }))
    } catch {
      setConversations(prev => prev.map(c =>
        c.id !== convId ? c : {
          ...c, messages: [...c.messages.slice(0, -1), {
            role: 'assistant' as const,
            content: 'Could not reach the backend. Make sure it is running on port 8000.',
          }],
        }
      ))
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: sidebarOpen ? 'var(--sidebar-width)' : '0',
        minWidth: sidebarOpen ? 'var(--sidebar-width)' : '0',
        overflow: 'hidden',
        transition: 'width 0.2s ease, min-width 0.2s ease',
        background: 'var(--surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
      }}>
        {/* Sidebar header */}
        <div style={{
          padding: '16px', borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '26px', height: '26px', borderRadius: '6px',
              background: 'var(--text-1)', display: 'flex',
              alignItems: 'center', justifyContent: 'center',
            }}>
              <GraduationCap size={14} color="white" />
            </div>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-1)' }}>
              KUST AI
            </span>
          </div>
          <button
            onClick={newConversation}
            style={{
              width: '28px', height: '28px', borderRadius: '6px',
              background: 'transparent', border: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', color: 'var(--text-2)',
            }}
          >
            <Plus size={14} />
          </button>
        </div>

        {/* Conversation list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          {conversations.length === 0 ? (
            <p style={{ padding: '16px 8px', fontSize: '12px', color: 'var(--text-3)', textAlign: 'center' }}>
              No conversations yet
            </p>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                onClick={() => setActiveId(conv.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '8px 10px', borderRadius: '8px', cursor: 'pointer',
                  background: conv.id === activeId ? 'var(--surface-hover)' : 'transparent',
                  marginBottom: '2px', group: 'conv',
                }}
                onMouseEnter={e => {
                  if (conv.id !== activeId)
                    (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)'
                }}
                onMouseLeave={e => {
                  if (conv.id !== activeId)
                    (e.currentTarget as HTMLElement).style.background = 'transparent'
                }}
              >
                <MessageSquare size={13} style={{ flexShrink: 0, color: 'var(--text-3)' }} />
                <span style={{
                  flex: 1, fontSize: '13px', color: 'var(--text-1)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {conv.title}
                </span>
                <button
                  onClick={e => { e.stopPropagation(); deleteConversation(conv.id) }}
                  style={{
                    flexShrink: 0, background: 'none', border: 'none',
                    cursor: 'pointer', color: 'var(--text-3)', padding: '2px',
                    borderRadius: '4px', display: 'flex', alignItems: 'center',
                    opacity: 0,
                  }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.opacity = '1'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.opacity = '0'}
                  onFocus={e => (e.currentTarget as HTMLElement).style.opacity = '1'}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Sidebar footer */}
        <div style={{
          padding: '12px 16px', borderTop: '1px solid var(--border)',
          fontSize: '11px', color: 'var(--text-3)',
        }}>
          Powered by GPT-4o-mini · Pinecone
        </div>
      </aside>

      {/* ── Main ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

        {/* Header */}
        <header style={{
          padding: '12px 20px', borderBottom: '1px solid var(--border)',
          background: 'var(--surface)', display: 'flex',
          alignItems: 'center', gap: '12px', flexShrink: 0,
        }}>
          <button
            onClick={() => setSidebarOpen(s => !s)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-2)', display: 'flex', padding: '4px',
            }}
          >
            {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
          </button>

          <div style={{ flex: 1 }}>
            <h1 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-1)' }}>
              {active ? active.title : 'KUST AI Assistant'}
            </h1>
            <p style={{ fontSize: '11px', color: 'var(--text-3)', marginTop: '1px' }}>
              Kohat University of Science & Technology
            </p>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '4px 10px', borderRadius: '20px', fontSize: '11px',
            background: 'var(--green-soft)', color: 'var(--green)',
            border: '1px solid #BBF7D0',
          }}>
            <span style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: 'var(--green)',
            }} />
            Live
          </div>
        </header>

        {/* Messages */}
        <main style={{ flex: 1, overflowY: 'auto' }}>
          <div style={{ maxWidth: '680px', margin: '0 auto', padding: '0 24px' }}>

            {(!active || active.messages.length === 0) ? (
              <div style={{
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                minHeight: 'calc(100vh - 200px)', textAlign: 'center',
              }}>
                <div style={{
                  width: '48px', height: '48px', borderRadius: '14px',
                  background: 'var(--surface)', border: '1.5px solid var(--border)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  marginBottom: '16px',
                }}>
                  <GraduationCap size={22} style={{ color: 'var(--text-1)' }} />
                </div>
                <h2 style={{
                  fontSize: '20px', fontWeight: 500, color: 'var(--text-1)',
                  marginBottom: '8px',
                }}>
                  Ask about KUST
                </h2>
                <p style={{
                  fontSize: '14px', color: 'var(--text-2)',
                  maxWidth: '360px', marginBottom: '32px', lineHeight: '1.6',
                }}>
                  Get answers from official university documents. Ask in English, Urdu, or Roman Urdu.
                </p>
                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr',
                  gap: '8px', width: '100%', maxWidth: '480px',
                }}>
                  {SUGGESTED.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => sendMessage(q)}
                      style={{
                        textAlign: 'left', padding: '12px 14px',
                        borderRadius: '10px', fontSize: '13px',
                        background: 'var(--surface)', color: 'var(--text-2)',
                        border: '1px solid var(--border)', cursor: 'pointer',
                        lineHeight: '1.5', transition: 'border-color 0.15s',
                      }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.borderColor = 'var(--text-3)'}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ paddingTop: '8px', paddingBottom: '8px' }}>
                {active.messages.map((msg, i) => (
                  <MessageBubble key={i} message={msg} animDelay={0} />
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </main>

        {/* Input */}
        <footer style={{
          padding: '16px 24px 20px',
          borderTop: '1px solid var(--border)',
          background: 'var(--surface)',
          flexShrink: 0,
        }}>
          <div style={{ maxWidth: '680px', margin: '0 auto' }}>
            <div style={{
              display: 'flex', alignItems: 'flex-end', gap: '8px',
              background: 'var(--bg)', border: '1.5px solid var(--border)',
              borderRadius: '12px', padding: '10px 12px',
              transition: 'border-color 0.15s',
            }}
              onFocusCapture={e => (e.currentTarget as HTMLElement).style.borderColor = 'var(--text-3)'}
              onBlurCapture={e => (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'}
            >
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask in English, Urdu, or Roman Urdu..."
                rows={1}
                disabled={loading}
                style={{
                  flex: 1, background: 'transparent', border: 'none',
                  outline: 'none', resize: 'none', fontSize: '14px',
                  color: 'var(--text-1)', lineHeight: '1.6',
                  fontFamily: 'inherit', maxHeight: '120px',
                }}
                onInput={e => {
                  const t = e.target as HTMLTextAreaElement
                  t.style.height = 'auto'
                  t.style.height = Math.min(t.scrollHeight, 120) + 'px'
                }}
              />
              <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
                <VoiceButton onTranscript={t => setInput(t)} disabled={loading} />
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || loading}
                  style={{
                    width: '34px', height: '34px', borderRadius: '8px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: input.trim() && !loading ? 'var(--text-1)' : 'var(--accent-soft)',
                    color: input.trim() && !loading ? 'white' : 'var(--text-3)',
                    border: '1px solid var(--border)', cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <Send size={13} />
                </button>
              </div>
            </div>
            <p style={{
              textAlign: 'center', fontSize: '11px', color: 'var(--text-3)',
              marginTop: '8px',
            }}>
              Answers grounded in official KUST documents · May contain errors
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}