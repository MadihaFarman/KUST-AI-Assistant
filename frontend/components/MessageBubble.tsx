'use client'
import { CitationBadge } from './CitationBadge'

interface Citation { source: string; page: number; score: number }
interface Message {
    role: 'user' | 'assistant'
    content: string
    citations?: Citation[]
    language?: string
    isStreaming?: boolean
}

export function MessageBubble({ message, animDelay = 0 }: { message: Message; animDelay?: number }) {
    const isUser = message.role === 'user'
    const isUrdu = message.language === 'ur'
    const isEmpty = !message.content && message.isStreaming

    if (isUser) {
        return (
            <div
                className="fade-up"
                style={{
                    animationDelay: `${animDelay}ms`,
                    display: 'flex',
                    justifyContent: 'flex-end',
                    padding: '8px 0',
                }}
            >
                <div style={{
                    maxWidth: '70%',
                    background: 'var(--text-1)',
                    color: 'white',
                    padding: '10px 14px',
                    borderRadius: '16px 16px 4px 16px',
                    fontSize: '14px',
                    lineHeight: '1.6',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                }}>
                    {message.content}
                </div>
            </div>
        )
    }

    return (
        <div
            className="fade-up"
            style={{
                animationDelay: `${animDelay}ms`,
                display: 'flex',
                gap: '10px',
                padding: '8px 0',
                alignItems: 'flex-start',
            }}
        >
            {/* K avatar */}
            <div style={{
                flexShrink: 0,
                width: '28px', height: '28px', borderRadius: '8px',
                background: 'var(--surface)', color: 'var(--text-1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '12px', fontWeight: 600,
                border: '1.5px solid var(--border)',
                marginTop: '2px',
            }}>
                K
            </div>

            {/* Message body */}
            <div style={{ flex: 1, minWidth: 0 }}>
                {isEmpty ? (
                    <div style={{ display: 'flex', gap: '4px', alignItems: 'center', height: '28px' }}>
                        {[0, 1, 2].map(i => (
                            <div key={i} style={{
                                width: '6px', height: '6px', borderRadius: '50%',
                                background: 'var(--text-3)',
                                animation: `blink 1.2s ${i * 0.2}s ease infinite`,
                            }} />
                        ))}
                    </div>
                ) : (
                    <>
                        <div
                            dir={isUrdu ? 'rtl' : 'ltr'}
                            style={{
                                color: 'var(--text-1)',
                                fontSize: '14px',
                                lineHeight: isUrdu ? '2.1' : '1.75',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                            }}
                        >
                            {message.content}
                            {message.isStreaming && <span className="cursor-blink" />}
                        </div>

                        {message.citations && message.citations.length > 0 && (
                            <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap' }}>
                                {message.citations.map((c, i) => (
                                    <CitationBadge key={i} citation={c} index={i} />
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}