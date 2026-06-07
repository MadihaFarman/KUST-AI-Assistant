'use client'
import { useState, useRef } from 'react'
import { Mic, MicOff, Loader2 } from 'lucide-react'

interface Props { onTranscript: (t: string) => void; disabled?: boolean }

export function VoiceButton({ onTranscript, disabled }: Props) {
    const [state, setState] = useState<'idle' | 'recording' | 'processing'>('idle')
    const recorderRef = useRef<MediaRecorder | null>(null)
    const chunksRef = useRef<Blob[]>([])

    const start = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
        chunksRef.current = []
        recorder.ondataavailable = e => chunksRef.current.push(e.data)
        recorder.onstop = async () => {
            setState('processing')
            const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
            const form = new FormData()
            form.append('audio', blob, 'audio.webm')
            try {
                const r = await fetch('http://localhost:8000/transcribe/', { method: 'POST', body: form })
                const d = await r.json()
                if (d.transcript) onTranscript(d.transcript)
            } finally {
                setState('idle')
                stream.getTracks().forEach(t => t.stop())
            }
        }
        recorder.start()
        recorderRef.current = recorder
        setState('recording')
    }

    const stop = () => recorderRef.current?.stop()

    return (
        <button
            onClick={state === 'idle' ? start : stop}
            disabled={disabled || state === 'processing'}
            style={{
                width: '34px', height: '34px', borderRadius: '8px', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: state === 'recording' ? '#FEE2E2' : 'var(--accent-soft)',
                color: state === 'recording' ? '#DC2626' : 'var(--text-2)',
                border: `1px solid ${state === 'recording' ? '#FECACA' : 'var(--border)'}`,
                cursor: 'pointer', transition: 'all 0.15s',
            }}
        >
            {state === 'processing'
                ? <Loader2 size={14} className="spin" />
                : state === 'recording'
                    ? <MicOff size={14} />
                    : <Mic size={14} />
            }
        </button>
    )
}