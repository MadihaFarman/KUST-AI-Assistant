interface Citation {
    source: string
    page: number
    score: number
}

export function CitationBadge({ citation, index }: { citation: Citation; index: number }) {
    const name = citation.source
        .replace(/_/g, ' ')
        .replace(/\d{6}_\d{6}/, '')
        .replace(/Final Revised /i, '')
        .trim()

    return (
        <span
            title={`${name} — Page ${citation.page}`}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                fontFamily: 'Geist Mono, monospace',
                background: 'var(--blue-soft)',
                color: 'var(--blue)',
                border: '1px solid #BFDBFE',
                marginRight: '4px',
                marginBottom: '4px',
                cursor: 'default',
            }}
        >
            [{index + 1}] p.{citation.page}
        </span>
    )
}