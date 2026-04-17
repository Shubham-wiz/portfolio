import { useMemo, useRef, useState, useEffect, useLayoutEffect } from 'react';
import { motion } from 'framer-motion';
import { runAttention } from '../../lib/mini-transformer';

const HEAD_COLORS = ['#c6ff3d', '#a78bfa', '#5b6cff', '#ff4d1f', '#22d3ee', '#f472b6'];

const TransformerDemo = () => {
  const [text, setText] = useState(
    'the quick brown fox jumps over the lazy dog in berlin'
  );
  const [nHeads, setNHeads] = useState(4);
  const [activeHead, setActiveHead] = useState(0);
  const [hoverToken, setHoverToken] = useState<number | null>(null);

  const { tokens, heads } = useMemo(() => runAttention(text, nHeads), [text, nHeads]);

  // keep active head valid when nHeads changes
  useEffect(() => {
    if (activeHead >= nHeads) setActiveHead(0);
  }, [nHeads, activeHead]);

  const attn = heads[activeHead];

  return (
    <div className="space-y-8">
      {/* Input */}
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
          / Input sequence
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={2}
          maxLength={160}
          className="w-full bg-ink border border-bone/15 rounded-2xl p-4 font-mono text-base text-bone focus:outline-none focus:border-acid transition-colors resize-none"
          placeholder="type a sentence…"
        />
        <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-bone/40">
          {tokens.length} tokens · max 16
        </div>
      </div>

      {/* Head controls */}
      <div className="flex flex-wrap items-center gap-5">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            Heads · {nHeads}
          </div>
          <input
            type="range"
            min={1}
            max={6}
            value={nHeads}
            onChange={(e) => setNHeads(parseInt(e.target.value))}
            className="w-48 accent-acid"
          />
        </div>

        <div className="flex-1 min-w-[240px]">
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            / Select head
          </div>
          <div className="flex gap-2 flex-wrap">
            {heads.map((_, i) => (
              <button
                key={i}
                data-cursor={`head ${i + 1}`}
                onClick={() => setActiveHead(i)}
                className={`px-3 py-1.5 rounded-full font-mono text-xs uppercase tracking-widest border transition ${
                  activeHead === i
                    ? 'bg-acid text-ink border-acid'
                    : 'border-bone/20 text-bone/70 hover:border-acid hover:text-acid'
                }`}
                style={activeHead === i ? {} : { color: HEAD_COLORS[i % HEAD_COLORS.length] }}
              >
                Head {i + 1}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Attention streams visualization */}
      <AttentionStreams
        tokens={tokens}
        attention={attn}
        hoverToken={hoverToken}
        setHoverToken={setHoverToken}
        color={HEAD_COLORS[activeHead % HEAD_COLORS.length]}
      />

      {/* Heatmap + head thumbnails */}
      <div className="grid lg:grid-cols-[1fr_260px] gap-6">
        <AttentionHeatmap tokens={tokens} attention={attn} hoverToken={hoverToken} />

        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-3">
            / All heads
          </div>
          <div className="grid grid-cols-2 gap-2">
            {heads.map((h, i) => (
              <button
                key={i}
                onClick={() => setActiveHead(i)}
                data-cursor={`open head ${i + 1}`}
                className={`relative rounded-lg overflow-hidden border transition ${
                  i === activeHead ? 'border-acid' : 'border-bone/15 hover:border-bone/40'
                }`}
                style={{ aspectRatio: '1' }}
              >
                <MiniHeatmap attention={h} color={HEAD_COLORS[i % HEAD_COLORS.length]} />
                <div className="absolute bottom-1 right-1 font-mono text-[9px] uppercase tracking-widest text-bone/80 bg-ink/60 px-1 rounded">
                  h{i + 1}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      <p className="text-bone/50 text-sm leading-relaxed">
        Real scaled dot-product attention · deterministic Q/K/V projections per head · sinusoidal positional encoding. Hover a token to trace where it "looks" — brighter streams mean stronger attention. The same mechanism underpins GPT, BERT and Claude.
      </p>
    </div>
  );
};

// -------------- Attention streams (bezier curves from hovered token) --------------

const AttentionStreams = ({
  tokens,
  attention,
  hoverToken,
  setHoverToken,
  color,
}: {
  tokens: string[];
  attention: number[][];
  hoverToken: number | null;
  setHoverToken: (i: number | null) => void;
  color: string;
}) => {
  const topRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [paths, setPaths] = useState<{ d: string; w: number; o: number }[]>([]);

  const recompute = () => {
    if (!topRef.current || !bottomRef.current || hoverToken == null || !attention[hoverToken]) {
      setPaths([]);
      return;
    }
    const container = topRef.current.parentElement?.getBoundingClientRect();
    if (!container) return;

    const srcEl = topRef.current.children[hoverToken] as HTMLElement | undefined;
    if (!srcEl) return;
    const srcR = srcEl.getBoundingClientRect();
    const sx = srcR.left + srcR.width / 2 - container.left;
    const sy = srcR.bottom - container.top;

    const row = attention[hoverToken];
    const newPaths: { d: string; w: number; o: number }[] = [];
    for (let j = 0; j < tokens.length; j++) {
      const dstEl = bottomRef.current.children[j] as HTMLElement | undefined;
      if (!dstEl) continue;
      const dstR = dstEl.getBoundingClientRect();
      const dx = dstR.left + dstR.width / 2 - container.left;
      const dy = dstR.top - container.top;
      const midY = (sy + dy) / 2;
      const d = `M ${sx} ${sy} C ${sx} ${midY + 20}, ${dx} ${midY - 20}, ${dx} ${dy}`;
      newPaths.push({ d, w: 0.6 + row[j] * 10, o: Math.max(0.06, row[j]) });
    }
    setPaths(newPaths);
  };

  useLayoutEffect(() => {
    recompute();
    const onResize = () => recompute();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hoverToken, attention, tokens]);

  return (
    <div className="relative bg-bone/[0.02] border border-bone/10 rounded-2xl p-6 overflow-hidden">
      <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-4">
        / Hover a token to see where it attends
      </div>

      <div ref={topRef} className="flex flex-wrap gap-2 mb-20 relative z-10">
        {tokens.map((t, i) => (
          <button
            key={`t-${i}`}
            data-cursor={`query ${t}`}
            onMouseEnter={() => setHoverToken(i)}
            onMouseLeave={() => setHoverToken(null)}
            className={`px-3 py-1.5 rounded-lg font-mono text-sm transition-all ${
              hoverToken === i
                ? 'bg-acid text-ink shadow-acid'
                : 'bg-bone/[0.06] text-bone hover:bg-bone/[0.12]'
            }`}
          >
            <span className="mr-1.5 text-[10px] opacity-60">{i}</span>
            {t}
          </button>
        ))}
      </div>

      <svg className="absolute inset-0 pointer-events-none" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        {paths.map((p, i) => (
          <path
            key={i}
            d={p.d}
            fill="none"
            stroke={color}
            strokeWidth={p.w}
            strokeOpacity={p.o}
            strokeLinecap="round"
          />
        ))}
      </svg>

      <div ref={bottomRef} className="flex flex-wrap gap-2 relative z-10">
        {tokens.map((t, j) => {
          const weight = hoverToken != null && attention[hoverToken] ? attention[hoverToken][j] : 0;
          return (
            <div
              key={`b-${j}`}
              className="px-3 py-1.5 rounded-lg font-mono text-sm text-bone relative overflow-hidden"
              style={{
                background: `linear-gradient(90deg, ${color}${Math.round(weight * 255)
                  .toString(16)
                  .padStart(2, '0')} 0%, ${color}10 ${weight * 100}%, rgba(255,255,255,0.04) ${weight * 100}%)`,
                border: `1px solid ${color}${Math.round(weight * 200)
                  .toString(16)
                  .padStart(2, '0')}`,
              }}
            >
              <span className="mr-1.5 text-[10px] opacity-60">{j}</span>
              {t}
              {hoverToken != null && (
                <span className="ml-2 font-mono text-[10px] opacity-60">
                  {(weight * 100).toFixed(0)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// -------------- Heatmap (big, for active head) --------------

const AttentionHeatmap = ({
  tokens,
  attention,
  hoverToken,
}: {
  tokens: string[];
  attention: number[][];
  hoverToken: number | null;
}) => {
  if (!attention || attention.length === 0) return null;
  const n = tokens.length;
  const CELL = Math.min(38, 520 / Math.max(8, n));
  const LABEL = 80;
  return (
    <div className="bg-bone/[0.02] border border-bone/10 rounded-2xl p-4 overflow-auto">
      <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-3 px-1">
        / Attention matrix · query × key
      </div>
      <svg width={LABEL + n * CELL + 10} height={LABEL + n * CELL + 10}>
        {tokens.map((t, i) => (
          <text
            key={`k-${i}`}
            x={LABEL + i * CELL + CELL / 2}
            y={LABEL - 6}
            textAnchor="end"
            transform={`rotate(-60 ${LABEL + i * CELL + CELL / 2} ${LABEL - 6})`}
            className="fill-bone/70 font-mono"
            fontSize={10}
          >
            {t}
          </text>
        ))}
        {tokens.map((t, i) => (
          <text
            key={`q-${i}`}
            x={LABEL - 6}
            y={LABEL + i * CELL + CELL / 2 + 3}
            textAnchor="end"
            className={`font-mono ${hoverToken === i ? 'fill-acid' : 'fill-bone/70'}`}
            fontSize={10}
          >
            {t}
          </text>
        ))}
        {attention.map((row, i) =>
          row.map((v, j) => {
            const a = Math.min(1, Math.max(0, v));
            return (
              <rect
                key={`c-${i}-${j}`}
                x={LABEL + j * CELL + 1}
                y={LABEL + i * CELL + 1}
                width={CELL - 2}
                height={CELL - 2}
                fill={`rgba(198, 255, 61, ${a.toFixed(3)})`}
                stroke={hoverToken === i ? 'rgba(198,255,61,0.6)' : 'rgba(244,241,234,0.05)'}
                strokeWidth={hoverToken === i ? 1.5 : 0.5}
                rx={2}
              />
            );
          })
        )}
      </svg>
    </div>
  );
};

// -------------- Mini heatmap for head thumbnails --------------

const MiniHeatmap = ({
  attention,
  color,
}: {
  attention: number[][];
  color: string;
}) => {
  const n = attention.length;
  if (n === 0) return <div className="w-full h-full bg-ink" />;
  return (
    <svg viewBox={`0 0 ${n} ${n}`} className="w-full h-full bg-ink" preserveAspectRatio="none">
      {attention.map((row, i) =>
        row.map((v, j) => {
          const a = Math.min(1, Math.max(0, v));
          const rgb = hexToRgb(color);
          return (
            <rect
              key={`${i}-${j}`}
              x={j}
              y={i}
              width={1.02}
              height={1.02}
              fill={`rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${a.toFixed(3)})`}
            />
          );
        })
      )}
    </svg>
  );
};

function hexToRgb(hex: string) {
  const h = hex.replace('#', '');
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

// satisfy unused-import lint
void motion;

export default TransformerDemo;
