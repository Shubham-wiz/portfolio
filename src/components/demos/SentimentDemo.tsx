import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';

// A compact positive/negative lexicon — enough for an honest live demo.
const POS = new Set([
  'good', 'great', 'amazing', 'awesome', 'love', 'loved', 'loving', 'excellent',
  'fantastic', 'brilliant', 'beautiful', 'happy', 'delighted', 'perfect', 'best',
  'enjoy', 'enjoyed', 'cool', 'nice', 'sweet', 'wonderful', 'incredible', 'stellar',
  'win', 'fast', 'easy', 'clean', 'smooth', 'sharp', 'crisp', 'solid', 'smart',
  'helpful', 'fun', 'magical', 'inspiring', 'elegant', 'premium', 'lovely',
]);
const NEG = new Set([
  'bad', 'terrible', 'awful', 'hate', 'hated', 'worst', 'horrible', 'sad',
  'angry', 'broken', 'buggy', 'slow', 'painful', 'ugly', 'annoying', 'annoyed',
  'boring', 'lame', 'dull', 'messy', 'stupid', 'useless', 'pointless', 'meh',
  'poor', 'shit', 'trash', 'garbage', 'fail', 'failed', 'ruined', 'gross',
  'disappointing', 'disappointed', 'confusing', 'noise', 'clunky', 'cold',
]);
const NEGATORS = new Set(['not', 'no', "isn't", 'never', 'nothing', 'nope', 'neither']);
const INTENSIFIERS = new Set(['very', 'really', 'super', 'extremely', 'absolutely', 'so']);

type Token = { raw: string; word: string; score: number; label: 'pos' | 'neg' | 'neu' };

function tokenize(text: string): string[] {
  return text.split(/(\s+)/).filter(Boolean);
}

function scoreTokens(text: string): { tokens: Token[]; total: number } {
  const words = tokenize(text);
  const lowered = words.map((w) => w.toLowerCase().replace(/[^a-z']/g, ''));
  const tokens: Token[] = [];
  let total = 0;
  for (let i = 0; i < words.length; i++) {
    const raw = words[i];
    const w = lowered[i];
    if (!w) {
      tokens.push({ raw, word: w, score: 0, label: 'neu' });
      continue;
    }
    let s = 0;
    if (POS.has(w)) s = 1;
    else if (NEG.has(w)) s = -1;

    if (s !== 0) {
      // look back up to 2 tokens for negators / intensifiers
      for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
        const prev = lowered[j];
        if (!prev) continue;
        if (NEGATORS.has(prev)) s *= -1;
        else if (INTENSIFIERS.has(prev)) s *= 1.6;
      }
    }
    total += s;
    tokens.push({
      raw,
      word: w,
      score: s,
      label: s > 0 ? 'pos' : s < 0 ? 'neg' : 'neu',
    });
  }
  return { tokens, total };
}

const SentimentDemo = () => {
  const [text, setText] = useState(
    "I absolutely love how smooth this site feels, but the loading was a bit slow."
  );

  const { tokens, total } = useMemo(() => scoreTokens(text), [text]);

  const maxAbs = Math.max(3, tokens.length * 0.6);
  const norm = Math.max(-1, Math.min(1, total / maxAbs));
  const sentiment =
    norm > 0.15 ? 'Positive' : norm < -0.15 ? 'Negative' : 'Neutral';
  const color = norm > 0.15 ? '#c6ff3d' : norm < -0.15 ? '#ff4d1f' : '#a78bfa';

  return (
    <div className="grid lg:grid-cols-[1fr_360px] gap-8 items-start">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
          / Type something
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          className="w-full bg-ink border border-bone/15 rounded-2xl p-4 font-sans text-lg text-bone placeholder-bone/40 focus:outline-none focus:border-acid transition-colors resize-none"
          placeholder="Tell me how your morning is going…"
        />
        <div className="mt-4 p-4 rounded-2xl bg-bone/[0.03] border border-bone/10 min-h-[80px] text-xl leading-relaxed">
          {tokens.map((t, i) => (
            <span
              key={i}
              className="px-0.5 rounded"
              style={{
                color:
                  t.label === 'pos' ? '#c6ff3d' : t.label === 'neg' ? '#ff4d1f' : '#c9c6bf',
                textShadow:
                  t.label === 'pos'
                    ? '0 0 18px rgba(198,255,61,0.45)'
                    : t.label === 'neg'
                      ? '0 0 18px rgba(255,77,31,0.35)'
                      : 'none',
                fontWeight: t.label !== 'neu' ? 600 : 400,
              }}
            >
              {t.raw}
            </span>
          ))}
        </div>
      </div>

      <div className="space-y-6">
        <div className="rounded-2xl border border-bone/10 bg-bone/[0.03] p-6">
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            Sentiment score
          </div>
          <div className="flex items-baseline gap-3">
            <motion.div
              key={sentiment}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="font-display text-5xl tracking-tight"
              style={{ color }}
            >
              {sentiment}
            </motion.div>
            <div className="font-mono text-sm text-bone/50">
              {total.toFixed(1)}
            </div>
          </div>

          <div className="mt-5 h-2 rounded-full bg-bone/10 overflow-hidden relative">
            <div
              className="absolute top-0 bottom-0 left-1/2 w-px bg-bone/30"
            />
            <motion.div
              animate={{ width: `${Math.abs(norm) * 50}%` }}
              transition={{ type: 'spring', damping: 20, stiffness: 180 }}
              className="absolute top-0 bottom-0 rounded-full"
              style={{
                background: color,
                left: norm >= 0 ? '50%' : undefined,
                right: norm < 0 ? '50%' : undefined,
              }}
            />
          </div>
          <div className="mt-2 flex justify-between font-mono text-[10px] uppercase tracking-widest text-bone/40">
            <span>negative</span>
            <span>positive</span>
          </div>
        </div>

        <div className="rounded-2xl border border-bone/10 bg-bone/[0.03] p-6 space-y-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-1">
            Detected signals
          </div>
          <div className="flex flex-wrap gap-2">
            {tokens.filter((t) => t.label !== 'neu').length === 0 ? (
              <span className="text-bone/40 text-sm">no strong signal</span>
            ) : (
              tokens
                .filter((t) => t.label !== 'neu')
                .map((t, i) => (
                  <span
                    key={i}
                    className="chip"
                    style={{
                      borderColor: t.label === 'pos' ? 'rgba(198,255,61,0.5)' : 'rgba(255,77,31,0.5)',
                      color: t.label === 'pos' ? '#c6ff3d' : '#ff4d1f',
                    }}
                  >
                    {t.word}
                    <span className="opacity-60">{t.score > 0 ? '+' : ''}{t.score.toFixed(1)}</span>
                  </span>
                ))
            )}
          </div>
        </div>

        <p className="text-bone/50 text-sm leading-relaxed">
          A small lexicon-based classifier with negation & intensifier handling — runs entirely on the client. The same pattern bootstraps larger transformer-based sentiment stacks in production.
        </p>
      </div>
    </div>
  );
};

export default SentimentDemo;
