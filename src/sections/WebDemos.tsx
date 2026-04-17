import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { Brain, Layers, Wind, MessageSquare, Network, Sparkles } from 'lucide-react';
import TextReveal from '../components/TextReveal';
import NeuralNetPlayground from '../components/demos/NeuralNetPlayground';
import KMeansDemo from '../components/demos/KMeansDemo';
import FlowField from '../components/demos/FlowField';
import SentimentDemo from '../components/demos/SentimentDemo';
import TransformerDemo from '../components/demos/TransformerDemo';
import LLMDemo from '../components/demos/LLMDemo';

type DemoId = 'transformer' | 'llm' | 'nn' | 'kmeans' | 'flow' | 'sentiment';

const demos: {
  id: DemoId;
  title: string;
  tagline: string;
  tag: string;
  icon: any;
  color: string;
}[] = [
  {
    id: 'transformer',
    title: 'Transformer · Attention',
    tagline: 'Real multi-head self-attention over your sentence — hover a token, trace the streams.',
    tag: 'Transformer',
    icon: Network,
    color: '#22d3ee',
  },
  {
    id: 'llm',
    title: 'LLM · Next Token',
    tagline: 'Word-level LM with temperature, top-K and top-P. Watch the distribution breathe.',
    tag: 'Language Model',
    icon: Sparkles,
    color: '#f472b6',
  },
  {
    id: 'nn',
    title: 'Neural Net · Playground',
    tagline: 'Hand-coded MLP with live backprop & decision boundary.',
    tag: 'Machine Learning',
    icon: Brain,
    color: '#c6ff3d',
  },
  {
    id: 'kmeans',
    title: 'K-Means · Interactive',
    tagline: 'Watch centroids crawl toward consensus — step by step.',
    tag: 'Unsupervised',
    icon: Layers,
    color: '#a78bfa',
  },
  {
    id: 'flow',
    title: 'Flow Field · Generative',
    tagline: 'Particles surfing a Perlin-noise vector field. Your mouse steers the storm.',
    tag: 'Creative Coding',
    icon: Wind,
    color: '#5b6cff',
  },
  {
    id: 'sentiment',
    title: 'Sentiment · Live',
    tagline: 'Negation-aware lexicon classifier. Type, see it score you in real time.',
    tag: 'NLP',
    icon: MessageSquare,
    color: '#ff4d1f',
  },
];

const WebDemos = () => {
  const [active, setActive] = useState<DemoId>('transformer');

  return (
    <section id="lab" className="relative py-32 md:py-40 bg-ink overflow-hidden">
      <div className="absolute inset-0 grid-lines opacity-30 pointer-events-none" />
      <div className="absolute top-24 -left-24 w-[500px] h-[500px] rounded-full bg-acid/10 blur-[160px] pointer-events-none" />
      <div className="absolute bottom-0 -right-24 w-[500px] h-[500px] rounded-full bg-violet/10 blur-[160px] pointer-events-none" />

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-16">
        <div className="flex items-center gap-4 mb-10 font-mono text-xs uppercase tracking-[0.3em] text-bone/50">
          <span className="w-2 h-2 rotate-45 bg-acid" />
          / 04 — The Lab
          <span className="flex-1 h-px bg-bone/10" />
          <span className="text-bone/30">Live, in your browser</span>
        </div>

        <div className="flex flex-col lg:flex-row justify-between gap-10 items-end mb-14">
          <TextReveal
            as="h2"
            text="Web demos. ML in your tab."
            className="font-display uppercase text-bone leading-[0.88] tracking-crushed text-[clamp(2.5rem,8vw,8rem)]"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-lg text-bone/65 text-lg"
          >
            No demo videos, no YouTube embeds. These are live JavaScript playgrounds — models, clustering, generative fields and NLP — all running on your machine right now.
          </motion.p>
        </div>

        {/* Tab bar */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-10">
          {demos.map((d) => {
            const isActive = active === d.id;
            return (
              <motion.button
                key={d.id}
                onClick={() => setActive(d.id)}
                whileHover={{ y: -4 }}
                data-cursor={d.tag.toLowerCase()}
                className={`group relative text-left p-5 rounded-2xl border transition-colors overflow-hidden ${
                  isActive
                    ? 'bg-bone/[0.06] border-acid'
                    : 'bg-bone/[0.02] border-bone/10 hover:border-bone/30'
                }`}
              >
                <div
                  className="absolute -top-8 -right-8 w-32 h-32 rounded-full blur-2xl opacity-40"
                  style={{ backgroundColor: d.color }}
                />
                <div className="relative flex items-start gap-3">
                  <div
                    className="w-10 h-10 rounded-xl grid place-items-center shrink-0"
                    style={{
                      background: isActive ? d.color : 'rgba(244,241,234,0.06)',
                      color: isActive ? '#050505' : d.color,
                    }}
                  >
                    <d.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40 mb-1">
                      {d.tag}
                    </div>
                    <div className="font-display uppercase text-bone text-xl leading-tight tracking-tight mb-1">
                      {d.title}
                    </div>
                    <div className="text-bone/55 text-xs leading-relaxed">
                      {d.tagline}
                    </div>
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Demo area */}
        <div className="relative bg-bone/[0.02] border border-bone/10 rounded-3xl p-6 md:p-10 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            >
              {active === 'transformer' && <TransformerDemo />}
              {active === 'llm' && <LLMDemo />}
              {active === 'nn' && <NeuralNetPlayground />}
              {active === 'kmeans' && <KMeansDemo />}
              {active === 'flow' && <FlowField />}
              {active === 'sentiment' && <SentimentDemo />}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="mt-6 font-mono text-[11px] uppercase tracking-[0.3em] text-bone/40 flex items-center gap-3">
          <span className="w-1.5 h-1.5 bg-acid rounded-full animate-pulse" />
          Running entirely on {typeof navigator !== 'undefined' ? navigator.platform : 'your device'} · no server calls
        </div>
      </div>
    </section>
  );
};

export default WebDemos;
