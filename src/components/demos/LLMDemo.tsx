import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Square, RotateCcw, Zap } from 'lucide-react';
import { NGramLM, Candidate } from '../../lib/ngram';

// A small AI-meta corpus. ~400 words of tech prose so generation stays on-theme.
const CORPUS = `
the transformer was a quiet revolution. it learned to attend to the past instead of marching through it. every token carried a query and every token answered with a key. the model learned which questions mattered and which answers were loudest. it was elegant and fast and hungry for compute.

a language model does not know the truth. it knows only what is likely. given a context of words it returns a distribution over every possible next word and we simply sample. temperature stretches or squeezes the distribution. top k and top p clip the long tail of unlikely candidates. creativity is just a slider.

in production a model must do more than complete text. it must route requests to the right expert. it must cache results that never change. it must retry on a bad sample and refuse when the prompt crosses a line. it must be observable. it must be boring to operate and exciting to use.

machine learning engineers build pipelines that run at three in the morning. they wake only when a metric crosses a threshold and a pager sings a sour note. they learn the shape of data drift. they learn the smell of a broken feature store. they learn that the hardest part of the job is the day after the model ships.

a model is a compressed hypothesis about the world. data engineers supply the world. they move bytes from source to sink. they stitch schemas together. they teach airflow to forgive and spark to forget. they do this again at dawn when a partition fails and again at dusk when a user clicks export.

agents are a new pattern. an agent plans. an agent uses tools. an agent remembers. an agent delegates to other agents. an agent fails in ways that look unreasonable until you read the trace. an agent is a language model with a todo list and a pager and a cup of coffee and a long weekend ahead.

the best systems are legible. a good dashboard says what broke. a good log says why. a good metric says how bad. a good alert says who. data engineering is the slow craft of turning noise into paragraphs a tired engineer can read at four in the morning without crying.

we are in the era of the thinking tool. the tool writes the code. the human writes the intent. the tool reads the log. the human reads the tool. somewhere between the two a product ships and a user smiles and a pager stays silent and the team goes home for a proper dinner.
`;

const LLMDemo = () => {
  const lm = useMemo(() => new NGramLM(CORPUS), []);
  const [prompt, setPrompt] = useState('the transformer learns by');
  const [output, setOutput] = useState('');
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [adjusted, setAdjusted] = useState<Candidate[]>([]);
  const [chosen, setChosen] = useState<string>('');

  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(10);
  const [topP, setTopP] = useState(0.92);
  const [maxTokens, setMaxTokens] = useState(60);

  const runningRef = useRef(false);

  const reset = () => {
    setOutput('');
    setStep(0);
    setCandidates([]);
    setAdjusted([]);
    setChosen('');
  };

  const onStep = () => {
    const context = lm.tokenize(prompt + ' ' + output);
    const dist = lm.distribution(context);
    if (dist.length === 0) return false;
    const { chosen: picked, filtered } = lm.sample(dist, {
      temperature,
      topK,
      topP,
    });
    setCandidates(dist.slice(0, 10));
    setAdjusted(filtered.slice(0, 10));
    setChosen(picked.token);
    setOutput((o) => appendToken(o, picked.token));
    setStep((s) => s + 1);
    return true;
  };

  const onGenerate = async () => {
    if (running) {
      runningRef.current = false;
      setRunning(false);
      return;
    }
    runningRef.current = true;
    setRunning(true);
    reset();
    await new Promise((r) => requestAnimationFrame(r));

    let accumulated = '';
    let stepsLeft = maxTokens;
    while (stepsLeft-- > 0 && runningRef.current) {
      const ctx = lm.tokenize(prompt + ' ' + accumulated);
      const dist = lm.distribution(ctx);
      if (dist.length === 0) break;
      const { chosen: picked, filtered } = lm.sample(dist, {
        temperature,
        topK,
        topP,
      });
      accumulated = appendToken(accumulated, picked.token);
      setOutput(accumulated);
      setCandidates(dist.slice(0, 10));
      setAdjusted(filtered.slice(0, 10));
      setChosen(picked.token);
      setStep((s) => s + 1);
      await new Promise((r) => setTimeout(r, 130));
    }
    runningRef.current = false;
    setRunning(false);
  };

  useEffect(() => {
    return () => {
      runningRef.current = false;
    };
  }, []);

  const wordsInPrompt = prompt.trim().split(/\s+/).length;
  const wordsOut = output.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="space-y-6">
      {/* Prompt */}
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
          / Prompt
        </div>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full bg-ink border border-bone/15 rounded-full px-5 py-3 font-mono text-base text-bone focus:outline-none focus:border-acid transition-colors"
          placeholder="enter a prompt…"
        />
      </div>

      <div className="grid lg:grid-cols-[1.15fr_1fr] gap-6">
        {/* Generation pane */}
        <div className="bg-bone/[0.02] border border-bone/10 rounded-2xl p-6 min-h-[360px] flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50">
              / Generation · step {step} / {maxTokens}
            </div>
            <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40">
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${running ? 'bg-acid animate-pulse' : 'bg-bone/30'}`} />
                {running ? 'streaming' : 'idle'}
              </span>
              <span>{wordsInPrompt + wordsOut}w</span>
            </div>
          </div>

          <div className="flex-1 font-serif-i text-xl md:text-2xl text-bone leading-relaxed">
            <span className="text-bone/60 font-sans not-italic font-mono text-base">
              {prompt}
            </span>{' '}
            <span>
              {output.split(/(\s+)/).map((tok, i) =>
                tok.trim() === '' ? (
                  tok
                ) : (
                  <motion.span
                    key={`${step}-${i}`}
                    initial={{ opacity: 0, y: 4, color: '#c6ff3d' }}
                    animate={{ opacity: 1, y: 0, color: '#f4f1ea' }}
                    transition={{ duration: 0.6, delay: 0.05 }}
                  >
                    {tok}
                  </motion.span>
                )
              )}
            </span>
            {running && (
              <motion.span
                animate={{ opacity: [1, 0.2, 1] }}
                transition={{ duration: 0.9, repeat: Infinity }}
                className="inline-block w-2 h-6 bg-acid ml-1 align-middle"
              />
            )}
          </div>

          {/* Controls */}
          <div className="mt-6 flex flex-wrap gap-2">
            <button
              data-cursor={running ? 'stop' : 'generate'}
              onClick={onGenerate}
              className="inline-flex items-center gap-2 px-4 py-2 bg-acid text-ink font-mono text-xs uppercase tracking-widest rounded-full"
            >
              {running ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {running ? 'Stop' : 'Generate'}
            </button>
            <button
              data-cursor="step"
              onClick={onStep}
              disabled={running}
              className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid disabled:opacity-40"
            >
              <Zap className="w-4 h-4" />
              Step once
            </button>
            <button
              data-cursor="reset"
              onClick={reset}
              disabled={running}
              className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid disabled:opacity-40"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          </div>
        </div>

        {/* Distribution pane */}
        <div className="bg-bone/[0.02] border border-bone/10 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50">
              / Next-token distribution
            </div>
            {chosen && (
              <AnimatePresence mode="wait">
                <motion.div
                  key={chosen + step}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="font-mono text-[10px] uppercase tracking-[0.25em] text-acid"
                >
                  sampled → {chosen}
                </motion.div>
              </AnimatePresence>
            )}
          </div>

          <div className="space-y-1.5 min-h-[280px]">
            {(candidates.length > 0 ? candidates : placeholder()).map((c, i) => {
              const adj = adjusted.find((x) => x.token === c.token);
              const adjP = adj?.p ?? 0;
              const isChosen = chosen === c.token;
              return (
                <div key={c.token + i} className="relative">
                  <div className="flex items-center justify-between font-mono text-xs mb-0.5">
                    <span className={isChosen ? 'text-acid' : 'text-bone/75'}>
                      {c.token}
                    </span>
                    <span className="text-bone/40 tabular-nums">
                      {(c.p * 100).toFixed(1)}%
                      <span className="text-bone/25"> → {(adjP * 100).toFixed(1)}%</span>
                    </span>
                  </div>
                  <div className="relative h-5 rounded bg-bone/[0.05] overflow-hidden">
                    <motion.div
                      animate={{ width: `${c.p * 100}%` }}
                      transition={{ type: 'spring', damping: 18, stiffness: 180 }}
                      className="absolute top-0 bottom-0 left-0 bg-bone/25"
                    />
                    <motion.div
                      animate={{ width: `${adjP * 100}%` }}
                      transition={{ type: 'spring', damping: 18, stiffness: 180 }}
                      className={`absolute top-0 bottom-0 left-0 ${isChosen ? 'bg-acid' : 'bg-violet'}`}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-4 flex items-center gap-3 font-mono text-[9px] uppercase tracking-widest text-bone/40">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-bone/40" /> raw
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-violet" /> after T/top-K/top-P
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-acid" /> sampled
            </span>
          </div>
        </div>
      </div>

      {/* Sliders */}
      <div className="grid md:grid-cols-4 gap-4">
        <Slider label="Temperature" value={temperature} min={0.05} max={2} step={0.05} onChange={setTemperature} format={(v) => v.toFixed(2)} />
        <Slider label="Top-K" value={topK} min={1} max={25} step={1} onChange={(v) => setTopK(Math.round(v))} />
        <Slider label="Top-P" value={topP} min={0.1} max={1} step={0.01} onChange={setTopP} format={(v) => v.toFixed(2)} />
        <Slider label="Max tokens" value={maxTokens} min={10} max={200} step={5} onChange={(v) => setMaxTokens(Math.round(v))} />
      </div>

      <p className="text-bone/50 text-sm leading-relaxed">
        A word-level trigram LM trained in your browser on a tiny AI-ish corpus. Exactly the same sampling pipeline that GPT-style models use — temperature reshapes the logits, <span className="text-acid">top-K</span> clips the tail, <span className="text-acid">top-P</span> keeps a nucleus of likely candidates, then we sample. Swap the n-gram table for 175B parameters and you have ChatGPT.
      </p>
    </div>
  );
};

const Slider = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) => (
  <div>
    <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.25em] text-bone/50 mb-2">
      <span>{label}</span>
      <span className="text-bone">{format ? format(value) : value}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-acid"
    />
  </div>
);

function placeholder(): Candidate[] {
  return [
    { token: '⎯ press generate ⎯', p: 0 },
  ];
}

function appendToken(prev: string, tok: string): string {
  if (/^[.,!?;:]$/.test(tok)) {
    return prev.trimEnd() + tok + ' ';
  }
  return prev + (prev && !prev.endsWith(' ') ? ' ' : '') + tok;
}

export default LLMDemo;
