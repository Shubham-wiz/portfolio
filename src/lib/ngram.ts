// Tiny word-level n-gram language model (trigram with bigram & unigram fallback).
// Built for the in-browser "LLM · Next Token" demo.

export type Candidate = { token: string; p: number };

export class NGramLM {
  private tri = new Map<string, Map<string, number>>();
  private bi = new Map<string, Map<string, number>>();
  private uni = new Map<string, number>();
  private vocab: string[] = [];

  constructor(corpus: string) {
    const tokens = this.tokenize(corpus);
    for (let i = 0; i < tokens.length; i++) {
      const w = tokens[i];
      this.uni.set(w, (this.uni.get(w) || 0) + 1);
      if (i >= 1) {
        const key = tokens[i - 1];
        if (!this.bi.has(key)) this.bi.set(key, new Map());
        const m = this.bi.get(key)!;
        m.set(w, (m.get(w) || 0) + 1);
      }
      if (i >= 2) {
        const key = tokens[i - 2] + '|' + tokens[i - 1];
        if (!this.tri.has(key)) this.tri.set(key, new Map());
        const m = this.tri.get(key)!;
        m.set(w, (m.get(w) || 0) + 1);
      }
    }
    this.vocab = Array.from(this.uni.keys());
  }

  tokenize(text: string): string[] {
    // keep punctuation as separate tokens for more lively generation
    return text
      .toLowerCase()
      .replace(/([.,!?;:])/g, ' $1 ')
      .split(/\s+/)
      .filter(Boolean);
  }

  distribution(context: string[]): Candidate[] {
    const last2 = context.slice(-2);
    const last1 = context.slice(-1);
    let map: Map<string, number> | undefined;
    if (last2.length === 2) map = this.tri.get(last2.join('|'));
    if (!map || map.size === 0) map = this.bi.get(last1[0] || '');
    if (!map || map.size === 0) map = this.uni;

    const total = Array.from(map.values()).reduce((a, b) => a + b, 0) || 1;
    const out: Candidate[] = [];
    for (const [tok, c] of map) out.push({ token: tok, p: c / total });
    out.sort((a, b) => b.p - a.p);
    return out;
  }

  /** Apply temperature, top-K, and top-P to a distribution. */
  sample(
    dist: Candidate[],
    opts: { temperature: number; topK: number; topP: number }
  ): { chosen: Candidate; filtered: Candidate[] } {
    const T = Math.max(0.05, opts.temperature);
    // Convert probs to "logits", apply temperature, re-softmax
    const logits = dist.map((c) => Math.log(c.p + 1e-9) / T);
    const m = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    let adjusted: Candidate[] = dist.map((c, i) => ({
      token: c.token,
      p: exps[i] / sum,
    }));
    adjusted.sort((a, b) => b.p - a.p);

    // top-K
    if (opts.topK > 0) adjusted = adjusted.slice(0, opts.topK);

    // top-P (nucleus)
    let acc = 0;
    const kept: Candidate[] = [];
    for (const c of adjusted) {
      kept.push(c);
      acc += c.p;
      if (acc >= opts.topP) break;
    }

    // renormalize
    const totalP = kept.reduce((a, b) => a + b.p, 0) || 1;
    const filtered = kept.map((c) => ({ token: c.token, p: c.p / totalP }));

    // sample
    const r = Math.random();
    let run = 0;
    let chosen = filtered[0];
    for (const c of filtered) {
      run += c.p;
      if (r <= run) {
        chosen = c;
        break;
      }
    }
    return { chosen, filtered };
  }
}
