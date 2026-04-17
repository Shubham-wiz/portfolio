// Tiny self-attention implementation used by the TransformerDemo.
// Deterministic random Q/K/V projections — the output attention patterns
// aren't semantically meaningful but faithfully show the mechanics of
// scaled dot-product multi-head attention.

const D_MODEL = 32;

// Seeded LCG for deterministic weights
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return ((s >>> 8) / 0xffffff) * 2 - 1; // in [-1, 1]
  };
}

function hashToken(token: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function embed(token: string): number[] {
  const rand = lcg(hashToken(token));
  const v = new Array<number>(D_MODEL);
  for (let i = 0; i < D_MODEL; i++) v[i] = rand() * 0.8;
  return v;
}

function positional(pos: number): number[] {
  const v = new Array<number>(D_MODEL);
  for (let i = 0; i < D_MODEL; i++) {
    const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / D_MODEL);
    v[i] = i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
  }
  return v;
}

function projection(seed: number, dHead: number): number[][] {
  const rand = lcg(seed);
  const W: number[][] = [];
  for (let i = 0; i < D_MODEL; i++) {
    const row: number[] = [];
    for (let j = 0; j < dHead; j++) {
      row.push((rand() * 2) / Math.sqrt(D_MODEL));
    }
    W.push(row);
  }
  return W;
}

function matmul(A: number[][], B: number[][]): number[][] {
  const R = A.length;
  const C = B[0].length;
  const K = B.length;
  const out: number[][] = [];
  for (let i = 0; i < R; i++) {
    const row = new Array<number>(C).fill(0);
    for (let k = 0; k < K; k++) {
      const a = A[i][k];
      for (let j = 0; j < C; j++) row[j] += a * B[k][j];
    }
    out.push(row);
  }
  return out;
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row);
  const exps = row.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / s);
}

export interface AttentionResult {
  tokens: string[];
  heads: number[][][]; // [head][query][key]
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, '')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 16);
}

export function runAttention(text: string, nHeads: number): AttentionResult {
  const tokens = tokenize(text);
  if (tokens.length === 0) return { tokens, heads: [] };
  const dHead = Math.max(4, Math.floor(D_MODEL / nHeads));

  // Input matrix: [seq, D_MODEL]
  const X: number[][] = tokens.map((t, i) => {
    const e = embed(t);
    const p = positional(i);
    return e.map((v, j) => v + p[j]);
  });

  const heads: number[][][] = [];
  for (let h = 0; h < nHeads; h++) {
    const Wq = projection(1000 + h * 7, dHead);
    const Wk = projection(2000 + h * 11, dHead);

    const Q = matmul(X, Wq);
    const K = matmul(X, Wk);

    // scores = Q K^T / sqrt(dHead)
    const scale = 1 / Math.sqrt(dHead);
    const scores: number[][] = [];
    for (let i = 0; i < Q.length; i++) {
      const row: number[] = [];
      for (let j = 0; j < K.length; j++) {
        let s = 0;
        for (let d = 0; d < dHead; d++) s += Q[i][d] * K[j][d];
        row.push(s * scale);
      }
      scores.push(softmaxRow(row));
    }
    heads.push(scores);
  }

  return { tokens, heads };
}
