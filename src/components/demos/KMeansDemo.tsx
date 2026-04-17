import { useEffect, useRef, useState } from 'react';
import { RotateCcw, Plus, Zap, Shuffle } from 'lucide-react';

type P = { x: number; y: number; c: number };

const PALETTE = ['#c6ff3d', '#a78bfa', '#ff4d1f', '#5b6cff', '#f472b6', '#22d3ee', '#facc15'];

const KMeansDemo = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ptsRef = useRef<P[]>([]);
  const centroidsRef = useRef<{ x: number; y: number }[]>([]);
  const targetsRef = useRef<{ x: number; y: number }[]>([]);
  const [k, setK] = useState(3);
  const [step, setStep] = useState(0);
  const rafRef = useRef<number | null>(null);

  const W = 520;
  const H = 520;

  const seed = (nClusters: number) => {
    const pts: P[] = [];
    const centers: { x: number; y: number }[] = [];
    for (let c = 0; c < nClusters; c++) {
      centers.push({ x: 80 + Math.random() * (W - 160), y: 80 + Math.random() * (H - 160) });
    }
    for (let i = 0; i < 220; i++) {
      const c = Math.floor(Math.random() * centers.length);
      const center = centers[c];
      const r = Math.random() * 50;
      const t = Math.random() * Math.PI * 2;
      pts.push({ x: center.x + Math.cos(t) * r, y: center.y + Math.sin(t) * r, c: -1 });
    }
    return pts;
  };

  const initCentroids = (nK: number) => {
    const c: { x: number; y: number }[] = [];
    for (let i = 0; i < nK; i++) {
      c.push({ x: 50 + Math.random() * (W - 100), y: 50 + Math.random() * (H - 100) });
    }
    return c;
  };

  useEffect(() => {
    ptsRef.current = seed(Math.max(3, k));
    centroidsRef.current = initCentroids(k);
    targetsRef.current = centroidsRef.current.map((c) => ({ ...c }));
    setStep(0);
  }, [k]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      // animate centroids toward target
      for (let i = 0; i < centroidsRef.current.length; i++) {
        const c = centroidsRef.current[i];
        const t = targetsRef.current[i];
        c.x += (t.x - c.x) * 0.08;
        c.y += (t.y - c.y) * 0.08;
      }
      // reassign based on CURRENT centroid positions for color preview
      for (const p of ptsRef.current) {
        let best = 0;
        let bestD = Infinity;
        for (let i = 0; i < centroidsRef.current.length; i++) {
          const c = centroidsRef.current[i];
          const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        }
        p.c = best;
      }

      // draw voronoi-ish soft regions via layered radial gradients
      for (let i = 0; i < centroidsRef.current.length; i++) {
        const c = centroidsRef.current[i];
        const col = PALETTE[i % PALETTE.length];
        const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, 220);
        g.addColorStop(0, col + '26');
        g.addColorStop(1, col + '00');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
      }

      // dots
      for (const p of ptsRef.current) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = PALETTE[p.c % PALETTE.length];
        ctx.fill();
        ctx.strokeStyle = 'rgba(5,5,5,0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // centroids
      for (let i = 0; i < centroidsRef.current.length; i++) {
        const c = centroidsRef.current[i];
        ctx.beginPath();
        ctx.arc(c.x, c.y, 12, 0, Math.PI * 2);
        ctx.fillStyle = PALETTE[i % PALETTE.length];
        ctx.fill();
        ctx.strokeStyle = '#050505';
        ctx.lineWidth = 3;
        ctx.stroke();
        // cross
        ctx.strokeStyle = '#050505';
        ctx.beginPath();
        ctx.moveTo(c.x - 6, c.y);
        ctx.lineTo(c.x + 6, c.y);
        ctx.moveTo(c.x, c.y - 6);
        ctx.lineTo(c.x, c.y + 6);
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    const handleClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ptsRef.current.push({ x, y, c: -1 });
    };
    canvas.addEventListener('click', handleClick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener('click', handleClick);
    };
  }, []);

  const iterate = () => {
    const centroids = targetsRef.current;
    // Assign
    for (const p of ptsRef.current) {
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < centroids.length; i++) {
        const c = centroids[i];
        const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      p.c = best;
    }
    // Update
    const newTargets = centroids.map(() => ({ x: 0, y: 0, n: 0 }));
    for (const p of ptsRef.current) {
      const t = newTargets[p.c];
      t.x += p.x;
      t.y += p.y;
      t.n += 1;
    }
    targetsRef.current = newTargets.map((t, i) =>
      t.n === 0 ? { ...centroids[i] } : { x: t.x / t.n, y: t.y / t.n }
    );
    setStep((s) => s + 1);
  };

  return (
    <div className="grid lg:grid-cols-[520px_1fr] gap-8 items-start">
      <div className="relative rounded-2xl overflow-hidden border border-bone/10 bg-ink">
        <canvas ref={canvasRef} />
        <div className="absolute top-3 left-3 font-mono text-[10px] uppercase tracking-[0.3em] text-ink/70 bg-acid px-2 py-1 rounded">
          K-Means · k = {k} · iter {step}
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            / Clusters · k
          </div>
          <div className="flex gap-2 flex-wrap">
            {[2, 3, 4, 5, 6, 7].map((n) => (
              <button
                key={n}
                data-cursor={`k=${n}`}
                onClick={() => setK(n)}
                className={`w-10 h-10 rounded-full font-display text-lg transition ${
                  k === n
                    ? 'bg-acid text-ink'
                    : 'border border-bone/20 text-bone/70 hover:border-acid hover:text-acid'
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            data-cursor="iterate"
            onClick={iterate}
            className="inline-flex items-center gap-2 px-4 py-2 bg-acid text-ink font-mono text-xs uppercase tracking-widest rounded-full"
          >
            <Zap className="w-4 h-4" />
            Step
          </button>
          <button
            data-cursor="run 20"
            onClick={() => {
              for (let i = 0; i < 20; i++) iterate();
            }}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
          >
            <Plus className="w-4 h-4" />
            Run × 20
          </button>
          <button
            data-cursor="reset"
            onClick={() => {
              centroidsRef.current = initCentroids(k);
              targetsRef.current = centroidsRef.current.map((c) => ({ ...c }));
              setStep(0);
            }}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
          >
            <RotateCcw className="w-4 h-4" />
            Re-seed centroids
          </button>
          <button
            data-cursor="new data"
            onClick={() => {
              ptsRef.current = seed(Math.max(3, k));
            }}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
          >
            <Shuffle className="w-4 h-4" />
            New data
          </button>
        </div>

        <p className="text-bone/50 text-sm leading-relaxed">
          Classic unsupervised clustering. Click the canvas to drop more points. Hit <span className="text-acid">Step</span> to run one iteration — watch each centroid crawl toward the mean of its cluster in real time.
        </p>
      </div>
    </div>
  );
};

export default KMeansDemo;
