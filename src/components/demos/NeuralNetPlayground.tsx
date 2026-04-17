import { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, Trash2, Shuffle } from 'lucide-react';
import { MLP } from '../../lib/mlp';

type Dataset = 'moons' | 'spiral' | 'circle' | 'xor';

type Point = { x: number; y: number; c: 0 | 1 };

const makeDataset = (kind: Dataset, n = 160): Point[] => {
  const pts: Point[] = [];
  if (kind === 'moons') {
    for (let i = 0; i < n / 2; i++) {
      const t = Math.PI * (i / (n / 2));
      pts.push({ x: Math.cos(t) * 0.5 - 0.2 + (Math.random() - 0.5) * 0.08, y: Math.sin(t) * 0.5 + (Math.random() - 0.5) * 0.08, c: 0 });
      pts.push({ x: 0.2 + Math.cos(t + Math.PI) * 0.5 + (Math.random() - 0.5) * 0.08, y: -Math.sin(t + Math.PI) * 0.5 + (Math.random() - 0.5) * 0.08 + 0.25, c: 1 });
    }
  } else if (kind === 'spiral') {
    const half = n / 2;
    for (let i = 0; i < half; i++) {
      const r = i / half * 0.8;
      const t = 1.75 * i / half * 2 * Math.PI;
      pts.push({ x: r * Math.sin(t) + (Math.random() - 0.5) * 0.04, y: r * Math.cos(t) + (Math.random() - 0.5) * 0.04, c: 0 });
      pts.push({ x: -r * Math.sin(t) + (Math.random() - 0.5) * 0.04, y: -r * Math.cos(t) + (Math.random() - 0.5) * 0.04, c: 1 });
    }
  } else if (kind === 'circle') {
    for (let i = 0; i < n; i++) {
      const r = Math.random();
      const t = Math.random() * Math.PI * 2;
      pts.push({ x: r * Math.cos(t) * 0.85, y: r * Math.sin(t) * 0.85, c: r < 0.45 ? 0 : 1 });
    }
  } else if (kind === 'xor') {
    for (let i = 0; i < n; i++) {
      const x = (Math.random() * 2 - 1) * 0.9;
      const y = (Math.random() * 2 - 1) * 0.9;
      pts.push({ x, y, c: x * y > 0 ? 0 : 1 });
    }
  }
  return pts;
};

const RES = 56; // grid resolution for decision boundary
const WIDTH = 520;
const HEIGHT = 520;

const NeuralNetPlayground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [running, setRunning] = useState(true);
  const [dataset, setDataset] = useState<Dataset>('moons');
  const [hidden, setHidden] = useState(2);
  const [neurons, setNeurons] = useState(8);
  const [lr, setLr] = useState(0.05);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.7);
  const netRef = useRef<MLP | null>(null);
  const pointsRef = useRef<Point[]>(makeDataset('moons'));
  const rafRef = useRef<number | null>(null);

  const rebuild = () => {
    const layers = [2, ...Array(hidden).fill(neurons), 1];
    netRef.current = new MLP(layers, 'tanh');
    setEpoch(0);
    setLoss(0.7);
  };

  useEffect(() => {
    rebuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hidden, neurons]);

  useEffect(() => {
    pointsRef.current = makeDataset(dataset);
    rebuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    canvas.style.width = WIDTH + 'px';
    canvas.style.height = HEIGHT + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const toPx = (p: { x: number; y: number }) => ({
      px: (p.x + 1) / 2 * WIDTH,
      py: (1 - (p.y + 1) / 2) * HEIGHT,
    });

    const draw = () => {
      const net = netRef.current!;
      const img = ctx.createImageData(RES, RES);
      for (let j = 0; j < RES; j++) {
        for (let i = 0; i < RES; i++) {
          const x = (i / (RES - 1)) * 2 - 1;
          const y = 1 - (j / (RES - 1)) * 2;
          const o = net.forward([x, y])[0];
          const idx = (j * RES + i) * 4;
          // gradient colors: acid-lime (198,255,61) for class 0, violet (167,139,250) for class 1
          const r = Math.round(198 + (167 - 198) * o);
          const g = Math.round(255 + (139 - 255) * o);
          const b = Math.round(61 + (250 - 61) * o);
          img.data[idx] = r;
          img.data[idx + 1] = g;
          img.data[idx + 2] = b;
          img.data[idx + 3] = 130;
        }
      }
      const off = document.createElement('canvas');
      off.width = RES;
      off.height = RES;
      off.getContext('2d')!.putImageData(img, 0, 0);
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, WIDTH, HEIGHT);

      // overlay grid
      ctx.strokeStyle = 'rgba(5,5,5,0.2)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 8; i++) {
        ctx.beginPath();
        ctx.moveTo((i / 8) * WIDTH, 0);
        ctx.lineTo((i / 8) * WIDTH, HEIGHT);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, (i / 8) * HEIGHT);
        ctx.lineTo(WIDTH, (i / 8) * HEIGHT);
        ctx.stroke();
      }

      // points
      for (const p of pointsRef.current) {
        const { px, py } = toPx(p);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = p.c === 0 ? '#c6ff3d' : '#a78bfa';
        ctx.fill();
        ctx.strokeStyle = '#050505';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    };

    const BATCH = 16;
    let localEpoch = 0;
    const tick = () => {
      if (running && netRef.current) {
        const net = netRef.current;
        const pts = pointsRef.current;
        let sum = 0;
        for (let i = 0; i < BATCH; i++) {
          const p = pts[Math.floor(Math.random() * pts.length)];
          sum += net.trainSample([p.x, p.y], p.c, lr);
        }
        localEpoch += 1;
        if (localEpoch % 3 === 0) {
          setEpoch((e) => e + 3);
          setLoss(sum / BATCH);
        }
      }
      draw();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    const handleClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = 1 - ((e.clientY - rect.top) / rect.height) * 2;
      const c: 0 | 1 = e.shiftKey ? 1 : 0;
      pointsRef.current.push({ x, y, c });
    };
    const handleContext = (e: MouseEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = 1 - ((e.clientY - rect.top) / rect.height) * 2;
      pointsRef.current.push({ x, y, c: 1 });
    };
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('contextmenu', handleContext);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener('click', handleClick);
      canvas.removeEventListener('contextmenu', handleContext);
    };
  }, [running, lr]);

  return (
    <div className="grid lg:grid-cols-[520px_1fr] gap-8 items-start">
      <div className="relative rounded-2xl overflow-hidden border border-bone/10 bg-ink">
        <canvas ref={canvasRef} className="block" />
        <div className="absolute top-3 left-3 font-mono text-[10px] uppercase tracking-[0.3em] text-ink/70 bg-acid px-2 py-1 rounded">
          Live boundary · {RES}×{RES}
        </div>
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.25em] text-bone/70">
          <span>Click: add <span className="text-acid">green</span> · Shift+click or right-click: <span className="text-violet">violet</span></span>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            / Dataset
          </div>
          <div className="flex flex-wrap gap-2">
            {(['moons', 'spiral', 'circle', 'xor'] as Dataset[]).map((d) => (
              <button
                key={d}
                data-cursor={d}
                onClick={() => setDataset(d)}
                className={`px-3 py-1.5 rounded-full border text-xs font-mono uppercase tracking-widest transition ${
                  dataset === d
                    ? 'bg-acid text-ink border-acid'
                    : 'border-bone/20 text-bone/70 hover:border-acid hover:text-acid'
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
              Hidden layers · {hidden}
            </div>
            <input
              type="range"
              min={1}
              max={4}
              value={hidden}
              onChange={(e) => setHidden(parseInt(e.target.value))}
              className="w-full accent-acid"
            />
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
              Neurons / layer · {neurons}
            </div>
            <input
              type="range"
              min={2}
              max={16}
              value={neurons}
              onChange={(e) => setNeurons(parseInt(e.target.value))}
              className="w-full accent-acid"
            />
          </div>
          <div className="col-span-2">
            <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
              Learning rate · {lr.toFixed(3)}
            </div>
            <input
              type="range"
              min={0.005}
              max={0.25}
              step={0.005}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full accent-acid"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-bone/[0.03] border border-bone/10 rounded-xl p-4">
            <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/50 mb-1">
              Epoch
            </div>
            <div className="font-display text-3xl tracking-tight">{epoch}</div>
          </div>
          <div className="bg-bone/[0.03] border border-bone/10 rounded-xl p-4">
            <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/50 mb-1">
              Loss · BCE
            </div>
            <div className="font-display text-3xl tracking-tight text-acid">{loss.toFixed(4)}</div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 pt-2">
          <button
            data-cursor={running ? 'pause' : 'play'}
            onClick={() => setRunning((r) => !r)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-acid text-ink font-mono text-xs uppercase tracking-widest rounded-full"
          >
            {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {running ? 'Pause' : 'Play'}
          </button>
          <button
            data-cursor="reset"
            onClick={rebuild}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
          >
            <RotateCcw className="w-4 h-4" />
            Reset weights
          </button>
          <button
            data-cursor="new data"
            onClick={() => {
              pointsRef.current = makeDataset(dataset);
              rebuild();
            }}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
          >
            <Shuffle className="w-4 h-4" />
            Resample
          </button>
          <button
            data-cursor="wipe"
            onClick={() => (pointsRef.current = [])}
            className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-flame hover:text-flame"
          >
            <Trash2 className="w-4 h-4" />
            Clear points
          </button>
        </div>

        <p className="text-bone/50 text-sm leading-relaxed">
          A real MLP written from scratch — no TensorFlow. Gradient descent with backprop runs live in your tab at ~50 mini-batches/sec. The background heatmap is the network's live decision surface.
        </p>
      </div>
    </div>
  );
};

export default NeuralNetPlayground;
