import { useEffect, useRef, useState } from 'react';
import { RotateCcw, Sparkles } from 'lucide-react';

// tiny hash-based pseudo noise
const hash = (x: number, y: number) => {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
};
const smoothstep = (t: number) => t * t * (3 - 2 * t);
const noise = (x: number, y: number) => {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = smoothstep(xf);
  const v = smoothstep(yf);
  const a = hash(xi, yi);
  const b = hash(xi + 1, yi);
  const c = hash(xi, yi + 1);
  const d = hash(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
};

const PALETTES = [
  ['#c6ff3d', '#a78bfa', '#ff4d1f'],
  ['#5b6cff', '#c6ff3d', '#ffffff'],
  ['#ff4d1f', '#f4f1ea', '#a78bfa'],
  ['#22d3ee', '#c6ff3d', '#f472b6'],
];

interface Particle {
  x: number;
  y: number;
  life: number;
  maxLife: number;
  hue: number;
}

const FlowField = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999, push: false });
  const [scale, setScale] = useState(0.004);
  const [count, setCount] = useState(600);
  const [palette, setPalette] = useState(0);
  const rafRef = useRef<number | null>(null);

  const W = 800;
  const H = 520;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = '100%';
    canvas.style.maxWidth = W + 'px';
    canvas.style.aspectRatio = `${W} / ${H}`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, W, H);

    const particles: Particle[] = [];
    const seed = () => {
      particles.length = 0;
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * W,
          y: Math.random() * H,
          life: Math.random() * 150,
          maxLife: 150 + Math.random() * 200,
          hue: Math.random(),
        });
      }
    };
    seed();

    let t = 0;
    const tick = () => {
      t += 0.6;
      ctx.fillStyle = 'rgba(5,5,5,0.06)';
      ctx.fillRect(0, 0, W, H);

      for (const p of particles) {
        const n = noise(p.x * scale, p.y * scale + t * scale * 10);
        const angle = n * Math.PI * 4;
        let vx = Math.cos(angle) * 1.2;
        let vy = Math.sin(angle) * 1.2;

        // mouse influence
        if (mouseRef.current.x > -9000) {
          const dx = p.x - mouseRef.current.x;
          const dy = p.y - mouseRef.current.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 140 * 140 && d2 > 0.5) {
            const d = Math.sqrt(d2);
            const f = (140 - d) / 140;
            const dir = mouseRef.current.push ? -1 : 1;
            vx += (dx / d) * f * 2.5 * dir;
            vy += (dy / d) * f * 2.5 * dir;
          }
        }

        p.x += vx;
        p.y += vy;
        p.life += 1;

        const pal = PALETTES[palette];
        const col = pal[Math.floor(p.hue * pal.length)];
        ctx.fillStyle = col + 'aa';
        ctx.fillRect(p.x, p.y, 1.4, 1.4);

        if (
          p.x < -20 ||
          p.x > W + 20 ||
          p.y < -20 ||
          p.y > H + 20 ||
          p.life > p.maxLife
        ) {
          p.x = Math.random() * W;
          p.y = Math.random() * H;
          p.life = 0;
          p.hue = Math.random();
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    const handleMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * W;
      mouseRef.current.y = ((e.clientY - rect.top) / rect.height) * H;
    };
    const handleLeave = () => {
      mouseRef.current.x = -9999;
      mouseRef.current.y = -9999;
    };
    const handleDown = () => (mouseRef.current.push = true);
    const handleUp = () => (mouseRef.current.push = false);

    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', handleLeave);
    canvas.addEventListener('mousedown', handleDown);
    canvas.addEventListener('mouseup', handleUp);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', handleLeave);
      canvas.removeEventListener('mousedown', handleDown);
      canvas.removeEventListener('mouseup', handleUp);
    };
  }, [scale, count, palette]);

  return (
    <div className="grid lg:grid-cols-[1fr_260px] gap-8 items-start">
      <div className="relative rounded-2xl overflow-hidden border border-bone/10 bg-ink">
        <canvas ref={canvasRef} className="block w-full" />
        <div className="absolute top-3 left-3 font-mono text-[10px] uppercase tracking-[0.3em] text-ink/80 bg-acid px-2 py-1 rounded">
          Perlin Flow Field
        </div>
        <div className="absolute bottom-3 left-3 font-mono text-[10px] uppercase tracking-[0.25em] text-bone/60">
          Drag to push · click + hold to pull
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            Turbulence · {(scale * 1000).toFixed(1)}
          </div>
          <input
            type="range"
            min={1}
            max={12}
            step={0.1}
            value={scale * 1000}
            onChange={(e) => setScale(parseFloat(e.target.value) / 1000)}
            className="w-full accent-acid"
          />
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            Particles · {count}
          </div>
          <input
            type="range"
            min={100}
            max={2000}
            step={50}
            value={count}
            onChange={(e) => setCount(parseInt(e.target.value))}
            className="w-full accent-acid"
          />
        </div>

        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/50 mb-2">
            Palette
          </div>
          <div className="flex gap-2">
            {PALETTES.map((p, i) => (
              <button
                key={i}
                data-cursor="palette"
                onClick={() => setPalette(i)}
                className={`flex h-10 w-14 rounded-lg overflow-hidden border ${
                  palette === i ? 'border-acid' : 'border-bone/20'
                }`}
              >
                {p.map((c) => (
                  <span key={c} className="flex-1" style={{ background: c }} />
                ))}
              </button>
            ))}
          </div>
        </div>

        <button
          data-cursor="reset"
          onClick={() => {
            setScale(0.004);
            setCount(600);
            setPalette(0);
          }}
          className="inline-flex items-center gap-2 px-4 py-2 border border-bone/20 text-bone font-mono text-xs uppercase tracking-widest rounded-full hover:border-acid hover:text-acid"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <div className="flex items-start gap-2 text-bone/50 text-sm leading-relaxed">
          <Sparkles className="w-4 h-4 text-acid mt-0.5 shrink-0" />
          <p>Generative art. A custom value-noise field pushes thousands of particles. The same math underpins procedural terrain and stylized fluid sims.</p>
        </div>
      </div>
    </div>
  );
};

export default FlowField;
