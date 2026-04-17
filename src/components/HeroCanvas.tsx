import { useEffect, useRef } from 'react';

interface Props {
  className?: string;
}

// Mouse-reactive particle field with connective lines – GPU-friendly.
const HeroCanvas = ({ className = '' }: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999, down: false });
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const particles: {
      x: number;
      y: number;
      vx: number;
      vy: number;
      r: number;
    }[] = [];

    const initParticles = () => {
      const count = Math.floor((width * height) / 13000);
      particles.length = 0;
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          r: Math.random() * 1.6 + 0.6,
        });
      }
    };

    resize();
    initParticles();

    const handleResize = () => {
      resize();
      initParticles();
    };

    const handleMouse = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = e.clientX - rect.left;
      mouseRef.current.y = e.clientY - rect.top;
    };
    const handleLeave = () => {
      mouseRef.current.x = -9999;
      mouseRef.current.y = -9999;
    };
    const handleDown = () => (mouseRef.current.down = true);
    const handleUp = () => (mouseRef.current.down = false);

    window.addEventListener('resize', handleResize);
    canvas.addEventListener('mousemove', handleMouse);
    canvas.addEventListener('mouseleave', handleLeave);
    canvas.addEventListener('mousedown', handleDown);
    canvas.addEventListener('mouseup', handleUp);

    const tick = () => {
      ctx.clearRect(0, 0, width, height);

      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      const push = mouseRef.current.down ? -1 : 1;

      // mouse glow halo
      if (mx > -9000) {
        const grad = ctx.createRadialGradient(mx, my, 0, mx, my, 260);
        grad.addColorStop(0, 'rgba(198, 255, 61, 0.10)');
        grad.addColorStop(1, 'rgba(198, 255, 61, 0)');
        ctx.fillStyle = grad;
        ctx.fillRect(mx - 260, my - 260, 520, 520);
      }

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // mouse force
        if (mx > -9000) {
          const dx = p.x - mx;
          const dy = p.y - my;
          const d2 = dx * dx + dy * dy;
          if (d2 < 160 * 160 && d2 > 0.1) {
            const f = (160 - Math.sqrt(d2)) / 160;
            p.vx += (dx / Math.sqrt(d2)) * f * 0.6 * push;
            p.vy += (dy / Math.sqrt(d2)) * f * 0.6 * push;
          }
        }

        p.vx *= 0.985;
        p.vy *= 0.985;
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x += width;
        else if (p.x > width) p.x -= width;
        if (p.y < 0) p.y += height;
        else if (p.y > height) p.y -= height;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(244, 241, 234, 0.75)';
        ctx.fill();
      }

      // connective lines
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 120 * 120) {
            const alpha = 1 - Math.sqrt(d2) / 120;
            ctx.strokeStyle = `rgba(198, 255, 61, ${alpha * 0.18})`;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('mousemove', handleMouse);
      canvas.removeEventListener('mouseleave', handleLeave);
      canvas.removeEventListener('mousedown', handleDown);
      canvas.removeEventListener('mouseup', handleUp);
    };
  }, []);

  return <canvas ref={canvasRef} className={`w-full h-full block ${className}`} />;
};

export default HeroCanvas;
