import { motion, useScroll, useTransform } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import HeroCanvas from '../components/HeroCanvas';
import MagneticButton from '../components/MagneticButton';
import Marquee from '../components/Marquee';

interface Props {
  onNavigate: (id: string) => void;
}

const Hero = ({ onNavigate }: Props) => {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  });

  const y = useTransform(scrollYProgress, [0, 1], ['0%', '30%']);
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [1, 1.15]);

  const [now, setNow] = useState<string>('');
  useEffect(() => {
    const update = () => {
      const d = new Date();
      setNow(
        d.toLocaleTimeString('en-GB', {
          timeZone: 'Europe/Berlin',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      );
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <section
      ref={ref as any}
      id="home"
      className="relative min-h-[110vh] bg-ink overflow-hidden"
    >
      {/* canvas layer */}
      <motion.div
        style={{ scale }}
        className="absolute inset-0 z-0"
      >
        <HeroCanvas />
      </motion.div>

      {/* dotted grid overlay */}
      <div className="absolute inset-0 dot-grid opacity-40 mix-blend-overlay z-[1]" />

      {/* ambient colored blobs */}
      <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-violet/10 blur-[160px] z-[1]" />
      <div className="absolute -bottom-40 -right-40 w-[600px] h-[600px] rounded-full bg-acid/10 blur-[160px] z-[1]" />

      {/* Content */}
      <motion.div
        style={{ y, opacity }}
        className="relative z-10 min-h-screen flex flex-col justify-center pt-28 px-6 sm:px-10 lg:px-16"
      >
        {/* top meta row */}
        <div className="flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] uppercase tracking-[0.25em] text-bone/55 mb-16">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-acid animate-pulse" />
            <span>Available · Q2 2026</span>
          </div>
          <div className="flex items-center gap-6">
            <span>Berlin · 52.52°N 13.40°E</span>
            <span className="text-acid">{now}</span>
          </div>
        </div>

        {/* big name */}
        <div className="relative">
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
            className="font-display leading-[0.82] tracking-crushed uppercase text-bone select-none"
            style={{ fontSize: 'clamp(4rem, 15vw, 18rem)' }}
          >
            <motion.span
              initial={{ y: '110%', opacity: 0 }}
              animate={{ y: '0%', opacity: 1 }}
              transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              className="block overflow-hidden"
            >
              <span className="block">Shubham</span>
            </motion.span>
            <motion.span
              initial={{ y: '110%', opacity: 0 }}
              animate={{ y: '0%', opacity: 1 }}
              transition={{ duration: 1, delay: 0.12, ease: [0.16, 1, 0.3, 1] }}
              className="block overflow-hidden"
            >
              <span className="block">
                Dw<span className="font-serif-i text-acid lowercase">i</span>ved
                <span className="font-serif-i lowercase text-violet">i</span>
              </span>
            </motion.span>
          </motion.h1>

          {/* floating badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8, rotate: -6 }}
            animate={{ opacity: 1, scale: 1, rotate: -6 }}
            transition={{ delay: 1, duration: 0.7 }}
            className="hidden md:flex absolute -top-6 right-2 md:right-8 items-center gap-2 px-4 py-2 bg-acid text-ink font-mono text-xs uppercase tracking-[0.2em] rounded-full shadow-acid"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-ink animate-pulse" />
            Data · AI · Architectures
          </motion.div>
        </div>

        {/* Subheadline */}
        <div className="mt-10 grid md:grid-cols-12 gap-6 items-start">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="md:col-span-5 md:col-start-1"
          >
            <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-bone/50 mb-3">
              / Manifesto
            </p>
            <p className="text-bone/80 text-xl md:text-2xl leading-snug">
              I build <span className="font-serif-i text-acid">living</span> data systems and
              <span className="font-serif-i text-violet"> thinking</span> agents — at the seam where
              raw pipelines become intelligent products.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.8 }}
            className="md:col-span-5 md:col-start-8 flex flex-col gap-4"
          >
            <div className="flex items-center gap-4">
              <span className="font-mono text-[11px] uppercase tracking-[0.3em] text-bone/50">
                / Currently
              </span>
              <span className="flex-1 h-px bg-bone/15" />
            </div>
            <p className="text-bone/80 text-base md:text-lg">
              Shipping multi-agent market intelligence at{' '}
              <span className="text-acid">Metamatics</span>. Previously ETL at terabyte scale for
              <span className="text-violet"> PERI GmbH</span> and <span className="text-flame">Quantizant</span>.
            </p>

            <div className="flex flex-wrap gap-3 mt-4">
              <MagneticButton
                onClick={() => onNavigate('work')}
                cursorLabel="see work"
                className="group relative inline-flex items-center gap-3 px-6 py-3 bg-acid text-ink font-display uppercase text-lg rounded-full overflow-hidden"
              >
                See the work
                <span aria-hidden className="inline-block transition-transform group-hover:translate-x-1">
                  →
                </span>
              </MagneticButton>
              <MagneticButton
                onClick={() => onNavigate('lab')}
                cursorLabel="play"
                className="group inline-flex items-center gap-3 px-6 py-3 border border-bone/25 text-bone hover:border-acid hover:text-acid transition-colors font-display uppercase text-lg rounded-full"
              >
                Play in the lab
                <span aria-hidden className="inline-block transition-transform group-hover:translate-x-1">
                  ↘
                </span>
              </MagneticButton>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* marquee at bottom */}
      <div className="absolute bottom-0 left-0 right-0 z-10 py-6 border-t border-bone/10 bg-ink/60 backdrop-blur-sm">
        <Marquee
          speed={42}
          items={[
            'DATA ENGINEER',
            'AI ARCHITECT',
            'MLOps',
            'MULTI-AGENT SYSTEMS',
            'APACHE SPARK',
            'LANGCHAIN',
            'PYTHON · TYPESCRIPT',
            'AVAILABLE FOR WORK',
          ]}
          separator={<span>✦</span>}
          className="font-display uppercase text-bone/85 text-2xl md:text-3xl tracking-tight"
        />
      </div>

      {/* scroll cue */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6 }}
        className="hidden md:block absolute bottom-24 right-10 z-10 font-mono text-[11px] uppercase tracking-[0.35em] text-bone/45"
      >
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 1.6, repeat: Infinity }}
          className="flex items-center gap-3"
        >
          <span className="block w-12 h-px bg-bone/40" />
          Scroll
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Hero;
