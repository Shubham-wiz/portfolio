import { motion, useMotionValueEvent, useScroll } from 'framer-motion';
import { useState } from 'react';

interface NavbarProps {
  onNavigate: (id: string) => void;
}

const items = [
  { id: 'about', label: '01 / About' },
  { id: 'work', label: '02 / Work' },
  { id: 'lab', label: '03 / Lab' },
  { id: 'experience', label: '04 / CV' },
  { id: 'writings', label: '05 / Writings' },
];

const Navbar = ({ onNavigate }: NavbarProps) => {
  const { scrollY } = useScroll();
  const [shrunk, setShrunk] = useState(false);

  useMotionValueEvent(scrollY, 'change', (v) => {
    setShrunk(v > 40);
  });

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay: 0.4, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className="fixed top-4 left-1/2 -translate-x-1/2 z-[9995] w-[min(1200px,calc(100%-2rem))]"
    >
      <div
        className={`flex items-center justify-between transition-all duration-500 ${
          shrunk
            ? 'bg-ink/75 backdrop-blur-md border border-bone/10 rounded-full px-4 py-2'
            : 'bg-transparent px-2 py-2'
        }`}
      >
        <button
          data-cursor="home"
          onClick={() => onNavigate('home')}
          className="flex items-center gap-2 group"
        >
          <div className="w-8 h-8 rounded-full bg-acid grid place-items-center text-ink font-display text-lg">
            S
          </div>
          <span className="font-mono text-xs uppercase tracking-widest text-bone/70 group-hover:text-acid transition-colors">
            Shubham/Dwivedi
          </span>
        </button>

        <div className="hidden md:flex items-center gap-1">
          {items.map((it) => (
            <button
              key={it.id}
              data-cursor="jump"
              onClick={() => onNavigate(it.id)}
              className="px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-widest text-bone/60 hover:text-acid hover:bg-bone/5 transition-colors"
            >
              {it.label}
            </button>
          ))}
        </div>

        <button
          data-cursor="say hi"
          onClick={() => onNavigate('contact')}
          className="relative px-4 py-2 rounded-full bg-acid text-ink font-display uppercase tracking-wider text-sm hover:bg-bone transition-colors"
        >
          Let's talk
        </button>
      </div>
    </motion.nav>
  );
};

export default Navbar;
