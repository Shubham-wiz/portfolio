import { useEffect, useRef, useState } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

const CustomCursor = () => {
  const x = useMotionValue(-100);
  const y = useMotionValue(-100);
  const sx = useSpring(x, { damping: 28, stiffness: 420, mass: 0.4 });
  const sy = useSpring(y, { damping: 28, stiffness: 420, mass: 0.4 });
  const ringX = useSpring(x, { damping: 22, stiffness: 140, mass: 0.8 });
  const ringY = useSpring(y, { damping: 22, stiffness: 140, mass: 0.8 });

  const [label, setLabel] = useState<string>('');
  const [hover, setHover] = useState(false);
  const [visible, setVisible] = useState(false);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      x.set(e.clientX);
      y.set(e.clientY);
      if (!visible) setVisible(true);
    };

    const handleOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const el = target.closest<HTMLElement>('[data-cursor]');
      if (el) {
        setHover(true);
        setLabel(el.dataset.cursor || '');
      } else {
        setHover(false);
        setLabel('');
      }
    };

    const handleLeave = () => setVisible(false);

    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseover', handleOver);
    document.addEventListener('mouseleave', handleLeave);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseover', handleOver);
      document.removeEventListener('mouseleave', handleLeave);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [x, y, visible]);

  return (
    <>
      <motion.div
        aria-hidden
        className="fixed top-0 left-0 pointer-events-none z-[100000] mix-blend-difference"
        style={{ x: sx, y: sy, translateX: '-50%', translateY: '-50%', opacity: visible ? 1 : 0 }}
      >
        <div className="w-2 h-2 rounded-full bg-white" />
      </motion.div>
      <motion.div
        aria-hidden
        className="fixed top-0 left-0 pointer-events-none z-[99998]"
        style={{
          x: ringX,
          y: ringY,
          translateX: '-50%',
          translateY: '-50%',
          opacity: visible ? 1 : 0,
        }}
      >
        <motion.div
          animate={{
            scale: hover ? 3.2 : 1,
            borderColor: hover ? 'rgba(198, 255, 61, 1)' : 'rgba(244, 241, 234, 0.45)',
            backgroundColor: hover ? 'rgba(198, 255, 61, 0.12)' : 'rgba(244, 241, 234, 0)',
          }}
          transition={{ type: 'spring', damping: 20, stiffness: 220 }}
          className="w-9 h-9 rounded-full border"
        />
        {label && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-2 py-1 bg-acid text-ink text-[10px] font-mono tracking-widest uppercase whitespace-nowrap rounded"
          >
            {label}
          </motion.div>
        )}
      </motion.div>
    </>
  );
};

export default CustomCursor;
