import { useRef, ReactNode } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

interface Tilt3DProps {
  children: ReactNode;
  className?: string;
  tiltMaxAngle?: number;
}

const Tilt3D = ({ children, className = '', tiltMaxAngle = 15 }: Tilt3DProps) => {
  const ref = useRef<HTMLDivElement>(null);

  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x);
  const mouseYSpring = useSpring(y);

  const rotateX = useTransform(
    mouseYSpring,
    [-0.5, 0.5],
    [`${tiltMaxAngle}deg`, `-${tiltMaxAngle}deg`]
  );
  const rotateY = useTransform(
    mouseXSpring,
    [-0.5, 0.5],
    [`-${tiltMaxAngle}deg`, `${tiltMaxAngle}deg`]
  );

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;

    const rect = ref.current.getBoundingClientRect();

    const width = rect.width;
    const height = rect.height;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;

    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateY,
        rotateX,
        transformStyle: 'preserve-3d',
        pointerEvents: 'auto',
      }}
      className={className}
    >
      <div style={{ transform: 'translateZ(75px)', transformStyle: 'preserve-3d', pointerEvents: 'auto' }}>
        {children}
      </div>
    </motion.div>
  );
};

export default Tilt3D;
