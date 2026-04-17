import { motion } from 'framer-motion';
import { useMemo, ReactNode } from 'react';

interface Props {
  text: string;
  as?: 'h1' | 'h2' | 'h3' | 'p' | 'span' | 'div';
  className?: string;
  stagger?: number;
  delay?: number;
  once?: boolean;
  children?: ReactNode;
}

const TextReveal = ({
  text,
  as = 'h2',
  className = '',
  stagger = 0.035,
  delay = 0,
  once = true,
}: Props) => {
  const words = useMemo(() => text.split(' '), [text]);
  const Tag: any = motion[as];

  return (
    <Tag
      initial="hidden"
      whileInView="visible"
      viewport={{ once, margin: '-10% 0px' }}
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: stagger, delayChildren: delay } },
      }}
      className={className}
      style={{ display: 'inline-block' }}
    >
      {words.map((w, i) => (
        <span
          key={i}
          style={{
            display: 'inline-block',
            overflow: 'hidden',
            verticalAlign: 'bottom',
            marginRight: '0.28em',
          }}
        >
          <motion.span
            variants={{
              hidden: { y: '110%', rotate: 6, opacity: 0 },
              visible: {
                y: '0%',
                rotate: 0,
                opacity: 1,
                transition: { duration: 0.85, ease: [0.16, 1, 0.3, 1] },
              },
            }}
            style={{ display: 'inline-block' }}
          >
            {w}
          </motion.span>
        </span>
      ))}
    </Tag>
  );
};

export default TextReveal;
