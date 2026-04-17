import { ReactNode } from 'react';

interface Props {
  items: string[];
  speed?: number; // seconds per loop
  className?: string;
  separator?: ReactNode;
  reverse?: boolean;
}

const Marquee = ({ items, speed = 38, className = '', separator, reverse = false }: Props) => {
  const loop = [...items, ...items];
  return (
    <div className={`relative overflow-hidden w-full ${className}`}>
      <div
        className="flex gap-10 whitespace-nowrap will-change-transform"
        style={{
          animation: `marquee ${speed}s linear infinite`,
          animationDirection: reverse ? 'reverse' : 'normal',
        }}
      >
        {loop.map((item, i) => (
          <span key={i} className="flex items-center gap-10 shrink-0">
            {item}
            <span className="opacity-40">{separator ?? '✦'}</span>
          </span>
        ))}
      </div>
    </div>
  );
};

export default Marquee;
