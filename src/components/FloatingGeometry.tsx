import { motion } from 'framer-motion';

const FloatingGeometry = () => {
  const shapes = [
    { size: 400, color: '#a78bfa', top: '10%', left: '5%', duration: 20, delay: 0 },
    { size: 350, color: '#d1e29d', top: '60%', right: '10%', duration: 25, delay: 2 },
    { size: 380, color: '#f472b6', top: '30%', right: '5%', duration: 22, delay: 4 },
    { size: 320, color: '#60a5fa', top: '75%', left: '15%', duration: 18, delay: 1 },
    { size: 360, color: '#4ade80', top: '45%', left: '50%', duration: 24, delay: 3 },
    { size: 300, color: '#a78bfa', top: '85%', right: '20%', duration: 23, delay: 2.5 },
    { size: 340, color: '#f472b6', top: '5%', right: '30%', duration: 21, delay: 1.5 },
    { size: 290, color: '#60a5fa', top: '50%', left: '80%', duration: 19, delay: 3.5 },
  ];

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 1 }}>
      {shapes.map((shape, index) => (
        <motion.div
          key={index}
          className="absolute"
          style={{
            width: shape.size,
            height: shape.size,
            top: shape.top,
            left: shape.left,
            right: shape.right,
          }}
          animate={{
            y: [0, -30, 0],
            rotate: [0, 360],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: shape.duration,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: shape.delay,
          }}
        >
          {/* Glowing orb - very faded */}
          <div
            className="w-full h-full rounded-full opacity-[0.02] blur-3xl"
            style={{
              background: `radial-gradient(circle, ${shape.color}, transparent)`,
            }}
          />
          
          {/* Wireframe Cube - very faded */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transformStyle: 'preserve-3d',
            }}
            animate={{
              rotateX: [0, 360],
              rotateY: [0, 360],
            }}
            transition={{
              duration: shape.duration * 1.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            <div
              className="border-2 opacity-[0.03]"
              style={{
                width: shape.size / 3,
                height: shape.size / 3,
                borderColor: shape.color,
                transform: 'translateZ(50px)',
              }}
            />
          </motion.div>
          
          {/* Rotating ring - very subtle */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transformStyle: 'preserve-3d',
            }}
            animate={{
              rotateZ: [0, 360],
            }}
            transition={{
              duration: shape.duration * 2,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            <div
              className="border border-dashed rounded-full opacity-[0.04]"
              style={{
                width: shape.size / 2,
                height: shape.size / 2,
                borderColor: shape.color,
              }}
            />
          </motion.div>
        </motion.div>
      ))}
    </div>
  );
};

export default FloatingGeometry;
