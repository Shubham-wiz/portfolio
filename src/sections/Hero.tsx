import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { useEffect, useState } from 'react';
import ParticleBackground from '../components/ParticleBackground';

const Hero = ({ onNavigate }: { onNavigate: (section: string) => void }) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // More dynamic letter animations with stagger and 3D effects
  const nameVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.2
      }
    }
  };

  const letterVariants = {
    hidden: { 
      opacity: 0, 
      y: 100,
      rotateX: -90,
      scale: 0.5,
    },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      rotateX: 0,
      scale: 1,
      transition: {
        duration: 0.8,
        delay: i * 0.05,
        ease: [0.16, 1, 0.3, 1],
        type: "spring",
        stiffness: 100,
        damping: 12
      }
    })
  };

  const glowVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: [0, 1, 0.8, 1],
      transition: {
        duration: 2,
        ease: "easeInOut",
        repeat: Infinity,
        repeatType: "reverse" as const
      }
    }
  };

  const name = "Shubham Dwivedi";
  const title = "Data Engineer & AI Architect";

  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center overflow-hidden bg-dark-bg">
      {/* Particle Background */}
      <ParticleBackground />
      
      {/* Dynamic gradient background that follows mouse */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute w-[600px] h-[600px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(167, 139, 250, 0.15) 0%, transparent 70%)',
            filter: 'blur(80px)',
          }}
          animate={{
            x: mousePosition.x - 300,
            y: mousePosition.y - 300,
          }}
          transition={{ type: "spring", damping: 50, stiffness: 100 }}
        />
        <motion.div
          className="absolute w-[500px] h-[500px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(209, 226, 157, 0.1) 0%, transparent 70%)',
            filter: 'blur(60px)',
          }}
          animate={{
            x: mousePosition.x - 250,
            y: mousePosition.y - 250,
          }}
          transition={{ type: "spring", damping: 30, stiffness: 80, delay: 0.1 }}
        />
      </div>

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 opacity-[0.03]">
        <div
          className="w-full h-full"
          style={{
            backgroundImage: `
              linear-gradient(rgba(167, 139, 250, 0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(167, 139, 250, 0.5) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px'
          }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-6"
        >
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="inline-block px-4 py-2 rounded-full border border-lime/20 bg-lime/5 mb-8"
          >
            <span className="text-lime text-sm font-mono">
              ✨ Available for consulting & collaboration
            </span>
          </motion.div>
        </motion.div>

        {/* Dynamic animated name with 3D perspective */}
        <div className="perspective-1000 mb-8">
          <motion.h1
            className="font-display text-5xl sm:text-7xl md:text-8xl lg:text-9xl font-bold mb-6"
            variants={nameVariants}
            initial="hidden"
            animate="visible"
            style={{ transformStyle: 'preserve-3d' }}
          >
            {name.split('').map((char, index) => (
              <motion.span
                key={index}
                custom={index}
                variants={letterVariants}
                className="inline-block"
                style={{
                  transformOrigin: 'center bottom',
                  transformStyle: 'preserve-3d',
                }}
                whileHover={{
                  y: -10,
                  scale: 1.2,
                  rotateZ: Math.random() * 20 - 10,
                  transition: { duration: 0.3 }
                }}
              >
                <span
                  className="relative inline-block text-gradient"
                  style={{
                    background: `linear-gradient(135deg, #a78bfa ${index * 3}%, #d1e29d ${100 - index * 2}%)`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}
                >
                  {char === ' ' ? '\u00A0' : char}
                  {/* Glow effect */}
                  <motion.span
                    variants={glowVariants}
                    initial="hidden"
                    animate="visible"
                    className="absolute inset-0 blur-xl"
                    style={{
                      background: `linear-gradient(135deg, #a78bfa ${index * 3}%, #d1e29d ${100 - index * 2}%)`,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      zIndex: -1
                    }}
                  >
                    {char === ' ' ? '\u00A0' : char}
                  </motion.span>
                </span>
              </motion.span>
            ))}
          </motion.h1>
        </div>

        {/* Animated title */}
        <motion.h2
          className="font-display text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-semibold mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          {title.split('').map((char, index) => (
            <motion.span
              key={index}
              initial={{ opacity: 0, y: 100, rotateX: -90 }}
              animate={{ opacity: 1, y: 0, rotateX: 0 }}
              transition={{
                duration: 0.6,
                delay: 1.2 + index * 0.04,
                ease: [0.16, 1, 0.3, 1]
              }}
              className="inline-block text-gradient glow-text"
              style={{ transformOrigin: 'center bottom' }}
            >
              {char === ' ' ? '\u00A0' : char}
            </motion.span>
          ))}
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.8, duration: 0.8 }}
          className="text-light-gray text-base sm:text-lg md:text-xl max-w-3xl mx-auto mb-12 leading-relaxed"
        >
          Building scalable data pipelines and intelligent AI systems that transform raw data into actionable insights. 
          Specializing in MLOps, distributed computing, and agentic AI architectures.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2.2, duration: 0.8 }}
          className="flex flex-wrap gap-4 justify-center"
        >
          <button
            onClick={() => onNavigate('projects')}
            className="group relative px-8 py-4 bg-gradient-to-r from-purple-500 to-lime text-dark-bg font-semibold rounded-full overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/50"
          >
            <span className="relative z-10">Explore My Work →</span>
            <div className="absolute inset-0 bg-gradient-to-r from-lime to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          </button>
          
          <button
            onClick={() => onNavigate('contact')}
            className="px-8 py-4 border-2 border-lime/30 text-lime font-semibold rounded-full hover:bg-lime/10 hover:border-lime transition-all duration-300 hover:scale-105"
          >
            Get in Touch
          </button>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.8 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10"
      >
        <motion.div
          animate={{ y: [0, 12, 0] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="flex flex-col items-center gap-2 text-light-gray/50 cursor-pointer hover:text-lime transition-colors"
          onClick={() => onNavigate('about')}
        >
          <span className="text-xs font-mono">SCROLL</span>
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </motion.div>

      {/* Location badge */}
      <div className="absolute bottom-8 left-8 z-10">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 2.5 }}
          className="font-mono text-xs text-light-gray/60 text-left"
        >
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 bg-lime rounded-full animate-pulse" />
            <span>GERMANY</span>
          </div>
          <div>51.1657° N, 10.4515° E</div>
        </motion.div>
      </div>

      {/* System info badge */}
      <div className="absolute top-8 right-8 z-10">
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 2.7 }}
          className="font-mono text-xs text-lime/60 text-right"
        >
          <div>SYSTEM ONLINE</div>
          <div className="text-light-gray/40">v2.0.0</div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
