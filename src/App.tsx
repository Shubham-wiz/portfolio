import { useEffect, useRef } from 'react';
import Lenis from 'lenis';
import Hero from './sections/Hero';
import About from './sections/About';
import Projects from './sections/Projects';
import WebDemos from './sections/WebDemos';
import Experience from './sections/Experience';
import Blog from './sections/Blog';
import Contact from './sections/Contact';
import CustomCursor from './components/CustomCursor';
import GrainOverlay from './components/GrainOverlay';
import ScrollProgress from './components/ScrollProgress';
import Navbar from './components/Navbar';
import Marquee from './components/Marquee';

function App() {
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    lenisRef.current = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      touchMultiplier: 2,
    });

    function raf(time: number) {
      lenisRef.current?.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return () => {
      lenisRef.current?.destroy();
    };
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element && lenisRef.current) {
      lenisRef.current.scrollTo(element, { offset: 0, duration: 1.5 });
    }
  };

  return (
    <div className="relative bg-ink">
      <CustomCursor />
      <GrainOverlay />
      <ScrollProgress />
      <Navbar onNavigate={scrollToSection} />

      <Hero onNavigate={scrollToSection} />

      <InterstitialMarquee
        items={[
          'DATA ENGINEERING',
          'AI ARCHITECTURES',
          'MULTI-AGENT SYSTEMS',
          'SPARK · AIRFLOW · K8s',
          'LANGCHAIN · LLAMA',
          'PRODUCTION MLOps',
        ]}
      />

      <About />

      <InterstitialMarquee
        reverse
        items={[
          'CASE FILES',
          'SHIPPED PRODUCTS',
          'PRODUCTION SYSTEMS',
          'FROM PIPELINE TO PRODUCT',
          'SCROLL →',
        ]}
      />

      <Projects />

      <InterstitialMarquee
        items={[
          'LIVE DEMOS',
          'IN YOUR BROWSER',
          'NO SERVER',
          'MACHINE LEARNING · CREATIVE CODING',
          'CLICK · DRAG · TRAIN',
        ]}
      />

      <WebDemos />

      <Experience />

      <Blog />

      <Contact />
    </div>
  );
}

const InterstitialMarquee = ({
  items,
  reverse,
}: {
  items: string[];
  reverse?: boolean;
}) => (
  <div className="relative border-y border-bone/10 bg-ink py-6 overflow-hidden">
    <Marquee
      items={items}
      speed={52}
      reverse={reverse}
      separator={<span>✦</span>}
      className="font-display uppercase text-bone tracking-tighter text-3xl md:text-5xl"
    />
  </div>
);

export default App;
