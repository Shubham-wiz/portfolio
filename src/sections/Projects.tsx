import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import { ArrowUpRight, X, TrendingUp, Code2, Layers } from 'lucide-react';
import { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { projects, Project } from '../data/projects-data';
import TextReveal from '../components/TextReveal';
import CodeBlock from '../components/CodeBlock';

const Projects = () => {
  const [selected, setSelected] = useState<Project | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [maxShift, setMaxShift] = useState(0);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end end'],
  });

  // Measure actual horizontal distance we need to travel so the pinned
  // scroll ends exactly when the last tile reaches the right edge.
  useLayoutEffect(() => {
    const update = () => {
      if (!trackRef.current) return;
      const trackW = trackRef.current.scrollWidth;
      const vw = window.innerWidth;
      setMaxShift(Math.max(0, trackW - vw + 64));
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  const x = useTransform(scrollYProgress, [0, 1], [0, -maxShift]);

  // Vertical scroll distance ≈ horizontal distance for 1:1 feel,
  // capped so there's no "dead" scroll at the end.
  const sectionHeight = `calc(100vh + ${maxShift}px)`;

  return (
    <section id="work" className="relative bg-ink">
      {/* Header band */}
      <div className="relative max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-16 pt-28 pb-10">
        <div className="flex items-center gap-4 mb-10 font-mono text-xs uppercase tracking-[0.3em] text-bone/50">
          <span className="w-2 h-2 rotate-45 bg-acid" />
          / 03 — Selected Work
          <span className="flex-1 h-px bg-bone/10" />
          <span className="text-bone/30">Case files · 2021 — now</span>
        </div>

        <div className="flex flex-col lg:flex-row justify-between gap-10 items-end">
          <TextReveal
            as="h2"
            text="Systems that ship."
            className="font-display uppercase text-bone leading-[0.88] tracking-crushed text-[clamp(3rem,10vw,10rem)]"
          />
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-md text-bone/65 text-lg"
          >
            A sample of production work — scroll horizontally. Click a case file to dig into architecture, metrics and actual code.
          </motion.div>
        </div>
      </div>

      {/* Horizontal pinned scroll container */}
      <div ref={ref} style={{ height: sectionHeight }} className="relative">
        <div className="sticky top-0 h-screen overflow-hidden flex items-center">
          <motion.div
            ref={trackRef}
            style={{ x }}
            className="flex gap-6 md:gap-10 pl-6 md:pl-16 pr-6 md:pr-16 will-change-transform"
          >
            {projects.map((p, i) => (
              <ProjectCard
                key={p.id}
                project={p}
                index={i}
                onClick={() => setSelected(p)}
              />
            ))}

            {/* final tile — slimmer so it doesn't eat a full card of scroll */}
            <div className="w-[60vw] md:w-[40vw] h-[75vh] shrink-0 flex items-center justify-center">
              <div className="text-center">
                <div className="font-mono text-[11px] uppercase tracking-[0.35em] text-bone/40 mb-3">
                  End of reel
                </div>
                <div className="font-display uppercase text-bone text-6xl md:text-8xl tracking-crushed leading-[0.88]">
                  more<br />soon.
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* floating scroll hint */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 font-mono text-[10px] uppercase tracking-[0.3em] text-bone/40 flex items-center gap-3 pointer-events-none">
          <span>Drag page down</span>
          <span className="block w-8 h-px bg-bone/30" />
          <span>Horizontal scroll</span>
        </div>
      </div>

      <ProjectModal project={selected} onClose={() => setSelected(null)} />
    </section>
  );
};

const ProjectCard = ({
  project,
  index,
  onClick,
}: {
  project: Project;
  index: number;
  onClick: () => void;
}) => {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ y: -8 }}
      data-cursor="open case"
      className="relative w-[85vw] md:w-[70vw] h-[75vh] shrink-0 rounded-3xl overflow-hidden border border-bone/10 bg-dark-card text-left group"
    >
      <div className="absolute inset-0">
        <img
          src={project.image}
          alt={project.title}
          className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-700"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/40 to-transparent" />
        <div
          className="absolute inset-0 opacity-30 mix-blend-multiply"
          style={{ background: `linear-gradient(135deg, ${project.color}00 30%, ${project.color}60 100%)` }}
        />
      </div>

      {/* index */}
      <div className="absolute top-6 left-6 font-mono text-[11px] uppercase tracking-[0.3em] text-bone/70">
        Case · {String(index + 1).padStart(2, '0')}
      </div>
      <div className="absolute top-6 right-6 flex items-center gap-3">
        <span
          className="px-3 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.2em]"
          style={{ backgroundColor: `${project.color}25`, color: project.color, border: `1px solid ${project.color}50` }}
        >
          {project.tags[0]}
        </span>
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-8 md:p-12">
        <h3 className="font-display uppercase text-bone text-4xl md:text-7xl lg:text-8xl leading-[0.88] tracking-crushed mb-6">
          {project.title}
        </h3>
        <p className="text-bone/75 text-base md:text-lg max-w-2xl line-clamp-2 mb-6">
          {project.description}
        </p>
        <div className="flex items-center gap-4">
          <div className="flex flex-wrap gap-2">
            {project.tags.slice(0, 4).map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        </div>
        <div className="mt-8 flex items-center justify-between">
          <div className="font-mono text-[11px] uppercase tracking-[0.3em] text-bone/50">
            {project.metrics.slice(0, 2).map((m) => (
              <span key={m.label} className="mr-6">
                <span className="text-bone">{m.value}</span> · {m.label}
              </span>
            ))}
          </div>
          <motion.div
            whileHover={{ rotate: 45 }}
            className="w-14 h-14 rounded-full bg-acid text-ink grid place-items-center"
          >
            <ArrowUpRight className="w-6 h-6" />
          </motion.div>
        </div>
      </div>
    </motion.button>
  );
};

// --- Modal restyled with new aesthetic + collapsible code blocks ---
const ProjectModal = ({ project, onClose }: { project: Project | null; onClose: () => void }) => {
  useEffect(() => {
    if (!project) return;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', onKey);
    };
  }, [project, onClose]);

  return createPortal(
    <AnimatePresence>
      {project && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          data-lenis-prevent
          className="fixed inset-0 bg-ink/85 backdrop-blur-lg z-[10001] flex items-start md:items-center justify-center p-0 md:p-6"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0, y: 30 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.96, opacity: 0, y: 30 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="relative bg-ink border border-bone/15 rounded-none md:rounded-3xl w-full md:max-w-5xl md:max-h-[92vh] h-screen md:h-auto overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={onClose}
              data-cursor="close"
              className="absolute top-4 right-4 md:top-6 md:right-6 z-30 w-11 h-11 rounded-full bg-ink/80 backdrop-blur border border-bone/20 text-bone hover:bg-acid hover:text-ink hover:border-acid grid place-items-center transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="overflow-y-auto flex-1">
              {/* Hero */}
              <div className="relative h-64 md:h-80">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover grayscale"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/50 to-ink/10" />
                <div
                  className="absolute inset-0 mix-blend-multiply opacity-40"
                  style={{
                    background: `linear-gradient(135deg, ${project.color}00 30%, ${project.color}90 100%)`,
                  }}
                />
                <div className="absolute bottom-0 left-0 right-0 p-6 md:p-10">
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.tags.map((t) => (
                      <span
                        key={t}
                        className="chip"
                        style={{ borderColor: `${project.color}50`, color: project.color, backgroundColor: `${project.color}10` }}
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                  <h1 className="font-display uppercase text-bone text-3xl md:text-6xl tracking-crushed leading-[0.9]">
                    {project.title}
                  </h1>
                </div>
              </div>

              <div className="px-6 md:px-12 py-10 md:py-14 max-w-4xl mx-auto">
                {/* Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-14">
                  {project.metrics.map((m) => (
                    <div
                      key={m.label}
                      className="bg-bone/[0.03] border border-bone/10 rounded-2xl p-4 text-center"
                    >
                      <div
                        className="font-display text-3xl md:text-4xl tracking-crushed"
                        style={{ color: project.color }}
                      >
                        {m.value}
                      </div>
                      <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/50 mt-2">
                        {m.label}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Overview */}
                <ModalSection
                  title="Overview"
                  icon={<TrendingUp className="w-5 h-5" style={{ color: project.color }} />}
                >
                  <p className="text-bone/75 text-lg leading-[1.8] whitespace-pre-line">
                    {project.detailedDescription}
                  </p>
                </ModalSection>

                {/* Architecture */}
                <ModalSection
                  title="Architecture"
                  icon={<Layers className="w-5 h-5" style={{ color: project.color }} />}
                >
                  <CodeBlock
                    code={project.architecture}
                    language="diagram"
                    title="System diagram"
                  />
                </ModalSection>

                {/* Key Features */}
                <ModalSection title="Key features">
                  <div className="grid md:grid-cols-2 gap-3">
                    {project.keyFeatures.map((f, i) => (
                      <div
                        key={i}
                        className="flex gap-3 bg-bone/[0.03] border border-bone/10 rounded-xl p-4"
                      >
                        <span className="text-acid text-xl leading-none shrink-0">▹</span>
                        <span className="text-bone/80 leading-relaxed">{f}</span>
                      </div>
                    ))}
                  </div>
                </ModalSection>

                {/* Implementation / code */}
                <ModalSection
                  title="Implementation"
                  icon={<Code2 className="w-5 h-5" style={{ color: project.color }} />}
                >
                  <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40 mb-3">
                    {project.codeSnippets.length} snippet{project.codeSnippets.length !== 1 ? 's' : ''} · click any to expand
                  </div>
                  {project.codeSnippets.map((s, i) => (
                    <CodeBlock
                      key={i}
                      code={s.code}
                      language={s.language}
                      title={s.title}
                    />
                  ))}
                </ModalSection>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
};

const ModalSection = ({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) => (
  <section className="mb-12">
    <h2 className="font-display uppercase text-bone text-2xl md:text-3xl tracking-crushed mb-5 flex items-center gap-3">
      {icon}
      {title}
    </h2>
    {children}
  </section>
);

export default Projects;
