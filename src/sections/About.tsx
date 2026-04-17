import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { Brain, Database, Code2, Zap, Sparkles } from 'lucide-react';
import TextReveal from '../components/TextReveal';
import AnimatedCounter from '../components/AnimatedCounter';

const skills = [
  { icon: Database, title: 'Data Engineering', desc: 'Spark · Airflow · Kafka · dbt. Pipelines that don\u2019t flinch at a petabyte.' },
  { icon: Brain, title: 'Machine Learning', desc: 'MLOps · feature stores · agentic RAG. Models that actually ship.' },
  { icon: Code2, title: 'Systems & APIs', desc: 'Python · FastAPI · Django · Go. Distributed services with tight SLOs.' },
  { icon: Zap, title: 'Cloud & DevOps', desc: 'AWS · GCP · K8s · Terraform. Infra as ergonomic code.' },
];

const stats = [
  { label: 'Years shipping data', to: 5, suffix: '+' },
  { label: 'Pipelines in prod', to: 48, suffix: '' },
  { label: 'Terabytes moved / wk', to: 120, suffix: 'TB' },
  { label: 'Coffees / day', to: 4.2, suffix: '', format: (n: number) => n.toFixed(1) },
];

const About = () => {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });
  const wordY = useTransform(scrollYProgress, [0, 1], ['-20%', '20%']);

  return (
    <section
      ref={ref as any}
      id="about"
      className="relative py-32 md:py-40 bg-ink overflow-hidden"
    >
      <div className="absolute inset-0 grid-lines opacity-50 pointer-events-none" />

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-16">
        {/* Section tag */}
        <div className="flex items-center gap-4 mb-12 font-mono text-xs uppercase tracking-[0.3em] text-bone/50">
          <span className="w-2 h-2 rotate-45 bg-acid" />
          / 02 — About
          <span className="flex-1 h-px bg-bone/10" />
          <span className="text-bone/30">Portrait of a builder</span>
        </div>

        {/* Huge kinetic statement */}
        <div className="relative mb-20 md:mb-28">
          <motion.div
            aria-hidden
            style={{ y: wordY }}
            className="absolute -top-10 right-0 font-display uppercase text-bone/[0.04] text-[16vw] leading-none pointer-events-none"
          >
            architect
          </motion.div>

          <TextReveal
            as="h2"
            text="I don't decorate dashboards. I wire the nervous system that powers them."
            className="font-display uppercase text-bone leading-[0.92] tracking-crushed text-[clamp(2.5rem,7vw,7rem)] max-w-[18ch]"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="mt-8 text-bone/65 text-lg md:text-xl max-w-2xl"
          >
            From raw event streams to LLM agents in production — I design, build, and babysit the full stack that turns data into decisions.
          </motion.p>
        </div>

        {/* Portrait + narrative */}
        <div className="grid lg:grid-cols-12 gap-10 md:gap-16 mb-24">
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-5 relative"
            data-cursor="that's me"
          >
            <div className="relative aspect-[4/5] overflow-hidden rounded-3xl border border-bone/10 group">
              <img
                src="/about-portrait.jpg"
                alt="Shubham Dwivedi"
                className="absolute inset-0 w-full h-full object-cover grayscale contrast-125 group-hover:grayscale-0 group-hover:contrast-100 transition-all duration-700"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/10 to-transparent" />
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.3em] text-bone/70">
                <span>EST. 2019</span>
                <span>Berlin, DE</span>
              </div>
              <div className="absolute bottom-5 left-5 right-5">
                <div className="flex items-center gap-2 text-acid font-mono text-[10px] uppercase tracking-[0.3em] mb-1">
                  <Sparkles className="w-3 h-3" />
                  Subject
                </div>
                <div className="font-display uppercase text-bone text-3xl leading-none tracking-tighter">
                  Shubham<br />Dwivedi
                </div>
              </div>
            </div>
            <div className="absolute -bottom-6 -right-6 w-32 h-32 rounded-full bg-acid text-ink grid place-items-center font-display uppercase text-center text-sm leading-tight rotate-12 animate-spin-slow">
              <span className="sr-only">Available for hire</span>
              <span aria-hidden>
                Available<br />for hire ✸
              </span>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="lg:col-span-7 flex flex-col gap-8"
          >
            <p className="text-bone/80 text-xl leading-relaxed">
              I'm a Data Engineer & AI Architect based in <span className="text-acid">Germany</span>. I've spent the last five years moving data that doesn't want to be moved and making models that actually do the job in production.
            </p>
            <p className="text-bone/65 text-lg leading-relaxed">
              At <span className="text-bone">Metamatics</span> I ship multi-agent systems for market intelligence. At <span className="text-bone">PERI GmbH</span> I rebuilt predictive modeling pipelines. At <span className="text-bone">Quantizant</span> I handled computer vision at the edge.
            </p>
            <p className="text-bone/65 text-lg leading-relaxed">
              MSc Data Analytics from <span className="text-violet">University of Hildesheim</span>. B.Tech CS from DIT University. I read papers on Sundays and break prod on Thursdays — never the other way around.
            </p>

            {/* live stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-bone/10">
              {stats.map((s) => (
                <div key={s.label}>
                  <div className="font-display text-4xl md:text-5xl text-bone tracking-tighter">
                    <AnimatedCounter to={s.to} suffix={s.suffix} format={s.format} />
                  </div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40 mt-1">
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* skill chips */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {skills.map((skill, i) => (
            <motion.div
              key={skill.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              whileHover={{ y: -6 }}
              data-cursor={skill.title.toLowerCase()}
              className="group relative bg-bone/[0.02] border border-bone/10 hover:border-acid rounded-2xl p-6 transition-colors overflow-hidden"
            >
              <div className="absolute -top-8 -right-8 w-32 h-32 rounded-full bg-acid/5 group-hover:bg-acid/10 blur-2xl transition-all" />
              <skill.icon className="w-8 h-8 text-bone/60 group-hover:text-acid transition-colors mb-5" />
              <h3 className="font-display uppercase text-2xl text-bone mb-2 tracking-tight">
                {skill.title}
              </h3>
              <p className="text-bone/55 text-sm leading-relaxed">{skill.desc}</p>
              <div className="mt-5 font-mono text-[10px] uppercase tracking-[0.3em] text-bone/30 group-hover:text-acid transition-colors flex items-center gap-2">
                <span className="block w-6 h-px bg-current" />
                Discipline · 0{i + 1}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default About;
