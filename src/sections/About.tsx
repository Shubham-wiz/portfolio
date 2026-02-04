import { motion } from 'framer-motion';
import { Code2, Database, Brain, Zap } from 'lucide-react';

const About = () => {
  const skills = [
    { icon: Database, title: "Data Engineering", desc: "Apache Spark, Airflow, ETL pipelines at scale" },
    { icon: Brain, title: "Machine Learning", desc: "MLOps, model deployment, agentic AI systems" },
    { icon: Code2, title: "Backend Development", desc: "Python, Django, FastAPI, distributed systems" },
    { icon: Zap, title: "Cloud & DevOps", desc: "AWS, Kubernetes, Docker, CI/CD automation" },
  ];

  return (
    <section id="about" className="relative py-32 bg-dark-card overflow-hidden">
      <div className="absolute inset-0 opacity-[0.03]">
        <div className="w-full h-full" style={{ backgroundImage: `linear-gradient(rgba(167, 139, 250, 0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(167, 139, 250, 0.5) 1px, transparent 1px)`, backgroundSize: '60px 60px' }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-7xl font-bold mb-6 text-gradient">About Me</h2>
          <p className="text-light-gray text-lg md:text-xl max-w-3xl mx-auto">
            Transforming complex data challenges into scalable, production-ready solutions
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center mb-20">
          <motion.div initial={{ opacity: 0, x: -50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} className="space-y-6">
            <p className="text-light-gray text-lg leading-relaxed">
              I'm a Data Engineer and AI Architect with a passion for building intelligent systems that process and analyze data at scale. Currently based in Germany, I specialize in creating robust data pipelines, deploying machine learning models, and developing AI-driven automation solutions.
            </p>
            <p className="text-light-gray text-lg leading-relaxed">
              With experience at companies like <span className="text-lime font-semibold">Metamatics</span>, <span className="text-lime font-semibold">PERI GmbH</span>, and <span className="text-lime font-semibold">Quantizant</span>, I've developed multi-agent AI systems, optimized ETL workflows processing terabyte-scale datasets, and implemented MLOps pipelines that drive measurable business impact.
            </p>
            <p className="text-light-gray text-lg leading-relaxed">
              I hold a Master's degree in Data Analytics from <span className="text-purple-500 font-semibold">University of Hildesheim</span> and a B.Tech in Computer Science from DIT University, India. My expertise spans AI process automation, predictive modeling, computer vision, and cloud-based data engineering.
            </p>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.9 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }} className="relative">
            <img src="/about-portrait.jpg" alt="Shubham Dwivedi" className="rounded-2xl border border-light-gray/20 w-full" />
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-purple-500/20 to-lime/20" />
          </motion.div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {skills.map((skill, index) => (
            <motion.div
              key={skill.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="bg-dark-bg border border-light-gray/10 rounded-xl p-6 hover:border-lime/30 transition-all duration-300 group"
            >
              <skill.icon className="w-12 h-12 text-lime mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="font-display text-xl font-bold mb-2">{skill.title}</h3>
              <p className="text-light-gray/70 text-sm">{skill.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default About;
