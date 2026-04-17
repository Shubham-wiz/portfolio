import { motion, AnimatePresence } from 'framer-motion';
import { Briefcase, MapPin, X, TrendingUp, AlertTriangle, CheckCircle2, Users, Layers } from 'lucide-react';
import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import TextReveal from '../components/TextReveal';

interface Experience {
  id: string;
  role: string;
  company: string;
  location: string;
  period: string;
  description: string[];
  highlights: { label: string; value: string }[];
  color: string;
  tagline: string;
  keyTech: string[];
  topAchievements: string[];
  detailedDescription?: string;
  allTechnologies?: string[];
  challenges?: string[];
  outcomes?: string[];
  teamInfo?: string;
}

const experiences: Experience[] = [
  {
    id: "metamatics",
    role: "Generative AI Developer",
    company: "Metamatics",
    location: "Remote, Germany",
    period: "Dec 2024 - Present",
    tagline: "AI-driven Market Intelligence & Multi-Agent Workflows",
    keyTech: ["LangChain", "FastAPI", "Django", "REST APIs"],
    topAchievements: [
      "Built Market Intelligence platform with multi-agent workflows",
      "Automated enterprise reporting & financial presentations"
    ],
    description: [
      "Architected AI-driven Market Intelligence platform using LangChain with REST API endpoints via FastAPI",
      "Developed automated dashboarding tool (Django + FastAPI) for market and competitor analysis",
      "Enhanced data reliability through automated fact-checking, increasing platform adoption by 40%"
    ],
    highlights: [{ label: "Research Time Saved", value: "70%" }, { label: "Adoption Increase", value: "40%" }],
    color: "#a78bfa",
    detailedDescription: "Led the development of an enterprise-grade Market Intelligence platform that leverages multi-agent AI workflows to automate research, analysis, and reporting. The platform integrates multiple LLM agents that collaborate to gather market data, validate information, and generate comprehensive reports. Implemented sophisticated fact-checking layers using ensemble validation techniques to ensure data reliability.",
    allTechnologies: ["LangChain", "FastAPI", "Django", "REST APIs", "Python", "PostgreSQL", "Redis", "Docker", "GPT-4", "Claude", "React", "TypeScript"],
    challenges: [
      "Ensuring factual accuracy across multi-agent outputs with conflicting information",
      "Optimizing LLM API costs while maintaining response quality and speed",
      "Designing scalable architecture to handle concurrent enterprise users"
    ],
    outcomes: [
      "Reduced manual research time by 70% for market analysts",
      "Achieved 40% increase in platform adoption across business units",
      "Processed 10,000+ automated reports with 95%+ accuracy rate"
    ],
    teamInfo: "Remote team of 4 engineers, collaborated with product managers and data scientists"
  },
  {
    id: "peri-thesis",
    role: "Master Thesis (Computer Vision)",
    company: "PERI GmbH",
    location: "Weißenhorn, Germany",
    period: "Jan 2024 - Aug 2024",
    tagline: "Real-Time Inventory Auditing with Computer Vision",
    keyTech: ["Computer Vision", "OpenCV", "PyTorch", "Real-Time Processing"],
    topAchievements: [
      "98% counting accuracy in inventory auditing system",
      "Improved supply chain visibility & reduced discrepancies"
    ],
    description: [
      "Developed real-time inventory auditing system with computer vision",
      "Achieved 98% counting accuracy, significantly improving supply chain visibility"
    ],
    highlights: [{ label: "Accuracy", value: "98%" }],
    color: "#d1e29d",
    detailedDescription: "Designed and implemented an end-to-end computer vision system for real-time inventory auditing in warehouse environments. The system uses custom-trained object detection models to automatically count and classify inventory items from video feeds, replacing manual counting processes. Developed robust detection algorithms that work under varying lighting conditions and camera angles.",
    allTechnologies: ["PyTorch", "OpenCV", "YOLO", "Python", "TensorFlow", "CUDA", "Docker", "Flask", "PostgreSQL", "NumPy", "Pandas"],
    challenges: [
      "Handling occlusion and overlapping objects in dense warehouse environments",
      "Maintaining accuracy across different lighting conditions and camera positions",
      "Optimizing inference speed for real-time processing on edge devices"
    ],
    outcomes: [
      "Achieved 98% counting accuracy across 50,000+ inventory items",
      "Reduced inventory discrepancies by 85% compared to manual processes",
      "Decreased audit time from 4 hours to 15 minutes per warehouse section"
    ],
    teamInfo: "Independent research project, collaborated with 2 warehouse supervisors and academic advisor"
  },
  {
    id: "peri-intern",
    role: "Data Engineer (Intern)",
    company: "PERI GmbH",
    location: "Weißenhorn, Germany",
    period: "Jul 2023 - Nov 2023",
    tagline: "Predictive Maintenance & NLP Automation on Azure",
    keyTech: ["Azure Databricks", "XGBoost", "LSTM", "BERT", "IoT"],
    topAchievements: [
      "7.5% operational uptime increase via predictive maintenance",
      "60% reduction in manual data entry with NLP automation"
    ],
    description: [
      "Developed predictive maintenance dashboard on Azure Databricks using XGBoost and LSTM",
      "Automated financial data extraction from PDFs using NLP (BERT/ROBERTa) with 95%+ accuracy",
      "Monitored IoT telemetry data, resulting in 20% reduction in operational downtime"
    ],
    highlights: [{ label: "Uptime Increase", value: "7.5%" }, { label: "Manual Entry Reduced", value: "60%" }],
    color: "#f472b6",
    detailedDescription: "Built predictive maintenance infrastructure on Azure Databricks to forecast equipment failures before they occur. Developed NLP pipelines to automatically extract structured financial data from unstructured PDF documents. Implemented real-time IoT monitoring dashboards to track equipment health metrics and trigger maintenance alerts.",
    allTechnologies: ["Azure Databricks", "XGBoost", "LSTM", "BERT", "ROBERTa", "Python", "PySpark", "Azure IoT Hub", "Power BI", "SQL", "Apache Spark", "MLflow"],
    challenges: [
      "Dealing with imbalanced failure data (only 2% failure rate in historical data)",
      "Extracting structured data from highly variable PDF formats across departments",
      "Integrating real-time IoT streams with batch prediction pipelines"
    ],
    outcomes: [
      "Increased operational uptime by 7.5% through early failure detection",
      "Reduced manual data entry by 60% with 95%+ extraction accuracy",
      "Decreased operational downtime by 20% via proactive maintenance"
    ],
    teamInfo: "Worked with 3 senior data engineers and 2 domain experts from operations"
  },
  {
    id: "quantizant",
    role: "Applied Data Analyst",
    company: "Quantizant",
    location: "Remote, Germany",
    period: "Apr 2023 - Jun 2023",
    tagline: "Big Data Processing & MLOps Pipeline Orchestration",
    keyTech: ["PySpark", "Hadoop", "Docker", "MLflow", "GitHub Actions"],
    topAchievements: [
      "30% reduction in computational costs via optimization",
      "Orchestrated MLOps pipelines with continuous training"
    ],
    description: [
      "Leveraged PySpark and Hadoop to process terabyte-scale datasets",
      "Optimized feature engineering, reducing computational costs by 30%",
      "Orchestrated MLOps pipelines using Docker, GitHub Actions, and MLflow"
    ],
    highlights: [{ label: "Cost Reduction", value: "30%" }],
    color: "#60a5fa",
    detailedDescription: "Processed and analyzed terabyte-scale datasets using distributed computing frameworks. Built end-to-end MLOps pipelines with automated model training, validation, and deployment. Optimized Spark jobs to reduce cluster costs while maintaining performance.",
    allTechnologies: ["PySpark", "Hadoop", "Docker", "MLflow", "GitHub Actions", "Python", "Apache Airflow", "AWS S3", "PostgreSQL", "scikit-learn", "Pandas"],
    challenges: [
      "Optimizing Spark jobs to process 5TB+ datasets within budget constraints",
      "Implementing automated feature engineering for high-dimensional data",
      "Setting up CI/CD for machine learning models with proper versioning"
    ],
    outcomes: [
      "Reduced computational costs by 30% through query optimization",
      "Processed 5TB+ of data daily with 99.9% pipeline reliability",
      "Achieved 40% faster model iteration cycles with automated MLOps"
    ],
    teamInfo: "Collaborated with 2 data scientists and DevOps team remotely"
  },
  {
    id: "lytiq",
    role: "Data Research Analyst (Intern)",
    company: "LYTIQ GmbH",
    location: "Düsseldorf, Germany",
    period: "Dec 2020 - Mar 2021",
    tagline: "Ensemble Models for Industrial Fault Detection",
    keyTech: ["Ensemble Models", "Machine Learning", "Classification", "Python"],
    topAchievements: [
      "94% classification accuracy in fault detection",
      "Minimized production bottlenecks in industrial systems"
    ],
    description: [
      "Developed ensemble models for industrial fault detection",
      "Achieved 94% classification accuracy to minimize production bottlenecks"
    ],
    highlights: [{ label: "Accuracy", value: "94%" }],
    color: "#4ade80",
    detailedDescription: "Researched and developed ensemble machine learning models for detecting faults in industrial manufacturing systems. Combined multiple classification algorithms (Random Forest, Gradient Boosting, SVM) to create robust fault detection system. Analyzed sensor data patterns to identify early warning signs of equipment failures.",
    allTechnologies: ["Python", "scikit-learn", "Random Forest", "Gradient Boosting", "SVM", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Jupyter"],
    challenges: [
      "Working with highly imbalanced datasets (normal vs fault cases)",
      "Distinguishing between similar fault patterns across different equipment",
      "Optimizing model performance for real-time industrial environments"
    ],
    outcomes: [
      "Achieved 94% fault classification accuracy across 12 fault types",
      "Reduced production bottlenecks by 35% through early detection",
      "Created reusable framework adopted for multiple production lines"
    ],
    teamInfo: "Research internship supervised by 1 senior data scientist"
  },
  {
    id: "iitk",
    role: "Machine Learning Research Scientist",
    company: "IIT-K",
    location: "India",
    period: "Jan 2020 - Dec 2020",
    tagline: "Smart Grid Cybersecurity & Graph-Based ML Research",
    keyTech: ["Graph ML", "Random Matrix Theory", "SMOTE", "Cybersecurity"],
    topAchievements: [
      "97.5% accuracy in detecting False Data Injection Attacks",
      "Published research in Springer on power system faults"
    ],
    description: [
      "Researched graph-based ML for critical infrastructure cybersecurity",
      "Achieved 97.5% accuracy in detecting False Data Injection Attacks in smart grids",
      "Published research in Springer on power system fault detection"
    ],
    highlights: [{ label: "Detection Accuracy", value: "97.5%" }],
    color: "#a78bfa",
    detailedDescription: "Conducted advanced research on detecting cyberattacks in smart grid systems using graph-based machine learning and Random Matrix Theory. Developed novel algorithms to identify False Data Injection Attacks (FDIA) that could compromise power grid integrity. Published research findings in peer-reviewed Springer journal.",
    allTechnologies: ["Python", "MATLAB", "Graph Neural Networks", "Random Matrix Theory", "SMOTE", "scikit-learn", "NetworkX", "TensorFlow", "NumPy", "SciPy"],
    challenges: [
      "Detecting sophisticated attacks designed to evade traditional anomaly detection",
      "Working with limited labeled attack data in critical infrastructure",
      "Balancing detection accuracy with false positive rates in real-time systems"
    ],
    outcomes: [
      "Achieved 97.5% accuracy in detecting False Data Injection Attacks",
      "Published peer-reviewed research paper in Springer journal",
      "Developed framework adopted by 2 power companies for pilot testing"
    ],
    teamInfo: "Collaborated with 3 PhD researchers and 1 professor in cybersecurity lab"
  }
];

const Experience = () => {
  const [selectedExperience, setSelectedExperience] = useState<Experience | null>(null);

  return (
    <section id="experience" className="relative py-28 md:py-36 bg-ink overflow-hidden">
      <div className="absolute inset-0 grid-lines opacity-30 pointer-events-none" />
      <div className="absolute top-24 -right-24 w-[500px] h-[500px] rounded-full bg-acid/5 blur-[140px] pointer-events-none" />

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-16">
        <div className="flex items-center gap-4 mb-10 font-mono text-xs uppercase tracking-[0.3em] text-bone/50">
          <span className="w-2 h-2 rotate-45 bg-acid" />
          / 05 — Track record
          <span className="flex-1 h-px bg-bone/10" />
          <span className="text-bone/30">2019 — present</span>
        </div>

        <div className="mb-14 flex flex-col lg:flex-row justify-between items-end gap-6">
          <TextReveal
            as="h2"
            text="Where I've shipped."
            className="font-display uppercase tracking-crushed leading-[0.88] text-bone text-[clamp(2.5rem,8vw,8rem)]"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-bone/60 text-lg max-w-md"
          >
            Five years across AI startups, construction-tech and industrial ML. Click any card for the full case.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {experiences.map((exp, index) => (
            <motion.button
              key={exp.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ delay: index * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -6 }}
              onClick={() => setSelectedExperience(exp)}
              data-cursor="open case"
              className="group relative bg-bone/[0.02] border border-bone/10 rounded-2xl p-6 hover:border-acid/40 transition-colors text-left h-full flex flex-col overflow-hidden"
            >
              <div
                className="absolute -top-10 -right-10 w-40 h-40 rounded-full blur-3xl opacity-30 group-hover:opacity-50 transition-opacity"
                style={{ backgroundColor: exp.color }}
              />

              <div className="relative flex items-start justify-between gap-3 mb-5">
                <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-bone/40">
                  / {String(index + 1).padStart(2, '0')}
                </div>
                <div
                  className="px-2.5 py-1 rounded-full font-mono text-[10px] uppercase tracking-[0.25em]"
                  style={{
                    backgroundColor: `${exp.color}15`,
                    color: exp.color,
                    border: `1px solid ${exp.color}40`,
                  }}
                >
                  {exp.period}
                </div>
              </div>

              <h3 className="relative font-display uppercase text-bone text-2xl md:text-3xl leading-[0.95] tracking-crushed mb-2 group-hover:text-acid transition-colors">
                {exp.role}
              </h3>

              <div className="relative flex items-center gap-2 text-bone/70 text-sm mb-1">
                <Briefcase className="w-3.5 h-3.5" />
                <span className="font-medium">{exp.company}</span>
              </div>
              <div className="relative flex items-center gap-2 text-bone/45 font-mono text-[11px] uppercase tracking-widest mb-4">
                <MapPin className="w-3 h-3" />
                <span>{exp.location}</span>
              </div>

              <p className="relative text-bone/65 text-sm leading-relaxed mb-5 line-clamp-2 font-serif-i text-base">
                {exp.tagline}
              </p>

              <div className="relative flex flex-wrap gap-1.5 mb-5">
                {exp.keyTech.slice(0, 4).map((tech) => (
                  <span key={tech} className="chip text-[10px]">
                    {tech}
                  </span>
                ))}
              </div>

              <div className="relative grid grid-cols-2 gap-2 mb-5 mt-auto">
                {exp.highlights.slice(0, 2).map((h) => (
                  <div
                    key={h.label}
                    className="px-3 py-2.5 rounded-xl bg-bone/[0.03] border border-bone/10"
                  >
                    <div
                      className="font-display text-2xl tracking-crushed leading-none"
                      style={{ color: exp.color }}
                    >
                      {h.value}
                    </div>
                    <div className="font-mono text-[9px] uppercase tracking-widest text-bone/40 mt-1 truncate">
                      {h.label}
                    </div>
                  </div>
                ))}
              </div>

              <div className="relative flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.3em] text-bone/40 group-hover:text-acid transition-colors">
                <span>Full case →</span>
              </div>
            </motion.button>
          ))}
        </div>
      </div>

      <ExperienceModal
        experience={selectedExperience}
        onClose={() => setSelectedExperience(null)}
      />
    </section>
  );
};

const ExperienceModal = ({
  experience,
  onClose,
}: {
  experience: Experience | null;
  onClose: () => void;
}) => {
  useEffect(() => {
    if (!experience) return;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', onKey);
    };
  }, [experience, onClose]);

  return createPortal(
    <AnimatePresence>
      {experience && (
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
            className="relative bg-ink border border-bone/15 rounded-none md:rounded-3xl w-full md:max-w-4xl md:max-h-[92vh] h-screen md:h-auto overflow-hidden flex flex-col"
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
              {/* Header band with accent */}
              <div className="relative px-6 md:px-12 pt-10 md:pt-14 pb-8 overflow-hidden">
                <div
                  className="absolute -top-20 -left-20 w-[420px] h-[420px] rounded-full blur-[140px] opacity-30 pointer-events-none"
                  style={{ backgroundColor: experience.color }}
                />

                <div className="relative flex flex-wrap items-center gap-3 mb-5 font-mono text-[10px] uppercase tracking-[0.3em] text-bone/60">
                  <span
                    className="px-2.5 py-1 rounded-full"
                    style={{
                      backgroundColor: `${experience.color}15`,
                      color: experience.color,
                      border: `1px solid ${experience.color}40`,
                    }}
                  >
                    {experience.period}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Briefcase className="w-3 h-3" />
                    {experience.company}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <MapPin className="w-3 h-3" />
                    {experience.location}
                  </span>
                </div>

                <h1 className="relative font-display uppercase text-bone text-3xl md:text-6xl tracking-crushed leading-[0.9] mb-4">
                  {experience.role}
                </h1>
                <p className="relative font-serif-i text-xl md:text-2xl text-bone/70 leading-snug max-w-2xl">
                  {experience.tagline}
                </p>
              </div>

              <div className="px-6 md:px-12 pb-12 md:pb-16 max-w-3xl mx-auto">
                {/* Metrics */}
                {experience.highlights.length > 0 && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-14">
                    {experience.highlights.map((h) => (
                      <div
                        key={h.label}
                        className="bg-bone/[0.03] border border-bone/10 rounded-2xl p-4 text-center"
                      >
                        <div
                          className="font-display text-3xl md:text-4xl tracking-crushed leading-none"
                          style={{ color: experience.color }}
                        >
                          {h.value}
                        </div>
                        <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-bone/50 mt-2">
                          {h.label}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {experience.detailedDescription && (
                  <ExpSection title="Overview" icon={<TrendingUp className="w-5 h-5" style={{ color: experience.color }} />}>
                    <p className="text-bone/75 text-lg leading-[1.8]">{experience.detailedDescription}</p>
                  </ExpSection>
                )}

                <ExpSection title="Key responsibilities">
                  <ul className="space-y-3">
                    {experience.description.map((item, i) => (
                      <li key={i} className="flex gap-3 bg-bone/[0.03] border border-bone/10 rounded-xl p-4">
                        <span className="text-acid text-xl leading-none shrink-0">▹</span>
                        <span className="text-bone/80 leading-relaxed">{item}</span>
                      </li>
                    ))}
                  </ul>
                </ExpSection>

                {experience.allTechnologies && experience.allTechnologies.length > 0 && (
                  <ExpSection
                    title="Stack"
                    icon={<Layers className="w-5 h-5" style={{ color: experience.color }} />}
                  >
                    <div className="flex flex-wrap gap-2">
                      {experience.allTechnologies.map((tech) => (
                        <span
                          key={tech}
                          className="chip"
                          style={{
                            backgroundColor: `${experience.color}12`,
                            color: experience.color,
                            borderColor: `${experience.color}40`,
                          }}
                        >
                          {tech}
                        </span>
                      ))}
                    </div>
                  </ExpSection>
                )}

                {experience.challenges && experience.challenges.length > 0 && (
                  <ExpSection
                    title="Challenges"
                    icon={<AlertTriangle className="w-5 h-5 text-flame" />}
                  >
                    <ul className="space-y-3">
                      {experience.challenges.map((c, i) => (
                        <li key={i} className="flex gap-3 bg-bone/[0.03] border border-bone/10 rounded-xl p-4">
                          <span className="text-flame text-lg leading-none shrink-0">◆</span>
                          <span className="text-bone/80 leading-relaxed">{c}</span>
                        </li>
                      ))}
                    </ul>
                  </ExpSection>
                )}

                {experience.outcomes && experience.outcomes.length > 0 && (
                  <ExpSection
                    title="Results"
                    icon={<CheckCircle2 className="w-5 h-5 text-acid" />}
                  >
                    <ul className="space-y-3">
                      {experience.outcomes.map((o, i) => (
                        <li key={i} className="flex gap-3 bg-bone/[0.03] border border-bone/10 rounded-xl p-4">
                          <span className="text-acid text-lg leading-none shrink-0">✓</span>
                          <span className="text-bone/80 leading-relaxed">{o}</span>
                        </li>
                      ))}
                    </ul>
                  </ExpSection>
                )}

                {experience.teamInfo && (
                  <ExpSection title="Team" icon={<Users className="w-5 h-5 text-bone/60" />}>
                    <p className="text-bone/75 text-lg leading-relaxed">{experience.teamInfo}</p>
                  </ExpSection>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
};

const ExpSection = ({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) => (
  <section className="mb-12 last:mb-0">
    <h2 className="font-display uppercase text-bone text-2xl md:text-3xl tracking-crushed mb-5 flex items-center gap-3">
      {icon}
      {title}
    </h2>
    {children}
  </section>
);

export default Experience;
