import { motion, AnimatePresence } from 'framer-motion';
import { Briefcase, MapPin, X, TrendingUp } from 'lucide-react';
import { useState, useEffect } from 'react';
import Tilt3D from '../components/Tilt3D';

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
    <section id="experience" className="relative py-32 bg-dark-card overflow-hidden">
      <div className="absolute inset-0 opacity-[0.02]">
        <div className="w-full h-full" style={{ backgroundImage: `linear-gradient(rgba(167, 139, 250, 0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(167, 139, 250, 0.5) 1px, transparent 1px)`, backgroundSize: '60px 60px' }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" style={{ zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-7xl font-bold mb-6 text-gradient">Experience</h2>
          <p className="text-light-gray text-lg md:text-xl">
            Building impactful solutions across leading tech companies
          </p>
        </motion.div>

        {/* Horizontal grid of compact cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {experiences.map((exp, index) => (
            <Tilt3D key={exp.id} tiltMaxAngle={10}>
              <motion.div
                initial={false}
                animate={{ opacity: 1, y: 0 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="group bg-dark-bg border border-light-gray/10 rounded-xl p-6 hover:border-lime/30 transition-all duration-300 cursor-pointer h-full flex flex-col"
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedExperience(exp)}
                style={{ transformStyle: 'preserve-3d', minHeight: '420px' }}
              >
              {/* Company & Role */}
              <div className="mb-3">
                <h3 className="font-display text-xl font-bold mb-2 group-hover:text-gradient transition-colors line-clamp-2">
                  {exp.role}
                </h3>
                <div className="flex items-center gap-2 text-light-gray/70 text-sm mb-1">
                  <Briefcase className="w-4 h-4" />
                  <span className="font-medium">{exp.company}</span>
                </div>
                <div className="flex items-center gap-2 text-light-gray/60 text-xs mb-2">
                  <MapPin className="w-3 h-3" />
                  <span>{exp.location}</span>
                </div>
                {/* Tagline */}
                <p className="text-light-gray/80 text-sm italic line-clamp-2">
                  {exp.tagline}
                </p>
              </div>

              {/* Period Badge */}
              <div className="px-3 py-1.5 rounded-full border border-lime/30 text-lime font-mono text-xs text-center mb-3">
                {exp.period}
              </div>

              {/* Key Technologies */}
              <div className="mb-3">
                <div className="text-xs font-semibold text-light-gray/70 mb-2">Tech Stack:</div>
                <div className="flex flex-wrap gap-1">
                  {exp.keyTech.slice(0, 4).map((tech) => (
                    <span
                      key={tech}
                      className="px-2 py-1 rounded text-xs font-medium"
                      style={{ 
                        backgroundColor: `${exp.color}10`, 
                        color: exp.color,
                        border: `1px solid ${exp.color}30`
                      }}
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>

              {/* Top Achievements Preview */}
              <div className="mb-3">
                <div className="text-xs font-semibold text-light-gray/70 mb-2">Key Impact:</div>
                <ul className="space-y-1">
                  {exp.topAchievements.slice(0, 2).map((achievement, i) => (
                    <li key={i} className="text-light-gray/70 text-xs flex items-start gap-1.5">
                      <span className="text-lime mt-0.5">▸</span>
                      <span className="line-clamp-1">{achievement}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Highlights Preview */}
              <div className="grid grid-cols-2 gap-2 mb-3">
                {exp.highlights.slice(0, 2).map((highlight) => (
                  <div 
                    key={highlight.label} 
                    className="px-3 py-2 rounded-lg text-center"
                    style={{ backgroundColor: `${exp.color}15`, border: `1px solid ${exp.color}30` }}
                  >
                    <div className="text-xs text-light-gray/70 truncate">{highlight.label}</div>
                    <div className="text-lg font-bold" style={{ color: exp.color }}>{highlight.value}</div>
                  </div>
                ))}
              </div>

              {/* View Details */}
              <div className="text-sm text-lime font-medium text-center opacity-0 group-hover:opacity-100 transition-opacity">
                Click for full details →
              </div>
              </motion.div>
            </Tilt3D>
          ))}
        </div>
      </div>

      {/* Experience Detail Modal */}
      <ExperienceModal
        experience={selectedExperience}
        onClose={() => setSelectedExperience(null)}
      />
    </section>
  );
};

const ExperienceModal = ({ experience, onClose }: { experience: Experience | null; onClose: () => void }) => {
  useEffect(() => {
    if (experience) {
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [experience]);

  const handleModalScroll = (e: React.WheelEvent) => {
    e.stopPropagation();
  };

  return (
    <AnimatePresence>
      {experience && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto"
          style={{ alignItems: 'center' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="relative bg-dark-card border border-light-gray/20 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
            onWheel={handleModalScroll}
          >
            {/* Header */}
            <div className="sticky top-0 z-10 bg-dark-card border-b border-light-gray/10 p-6">
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-2 bg-dark-bg/80 backdrop-blur-sm rounded-full text-white hover:bg-lime/20 hover:text-lime transition-colors"
              >
                <X className="w-6 h-6" />
              </button>

              <h1 className="font-display text-3xl md:text-4xl font-bold text-white mb-3">
                {experience.role}
              </h1>
              
              <div className="flex flex-wrap items-center gap-4 text-light-gray/70 mb-3">
                <div className="flex items-center gap-2">
                  <Briefcase className="w-5 h-5" />
                  <span className="font-medium text-lg">{experience.company}</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  <span>{experience.location}</span>
                </div>
              </div>

              <div className="inline-block px-4 py-2 rounded-full border border-lime/30 text-lime font-mono text-sm">
                {experience.period}
              </div>
            </div>

            {/* Content */}
            <div className="p-8">
              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
                {experience.highlights.map((highlight) => (
                  <div
                    key={highlight.label}
                    className="bg-dark-bg border border-light-gray/10 rounded-lg p-4 text-center"
                  >
                    <div className="text-3xl font-bold mb-1" style={{ color: experience.color }}>
                      {highlight.value}
                    </div>
                    <div className="text-sm text-light-gray/70">{highlight.label}</div>
                  </div>
                ))}
              </div>

              {/* Detailed Overview */}
              {experience.detailedDescription && (
                <div className="mb-10">
                  <h2 className="text-2xl font-display font-bold text-white mb-4">Overview</h2>
                  <p className="text-light-gray text-lg leading-relaxed">
                    {experience.detailedDescription}
                  </p>
                </div>
              )}

              {/* Responsibilities */}
              <div className="mb-10">
                <h2 className="text-2xl font-display font-bold text-white mb-6 flex items-center gap-2">
                  <TrendingUp style={{ color: experience.color }} />
                  Key Responsibilities & Achievements
                </h2>
                <ul className="space-y-4">
                  {experience.description.map((item, i) => (
                    <li key={i} className="text-light-gray text-lg flex items-start gap-3">
                      <span className="text-lime mt-1 text-xl">▹</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Complete Tech Stack */}
              {experience.allTechnologies && (
                <div className="mb-10">
                  <h2 className="text-2xl font-display font-bold text-white mb-4">Complete Technology Stack</h2>
                  <div className="flex flex-wrap gap-2">
                    {experience.allTechnologies.map((tech) => (
                      <span
                        key={tech}
                        className="px-3 py-2 rounded-lg text-sm font-medium"
                        style={{ 
                          backgroundColor: `${experience.color}15`, 
                          color: experience.color,
                          border: `1px solid ${experience.color}30`
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Challenges */}
              {experience.challenges && experience.challenges.length > 0 && (
                <div className="mb-10">
                  <h2 className="text-2xl font-display font-bold text-white mb-6">Challenges & Solutions</h2>
                  <ul className="space-y-4">
                    {experience.challenges.map((challenge, i) => (
                      <li key={i} className="text-light-gray text-lg flex items-start gap-3">
                        <span className="text-orange-400 mt-1 text-xl">◆</span>
                        <span>{challenge}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Outcomes */}
              {experience.outcomes && experience.outcomes.length > 0 && (
                <div className="mb-10">
                  <h2 className="text-2xl font-display font-bold text-white mb-6">Results & Impact</h2>
                  <ul className="space-y-4">
                    {experience.outcomes.map((outcome, i) => (
                      <li key={i} className="text-light-gray text-lg flex items-start gap-3">
                        <span className="text-lime mt-1 text-xl">✓</span>
                        <span>{outcome}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Team Info */}
              {experience.teamInfo && (
                <div className="mb-8">
                  <h2 className="text-2xl font-display font-bold text-white mb-4">Team & Collaboration</h2>
                  <p className="text-light-gray text-lg leading-relaxed">
                    {experience.teamInfo}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Experience;
