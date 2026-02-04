import { motion, AnimatePresence } from 'framer-motion';
import { Code2, X, TrendingUp } from 'lucide-react';
import { useState, useEffect } from 'react';
import { projects, Project } from '../data/projects-data';
import Tilt3D from '../components/Tilt3D';

const Projects = () => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  return (
    <section id="projects" className="relative py-32 bg-dark-bg overflow-hidden">
      <div className="absolute top-20 right-10 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-20 left-10 w-96 h-96 bg-lime/10 rounded-full blur-[120px]" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" style={{ zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-7xl font-bold mb-6 text-gradient">Featured Projects</h2>
          <p className="text-light-gray text-lg md:text-xl max-w-2xl mx-auto">
            Building production-grade systems that solve real-world problems
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <Tilt3D key={project.id} tiltMaxAngle={8}>
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="group relative bg-dark-card border border-light-gray/10 rounded-2xl overflow-hidden hover:border-lime/30 transition-all duration-500 cursor-pointer h-full flex flex-col"
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedProject(project)}
                style={{ transformStyle: 'preserve-3d', minHeight: '550px' }}
              >
              <div className="relative h-72 overflow-hidden">
                <motion.img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover"
                  whileHover={{ scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-dark-card to-transparent" />
              </div>

              <div className="p-8 flex-1 flex flex-col">
                <h3 className="font-display text-3xl font-bold mb-4 group-hover:text-gradient transition-colors">
                  {project.title}
                </h3>
                <p className="text-light-gray text-lg mb-6 flex-1">{project.description}</p>
                
                <div className="flex flex-wrap gap-2 mb-6">
                  {project.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 rounded-full text-sm font-medium border"
                      style={{ borderColor: `${project.color}40`, color: project.color }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex items-center gap-2 text-lime font-medium group-hover:gap-4 transition-all">
                  <Code2 className="w-5 h-5" />
                  <span>View Technical Details</span>
                  <span className="transform group-hover:translate-x-2 transition-transform">→</span>
                </div>
              </div>
              </motion.div>
            </Tilt3D>
          ))}
        </div>
      </div>

      {/* Project Detail Modal */}
      <ProjectModal
        project={selectedProject}
        onClose={() => setSelectedProject(null)}
      />
    </section>
  );
};

const ProjectModal = ({ project, onClose }: { project: Project | null; onClose: () => void }) => {
  useEffect(() => {
    if (project) {
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }
    return () => {
      // Re-enable body scroll when modal closes
      document.body.style.overflow = 'unset';
    };
  }, [project]);

  const handleModalScroll = (e: React.WheelEvent) => {
    // Prevent scroll from bubbling to background
    e.stopPropagation();
  };

  return (
    <AnimatePresence>
      {project && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-hidden"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="relative bg-dark-card border border-light-gray/20 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
            onWheel={handleModalScroll}
          >
            {/* Header */}
            <div className="relative h-64">
              <img
                src={project.image}
                alt={project.title}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-dark-card to-transparent" />
              
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-2 bg-dark-bg/80 backdrop-blur-sm rounded-full text-white hover:bg-lime/20 hover:text-lime transition-colors"
              >
                <X className="w-6 h-6" />
              </button>

              <div className="absolute bottom-6 left-6 right-6">
                <div className="flex flex-wrap gap-2 mb-3">
                  {project.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1.5 rounded-full font-medium text-sm"
                      style={{
                        backgroundColor: `${project.color}20`,
                        color: project.color
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <h1 className="font-display text-4xl font-bold text-white mb-2">
                  {project.title}
                </h1>
                <p className="text-light-gray text-lg">{project.description}</p>
              </div>
            </div>

            {/* Content */}
            <div className="p-8">
              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
                {project.metrics.map((metric) => (
                  <div
                    key={metric.label}
                    className="bg-dark-bg border border-light-gray/10 rounded-lg p-4 text-center"
                  >
                    <div className="text-3xl font-bold mb-1" style={{ color: project.color }}>
                      {metric.value}
                    </div>
                    <div className="text-sm text-light-gray/70">{metric.label}</div>
                  </div>
                ))}
              </div>

              {/* Detailed Description */}
              <div className="mb-12">
                <h2 className="text-2xl font-display font-bold text-white mb-4 flex items-center gap-2">
                  <TrendingUp style={{ color: project.color }} />
                  Project Overview
                </h2>
                <p className="text-light-gray leading-relaxed text-lg whitespace-pre-line">
                  {project.detailedDescription}
                </p>
              </div>

              {/* Architecture */}
              <div className="mb-12">
                <h2 className="text-2xl font-display font-bold text-white mb-4">
                  System Architecture
                </h2>
                <div className="bg-dark-bg border border-light-gray/10 rounded-lg p-6">
                  <pre className="text-light-gray font-mono text-sm overflow-x-auto whitespace-pre-wrap">
                    {project.architecture}
                  </pre>
                </div>
              </div>

              {/* Key Features */}
              <div className="mb-12">
                <h2 className="text-2xl font-display font-bold text-white mb-4">
                  Key Features
                </h2>
                <div className="grid md:grid-cols-2 gap-4">
                  {project.keyFeatures.map((feature, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-3 bg-dark-bg border border-light-gray/10 rounded-lg p-4"
                    >
                      <span className="text-lime text-2xl">▹</span>
                      <span className="text-light-gray">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Code Snippets */}
              <div className="space-y-8">
                <h2 className="text-2xl font-display font-bold text-white mb-4 flex items-center gap-2">
                  <Code2 style={{ color: project.color }} />
                  Implementation Examples
                </h2>
                {project.codeSnippets.map((snippet, index) => (
                  <div key={index} className="bg-dark-bg border border-light-gray/10 rounded-lg overflow-hidden">
                    <div className="px-4 py-2 border-b border-light-gray/10 flex items-center justify-between">
                      <h3 className="font-semibold text-white">{snippet.title}</h3>
                      <span className="text-sm text-light-gray/50">{snippet.language}</span>
                    </div>
                    <pre className="p-4 overflow-x-auto">
                      <code className="text-lime font-mono text-sm">{snippet.code}</code>
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Projects;
