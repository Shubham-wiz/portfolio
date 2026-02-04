import { motion, AnimatePresence } from 'framer-motion';
import { Calendar, Clock, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { blogArticles, BlogArticle } from '../data/blog-articles';
import Tilt3D from '../components/Tilt3D';

const categories = ["All", "AI/ML", "Data Engineering", "MLOps"];

const Blog = () => {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [selectedArticle, setSelectedArticle] = useState<BlogArticle | null>(null);

  const filteredPosts = selectedCategory === "All"
    ? blogArticles
    : blogArticles.filter(post => post.category === selectedCategory);

  return (
    <section id="blog" className="relative py-32 bg-dark-bg overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-lime/10 rounded-full blur-[120px]" />
      </div>

      <div className="absolute inset-0 opacity-[0.02]">
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

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" style={{ zIndex: 10 }}>
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.h2
            className="font-display text-5xl md:text-7xl font-bold mb-6"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <span className="text-gradient">Latest Insights</span>
          </motion.h2>
          <p className="text-light-gray text-lg md:text-xl max-w-2xl mx-auto">
            Deep dives into data engineering, MLOps, and AI architecture
          </p>
        </motion.div>

        {/* Category filter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex flex-wrap justify-center gap-4 mb-16"
        >
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-6 py-3 rounded-full font-medium transition-all duration-300 ${
                selectedCategory === category
                  ? 'bg-gradient-to-r from-purple-500 to-lime text-dark-bg'
                  : 'border border-light-gray/20 text-light-gray hover:border-lime/50 hover:text-lime'
              }`}
            >
              {category}
            </button>
          ))}
        </motion.div>

        {/* Blog grid - BIGGER cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {filteredPosts.map((post, index) => (
            <Tilt3D key={post.id} tiltMaxAngle={8}>
              <BlogCard
                post={post}
                index={index}
                onClick={() => setSelectedArticle(post)}
              />
            </Tilt3D>
          ))}
        </div>


      </div>

      {/* Article Modal */}
      <ArticleModal
        article={selectedArticle}
        onClose={() => setSelectedArticle(null)}
      />
    </section>
  );
};

const BlogCard = ({ post, index, onClick }: { post: BlogArticle; index: number; onClick: () => void }) => {
  return (
    <motion.article
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ delay: index * 0.1 }}
      className="group relative bg-dark-card border border-light-gray/10 rounded-2xl overflow-hidden hover:border-lime/30 transition-all duration-500 cursor-pointer h-full flex flex-col"
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      style={{ minHeight: '520px' }}
    >
      {/* Image container - BIGGER */}
      <div className="relative h-80 overflow-hidden">
        <motion.img
          src={post.image}
          alt={post.title}
          className="w-full h-full object-cover"
          whileHover={{ scale: 1.1 }}
          transition={{ duration: 0.6 }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-dark-card via-dark-card/50 to-transparent" />
        
        {/* Badges on image */}
        <div className="absolute top-4 left-4 flex gap-2">
          {post.featured && (
            <span className="px-3 py-1 bg-gradient-to-r from-purple-500 to-lime text-dark-bg text-xs font-bold rounded-full">
              FEATURED
            </span>
          )}
          {post.mustRead && (
            <span className="px-3 py-1 bg-lime/20 border border-lime text-lime text-xs font-bold rounded-full">
              MUST READ
            </span>
          )}
        </div>
      </div>

      {/* Content - BIGGER padding and text */}
      <div className="p-8">
        <div className="flex items-center gap-4 mb-4 text-sm">
          <span
            className="px-4 py-1.5 rounded-full font-medium"
            style={{
              backgroundColor: `${post.color}20`,
              color: post.color
            }}
          >
            {post.category}
          </span>
          <div className="flex items-center gap-4 text-light-gray/50">
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              <span>{post.date}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{post.readTime}</span>
            </div>
          </div>
        </div>

        <h2 className="font-display text-3xl font-bold text-white mb-4 group-hover:text-gradient transition-colors line-clamp-2">
          {post.title}
        </h2>

        <p className="text-light-gray text-lg mb-6 line-clamp-3 flex-1">
          {post.excerpt}
        </p>

        <div className="flex items-center gap-2 text-lime font-medium group-hover:gap-4 transition-all mt-auto">
          <span>Read Full Article</span>
          <span className="transform group-hover:translate-x-2 transition-transform">→</span>
        </div>
      </div>

      {/* Glow effect on hover */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 50% 50%, ${post.color}15, transparent 70%)`
        }}
      />
    </motion.article>
  );
};

const ArticleModal = ({ article, onClose }: { article: BlogArticle | null; onClose: () => void }) => {
  useEffect(() => {
    if (article) {
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }
    return () => {
      // Re-enable body scroll when modal closes
      document.body.style.overflow = 'unset';
    };
  }, [article]);

  const handleModalScroll = (e: React.WheelEvent) => {
    // Prevent scroll from bubbling to background
    e.stopPropagation();
  };

  return (
    <AnimatePresence>
      {article && (
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
            className="relative bg-dark-card border border-light-gray/20 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
            onWheel={handleModalScroll}
          >
            {/* Header with image */}
            <div className="relative h-64 md:h-80">
              <img
                src={article.image}
                alt={article.title}
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
                <div className="flex items-center gap-3 mb-3">
                  <span
                    className="px-4 py-1.5 rounded-full font-medium text-sm"
                    style={{
                      backgroundColor: `${article.color}20`,
                      color: article.color
                    }}
                  >
                    {article.category}
                  </span>
                  <div className="flex items-center gap-3 text-sm text-white/70">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>{article.date}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{article.readTime}</span>
                    </div>
                  </div>
                </div>
                <h1 className="font-display text-3xl md:text-4xl font-bold text-white">
                  {article.title}
                </h1>
              </div>
            </div>

            {/* Scrollable content */}
            <div className="p-8">
              <div className="prose prose-invert prose-lg max-w-none">
                <ReactMarkdown
                  components={{
                    h1: ({ children }) => (
                      <h2 className="text-3xl font-display font-bold mt-12 mb-4 text-white">{children}</h2>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-3xl font-display font-bold mt-12 mb-4 text-white">{children}</h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-2xl font-display font-semibold mt-8 mb-3 text-lime">{children}</h3>
                    ),
                    p: ({ children }) => (
                      <p className="text-light-gray leading-relaxed mb-6 text-lg">{children}</p>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-disc list-inside mb-6 space-y-2 text-light-gray">{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-decimal list-inside mb-6 space-y-2 text-light-gray">{children}</ol>
                    ),
                    code: ({ className, children }) => {
                      const isBlock = className?.includes('language-');
                      return isBlock ? (
                        <pre className="bg-dark-bg border border-light-gray/10 rounded-lg p-4 overflow-x-auto mb-6">
                          <code className="text-lime font-mono text-sm">{children}</code>
                        </pre>
                      ) : (
                        <code className="bg-dark-bg px-2 py-1 rounded text-lime font-mono text-sm">{children}</code>
                      );
                    },
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-lime pl-4 italic text-light-gray/80 my-6">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {article.content}
                </ReactMarkdown>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Blog;
