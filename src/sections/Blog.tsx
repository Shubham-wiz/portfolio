import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { Calendar, Clock, X, ArrowUpRight } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { blogArticles, BlogArticle } from '../data/blog-articles';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';
import TextReveal from '../components/TextReveal';

const categories = ['All', 'AI/ML', 'Data Engineering', 'MLOps'];

const Blog = () => {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedArticle, setSelectedArticle] = useState<BlogArticle | null>(null);

  const filteredPosts =
    selectedCategory === 'All'
      ? blogArticles
      : blogArticles.filter((post) => post.category === selectedCategory);

  return (
    <section id="writings" className="relative py-28 md:py-36 bg-ink overflow-hidden">
      <div className="absolute top-20 left-10 w-96 h-96 bg-violet/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-acid/10 rounded-full blur-[120px]" />
      <div className="absolute inset-0 grid-lines opacity-30 pointer-events-none" />

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-16">
        <div className="flex items-center gap-4 mb-10 font-mono text-xs uppercase tracking-[0.3em] text-bone/50">
          <span className="w-2 h-2 rotate-45 bg-acid" />
          / 06 — Writings
          <span className="flex-1 h-px bg-bone/10" />
          <span className="text-bone/30">Notes on systems, models, people</span>
        </div>

        <div className="mb-14 flex flex-col lg:flex-row justify-between items-end gap-6">
          <TextReveal
            as="h2"
            text="Long reads. Loud opinions."
            className="font-display uppercase tracking-crushed leading-[0.88] text-bone text-[clamp(2.5rem,8vw,8rem)]"
          />
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-lg text-bone/60 text-lg"
          >
            Deep dives into data engineering, MLOps, and AI architecture — with the ink still wet.
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex flex-wrap gap-3 mb-12"
        >
          {categories.map((category) => (
            <button
              key={category}
              data-cursor={category.toLowerCase()}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full font-mono text-xs uppercase tracking-widest transition-colors ${
                selectedCategory === category
                  ? 'bg-acid text-ink border border-acid'
                  : 'border border-bone/20 text-bone/70 hover:border-acid hover:text-acid'
              }`}
            >
              {category}
            </button>
          ))}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {filteredPosts.map((post, index) => (
            <BlogCard
              key={post.id}
              post={post}
              index={index}
              onClick={() => setSelectedArticle(post)}
            />
          ))}
        </div>
      </div>

      <ArticleModal article={selectedArticle} onClose={() => setSelectedArticle(null)} />
    </section>
  );
};

// -------------- Blog card --------------

const BlogCard = ({
  post,
  index,
  onClick,
}: {
  post: BlogArticle;
  index: number;
  onClick: () => void;
}) => {
  return (
    <motion.button
      onClick={onClick}
      data-cursor="open article"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ delay: index * 0.08, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -6 }}
      className="group relative bg-bone/[0.02] border border-bone/10 rounded-3xl overflow-hidden hover:border-acid/40 transition-colors text-left flex flex-col"
    >
      <div className="relative h-64 md:h-72 overflow-hidden">
        <motion.img
          src={post.image}
          alt={post.title}
          className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-700"
          whileHover={{ scale: 1.06 }}
          transition={{ duration: 0.7 }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/30 to-transparent" />
        <div
          className="absolute inset-0 opacity-25 mix-blend-multiply transition-opacity duration-500 group-hover:opacity-10"
          style={{ background: `linear-gradient(135deg, ${post.color}00 40%, ${post.color}60 100%)` }}
        />

        <div className="absolute top-4 left-4 flex items-center gap-2 flex-wrap">
          {post.featured && (
            <span className="chip chip-acid">Featured</span>
          )}
          {post.mustRead && (
            <span className="chip border-acid/60 text-acid">Must read</span>
          )}
        </div>

        <div className="absolute top-4 right-4 font-mono text-[10px] uppercase tracking-[0.3em] text-bone/60">
          / {String(index + 1).padStart(2, '0')}
        </div>
      </div>

      <div className="p-7 md:p-8 flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4 font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40">
          <span
            className="px-2.5 py-1 rounded-full"
            style={{
              backgroundColor: `${post.color}20`,
              color: post.color,
              border: `1px solid ${post.color}40`,
            }}
          >
            {post.category}
          </span>
          <div className="flex items-center gap-1.5">
            <Calendar className="w-3 h-3" />
            <span>{post.date}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Clock className="w-3 h-3" />
            <span>{post.readTime}</span>
          </div>
        </div>

        <h3 className="font-display uppercase text-bone text-2xl md:text-3xl leading-tight tracking-tight mb-4 group-hover:text-acid transition-colors">
          {post.title}
        </h3>

        <p className="text-bone/65 text-base md:text-lg leading-relaxed line-clamp-3 mb-6 flex-1">
          {post.excerpt}
        </p>

        <div className="flex items-center justify-between gap-4">
          <span className="font-mono text-xs uppercase tracking-[0.3em] text-bone/60 group-hover:text-acid transition-colors">
            Read essay
          </span>
          <motion.div
            whileHover={{ rotate: 45 }}
            className="w-10 h-10 rounded-full bg-bone/[0.06] border border-bone/15 grid place-items-center group-hover:bg-acid group-hover:text-ink transition-colors"
          >
            <ArrowUpRight className="w-4 h-4" />
          </motion.div>
        </div>
      </div>
    </motion.button>
  );
};

// -------------- Article modal --------------

const ArticleModal = ({
  article,
  onClose,
}: {
  article: BlogArticle | null;
  onClose: () => void;
}) => {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: scrollerRef });
  const progress = useSpring(scrollYProgress, {
    stiffness: 200,
    damping: 30,
    restDelta: 0.001,
  });

  useEffect(() => {
    if (!article) return;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', onKey);
    };
  }, [article, onClose]);

  return createPortal(
    <AnimatePresence>
      {article && (
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
            {/* Reading progress bar */}
            <motion.div
              style={{ scaleX: progress }}
              className="absolute top-0 left-0 right-0 h-[2px] bg-acid origin-left z-20"
            />

            {/* Close button - floating */}
            <button
              onClick={onClose}
              data-cursor="close"
              className="absolute top-4 right-4 md:top-6 md:right-6 z-30 w-11 h-11 rounded-full bg-ink/80 backdrop-blur border border-bone/20 text-bone hover:bg-acid hover:text-ink hover:border-acid grid place-items-center transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Scrollable body */}
            <div ref={scrollerRef} className="overflow-y-auto flex-1">
              {/* Hero image */}
              <div className="relative h-64 md:h-80">
                <img
                  src={article.image}
                  alt={article.title}
                  className="w-full h-full object-cover grayscale"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/50 to-ink/10" />
                <div
                  className="absolute inset-0 mix-blend-multiply opacity-40"
                  style={{
                    background: `linear-gradient(135deg, ${article.color}00 30%, ${article.color}80 100%)`,
                  }}
                />

                <div className="absolute bottom-0 left-0 right-0 p-6 md:p-10">
                  <div className="flex items-center flex-wrap gap-3 mb-4 font-mono text-[10px] uppercase tracking-[0.3em] text-bone/70">
                    <span
                      className="px-2.5 py-1 rounded-full"
                      style={{
                        backgroundColor: `${article.color}25`,
                        color: article.color,
                        border: `1px solid ${article.color}50`,
                      }}
                    >
                      {article.category}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <Calendar className="w-3 h-3" />
                      {article.date}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Clock className="w-3 h-3" />
                      {article.readTime}
                    </div>
                  </div>
                  <h1 className="font-display uppercase text-bone text-3xl md:text-6xl leading-[0.9] tracking-crushed">
                    {article.title}
                  </h1>
                </div>
              </div>

              {/* Article body */}
              <article className="px-6 md:px-14 py-10 md:py-14 max-w-3xl mx-auto">
                <p className="font-serif-i text-xl md:text-2xl leading-relaxed text-bone/80 mb-10 border-l-2 border-acid pl-5">
                  {article.excerpt}
                </p>

                <div className="article-prose">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ children }) => (
                        <h2 className="font-display uppercase text-bone text-2xl md:text-4xl tracking-crushed leading-tight mt-14 mb-5">
                          {children}
                        </h2>
                      ),
                      h2: ({ children }) => (
                        <h2 className="font-display uppercase text-bone text-2xl md:text-4xl tracking-crushed leading-tight mt-14 mb-5">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="font-display uppercase text-acid text-xl md:text-2xl tracking-tight leading-snug mt-10 mb-4">
                          {children}
                        </h3>
                      ),
                      p: ({ children }) => (
                        <p className="text-bone/75 text-lg leading-[1.8] mb-6">{children}</p>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-none space-y-2.5 mb-6 text-bone/75 text-lg">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal pl-6 space-y-2.5 mb-6 text-bone/75 text-lg marker:text-acid marker:font-mono marker:text-sm">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="flex gap-3">
                          <span className="text-acid mt-1.5 shrink-0">▹</span>
                          <span className="flex-1 leading-[1.75]">{children}</span>
                        </li>
                      ),
                      strong: ({ children }) => (
                        <strong className="text-bone font-semibold">{children}</strong>
                      ),
                      em: ({ children }) => (
                        <em className="text-acid/90 font-serif-i not-italic">{children}</em>
                      ),
                      a: ({ children, href }) => (
                        <a
                          href={href}
                          className="text-acid underline decoration-acid/40 underline-offset-4 hover:decoration-acid"
                          target="_blank"
                          rel="noreferrer"
                        >
                          {children}
                        </a>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-acid pl-5 my-8 font-serif-i text-xl text-bone/80 leading-relaxed">
                          {children}
                        </blockquote>
                      ),
                      code: ({ className, children }) => {
                        const match = /language-(\w+)/.exec(className || '');
                        const isBlock = Boolean(match);
                        if (isBlock) {
                          const lang = match![1];
                          const code = Array.isArray(children)
                            ? children.join('')
                            : String(children ?? '');
                          if (lang === 'mermaid') {
                            return <MermaidDiagram code={code} />;
                          }
                          return (
                            <CodeBlock
                              code={code}
                              language={lang}
                              title={`${lang} snippet`}
                            />
                          );
                        }
                        return (
                          <code className="px-1.5 py-0.5 rounded bg-bone/[0.08] border border-bone/10 text-acid font-mono text-[0.92em]">
                            {children}
                          </code>
                        );
                      },
                      pre: ({ children }) => <>{children}</>,
                      hr: () => <hr className="my-12 border-bone/10" />,
                      table: ({ children }) => (
                        <div className="not-prose my-8 overflow-x-auto rounded-2xl border border-bone/10 bg-bone/[0.02]">
                          <table className="w-full border-collapse text-left">{children}</table>
                        </div>
                      ),
                      thead: ({ children }) => (
                        <thead className="bg-bone/[0.04] border-b border-bone/10">{children}</thead>
                      ),
                      tbody: ({ children }) => <tbody>{children}</tbody>,
                      tr: ({ children }) => (
                        <tr className="border-b border-bone/5 last:border-0 hover:bg-bone/[0.02] transition-colors">
                          {children}
                        </tr>
                      ),
                      th: ({ children }) => (
                        <th className="px-4 py-3 font-mono text-[10px] uppercase tracking-[0.2em] text-acid font-semibold">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="px-4 py-3 text-bone/80 text-[15px] align-top">{children}</td>
                      ),
                    }}
                  >
                    {article.content}
                  </ReactMarkdown>
                </div>

                <div className="mt-16 pt-8 border-t border-bone/10 flex flex-wrap items-center justify-between gap-4">
                  <div className="font-mono text-xs uppercase tracking-[0.3em] text-bone/40">
                    End of essay
                  </div>
                  <button
                    onClick={() => scrollerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })}
                    data-cursor="top"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-bone/20 text-bone/80 font-mono text-[10px] uppercase tracking-[0.25em] hover:border-acid hover:text-acid transition-colors"
                  >
                    ↑ Back to top
                  </button>
                </div>
              </article>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
};

export default Blog;
