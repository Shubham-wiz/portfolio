import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Copy, Check, Code2 } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  defaultOpen?: boolean;
}

/**
 * Collapsible code block with copy button, language badge and line count.
 * Collapsed by default. Used inside modals (blog articles, project cases).
 */
const CodeBlock = ({
  code,
  language = 'text',
  title,
  defaultOpen = false,
}: CodeBlockProps) => {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  const trimmed = code.replace(/^\n+|\n+$/g, '');
  const lines = trimmed.split('\n');
  const lineCount = lines.length;

  const preview = lines.slice(0, 2).join('\n');

  const onCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(trimmed);
      setCopied(true);
      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // ignore
    }
  };

  return (
    <div className="relative not-prose my-6 rounded-2xl border border-bone/10 bg-ink/80 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        data-cursor={open ? 'collapse' : 'expand'}
        className="w-full flex items-center justify-between gap-4 px-4 py-3 hover:bg-bone/[0.03] transition-colors group"
      >
        <div className="flex items-center gap-3 min-w-0">
          <motion.div
            animate={{ rotate: open ? 0 : -90 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="shrink-0 text-bone/50 group-hover:text-acid"
          >
            <ChevronDown className="w-4 h-4" />
          </motion.div>
          <Code2 className="w-4 h-4 text-acid shrink-0" />
          <span className="font-mono text-xs uppercase tracking-[0.25em] text-bone/70 truncate">
            {title ?? 'Code'}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden sm:inline font-mono text-[10px] uppercase tracking-[0.25em] text-bone/40">
            {lineCount} {lineCount === 1 ? 'line' : 'lines'}
          </span>
          <span
            className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-widest border border-bone/15 text-bone/60"
          >
            {language}
          </span>
        </div>
      </button>

      {/* Preview (when collapsed) */}
      {!open && lineCount > 0 && (
        <div
          className="relative px-4 pb-3 cursor-pointer"
          onClick={() => setOpen(true)}
          data-cursor="expand"
        >
          <pre className="font-mono text-[12.5px] text-bone/35 leading-relaxed overflow-hidden max-h-[3.2em]">
            <code>{preview}</code>
          </pre>
          <div className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-ink to-transparent pointer-events-none" />
          <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.25em] text-bone/35 group-hover:text-acid">
            ↓ Click to expand
          </div>
        </div>
      )}

      {/* Full code (when open) */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="full"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="relative border-t border-bone/10">
              <button
                onClick={onCopy}
                data-cursor={copied ? 'copied' : 'copy'}
                className="absolute top-3 right-3 z-10 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-bone/[0.06] hover:bg-acid hover:text-ink border border-bone/15 text-bone/80 font-mono text-[10px] uppercase tracking-[0.25em] transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    Copy
                  </>
                )}
              </button>

              <div className="overflow-auto max-h-[60vh]">
                <pre className="p-5 pr-24 font-mono text-[13px] leading-relaxed text-bone/85 tabular-nums">
                  {lines.map((line, i) => (
                    <div key={i} className="flex gap-4">
                      <span className="select-none text-bone/20 w-8 text-right shrink-0">
                        {i + 1}
                      </span>
                      <span className="whitespace-pre flex-1">{line || '\u00A0'}</span>
                    </div>
                  ))}
                </pre>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CodeBlock;
