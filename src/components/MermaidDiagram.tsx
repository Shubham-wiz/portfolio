import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { motion } from 'framer-motion';
import { Network } from 'lucide-react';

let initialized = false;
function initMermaid() {
  if (initialized) return;
  initialized = true;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'base',
    fontFamily: '"JetBrains Mono", ui-monospace, monospace',
    themeVariables: {
      background: '#0a0a0a',
      primaryColor: '#111111',
      primaryTextColor: '#f4f1ea',
      primaryBorderColor: '#c6ff3d',
      lineColor: '#5b6cff',
      secondaryColor: '#1a1a1a',
      tertiaryColor: '#222222',
      textColor: '#f4f1ea',
      mainBkg: '#111111',
      nodeBorder: '#c6ff3d',
      clusterBkg: '#0f0f0f',
      clusterBorder: '#2b2b2b',
      edgeLabelBackground: '#0a0a0a',
      fontSize: '14px',
    },
    flowchart: { htmlLabels: true, curve: 'basis' },
    sequence: { actorMargin: 50 },
  });
}

interface Props {
  code: string;
}

/**
 * Renders a Mermaid diagram inline (in blog articles, project cases).
 * Falls back to source + error message if parsing fails.
 */
const MermaidDiagram = ({ code }: Props) => {
  const hostRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const idRef = useRef(`mmd-${Math.random().toString(36).slice(2, 10)}`);

  useEffect(() => {
    initMermaid();
    let cancelled = false;
    const src = code.replace(/^\n+|\n+$/g, '');

    (async () => {
      try {
        const { svg: rendered } = await mermaid.render(idRef.current, src);
        if (!cancelled) {
          setSvg(rendered);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code]);

  return (
    <motion.figure
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="not-prose my-8 rounded-2xl border border-bone/10 bg-ink/80 overflow-hidden"
    >
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-bone/10">
        <div className="flex items-center gap-2 text-bone/70">
          <Network className="w-4 h-4 text-acid" />
          <span className="font-mono text-xs uppercase tracking-[0.25em]">Diagram</span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-widest text-bone/40 border border-bone/15 rounded-full px-2 py-0.5">
          mermaid
        </span>
      </div>

      {error ? (
        <div className="p-5">
          <div className="font-mono text-xs uppercase tracking-widest text-flame mb-2">
            Diagram failed to render
          </div>
          <pre className="font-mono text-xs text-bone/50 whitespace-pre-wrap">{error}</pre>
        </div>
      ) : (
        <div
          ref={hostRef}
          className="mermaid-host p-5 md:p-8 overflow-x-auto flex items-center justify-center min-h-[120px]"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </motion.figure>
  );
};

export default MermaidDiagram;
