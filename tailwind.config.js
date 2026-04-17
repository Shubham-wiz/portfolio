/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ink': '#050505',
        'bone': '#f4f1ea',
        'dark-bg': '#050505',
        'dark-card': '#0e0e0e',
        'light-gray': '#c9c6bf',
        'acid': '#c6ff3d',
        'lime': '#c6ff3d',
        'electric': '#5b6cff',
        'flame': '#ff4d1f',
        'violet': '#a78bfa',
        'purple': {
          500: '#a78bfa',
        }
      },
      fontFamily: {
        'display': ['"Anton"', '"Bricolage Grotesque"', 'sans-serif'],
        'body': ['"Bricolage Grotesque"', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        'sans': ['"Bricolage Grotesque"', 'sans-serif'],
        'serif': ['"Instrument Serif"', 'Georgia', 'serif'],
      },
      letterSpacing: {
        'tighter2': '-0.04em',
        'crushed': '-0.06em',
      },
      animation: {
        'marquee': 'marquee 30s linear infinite',
        'spin-slow': 'spin 12s linear infinite',
        'pulse-slow': 'pulse 6s ease-in-out infinite',
      },
      keyframes: {
        marquee: {
          '0%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(-50%)' },
        },
      },
    },
  },
  plugins: [],
}
