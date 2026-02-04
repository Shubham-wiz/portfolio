/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0a0a0a',
        'dark-card': '#1a1a1a',
        'light-gray': '#d1d5db',
        'lime': '#d1e29d',
        'purple': {
          500: '#a78bfa',
        }
      },
      fontFamily: {
        'display': ['Bricolage Grotesque', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
