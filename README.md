# Shubham Dwivedi - Portfolio Website v2.0

Modern portfolio website for Data Engineer & AI Architect, featuring dynamic animations, full-length blog articles, and an easy content management system.

## ✨ Features

- **Dynamic Name Introduction**: 3D animated name with interactive hover effects
- **Bigger Cards**: Enhanced project and blog cards for better visual impact
- **Full-Length Articles**: Complete blog posts with beautiful markdown rendering
- **Easy Content Management**: Simple system to add new blog articles
- **SEO Optimized**: Meta tags, Open Graph, Twitter Cards
- **Smooth Scrolling**: Lenis smooth scroll integration
- **Responsive Design**: Works perfectly on all devices

## 🚀 Quick Start

### Development
```bash
npm install
npm run dev
```

Visit `http://localhost:5173` to see your site.

### Production Build
```bash
npm run build
npm run preview
```

## 📝 How to Add New Blog Articles (Must-Reads)

Adding new blog articles is super easy! Just follow these steps:

### Step 1: Open the blog articles file
```
src/data/blog-articles.ts
```

### Step 2: Add your new article to the array

Add a new object to the `blogArticles` array with this structure:

```typescript
{
  id: "your-article-slug",
  title: "Your Article Title",
  excerpt: "A short description (2-3 sentences) shown on the card",
  content: `# Your Full Article Content

Write your full article here in Markdown format.

## Section Headings

You can use:
- **Bold text**
- *Italic text*
- \`code snippets\`
- [Links](https://example.com)
- Lists and more!

### Code Blocks

\`\`\`python
def hello_world():
    print("Hello, World!")
\`\`\`

The article can be as long as you want!
  `,
  image: "/your-blog-image.jpg",  // Add image to public/ folder
  date: "Feb 3, 2025",
  readTime: "10 min",
  category: "AI/ML",  // or "Data Engineering" or "MLOps"
  featured: false,  // Set to true for featured badge
  mustRead: true,   // Set to true to show in Must-Read section
  color: "#a78bfa"  // Choose: #a78bfa (purple), #d1e29d (lime), #f472b6 (pink), #60a5fa (blue)
}
```

### Step 3: Add your blog image

1. Add your blog image (JPG recommended) to the `public/` folder
2. Name it something descriptive like `blog-your-topic.jpg`
3. Reference it in the `image` field as `/blog-your-topic.jpg`

### Step 4: Build and deploy

```bash
npm run build
```

That's it! Your new article will automatically appear in:
- The main blog grid (if category matches filter)
- The "Must-Read Articles" section (if `mustRead: true`)

## 📂 Project Structure

```
├── public/              # Static assets (images, favicon)
├── src/
│   ├── data/
│   │   └── blog-articles.ts    # 📝 Add your blog articles here!
│   ├── sections/
│   │   ├── Hero.tsx            # Dynamic animated homepage
│   │   ├── About.tsx
│   │   ├── Projects.tsx
│   │   ├── Experience.tsx
│   │   ├── Blog.tsx            # Blog section with modal
│   │   └── Contact.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html          # SEO meta tags here
└── package.json
```

## 🎨 Customization

### Colors
Edit `tailwind.config.js` to change the color scheme:
```js
colors: {
  'dark-bg': '#0a0a0a',
  'dark-card': '#1a1a1a',
  'lime': '#d1e29d',
  'purple': { 500: '#a78bfa' }
}
```

### Fonts
The site uses "Bricolage Grotesque" from Google Fonts. Change in `src/index.css`:
```css
@import "https://fonts.googleapis.com/css2?family=Your+Font&display=swap";
```

### Personal Information
Update contact details in `src/sections/Contact.tsx`

### Experience & Projects
Edit the data arrays in `src/sections/Experience.tsx` and `src/sections/Projects.tsx`

## 🔧 Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Lenis** - Smooth scrolling
- **React Markdown** - Article rendering
- **Lucide React** - Icons

## 📱 Responsive Breakpoints

- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## 🐛 Troubleshooting

### Build fails with TypeScript errors
```bash
# Check for unused imports or type errors
npm run build
```

### Images not showing
- Make sure images are in the `public/` folder
- Reference them with `/image-name.jpg` (starting with `/`)

### Smooth scroll not working
- Check that Lenis is properly initialized in `App.tsx`
- Ensure section IDs match navigation links

## 📄 License

This is a personal portfolio project. Feel free to use it as inspiration!

## 🚀 Deployment

### Deploy to Netlify/Vercel
1. Connect your Git repository
2. Build command: `npm run build`
3. Publish directory: `dist`

### Deploy manually
```bash
npm run build
# Upload the 'dist' folder to your hosting provider
```

---

**Note**: The coordinates in the Hero section now correctly show Berlin's location: `52.5200° N, 13.4050° E`

Built with ❤️ by Shubham Dwivedi

## 📊 Mermaid Diagrams in Articles

The blog articles include Mermaid diagrams for visual explanations. Currently, they display as code blocks. To render them as actual diagrams:

### Option 1: Add Mermaid Support (Recommended)

Install the mermaid plugin:
```bash
npm install react-markdown-mermaid
```

Then update `src/sections/Blog.tsx` to include the Mermaid component in the ReactMarkdown configuration.

### Option 2: Keep as Code Blocks

The diagrams are well-structured and readable as code. Readers familiar with Mermaid will understand the flow.

### Current Articles with Diagrams:
- **Council of LLMs**: System architecture, use case flows, performance trade-offs
- **Kolmogorov-Arnold Networks**: MLP vs KAN comparison, implementation architecture

---

## 🎯 Current Must-Read Articles

The portfolio now includes **5 comprehensive technical articles**:

1. **Agentic AI in Data Engineering** (8 min) - Multi-agent systems for data pipelines
2. **Apache Spark ETL Best Practices** (12 min) - Production-grade Spark optimization
3. **MLOps for Production** (15 min) - End-to-end ML deployment
4. **Council of LLMs** (14 min) - Multi-agent consensus systems with Mermaid diagrams
5. **Kolmogorov-Arnold Networks** (16 min) - Revolutionary neural network architecture

All articles include:
- ✅ Real code examples
- ✅ Architecture diagrams
- ✅ Production insights
- ✅ Best practices
- ✅ Performance metrics

Total: **65+ minutes** of in-depth technical content!
