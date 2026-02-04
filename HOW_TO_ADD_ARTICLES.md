# How to Add New Must-Read Blog Articles

## Quick Guide (3 Easy Steps!)

### Step 1️⃣: Add your image
Put your blog image (JPG/PNG) in the `public/` folder.
Example: `public/blog-my-new-article.jpg`

### Step 2️⃣: Edit the blog articles file
Open: `src/data/blog-articles.ts`

Add this template at the end of the `blogArticles` array (before the closing `]`):

```typescript
,
{
  id: "my-new-article",
  title: "My Amazing Article Title",
  excerpt: "A brief description of what this article is about. Keep it to 2-3 sentences.",
  content: `# Main Article Title

Write your full article content here using Markdown!

## Why This Is Cool

You can include:
- Bullet points
- **Bold text**
- *Italic text*
- \`inline code\`

### Code Examples

\`\`\`python
def example():
    return "You can add code blocks too!"
\`\`\`

## Another Section

Write as much as you want! The full article appears in a beautiful modal when users click the card.

### Key Takeaways

1. First point
2. Second point
3. Third point

*Make it as long as you need - users will enjoy reading the full content!*
  `,
  image: "/blog-my-new-article.jpg",
  date: "Feb 3, 2025",
  readTime: "8 min",
  category: "AI/ML",  // Options: "AI/ML", "Data Engineering", "MLOps"
  featured: false,    // true = shows FEATURED badge
  mustRead: true,     // true = appears in Must-Read section
  color: "#a78bfa"    // Purple: #a78bfa, Lime: #d1e29d, Pink: #f472b6, Blue: #60a5fa
}
```

### Step 3️⃣: Build and see it!
```bash
npm run build
```

## 🎨 Choosing Colors

- **Purple** `#a78bfa` - AI/ML topics
- **Lime** `#d1e29d` - Data Engineering
- **Pink** `#f472b6` - MLOps
- **Blue** `#60a5fa` - General tech

## 📝 Markdown Tips

### Headings
```markdown
# H1 - Main title
## H2 - Section
### H3 - Subsection
```

### Lists
```markdown
- Bullet point
- Another point

1. Numbered item
2. Another item
```

### Links
```markdown
[Link text](https://example.com)
```

### Emphasis
```markdown
**bold text**
*italic text*
`code`
```

### Code Blocks
````markdown
```python
def hello():
    print("Hello!")
```
````

### Quotes
```markdown
> This is a quote
```

## 🚀 Categories Explained

- **"AI/ML"** - Machine learning, AI, neural networks, etc.
- **"Data Engineering"** - ETL, pipelines, data processing
- **"MLOps"** - ML deployment, monitoring, production

## ⭐ Featured vs Must-Read

- **featured: true** → Shows a "FEATURED" badge on the card (use for your best content)
- **mustRead: true** → Article appears in the special "Must-Read Articles" section at the bottom

You can set both to `true` for your absolute best articles!

## 💡 Pro Tips

1. **Keep excerpts short** - They show on the card, so 2-3 sentences max
2. **Write long content** - The modal shows the full article, so make it comprehensive!
3. **Use headers** - Break up your content with ## and ### headers
4. **Add code examples** - Use code blocks for technical articles
5. **Include images** - Put them in `public/` and reference as `/image.jpg`
6. **Accurate read time** - Estimate ~200 words per minute

## 📊 Example of a Great Article

```typescript
{
  id: "kubernetes-ml",
  title: "Deploying ML Models on Kubernetes",
  excerpt: "A complete guide to containerizing and deploying machine learning models using Kubernetes. Learn best practices for scaling, monitoring, and managing ML workloads.",
  content: `# Deploying ML Models on Kubernetes

Kubernetes has become the standard for deploying ML models at scale...

## Why Kubernetes for ML?

1. **Scalability** - Auto-scaling based on load
2. **Reliability** - Self-healing containers
3. **Flexibility** - Works with any ML framework

[Full detailed content here...]
  `,
  image: "/blog-kubernetes-ml.jpg",
  date: "Feb 3, 2025",
  readTime: "12 min",
  category: "MLOps",
  featured: true,
  mustRead: true,
  color: "#f472b6"
}
```

---

**Questions?** Check the main README.md for more details!
