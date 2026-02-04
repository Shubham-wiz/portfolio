# 🚀 Deployment Guide for shubhamdwivedi.me / .tech

You have everything you need! Just follow these steps:

## ✅ What You Already Have
- ✅ Built website (in `dist/` folder)
- ✅ Two domains: `shubhamdwivedi.me` and `shubhamdwivedi.tech`

## 🎯 What You Need (All FREE!)
- ✅ **Hosting Platform** (Netlify, Vercel, or Cloudflare Pages - all free)
- ✅ **SSL Certificate** (provided automatically by hosting platforms)
- ✅ **Git Repository** (optional but recommended - GitHub/GitLab)

---

## 🏆 Recommended: Netlify (Easiest!)

Netlify is perfect for static sites like yours. Everything is free and takes 10 minutes.

### Step 1: Deploy to Netlify

#### Option A: Drag & Drop (Fastest - 2 minutes)

1. **Go to**: https://app.netlify.com/drop
2. **Drag the `dist/` folder** onto the page
3. **Done!** Your site is live at: `random-name-123.netlify.app`

#### Option B: Git Deploy (Recommended for updates)

1. **Create GitHub Repository**:
   ```bash
   cd /mnt/c/Users/DOJO/Downloads/websites/Kimi_Agent_Deployment_v2
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/portfolio.git
   git push -u origin main
   ```

2. **Connect to Netlify**:
   - Go to: https://app.netlify.com
   - Click "Add new site" → "Import an existing project"
   - Choose GitHub
   - Select your repository
   - **Build settings**:
     - Build command: `npm run build`
     - Publish directory: `dist`
   - Click "Deploy site"

### Step 2: Connect Your Custom Domains

#### For shubhamdwivedi.tech (Primary)

1. **In Netlify**:
   - Go to: Site settings → Domain management
   - Click "Add custom domain"
   - Enter: `shubhamdwivedi.tech`
   - Click "Verify"

2. **In Namecheap**:
   - Log in to Namecheap
   - Go to Domain List → Manage `shubhamdwivedi.tech`
   - Navigate to: Advanced DNS
   - **Delete all existing records**
   - **Add these records**:

   | Type  | Host | Value                | TTL  |
   |-------|------|----------------------|------|
   | A     | @    | 75.2.60.5           | Auto |
   | CNAME | www  | YOUR-SITE.netlify.app | Auto |

   *(Replace `YOUR-SITE` with your actual Netlify subdomain)*

3. **Add www subdomain** (optional but recommended):
   - In Netlify: Add `www.shubhamdwivedi.tech` as domain alias
   - Already configured in DNS above

#### For shubhamdwivedi.me (Secondary/Redirect)

1. **In Netlify**:
   - Add `shubhamdwivedi.me` as domain alias
   - Netlify will auto-redirect to your primary domain

2. **In Namecheap**:
   - Same DNS settings as above for `.me` domain

### Step 3: Enable HTTPS (Automatic!)

- Netlify provides **free SSL certificate** automatically
- Takes 10-30 minutes to activate
- Force HTTPS in Netlify settings → Domain management

### Step 4: Performance Optimizations (Optional)

In Netlify settings:

- ✅ **Asset Optimization**: Enable
- ✅ **Pretty URLs**: Enable
- ✅ **Prerendering**: Enable

---

## 🔥 Alternative: Vercel (Also Excellent!)

Vercel is equally good and just as easy.

### Deploy to Vercel

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy**:
   ```bash
   cd /mnt/c/Users/DOJO/Downloads/websites/Kimi_Agent_Deployment_v2
   vercel
   ```
   - Follow prompts
   - Select: `dist` as output directory

3. **Connect Custom Domain**:
   - Go to: https://vercel.com/dashboard
   - Select your project → Settings → Domains
   - Add: `shubhamdwivedi.tech`

4. **Namecheap DNS Settings**:

   | Type  | Host | Value                      | TTL  |
   |-------|------|----------------------------|------|
   | A     | @    | 76.76.21.21               | Auto |
   | CNAME | www  | cname.vercel-dns.com      | Auto |

---

## ⚡ Alternative: Cloudflare Pages (Best Performance!)

Cloudflare Pages is super fast with global CDN.

### Deploy to Cloudflare Pages

1. **Go to**: https://pages.cloudflare.com
2. **Create account** (free)
3. **Create project**:
   - Connect GitHub repo OR
   - Upload `dist/` folder directly
4. **Build settings**:
   - Build command: `npm run build`
   - Output directory: `dist`

5. **Connect Domain**:
   - Cloudflare can manage your DNS automatically
   - Transfer nameservers OR use CNAME setup

---

## 📊 Comparison Table

| Feature           | Netlify | Vercel | Cloudflare Pages |
|-------------------|---------|--------|------------------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Free SSL**      | ✅      | ✅     | ✅               |
| **CDN**           | ✅      | ✅     | ✅ (Best!)       |
| **Build Minutes** | 300/mo  | 6000/mo| Unlimited        |
| **Git Deploy**    | ✅      | ✅     | ✅               |
| **Custom Headers**| ✅      | ✅     | ✅               |
| **Analytics**     | ✅      | ✅     | ✅               |

**Recommendation**: Start with **Netlify** (easiest), switch to Cloudflare later if you want max performance.

---

## 🎯 Complete Deployment Checklist

### Before Deployment
- [x] Website built (`npm run build`)
- [x] Test locally (`npm run preview`)
- [x] All images in `public/` folder
- [x] Update contact info in code (if needed)

### DNS Configuration (Namecheap)
- [ ] Log in to Namecheap
- [ ] Go to Advanced DNS for `shubhamdwivedi.tech`
- [ ] Add A record: `@` → `75.2.60.5` (Netlify IP)
- [ ] Add CNAME: `www` → `your-site.netlify.app`
- [ ] Repeat for `shubhamdwivedi.me`
- [ ] Wait 5-30 minutes for DNS propagation

### Hosting Platform
- [ ] Choose platform (Netlify recommended)
- [ ] Deploy site
- [ ] Add custom domains
- [ ] Enable HTTPS (automatic)
- [ ] Test: https://shubhamdwivedi.tech

### Post-Deployment
- [ ] Test all pages and links
- [ ] Test on mobile devices
- [ ] Check blog article modals
- [ ] Verify images load correctly
- [ ] Test contact form/links
- [ ] Check SEO meta tags (View Page Source)

---

## 🔍 DNS Propagation Check

After updating DNS, check status:
- https://dnschecker.org
- Enter: `shubhamdwivedi.tech`
- Should see your hosting provider's IP globally

Typical propagation time: **5-30 minutes** (can be up to 48 hours)

---

## 🚨 Troubleshooting

### "Domain not working after 24 hours"
- Check DNS records in Namecheap (no typos)
- Clear browser cache (Ctrl + Shift + Delete)
- Try incognito/private browsing
- Check https://dnschecker.org

### "SSL Certificate Error"
- Wait 30 minutes after adding domain
- Check Netlify/Vercel dashboard for SSL status
- May need to click "Provision certificate" manually

### "Page shows Netlify 404"
- Check build output directory is `dist`
- Verify `index.html` is in root of `dist/`
- Rebuild site: `npm run build`

### "Images not loading"
- Check images are in `public/` folder
- Verify image paths start with `/` (e.g., `/blog-mlops.jpg`)
- Rebuild and redeploy

---

## 🎨 Custom Configurations

### Redirect www to non-www (or vice versa)

**In Netlify**, create `public/_redirects`:
```
https://www.shubhamdwivedi.tech/* https://shubhamdwivedi.tech/:splat 301!
```

**In Vercel**, add to `vercel.json`:
```json
{
  "redirects": [
    {
      "source": "https://www.shubhamdwivedi.tech/:path*",
      "destination": "https://shubhamdwivedi.tech/:path*",
      "permanent": true
    }
  ]
}
```

### Custom Headers (Security)

Create `public/_headers`:
```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  X-XSS-Protection: 1; mode=block
  Referrer-Policy: strict-origin-when-cross-origin
```

---

## 📈 After Launch

### Analytics (Free Options)
1. **Cloudflare Analytics** (if using Cloudflare)
2. **Netlify Analytics** ($9/month)
3. **Google Analytics** (free - add to `index.html`)
4. **Plausible Analytics** (privacy-focused, paid)

### Monitor Uptime
- **UptimeRobot** (free): https://uptimerobot.com
- Set up monitoring for both domains

### SEO
- Submit sitemap to Google Search Console
- Submit to Bing Webmaster Tools
- Share on LinkedIn, Twitter

---

## 🎉 Quick Start (TL;DR)

**Fastest path to live website (10 minutes):**

1. **Deploy**:
   ```bash
   # Go to: https://app.netlify.com/drop
   # Drag 'dist' folder
   ```

2. **Update DNS** (Namecheap):
   - A record: `@` → `75.2.60.5`
   - CNAME: `www` → `your-site.netlify.app`

3. **Add Domain** (Netlify):
   - Site settings → Add custom domain
   - Enter: `shubhamdwivedi.tech`

4. **Wait 10-30 minutes** for DNS + SSL

5. **Done!** Visit: https://shubhamdwivedi.tech

---

## 💡 Pro Tips

1. **Use both domains**:
   - `.tech` for main portfolio (more professional)
   - `.me` for personal brand/redirect

2. **Set up email forwarding**:
   - Namecheap offers free email forwarding
   - Forward `hello@shubhamdwivedi.tech` → your email

3. **Add a custom 404 page**:
   - Create `public/404.html`
   - Style it to match your site

4. **Enable form submissions**:
   - Netlify Forms (free, built-in)
   - Add to contact form: `data-netlify="true"`

5. **Continuous deployment**:
   - Push to GitHub → Auto-deploy
   - No manual uploads needed

---

## 📞 Need Help?

- **Netlify Docs**: https://docs.netlify.com
- **Vercel Docs**: https://vercel.com/docs
- **Namecheap Support**: https://www.namecheap.com/support/

**Common Issues**: Check the Troubleshooting section above first!

---

**Estimated Total Time**: 15-30 minutes (including DNS propagation)

**Total Cost**: $0/month (everything is free!)

Good luck with your launch! 🚀
