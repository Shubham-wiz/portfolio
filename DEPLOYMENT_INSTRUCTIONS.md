# 🚀 Deploy Your Portfolio to Netlify

## Quick Deployment (5 minutes)

### **Option 1: Drag & Drop (Easiest)**

1. **Build your site:**
   ```bash
   npm run build
   ```
   This creates a `dist` folder with your production website.

2. **Go to Netlify:**
   - Visit: https://app.netlify.com/
   - Sign up (free) using GitHub, GitLab, or email

3. **Deploy:**
   - Click **"Add new site"** → **"Deploy manually"**
   - **Drag & drop** the entire `dist` folder
   - Wait 30 seconds... **Done!** ✅

4. **Your site is live!**
   - You'll get a random URL like: `https://random-name-123.netlify.app`

---

### **Option 2: Connect GitHub (Automatic Updates)**

1. **Initialize Git (if not already):**
   ```bash
   cd /mnt/c/Users/DOJO/Downloads/websites/Kimi_Agent_Deployment_v2
   git init
   git add .
   git commit -m "Initial commit - Portfolio website"
   ```

2. **Create GitHub repo:**
   - Go to https://github.com/new
   - Create a new repository (e.g., "portfolio")
   - **Don't** initialize with README

3. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/portfolio.git
   git branch -M main
   git push -u origin main
   ```

4. **Connect to Netlify:**
   - Go to https://app.netlify.com/
   - Click **"Add new site"** → **"Import an existing project"**
   - Choose **GitHub**
   - Select your portfolio repository
   
5. **Configure build settings:**
   ```
   Build command: npm run build
   Publish directory: dist
   ```

6. **Click "Deploy site"**
   - Netlify auto-deploys every time you push to GitHub! 🎉

---

## 🌐 Add Your Custom Domains

### **After deployment, add your domains:**

1. **In Netlify Dashboard:**
   - Go to **Site settings** → **Domain management**
   - Click **"Add custom domain"**

2. **Add both domains:**
   - `shubhamdwivedi.tech`
   - `shubhamdwivedi.me`

3. **Update DNS settings:**

   **For each domain, add these DNS records at your domain registrar:**

   **A Record:**
   ```
   Type: A
   Name: @
   Value: 75.2.60.5
   ```

   **CNAME Record (www):**
   ```
   Type: CNAME
   Name: www
   Value: YOUR_SITE.netlify.app
   ```

   **Alternative (if CNAME for root isn't supported):**
   Add these 4 A records:
   ```
   Type: A, Name: @, Value: 75.2.60.5
   Type: A, Name: @, Value: 99.83.190.102
   Type: A, Name: @, Value: 13.248.212.111
   Type: A, Name: @, Value: 3.33.246.48
   ```

4. **Enable HTTPS:**
   - Netlify automatically provides free SSL certificates
   - Click **"Verify DNS configuration"**
   - Click **"Provision certificate"**
   - Wait 1-24 hours for DNS propagation

---

## 📧 Enable Contact Form Emails

### **After deployment:**

1. **Go to Netlify Dashboard**
   - Navigate to your site
   - Go to **Forms** tab

2. **Add email notification:**
   - Click **"Settings and Usage"**
   - Click **"Add notification"** → **"Email notification"**
   - Enter: `sdwivedi@gmx.de`
   - Select: "New form submission"
   - Save

3. **Test the form:**
   - Go to your live site
   - Fill out the contact form
   - Submit
   - Check your email (sdwivedi@gmx.de)
   - **Note**: Form won't work on localhost, only on deployed site!

---

## ✅ Post-Deployment Checklist

After deploying, verify:

- [ ] Site loads on your Netlify URL
- [ ] All sections appear correctly (Hero, About, Experience, Projects, Blog, Contact)
- [ ] 3D tilt effects work on cards
- [ ] Modal scrolling works (background locked)
- [ ] Blog article images load (KAN, Council headers)
- [ ] Contact form appears
- [ ] Custom domains pointing correctly (if configured)
- [ ] HTTPS enabled (green padlock)
- [ ] Contact form sends emails to sdwivedi@gmx.de

---

## 🔄 Update Your Site Later

### **Option 1: If using Drag & Drop:**
```bash
# Make your changes, then:
npm run build
# Drag & drop the new dist folder to Netlify
```

### **Option 2: If using GitHub:**
```bash
# Make your changes, then:
git add .
git commit -m "Update portfolio"
git push
# Netlify automatically rebuilds and deploys!
```

---

## 💰 Cost

**Netlify Free Tier includes:**
- ✅ 100 GB bandwidth/month
- ✅ 300 build minutes/month
- ✅ 100 form submissions/month
- ✅ Free SSL certificates
- ✅ Custom domains
- ✅ Automatic deployments

**Perfect for a personal portfolio! $0/month** 🎉

---

## 🆘 Troubleshooting

### **Contact form not working:**
- ✅ Make sure you're testing on the **live deployed site**, not localhost
- ✅ Check Netlify Dashboard → Forms → See if submissions appear
- ✅ Add email notification in Netlify settings
- ✅ Check spam folder

### **Custom domain not working:**
- ✅ Wait 24 hours for DNS propagation
- ✅ Double-check DNS records at your registrar
- ✅ Use https://dnschecker.org to verify DNS

### **Site not updating:**
- ✅ Clear browser cache (Ctrl+Shift+R)
- ✅ Check Netlify deploy logs for errors
- ✅ Rebuild: Deploys → Trigger deploy → Deploy site

---

## 📱 Next Steps

1. Deploy to Netlify (5 minutes)
2. Configure custom domains (30 minutes + DNS wait)
3. Test contact form on live site
4. Share your portfolio on LinkedIn!

**Your portfolio URL will be:**
- Netlify: `https://YOUR_SITE.netlify.app`
- Custom: `https://shubhamdwivedi.tech`
- Custom: `https://shubhamdwivedi.me`

---

**Need help?** Netlify has excellent docs: https://docs.netlify.com
