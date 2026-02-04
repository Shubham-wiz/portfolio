# 🚀 Quick Start - Get Live in 10 Minutes!

## You Need (All FREE):
1. ✅ Domains: `shubhamdwivedi.tech` + `shubhamdwivedi.me` (you have these!)
2. ✅ Website: Built and ready in `dist/` folder (done!)
3. ⏳ Netlify account (free, takes 1 minute)

---

## Step 1: Deploy to Netlify (2 minutes)

### Option A: Drag & Drop (Easiest!)

1. **Open**: https://app.netlify.com/drop
2. **Sign up** with GitHub/Email (free)
3. **Drag the `dist` folder** onto the page
4. **Done!** Your site is now at: `https://random-name.netlify.app`

### Option B: Git Deploy (For Future Updates)

```bash
# Initialize git (if not done)
cd /mnt/c/Users/DOJO/Downloads/websites/Kimi_Agent_Deployment_v2
git init
git add .
git commit -m "Portfolio website"

# Push to GitHub (create repo first at github.com)
git remote add origin https://github.com/YOUR_USERNAME/portfolio.git
git push -u origin main

# Then connect to Netlify via dashboard
```

---

## Step 2: Connect Your Domains (5 minutes)

### In Netlify Dashboard:

1. Go to: **Site settings** → **Domain management**
2. Click: **Add custom domain**
3. Enter: `shubhamdwivedi.tech`
4. Copy the Netlify DNS info shown (something like `your-site.netlify.app`)

### In Namecheap:

1. **Log in** to Namecheap
2. Go to: **Domain List** → **Manage** (for `shubhamdwivedi.tech`)
3. Click: **Advanced DNS**
4. **Delete all existing records**
5. **Add these 2 records**:

```
Type: A Record
Host: @
Value: 75.2.60.5
TTL: Automatic

Type: CNAME Record
Host: www
Value: YOUR-SITE.netlify.app  (replace with your actual Netlify subdomain)
TTL: Automatic
```

6. **Repeat for `shubhamdwivedi.me`** domain

---

## Step 3: Wait (10-30 minutes)

- ⏰ DNS propagation: 5-30 minutes
- 🔒 SSL certificate: Activates automatically
- ✅ Check status at: https://dnschecker.org

---

## Step 4: Test Your Site!

After 30 minutes, visit:
- ✅ https://shubhamdwivedi.tech
- ✅ https://www.shubhamdwivedi.tech
- ✅ https://shubhamdwivedi.me

**All should redirect to your main domain with HTTPS!**

---

## 🎉 You're Live!

### What You Get:
- ✅ **HTTPS/SSL**: Automatic & free
- ✅ **Global CDN**: Fast worldwide
- ✅ **Unlimited bandwidth**: No limits
- ✅ **Auto-deploy**: Push to GitHub = auto-update
- ✅ **100/month builds**: More than enough

### Total Cost: **$0/month** 🎁

---

## 🔥 Next Steps (Optional):

### 1. Set up Email Forwarding
In Namecheap:
- Go to: Domain → Email Forwarding
- Create: `hello@shubhamdwivedi.tech` → your email
- Create: `contact@shubhamdwivedi.tech` → your email

### 2. Update Contact Info
Edit `src/sections/Contact.tsx`:
```typescript
const socials = [
  { ..., value: "your-actual-email@gmail.com", href: "mailto:your-email" },
  { ..., value: "linkedin.com/in/YOUR-LINKEDIN", href: "https://..." },
  // etc.
];
```

Then rebuild: `npm run build` and redeploy

### 3. Enable Analytics
In Netlify:
- Go to: **Analytics** (paid) or
- Add Google Analytics to `index.html` (free)

### 4. Monitor Uptime
- Sign up: https://uptimerobot.com (free)
- Add monitor for: `shubhamdwivedi.tech`
- Get alerts if site goes down

---

## 🚨 Troubleshooting:

### "My domain shows 'DNS_PROBE_FINISHED_NXDOMAIN'"
- DNS hasn't propagated yet (wait 30 more minutes)
- Check DNS records in Namecheap (no typos in A record/CNAME)
- Clear browser cache: Ctrl+Shift+Delete

### "Site shows 'Not Secure' warning"
- Wait 30 minutes for SSL to activate
- In Netlify: Domain management → Click "Verify DNS configuration"
- Force HTTPS in Netlify settings

### "Images not showing"
- Check images are in `public/` folder
- Rebuild: `npm run build`
- Redeploy to Netlify

---

## 📞 Need Help?

Check the full **DEPLOYMENT_GUIDE.md** for detailed troubleshooting.

**Questions?**
- Netlify Docs: https://docs.netlify.com
- Namecheap Support: https://www.namecheap.com/support/

---

## ✅ Deployment Checklist:

- [ ] Netlify account created
- [ ] Site deployed (drag & drop `dist/`)
- [ ] Custom domain added in Netlify
- [ ] DNS records updated in Namecheap (A + CNAME)
- [ ] Waited 30 minutes for DNS propagation
- [ ] Site loads at https://shubhamdwivedi.tech ✨
- [ ] SSL shows padlock icon 🔒
- [ ] All pages working (test blog articles!)
- [ ] Mobile responsive test
- [ ] Share on LinkedIn! 🎉

---

**That's it! Welcome to the internet! 🌐**
