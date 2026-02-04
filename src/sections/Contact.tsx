import { motion } from 'framer-motion';
import { Linkedin, MapPin, Send } from 'lucide-react';
import { useState } from 'react';

const Contact = () => {
  const [formStatus, setFormStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');

  const socials = [
    { icon: Linkedin, label: "LinkedIn", value: "linkedin.com/in/shubhamdwivedi", href: "https://linkedin.com/in/shubhamdwivedi" },
    { icon: MapPin, label: "Location", value: "Germany", href: "https://www.google.com/maps/place/Germany" },
  ];

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setFormStatus('submitting');

    const form = e.currentTarget;
    const formData = new FormData(form);

    try {
      const response = await fetch('/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(formData as any).toString(),
      });

      if (response.ok) {
        setFormStatus('success');
        form.reset();
        setTimeout(() => setFormStatus('idle'), 5000);
      } else {
        setFormStatus('error');
      }
    } catch (error) {
      setFormStatus('error');
      setTimeout(() => setFormStatus('idle'), 5000);
    }
  };

  return (
    <section id="contact" className="relative py-32 bg-dark-bg overflow-hidden">
      <div className="absolute top-20 left-10 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-lime/10 rounded-full blur-[120px]" />

      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-7xl font-bold mb-6 text-gradient">Let's Connect</h2>
          <p className="text-light-gray text-lg md:text-xl max-w-2xl mx-auto">
            Interested in collaboration? Drop me a message and let's build something amazing together.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-dark-card border border-light-gray/10 rounded-2xl p-8"
          >
            <h3 className="text-2xl font-display font-bold mb-6 text-white">Send a Message</h3>
            
            <form 
              name="contact" 
              method="POST" 
              data-netlify="true"
              netlify-honeypot="bot-field"
              onSubmit={handleSubmit}
              className="space-y-6"
            >
              {/* Hidden fields for Netlify */}
              <input type="hidden" name="form-name" value="contact" />
              <input type="hidden" name="bot-field" />

              <div>
                <label htmlFor="name" className="block text-sm font-medium text-light-gray mb-2">
                  Your Name *
                </label>
                <input
                  type="text"
                  name="name"
                  id="name"
                  required
                  className="w-full px-4 py-3 bg-dark-bg border border-light-gray/20 rounded-lg text-white placeholder-light-gray/50 focus:outline-none focus:border-lime transition-colors"
                  placeholder="John Doe"
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-light-gray mb-2">
                  Your Email *
                </label>
                <input
                  type="email"
                  name="email"
                  id="email"
                  required
                  className="w-full px-4 py-3 bg-dark-bg border border-light-gray/20 rounded-lg text-white placeholder-light-gray/50 focus:outline-none focus:border-lime transition-colors"
                  placeholder="john@example.com"
                />
              </div>

              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-light-gray mb-2">
                  Subject *
                </label>
                <input
                  type="text"
                  name="subject"
                  id="subject"
                  required
                  className="w-full px-4 py-3 bg-dark-bg border border-light-gray/20 rounded-lg text-white placeholder-light-gray/50 focus:outline-none focus:border-lime transition-colors"
                  placeholder="Project Collaboration"
                />
              </div>

              <div>
                <label htmlFor="message" className="block text-sm font-medium text-light-gray mb-2">
                  Message *
                </label>
                <textarea
                  name="message"
                  id="message"
                  rows={5}
                  required
                  className="w-full px-4 py-3 bg-dark-bg border border-light-gray/20 rounded-lg text-white placeholder-light-gray/50 focus:outline-none focus:border-lime transition-colors resize-none"
                  placeholder="Tell me about your project or inquiry..."
                />
              </div>

              <button
                type="submit"
                disabled={formStatus === 'submitting'}
                className="w-full py-4 bg-gradient-to-r from-purple-500 to-lime text-dark-bg font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {formStatus === 'submitting' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-dark-bg border-t-transparent rounded-full animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Send Message
                  </>
                )}
              </button>

              {formStatus === 'success' && (
                <div className="p-4 bg-lime/10 border border-lime/30 rounded-lg text-lime text-center">
                  ✓ Message sent successfully! I'll get back to you soon.
                </div>
              )}

              {formStatus === 'error' && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-center">
                  ✗ Failed to send message. Please try again or reach out via LinkedIn.
                </div>
              )}
            </form>
          </motion.div>

          {/* Social Links */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-dark-card border border-light-gray/10 rounded-2xl p-8">
              <h3 className="text-2xl font-display font-bold mb-6 text-white">Connect With Me</h3>
              <div className="space-y-4">
                {socials.map((social, index) => (
                  <motion.a
                    key={social.label}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className="group flex items-center gap-4 p-4 rounded-lg hover:bg-dark-bg transition-colors"
                    whileHover={{ x: 5 }}
                  >
                    <div className="w-12 h-12 rounded-full bg-lime/10 flex items-center justify-center group-hover:bg-lime/20 transition-colors">
                      <social.icon className="w-6 h-6 text-lime" />
                    </div>
                    <div>
                      <div className="font-display font-semibold text-lg mb-1">{social.label}</div>
                      <div className="text-light-gray/70 text-sm">{social.value}</div>
                    </div>
                  </motion.a>
                ))}
              </div>
            </div>

            <div className="bg-dark-card border border-light-gray/10 rounded-2xl p-8">
              <h3 className="text-xl font-display font-bold mb-4 text-white">Quick Info</h3>
              <div className="space-y-3 text-light-gray">
                <p className="flex items-start gap-2">
                  <span className="text-lime mt-1">▹</span>
                  <span>Based in Hildesheim, Germany</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-lime mt-1">▹</span>
                  <span>Available for remote opportunities</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-lime mt-1">▹</span>
                  <span>Open to freelance projects</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-lime mt-1">▹</span>
                  <span>Response time: 24-48 hours</span>
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="text-center">
          <p className="text-light-gray/50 text-sm mb-4">
            © 2025 Shubham Dwivedi. All rights reserved.
          </p>
          <div className="flex justify-center gap-6 text-sm text-light-gray/50">
            {["Home", "About", "Projects", "Blog"].map((link) => (
              <a key={link} href={`#${link.toLowerCase()}`} className="hover:text-lime transition-colors">
                {link}
              </a>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Contact;
