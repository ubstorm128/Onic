/**
 * Robust Multimodal AI for Multi-Cancer Classification
 * Interactive Logic, Navigation Spy, Cursor Spotlight & Real-time Diagnostic Simulation
 */

(function () {
  'use strict';

  /* ==========================================================================
     Theme Management with Smooth View Transitions
     ========================================================================== */
  const root = document.documentElement;
  const themeToggle = document.getElementById('tt');

  function getActiveTheme() {
    const savedTheme = localStorage.getItem('theme_preference');
    if (savedTheme) return savedTheme;
    const attr = root.getAttribute('data-theme');
    if (attr) return attr;
    return 'light'; // Default to light mode
  }

  function syncThemeUI(theme) {
    if (themeToggle) {
      themeToggle.classList.toggle('flip', theme === 'dark');
      themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    }
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem('theme_preference', theme);
    syncThemeUI(theme);
  }

  function triggerButtonRipple(button, event, targetTheme) {
    if (!button) return;
    const rect = button.getBoundingClientRect();
    const ripple = document.createElement('span');
    ripple.className = 'btn-ripple';

    const diameter = Math.max(rect.width, rect.height) * 1.6;
    ripple.style.width = `${diameter}px`;
    ripple.style.height = `${diameter}px`;

    let clickX = rect.width / 2;
    let clickY = rect.height / 2;
    if (event && event.clientX && event.clientY) {
      clickX = event.clientX - rect.left;
      clickY = event.clientY - rect.top;
    }

    ripple.style.left = `${clickX - diameter / 2}px`;
    ripple.style.top = `${clickY - diameter / 2}px`;

    ripple.style.background = targetTheme === 'dark'
      ? 'rgba(56, 189, 248, 0.35)'
      : 'rgba(0, 75, 181, 0.25)';

    button.appendChild(ripple);
    setTimeout(() => {
      ripple.remove();
    }, 650);
  }

  // Initialize theme
  const initialTheme = getActiveTheme();
  applyTheme(initialTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', function (e) {
      const currentTheme = getActiveTheme();
      const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';

      // Physical tactile ripple on the toggle button
      triggerButtonRipple(themeToggle, e, nextTheme);

      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      if (!document.startViewTransition || reduceMotion) {
        root.classList.add('theme-transitioning');
        applyTheme(nextTheme);
        setTimeout(() => root.classList.remove('theme-transitioning'), 450);
        return;
      }

      const rect = themeToggle.getBoundingClientRect();
      const x = Math.round(rect.left + rect.width / 2);
      const y = Math.round(rect.top + rect.height / 2);
      const endRadius = Math.ceil(
        Math.hypot(
          Math.max(x, window.innerWidth - x),
          Math.max(y, window.innerHeight - y)
        )
      );

      const transition = document.startViewTransition(() => {
        applyTheme(nextTheme);
      });

      transition.ready.then(() => {
        root.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`
            ]
          },
          {
            duration: 520,
            easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
            pseudoElement: '::view-transition-new(root)'
          }
        );
      }).catch(() => {
        applyTheme(nextTheme);
      });
    });
  }

  /* ==========================================================================
     Scroll Progress Indicator & Sticky Navigation Dynamics
     ========================================================================== */
  const progressBar = document.getElementById('scroll-progress');
  const navWrapper = document.getElementById('nav-wrapper') || document.querySelector('.nav-wrapper');
  const navbar = document.getElementById('navbar');
  const navIndicator = document.querySelector('.nav-indicator');
  const navLinks = document.querySelectorAll('nav .nav-links a.nav-link');
  const sections = document.querySelectorAll('section[id]');

  function updateScrollState() {
    const scrollY = window.scrollY;

    // Update Progress Bar
    if (progressBar) {
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = docHeight > 0 ? (scrollY / docHeight) * 100 : 0;
      progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
    }

    // Toggle Scrolled Class for Full-Width Blurred Navbar Line
    const isScrolled = scrollY > 15;
    if (navWrapper) {
      navWrapper.classList.toggle('scrolled', isScrolled);
    }
    if (navbar) {
      navbar.classList.toggle('scrolled', isScrolled);
    }
  }

  window.addEventListener('scroll', updateScrollState, { passive: true });
  updateScrollState();

  /* ==========================================================================
     Sliding Active Capsule Indicator for Navigation
     ========================================================================== */
  function moveNavIndicator(activeLink) {
    if (!navIndicator || !activeLink) return;
    const navLinksContainer = activeLink.parentElement;
    const linkRect = activeLink.getBoundingClientRect();
    const containerRect = navLinksContainer.getBoundingClientRect();

    const left = linkRect.left - containerRect.left;
    const width = linkRect.width;

    navIndicator.style.transform = `translateX(${left}px)`;
    navIndicator.style.width = `${width}px`;
    navIndicator.style.opacity = '1';
  }

  // Initialize indicator on the active link
  const currentActiveLink = document.querySelector('nav .nav-links a.nav-link.active');
  if (currentActiveLink) {
    setTimeout(() => moveNavIndicator(currentActiveLink), 150);
  }

  // Update on window resize
  window.addEventListener('resize', () => {
    const active = document.querySelector('nav .nav-links a.nav-link.active');
    if (active) moveNavIndicator(active);
  }, { passive: true });

  // Scrollspy via IntersectionObserver
  if (sections.length && navLinks.length) {
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -65% 0px',
      threshold: 0
    };

    const sectionObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const currentId = entry.target.getAttribute('id');
          navLinks.forEach((link) => {
            const href = link.getAttribute('href').replace('#', '');
            if (href === currentId) {
              link.classList.add('active');
              moveNavIndicator(link);
            } else {
              link.classList.remove('active');
            }
          });

          // Sync mobile links too
          document.querySelectorAll('.mobile-drawer .mobile-link').forEach((mLink) => {
            const mHref = mLink.getAttribute('href').replace('#', '');
            mLink.classList.toggle('active', mHref === currentId);
          });
        }
      });
    }, observerOptions);

    sections.forEach((sec) => sectionObserver.observe(sec));
  }

  // Precise & Smooth Section Jump for Navigation Links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#' || targetId === '') return;
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();

        if (targetId === '#hero') {
          window.scrollTo({
            top: 0,
            behavior: 'smooth'
          });
          closeMobileMenu();
          return;
        }

        // Calculate exact offset so the section heading or content aligns neatly below the navbar
        const navHeight = navWrapper ? navWrapper.getBoundingClientRect().height : 60;
        const heading = targetElement.querySelector('h1, h2, .section-tag') || targetElement;
        const headingTop = heading.getBoundingClientRect().top + window.pageYOffset;
        const scrollTarget = Math.max(0, Math.round(headingTop - navHeight - 16));

        window.scrollTo({
          top: scrollTarget,
          behavior: 'smooth'
        });

        // Sync active state immediately on click
        navLinks.forEach((link) => {
          const href = link.getAttribute('href');
          const isMatch = href === targetId;
          link.classList.toggle('active', isMatch);
          if (isMatch) moveNavIndicator(link);
        });

        document.querySelectorAll('.mobile-drawer .mobile-link').forEach((mLink) => {
          mLink.classList.toggle('active', mLink.getAttribute('href') === targetId);
        });

        if (history.pushState) {
          history.pushState(null, null, targetId);
        }

        // Close mobile drawer if open
        closeMobileMenu();
      }
    });
  });

  /* ==========================================================================
     Mobile Drawer Menu Navigation
     ========================================================================== */
  const mobileToggle = document.getElementById('mobile-toggle');
  const mobileMenu = document.getElementById('mobile-menu');

  function openMobileMenu() {
    if (!mobileMenu || !mobileToggle) return;
    mobileMenu.classList.add('open');
    mobileMenu.setAttribute('aria-hidden', 'false');
    mobileToggle.setAttribute('aria-expanded', 'true');
  }

  function closeMobileMenu() {
    if (!mobileMenu || !mobileToggle) return;
    mobileMenu.classList.remove('open');
    mobileMenu.setAttribute('aria-hidden', 'true');
    mobileToggle.setAttribute('aria-expanded', 'false');
  }

  if (mobileToggle) {
    mobileToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = mobileMenu && mobileMenu.classList.contains('open');
      if (isOpen) {
        closeMobileMenu();
      } else {
        openMobileMenu();
      }
    });
  }

  // Close mobile drawer on outside click
  document.addEventListener('click', (e) => {
    if (mobileMenu && mobileMenu.classList.contains('open')) {
      if (!mobileMenu.contains(e.target) && !mobileToggle.contains(e.target)) {
        closeMobileMenu();
      }
    }
  });

  // Close mobile drawer on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mobileMenu && mobileMenu.classList.contains('open')) {
      closeMobileMenu();
    }
  });

  /* ==========================================================================
     Interactive Card Spotlight (Cursor Tracking Effect)
     ========================================================================== */
  const interactiveCards = document.querySelectorAll('.tile, .step, .person-card, .benchmark-card, .stage, .demo-result-card');

  interactiveCards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);
    });
  });

  /* ==========================================================================
     Back to Top Smooth Scroll
     ========================================================================== */
  const backToTopBtn = document.getElementById('back-to-top');
  if (backToTopBtn) {
    backToTopBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  /* ==========================================================================
     Intersection Reveal (.rv) Animations
     ========================================================================== */
  const revealElements = document.querySelectorAll('.rv');
  if (revealElements.length) {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          revealObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.01,
      rootMargin: '80px 0px 80px 0px'
    });

    revealElements.forEach((el, index) => {
      el.style.transitionDelay = `${(index % 4) * 60}ms`;
      revealObserver.observe(el);
    });

    // Initial check to reveal elements already in view
    setTimeout(() => {
      revealElements.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight + 100) {
          el.classList.add('in');
        }
      });
    }, 100);
  }

  /* ==========================================================================
     Animated Number Counters on Scroll
     ========================================================================== */
  const counterElements = document.querySelectorAll('[data-counter-target]');
  if (counterElements.length) {
    const counterObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const target = parseFloat(el.getAttribute('data-counter-target'));
          const suffix = el.getAttribute('data-counter-suffix') || '';
          const decimals = parseInt(el.getAttribute('data-counter-decimals') || '0', 10);
          animateCount(el, 0, target, 1600, decimals, suffix);
          counterObserver.unobserve(el);
        }
      });
    }, { threshold: 0.2 });

    counterElements.forEach((el) => counterObserver.observe(el));
  }

  function animateCount(el, start, end, duration, decimals, suffix) {
    let startTimestamp = null;
    function step(timestamp) {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const easeVal = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * easeVal;
      el.textContent = current.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      }) + suffix;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    }
    window.requestAnimationFrame(step);
  }

  /* ==========================================================================
     Interactive 4-Omics Multimodal Stage & AI Prediction Simulation
     ========================================================================== */
  const API_BASE = 'http://127.0.0.1:8000';

  // Modality States: [0: mRNA (3,217), 1: miRNA (383), 2: CNV (3,105), 3: Methy (3,139)]
  const modalityStates = [true, true, true, true];
  let activePatientIdx = 0;
  let isApiOnline = false;
  let currentConfidence = 99.6;
  let confidenceAnimFrame = null;
  let apiDebounceTimeout = null;

  // Curated Test Cohort Patient Profiles
  const TEST_PATIENTS = {
    0: {
      name: 'Patient #101',
      code: 'BRCA',
      className: 'Breast Invasive Carcinoma',
      organ: 'Breast',
      sampleId: 'TCGA-AO-A0JJ-01',
      baseConfidence: 99.6,
      subtext: 'High-confidence somatic ESR1/ERBB2 and GATA3 signaling markers with concordant multi-omic confirmation.',
      biomarkers: [
        { feature: 'SFRP1', mod: 'mRNA' },
        { feature: 'C5orf46', mod: 'mRNA' },
        { feature: 'STAC2', mod: 'mRNA' },
        { feature: 'MIA', mod: 'mRNA' }
      ]
    },
    1: {
      name: 'Patient #142',
      code: 'HNSC',
      className: 'Head and Neck Squamous Cell Carcinoma',
      organ: 'Head & Neck',
      sampleId: 'TCGA-DQ-5629-01',
      baseConfidence: 97.4,
      subtext: 'Squamous lineage biomarkers with CDKN2A deletion and high TP53 mutation signature.',
      biomarkers: [
        { feature: 'CDKN2A', mod: 'CNV' },
        { feature: 'TP53', mod: 'mRNA' },
        { feature: 'PIK3CA', mod: 'CNV' },
        { feature: 'FAT1', mod: 'Methy' }
      ]
    },
    6: {
      name: 'Patient #188',
      code: 'LGG',
      className: 'Brain Lower Grade Glioma',
      organ: 'Central Nervous System',
      sampleId: 'TCGA-DB-5273-01',
      baseConfidence: 100.0,
      subtext: 'IDH1 R132H mutation profile coupled with 1p/19q co-deletion and hypermethylated glioma phenotype.',
      biomarkers: [
        { feature: 'IDH1', mod: 'mRNA' },
        { feature: 'TP53', mod: 'mRNA' },
        { feature: 'ATRX', mod: 'Methy' },
        { feature: 'CIC', mod: 'CNV' }
      ]
    },
    2: {
      name: 'Patient #215',
      code: 'THCA',
      className: 'Thyroid Carcinoma',
      organ: 'Thyroid',
      sampleId: 'TCGA-EM-A3AI-01',
      baseConfidence: 100.0,
      subtext: 'BRAF V600E signature concordant with high thyroid differentiation score and RET/PTC alterations.',
      biomarkers: [
        { feature: 'BRAF', mod: 'mRNA' },
        { feature: 'TG', mod: 'Methy' },
        { feature: 'RET', mod: 'CNV' },
        { feature: 'TSHR', mod: 'mRNA' }
      ]
    },
    5: {
      name: 'Patient #107',
      code: 'BRCA',
      className: 'Breast Invasive Carcinoma',
      organ: 'Breast',
      sampleId: 'TCGA-AR-A24T-01',
      baseConfidence: 99.2,
      subtext: 'Luminal A phenotype characterized by ESR1 overexpression and minimal copy number instability.',
      biomarkers: [
        { feature: 'ESR1', mod: 'mRNA' },
        { feature: 'GATA3', mod: 'mRNA' },
        { feature: 'FOXA1', mod: 'mRNA' },
        { feature: 'CDH1', mod: 'CNV' }
      ]
    }
  };

  // Condition Metadata & Baselines
  const CONDITION_MAP = {
    '1-1-1-1': { id: 'C0', desc: 'Complete Quad-Modal (Reference Standard)', status: 'Optimal Quad-Modal Synergy', statusClass: 'success' },
    '0-1-1-1': { id: 'C1', desc: 'mRNA Missing (miRNA + CNV + Methy)', status: 'Robust 3-Omic (mRNA Imputed)', statusClass: 'success' },
    '1-0-1-1': { id: 'C2', desc: 'miRNA Missing (mRNA + CNV + Methy)', status: 'Robust 3-Omic (miRNA Imputed)', statusClass: 'success' },
    '1-1-0-1': { id: 'C3', desc: 'CNV Missing (mRNA + miRNA + Methy)', status: 'Robust 3-Omic (CNV Imputed)', statusClass: 'success' },
    '1-1-1-0': { id: 'C4', desc: 'Methy Missing (mRNA + miRNA + CNV)', status: 'Robust 3-Omic (Methy Imputed)', statusClass: 'success' },
    '0-0-1-1': { id: 'C5', desc: 'mRNA + miRNA Missing (CNV + Methy)', status: 'Dual-Stream (CNV + Epigenetics)', statusClass: 'warning' },
    '0-1-0-1': { id: 'C6', desc: 'mRNA + CNV Missing (miRNA + Methy)', status: 'Dual-Stream (miRNA + Epigenetics)', statusClass: 'warning' },
    '0-1-1-0': { id: 'C7', desc: 'mRNA + Methy Missing (miRNA + CNV)', status: 'Dual-Stream (miRNA + CNV)', statusClass: 'warning' },
    '1-0-0-1': { id: 'C8', desc: 'miRNA + CNV Missing (mRNA + Methy)', status: 'Dual-Stream (mRNA + Epigenetics)', statusClass: 'warning' },
    '1-0-1-0': { id: 'C9', desc: 'miRNA + Methy Missing (mRNA + CNV)', status: 'Dual-Stream (mRNA + CNV)', statusClass: 'warning' },
    '1-1-0-0': { id: 'C10', desc: 'CNV + Methy Missing (mRNA + miRNA)', status: 'Dual-Stream (RNA Expressions)', statusClass: 'warning' },
    '0-0-0-1': { id: 'C11', desc: 'Single Modality: Only DNA Methylation', status: 'Single-Stream (Methy Only)', statusClass: 'warning' },
    '0-0-1-0': { id: 'C12', desc: 'Single Modality: Only Genomic CNV (+8.5% Gain)', status: 'CNV Stress Test (S2 Dropout Gain)', statusClass: 'warning' },
    '0-1-0-0': { id: 'C13', desc: 'Single Modality: Only miRNA Present', status: 'Single-Stream (miRNA Only)', statusClass: 'warning' },
    '1-0-0-0': { id: 'C14', desc: 'Single Modality: Only mRNA Present', status: 'Single-Stream (mRNA Only)', statusClass: 'warning' },
    '0-0-0-0': { id: 'C_IDLE', desc: 'Pipeline Idle (No Active Input Streams)', status: 'Pipeline Idle (No Input)', statusClass: 'danger' }
  };

  // Base empirical attention weights from TCGA explainability report
  const BASE_WEIGHTS = { mRNA: 34.0, miRNA: 26.1, CNV: 5.7, Methy: 34.2 };

  // DOM Elements
  const meterBar = document.getElementById('mb');
  const meterVal = document.getElementById('mv');
  const statusBadge = document.getElementById('stage-status-badge');
  const apiStatusPill = document.getElementById('api-status-pill');
  const apiStatusLabel = document.getElementById('api-status-label');
  const conditionBadge = document.getElementById('condition-badge');
  const latencyBadge = document.getElementById('latency-badge');
  const diagnosisEl = document.getElementById('result-diagnosis-text');
  const resultMetaEl = document.getElementById('result-meta-text');
  const biomarkerTagsContainer = document.getElementById('biomarker-tags');

  const contribMrna = document.getElementById('contrib-mrna');
  const contribMirna = document.getElementById('contrib-mirna');
  const contribCnv = document.getElementById('contrib-cnv');
  const contribMethy = document.getElementById('contrib-methy');

  const chips = document.querySelectorAll('.chip[data-m]');
  const presetButtons = document.querySelectorAll('.preset-btn[data-preset]');
  const patientChips = document.querySelectorAll('.patient-chip[data-pidx]');
  const coreCircle = document.querySelector('.core');

  // Compute normalized modality contributions based on active flags
  function computeContributions(activeFlags) {
    const raw = {
      mRNA: activeFlags[0] ? BASE_WEIGHTS.mRNA : 0,
      miRNA: activeFlags[1] ? BASE_WEIGHTS.miRNA : 0,
      CNV: activeFlags[2] ? BASE_WEIGHTS.CNV : 0,
      Methy: activeFlags[3] ? BASE_WEIGHTS.Methy : 0
    };
    const total = raw.mRNA + raw.miRNA + raw.CNV + raw.Methy;
    if (total <= 0) return { mRNA: 0, miRNA: 0, CNV: 0, Methy: 0 };
    return {
      mRNA: Number(((raw.mRNA / total) * 100).toFixed(1)),
      miRNA: Number(((raw.miRNA / total) * 100).toFixed(1)),
      CNV: Number(((raw.CNV / total) * 100).toFixed(1)),
      Methy: Number(((raw.Methy / total) * 100).toFixed(1))
    };
  }

  // Render Biomarker Chips
  function renderBiomarkers(biomarkers) {
    if (!biomarkerTagsContainer) return;
    biomarkerTagsContainer.innerHTML = '';
    if (!biomarkers || biomarkers.length === 0) {
      const emptySpan = document.createElement('span');
      emptySpan.className = 'biomarker-tag';
      emptySpan.textContent = 'None (Inputs Inactive)';
      biomarkerTagsContainer.appendChild(emptySpan);
      return;
    }
    biomarkers.forEach((bm) => {
      const tag = document.createElement('span');
      tag.className = 'biomarker-tag';
      tag.textContent = bm.mod ? `${bm.feature} (${bm.mod})` : `${bm.feature} (${bm.modality})`;
      biomarkerTagsContainer.appendChild(tag);
    });
  }

  // Check Backend Health
  async function checkApiHealth() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);
      const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        isApiOnline = true;
        setApiPillState(true, 'Live Model Online');
        return true;
      }
    } catch (e) {
      // Offline fallback
    }
    isApiOnline = false;
    setApiPillState(false, 'Offline Benchmark Mode');
    return false;
  }

  function setApiPillState(online, label) {
    if (!apiStatusPill || !apiStatusLabel) return;
    apiStatusPill.classList.toggle('online', online);
    apiStatusPill.classList.toggle('offline', !online);
    apiStatusLabel.textContent = label;
  }

  // Fetch prediction from live API or use calibrated fallback
  async function evaluatePrediction() {
    const key = `${modalityStates[0] ? 1 : 0}-${modalityStates[1] ? 1 : 0}-${modalityStates[2] ? 1 : 0}-${modalityStates[3] ? 1 : 0}`;
    const conditionMeta = CONDITION_MAP[key] || CONDITION_MAP['0-0-0-0'];
    const activeCount = modalityStates.filter(Boolean).length;
    const currentPatient = TEST_PATIENTS[activePatientIdx] || TEST_PATIENTS[0];

    // Attempt live API request
    if (isApiOnline) {
      try {
        const t0 = performance.now();
        const response = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modalities: modalityStates.map((v) => (v ? 1 : 0)),
            patient_idx: activePatientIdx,
            strategy: 's2'
          })
        });

        if (response.ok) {
          const data = await response.json();
          renderPredictionResult({
            conditionId: data.condition_id,
            conditionDesc: data.condition_description,
            latencyText: `Inference: ${data.latency_ms} ms`,
            predictedClass: data.predicted_class,
            className: data.class_name,
            organ: data.organ,
            patientId: data.patient_id,
            groundTruth: data.ground_truth,
            confidence: data.confidence_pct,
            statusText: data.status_text,
            statusClass: data.status_class,
            contributions: data.modality_contributions,
            biomarkers: data.biomarkers,
            activeCount: activeCount
          });
          setApiPillState(true, `Live Model (${data.latency_ms}ms)`);
          return;
        }
      } catch (err) {
        console.warn('Live API request failed, switching to calibrated offline fallback.');
        isApiOnline = false;
        setApiPillState(false, 'Offline Benchmark Mode');
      }
    }

    // Calibrated Local Fallback Execution
    let fallbackConfidence = 0.0;
    if (activeCount === 4) {
      fallbackConfidence = currentPatient.baseConfidence;
    } else if (activeCount === 3) {
      fallbackConfidence = Number((currentPatient.baseConfidence * 0.985).toFixed(1));
    } else if (activeCount === 2) {
      fallbackConfidence = Number((currentPatient.baseConfidence * 0.925).toFixed(1));
    } else if (activeCount === 1) {
      // Condition C12 special gain (74.12%) vs others
      if (modalityStates[2]) {
        fallbackConfidence = 74.1; // C12 CNV stress test
      } else if (modalityStates[3]) {
        fallbackConfidence = 93.4; // C11 Methy fallback
      } else {
        fallbackConfidence = 85.0;
      }
    } else {
      fallbackConfidence = 0.0;
    }

    const fallbackContributions = computeContributions(modalityStates);
    const activeBiomarkers = currentPatient.biomarkers.filter((b) => {
      if (b.mod === 'mRNA') return modalityStates[0];
      if (b.mod === 'miRNA') return modalityStates[1];
      if (b.mod === 'CNV') return modalityStates[2];
      if (b.mod === 'Methy') return modalityStates[3];
      return true;
    });

    renderPredictionResult({
      conditionId: conditionMeta.id,
      conditionDesc: conditionMeta.desc,
      latencyText: 'Calibrated Benchmark (Offline)',
      predictedClass: activeCount > 0 ? currentPatient.code : 'INDETERMINATE',
      className: activeCount > 0 ? currentPatient.className : 'No Active Modalities',
      organ: activeCount > 0 ? currentPatient.organ : 'None',
      patientId: currentPatient.sampleId,
      groundTruth: currentPatient.code,
      confidence: fallbackConfidence,
      statusText: conditionMeta.status,
      statusClass: conditionMeta.statusClass,
      contributions: fallbackContributions,
      biomarkers: activeBiomarkers,
      activeCount: activeCount
    });
  }

  // Render Result in UI
  function renderPredictionResult(p) {
    // Condition Badge
    if (conditionBadge) {
      conditionBadge.textContent = `${p.conditionId}: ${p.conditionDesc}`;
    }

    // Latency Badge
    if (latencyBadge) {
      latencyBadge.textContent = p.latencyText;
    }

    // Diagnosis & Meta Subtext
    if (diagnosisEl) {
      diagnosisEl.textContent = p.activeCount > 0 ? `${p.className} (${p.predictedClass})` : 'No Active Modalities';
    }
    if (resultMetaEl) {
      resultMetaEl.textContent = p.activeCount > 0
        ? `Primary Organ: ${p.organ} · TCGA Sample: ${p.patientId} (Ground Truth: ${p.groundTruth})`
        : 'Activate at least one omic modality above to compute patient diagnostic classification.';
    }

    // Status Badge
    if (statusBadge) {
      statusBadge.textContent = p.statusText;
      statusBadge.className = `status-badge ${p.statusClass}`;
    }

    // Confidence Meter
    if (meterBar) {
      meterBar.style.width = `${p.confidence}%`;
      meterBar.classList.toggle('degraded', p.activeCount === 1);
      meterBar.classList.toggle('zero', p.activeCount === 0);
    }
    if (meterVal) {
      animateValue(meterVal, currentConfidence, p.confidence, 350);
      currentConfidence = p.confidence;
    }

    // Contributions Grid (mRNA, miRNA, CNV, Methy)
    if (contribMrna) contribMrna.textContent = `${p.contributions.mRNA || 0}%`;
    if (contribMirna) contribMirna.textContent = `${p.contributions.miRNA || 0}%`;
    if (contribCnv) contribCnv.textContent = `${p.contributions.CNV || 0}%`;
    if (contribMethy) contribMethy.textContent = `${p.contributions.Methy || 0}%`;

    // Biomarkers
    renderBiomarkers(p.biomarkers);
  }

  // Update SVG Topology, Chips, and Core
  function updateDiagnosticUI() {
    const activeCount = modalityStates.filter(Boolean).length;
    const key = `${modalityStates[0] ? 1 : 0}-${modalityStates[1] ? 1 : 0}-${modalityStates[2] ? 1 : 0}-${modalityStates[3] ? 1 : 0}`;

    // Update Modality Chips
    chips.forEach((chip) => {
      const idx = parseInt(chip.getAttribute('data-m'), 10);
      chip.setAttribute('aria-pressed', modalityStates[idx]);
    });

    // Update SVG Elements (Flows, Nodes, Labels)
    document.querySelectorAll('[data-m]').forEach((el) => {
      const idx = parseInt(el.getAttribute('data-m'), 10);
      const isActive = modalityStates[idx];
      if (!el.classList.contains('chip')) {
        el.classList.toggle('off', !isActive);
      }
    });

    // Update Core glow/pulse
    if (coreCircle) {
      coreCircle.classList.toggle('off', activeCount === 0);
      coreCircle.classList.toggle('degraded', activeCount === 1);
    }

    // Sync Preset Buttons
    syncPresetButtons(key);

    // Debounced Evaluation Call
    if (apiDebounceTimeout) clearTimeout(apiDebounceTimeout);
    apiDebounceTimeout = setTimeout(evaluatePrediction, 50);
  }

  function animateValue(element, start, end, duration) {
    if (confidenceAnimFrame) cancelAnimationFrame(confidenceAnimFrame);
    const startTime = performance.now();
    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const val = start + (end - start) * ease;
      element.textContent = val.toFixed(1) + '%';
      if (progress < 1) {
        confidenceAnimFrame = requestAnimationFrame(tick);
      }
    }
    confidenceAnimFrame = requestAnimationFrame(tick);
  }

  function syncPresetButtons(activeKey) {
    presetButtons.forEach((btn) => {
      const preset = btn.getAttribute('data-preset');
      let matches = false;
      if (preset === 'all' && activeKey === '1-1-1-1') matches = true;
      if (preset === 'no-mrna' && activeKey === '0-1-1-1') matches = true;
      if (preset === 'methy-only' && activeKey === '0-0-0-1') matches = true;
      if (preset === 'cnv-only' && activeKey === '0-0-1-0') matches = true;
      btn.classList.toggle('active', matches);
    });
  }

  // Modality Chip Click Listeners
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const idx = parseInt(chip.getAttribute('data-m'), 10);
      modalityStates[idx] = !modalityStates[idx];
      updateDiagnosticUI();
    });
  });

  // Preset Button Click Listeners
  presetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      if (preset === 'all') {
        modalityStates[0] = true;
        modalityStates[1] = true;
        modalityStates[2] = true;
        modalityStates[3] = true;
      } else if (preset === 'no-mrna') {
        modalityStates[0] = false;
        modalityStates[1] = true;
        modalityStates[2] = true;
        modalityStates[3] = true;
      } else if (preset === 'methy-only') {
        modalityStates[0] = false;
        modalityStates[1] = false;
        modalityStates[2] = false;
        modalityStates[3] = true;
      } else if (preset === 'cnv-only') {
        modalityStates[0] = false;
        modalityStates[1] = false;
        modalityStates[2] = true;
        modalityStates[3] = false;
      }
      updateDiagnosticUI();
    });
  });

  // Patient Selector Chip Listeners
  patientChips.forEach((btn) => {
    btn.addEventListener('click', () => {
      patientChips.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      activePatientIdx = parseInt(btn.getAttribute('data-pidx'), 10);
      updateDiagnosticUI();
    });
  });

  // Direct SVG Node Click Listeners (allows clicking directly on the SVG boxes to toggle)
  document.querySelectorAll('svg rect.node[data-m]').forEach((rect) => {
    rect.addEventListener('click', () => {
      const idx = parseInt(rect.getAttribute('data-m'), 10);
      modalityStates[idx] = !modalityStates[idx];
      updateDiagnosticUI();
    });
  });

  // API Status Pill Click Listener (Manual Re-Ping)
  if (apiStatusPill) {
    apiStatusPill.addEventListener('click', async () => {
      setApiPillState(false, 'Pinging Model API...');
      await checkApiHealth();
      evaluatePrediction();
    });
  }

  /* ==========================================================================
     Clinical Prediction Studio Controller (File Upload, Validation & Inference)
     ========================================================================== */
  const tabStudioBtn = document.getElementById('tab-studio-btn');
  const tabTopologyBtn = document.getElementById('tab-topology-btn');
  const studioPanel = document.getElementById('studio-panel');
  const topologyPanel = document.getElementById('topology-panel');

  const MODALITIES = ['mRNA', 'miRNA', 'CNV', 'Methy'];
  const MODALITY_DIMS = { mRNA: 3217, miRNA: 383, CNV: 3105, Methy: 3139 };
  const PATIENT_PREFIX_MAP = {
    0: 'BRCA_Patient101',
    1: 'HNSC_Patient142',
    6: 'LGG_Patient188',
    2: 'THCA_Patient215',
    5: 'BRCA_Patient101'
  };

  let activeStudioPatientIdx = 0;
  const selectedStudioFiles = { mRNA: null, miRNA: null, CNV: null, Methy: null };

  // Tab Switching
  if (tabStudioBtn && tabTopologyBtn && studioPanel && topologyPanel) {
    tabStudioBtn.addEventListener('click', () => {
      tabStudioBtn.classList.add('active');
      tabTopologyBtn.classList.remove('active');
      studioPanel.style.display = 'block';
      topologyPanel.style.display = 'none';
    });

    tabTopologyBtn.addEventListener('click', () => {
      tabTopologyBtn.classList.add('active');
      tabStudioBtn.classList.remove('active');
      studioPanel.style.display = 'none';
      topologyPanel.style.display = 'block';
    });
  }

  // Modality Checkbox Changes & UI Sync
  MODALITIES.forEach((mod) => {
    const chk = document.getElementById(`check-mod-${mod.toLowerCase()}`);
    const card = document.getElementById(`card-mod-${mod.toLowerCase()}`);
    const badge = document.getElementById(`badge-mod-${mod.toLowerCase()}`);

    if (chk && card && badge) {
      chk.addEventListener('change', () => {
        const isPresent = chk.checked;
        card.classList.toggle('missing', !isPresent);
        if (!isPresent) {
          badge.className = 'mod-status-badge status-missing';
          badge.textContent = 'Missing (Masked)';
        } else {
          if (selectedStudioFiles[mod]) {
            badge.className = 'mod-status-badge status-ready';
            badge.textContent = `File Ready`;
          } else {
            badge.className = 'mod-status-badge status-awaiting';
            badge.textContent = 'Awaiting Input';
          }
        }
      });
    }
  });

  // Drag and Drop & File Upload Handlers
  MODALITIES.forEach((mod) => {
    const modLower = mod.toLowerCase();
    const dropzone = document.getElementById(`dropzone-${modLower}`);
    const fileInput = document.getElementById(`file-input-${modLower}`);
    const previewPill = document.getElementById(`preview-pill-${modLower}`);
    const filenameSpan = previewPill ? previewPill.querySelector('.pill-name-text') : null;
    const filesizeSpan = document.getElementById(`filesize-${modLower}`);
    const removeBtn = previewPill ? previewPill.querySelector('.file-remove-btn') : null;
    const badge = document.getElementById(`badge-mod-${modLower}`);
    const card = document.getElementById(`card-mod-${modLower}`);

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function setFile(file) {
      if (!file) return;
      // Validate file extension
      const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      if (!['.csv', '.tsv', '.txt'].includes(ext)) {
        showStudioAlert(`Unsupported file format '${ext}' for ${mod}. Please upload a .csv, .tsv, or .txt file.`, 'danger');
        return;
      }
      // Validate file size (< 25MB)
      if (file.size > 25 * 1024 * 1024) {
        showStudioAlert(`File '${file.name}' exceeds maximum allowed size of 25 MB.`, 'danger');
        return;
      }

      selectedStudioFiles[mod] = file;

      // Update UI
      if (filenameSpan) filenameSpan.textContent = file.name;
      if (filesizeSpan) filesizeSpan.textContent = formatBytes(file.size);
      if (previewPill) previewPill.style.display = 'flex';
      if (dropzone) dropzone.style.display = 'none';

      if (badge) {
        badge.className = 'mod-status-badge status-ready';
        badge.textContent = 'File Ready';
      }
      clearStudioAlert();
    }

    function removeFile() {
      selectedStudioFiles[mod] = null;
      if (fileInput) fileInput.value = '';
      if (previewPill) previewPill.style.display = 'none';
      if (dropzone) dropzone.style.display = 'flex';
      if (badge) {
        const chk = document.getElementById(`check-mod-${modLower}`);
        if (chk && !chk.checked) {
          badge.className = 'mod-status-badge status-missing';
          badge.textContent = 'Missing (Masked)';
        } else {
          badge.className = 'mod-status-badge status-awaiting';
          badge.textContent = 'Awaiting Input';
        }
      }
    }

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files[0]) {
          setFile(fileInput.files[0]);
        }
      });

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dropzone-dragover');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dropzone-dragover');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dropzone-dragover');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
          setFile(e.dataTransfer.files[0]);
        }
      });
    }

    if (removeBtn) {
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeFile();
      });
    }

    // Template CSV Download Handler
    const templateBtn = card ? card.querySelector('[data-mod-template]') : null;
    if (templateBtn) {
      templateBtn.addEventListener('click', async (e) => {
        if (isApiOnline) {
          e.preventDefault();
          try {
            const res = await fetch(`${API_BASE}/api/template/${mod}`);
            if (res.ok) {
              const blob = await res.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `template_${mod}.csv`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              window.URL.revokeObjectURL(url);
              return;
            }
          } catch (err) {
            // Fall back to direct static file download via href
            window.location.href = templateBtn.getAttribute('href') || `templates/template_${mod}.csv`;
          }
        }
      });
    }

    // Load Demo Sample for this Modality
    const loadSampleBtn = card ? card.querySelector('[data-load-mod-sample]') : null;
    if (loadSampleBtn) {
      loadSampleBtn.addEventListener('click', async () => {
        const prefix = PATIENT_PREFIX_MAP[activeStudioPatientIdx] || 'BRCA_Patient101';
        try {
          loadSampleBtn.textContent = 'Loading...';
          let res = await fetch(`sample_inputs/${prefix}_${mod}.csv`);
          if (!res.ok && isApiOnline) {
            res = await fetch(`${API_BASE}/sample_inputs/${prefix}_${mod}.csv`);
          }
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const blob = await res.blob();
          const file = new File([blob], `${prefix}_${mod}.csv`, { type: 'text/csv' });
          setFile(file);
          loadSampleBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>
            Load Sample
          `;
        } catch (err) {
          loadSampleBtn.textContent = 'Load Sample';
          showStudioAlert(`Failed to load demo file: ${err.message}`, 'warning');
        }
      });
    }
  });

  // Manual Vector Accordion Toggle
  const manualToggle = document.getElementById('manual-accordion-toggle');
  const manualAccordion = document.getElementById('manual-vector-accordion');
  const accordionArrow = document.getElementById('accordion-arrow-icon');
  if (manualToggle && manualAccordion) {
    manualToggle.addEventListener('click', () => {
      const isOpen = manualAccordion.classList.toggle('open');
      if (accordionArrow) accordionArrow.textContent = isOpen ? '▲' : '▼';
    });
  }

  // Quick Test Cohort Demonstrations & Presets
  const demoPatientBtns = document.querySelectorAll('[data-load-patient]');
  const demoPresetBtns = document.querySelectorAll('[data-load-preset]');

  demoPatientBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      demoPatientBtns.forEach((b) => b.classList.remove('active'));
      demoPresetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      activeStudioPatientIdx = parseInt(btn.getAttribute('data-load-patient'), 10);
      // Ensure all 4 modalities are checked for complete patient case
      MODALITIES.forEach((m) => {
        const chk = document.getElementById(`check-mod-${m.toLowerCase()}`);
        if (chk && !chk.checked) {
          chk.checked = true;
          chk.dispatchEvent(new Event('change'));
        }
      });

      // Clear any custom uploaded files so it evaluates this patient case
      resetStudioFiles();

      const sampleIdInput = document.getElementById('studio-sample-id');
      if (sampleIdInput) {
        sampleIdInput.value = btn.textContent.trim();
      }

      // Execute live prediction immediately for demonstration
      runStudioPrediction();
    });
  });

  demoPresetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      demoPatientBtns.forEach((b) => b.classList.remove('active'));
      demoPresetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      const preset = btn.getAttribute('data-load-preset');
      if (preset === 'no-mrna') {
        const chkMrna = document.getElementById('check-mod-mrna');
        if (chkMrna) { chkMrna.checked = false; chkMrna.dispatchEvent(new Event('change')); }
        ['mirna', 'cnv', 'methy'].forEach((m) => {
          const c = document.getElementById(`check-mod-${m}`);
          if (c && !c.checked) { c.checked = true; c.dispatchEvent(new Event('change')); }
        });
      } else if (preset === 'cnv-only') {
        ['mrna', 'mirna', 'methy'].forEach((m) => {
          const c = document.getElementById(`check-mod-${m}`);
          if (c && c.checked) { c.checked = false; c.dispatchEvent(new Event('change')); }
        });
        const chkCnv = document.getElementById('check-mod-cnv');
        if (chkCnv && !chkCnv.checked) { chkCnv.checked = true; chkCnv.dispatchEvent(new Event('change')); }
      }

      resetStudioFiles();
      runStudioPrediction();
    });
  });

  function resetStudioFiles() {
    MODALITIES.forEach((mod) => {
      selectedStudioFiles[mod] = null;
      const fileInput = document.getElementById(`file-input-${mod.toLowerCase()}`);
      if (fileInput) fileInput.value = '';
      const previewPill = document.getElementById(`preview-pill-${mod.toLowerCase()}`);
      if (previewPill) previewPill.style.display = 'none';
      const dropzone = document.getElementById(`dropzone-${mod.toLowerCase()}`);
      if (dropzone) dropzone.style.display = 'flex';
      const badge = document.getElementById(`badge-mod-${mod.toLowerCase()}`);
      const chk = document.getElementById(`check-mod-${mod.toLowerCase()}`);
      if (badge && chk) {
        badge.className = chk.checked ? 'mod-status-badge status-awaiting' : 'mod-status-badge status-missing';
        badge.textContent = chk.checked ? 'Awaiting Input' : 'Missing (Masked)';
      }
    });
  }

  // Alerts
  const studioAlertBox = document.getElementById('studio-alert-box');
  function showStudioAlert(message, type = 'danger') {
    if (!studioAlertBox) return;
    studioAlertBox.textContent = message;
    studioAlertBox.className = `studio-alert-box ${type}`;
    studioAlertBox.style.display = 'block';
  }

  function clearStudioAlert() {
    if (!studioAlertBox) return;
    studioAlertBox.style.display = 'none';
    studioAlertBox.textContent = '';
  }

  // Prediction Execution Logic
  const btnStudioPredict = document.getElementById('btn-studio-predict');
  const btnPredictLabel = document.getElementById('btn-predict-label');
  const studioPredictSpinner = document.getElementById('studio-predict-spinner');
  const studioResultsPanel = document.getElementById('studio-results-panel');

  function setPredictLoading(isLoading) {
    if (!btnStudioPredict) return;
    btnStudioPredict.disabled = isLoading;
    btnStudioPredict.classList.toggle('loading', isLoading);
    if (btnPredictLabel) {
      btnPredictLabel.textContent = isLoading ? 'Executing PyTorch Forward Pass...' : 'Run Cancer Prediction';
    }
  }

  async function runStudioPrediction() {
    clearStudioAlert();

    // 1. Gather active modalities
    const activeFlags = MODALITIES.map((m) => {
      const chk = document.getElementById(`check-mod-${m.toLowerCase()}`);
      return chk && chk.checked ? 1 : 0;
    });

    const activeCount = activeFlags.filter((v) => v === 1).length;
    if (activeCount === 0) {
      showStudioAlert('Input Validation Error: At least one omic modality must be selected to compute cancer prediction.', 'danger');
      if (studioResultsPanel) studioResultsPanel.classList.remove('visible');
      return;
    }

    // Strategy
    const stratInput = document.querySelector('input[name="studio-strategy"]:checked');
    const strategy = stratInput ? stratInput.value : 's2';

    // Sample ID
    const sampleIdInput = document.getElementById('studio-sample-id');
    const sampleId = (sampleIdInput && sampleIdInput.value.trim()) || `Patient-Sample-${activeStudioPatientIdx + 101}`;

    setPredictLoading(true);

    try {
      // Check if files were provided
      const hasUploadedFiles = MODALITIES.some((m, idx) => activeFlags[idx] === 1 && selectedStudioFiles[m] !== null);

      // Check manual textareas
      let manualPayload = null;
      MODALITIES.forEach((m, idx) => {
        if (activeFlags[idx] === 1 && !selectedStudioFiles[m]) {
          const txt = document.getElementById(`manual-${m.toLowerCase()}`);
          if (txt && txt.value.trim().length > 0) {
            if (!manualPayload) manualPayload = {};
            const rawVal = txt.value.trim();
            if (rawVal.startsWith('{')) {
              try {
                manualPayload[m] = JSON.parse(rawVal);
              } catch (e) {
                throw new Error(`Invalid JSON syntax in manual ${m} input: ${e.message}`);
              }
            } else {
              const nums = rawVal.split(',').map((s) => parseFloat(s.trim()));
              if (nums.some(isNaN)) {
                throw new Error(`Non-numeric value detected in manual ${m} input vector.`);
              }
              manualPayload[m] = nums;
            }
          }
        }
      });

      let responseData = null;

      // Path A: File Upload (multipart/form-data)
      if (hasUploadedFiles) {
        const formData = new FormData();
        MODALITIES.forEach((m, idx) => {
          if (activeFlags[idx] === 1) {
            if (selectedStudioFiles[m]) {
              formData.append(`file_${m}`, selectedStudioFiles[m]);
            } else {
              // Active without file
              throw new Error(`Modality '${m}' is selected as active, but no CSV file was uploaded. Please upload a file, paste manual vector, or uncheck '${m}'.`);
            }
          }
        });

        formData.append('modalities', JSON.stringify(MODALITIES.filter((_, idx) => activeFlags[idx] === 1)));
        formData.append('strategy', strategy);
        formData.append('sample_id', sampleId);

        const res = await fetch(`${API_BASE}/api/predict/upload`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
          throw new Error(errData.detail || 'File upload prediction failed.');
        }
        responseData = await res.json();
      }
      // Path B: Manual Feature Entry (JSON)
      else if (manualPayload && Object.keys(manualPayload).length > 0) {
        const res = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modalities: activeFlags,
            inputs: manualPayload,
            strategy: strategy
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
          throw new Error(errData.detail || 'Manual vector prediction failed.');
        }
        responseData = await res.json();
      }
      // Path C: Test Cohort Case Real Inference
      else {
        const res = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modalities: activeFlags,
            patient_idx: activeStudioPatientIdx,
            strategy: strategy
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
          throw new Error(errData.detail || 'Inference pipeline failed.');
        }
        responseData = await res.json();
      }

      // Render Results Panel
      renderStudioResults(responseData);
      setApiPillState(true, `Live Model (${responseData.latency_ms}ms)`);

    } catch (err) {
      showStudioAlert(err.message, 'danger');
      if (studioResultsPanel) studioResultsPanel.classList.remove('visible');
    } finally {
      setPredictLoading(false);
    }
  }

  function renderStudioResults(data) {
    if (!studioResultsPanel) return;

    // Header Meta
    const condBadge = document.getElementById('studio-condition-badge');
    if (condBadge) condBadge.textContent = `${data.condition_id}: ${data.condition_description}`;

    const ckptBadge = document.getElementById('studio-checkpoint-badge');
    if (ckptBadge) ckptBadge.textContent = `Model: ${data.model_checkpoint}`;

    const latBadge = document.getElementById('studio-latency-badge');
    if (latBadge) latBadge.textContent = `Inference: ${data.latency_ms} ms`;

    // Malignancy Card
    const malCode = document.getElementById('studio-malignancy-code');
    if (malCode) malCode.textContent = data.predicted_class;

    const malName = document.getElementById('studio-malignancy-name');
    if (malName) malName.textContent = data.class_name;

    const malMeta = document.getElementById('studio-malignancy-meta');
    if (malMeta) {
      malMeta.textContent = `Primary Organ: ${data.organ} · Sample: ${data.patient_id} (Ground Truth: ${data.ground_truth})`;
    }

    const confVal = document.getElementById('studio-confidence-val');
    if (confVal) confVal.textContent = `${data.confidence_pct}%`;

    // Horizontal Class Probability Bar Chart (BRCA, HNSC, LGG, THCA)
    const classCodes = ['BRCA', 'HNSC', 'LGG', 'THCA'];
    classCodes.forEach((code) => {
      const prob = data.probabilities[code] !== undefined ? data.probabilities[code] : 0.0;
      const pct = (prob * 100).toFixed(1);
      const row = document.getElementById(`prob-row-${code}`);
      const bar = document.getElementById(`prob-bar-${code}`);
      const valText = document.getElementById(`prob-val-${code}`);

      if (bar) bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
      if (valText) valText.textContent = `${pct}%`;
      if (row) {
        row.classList.toggle('winner', code === data.predicted_class);
      }
    });

    // Modality Breakdown Pills
    const activePillsContainer = document.getElementById('studio-active-modalities-pills');
    if (activePillsContainer) {
      activePillsContainer.innerHTML = '';
      if (data.active_modalities && data.active_modalities.length > 0) {
        data.active_modalities.forEach((m) => {
          const pill = document.createElement('span');
          pill.className = 'summary-pill active';
          pill.textContent = `${m} (${MODALITY_DIMS[m] || '—'} features)`;
          activePillsContainer.appendChild(pill);
        });
      } else {
        activePillsContainer.innerHTML = '<span class="summary-pill missing">None</span>';
      }
    }

    const missingPillsContainer = document.getElementById('studio-missing-modalities-pills');
    if (missingPillsContainer) {
      missingPillsContainer.innerHTML = '';
      if (data.missing_modalities && data.missing_modalities.length > 0) {
        data.missing_modalities.forEach((m) => {
          const pill = document.createElement('span');
          pill.className = 'summary-pill missing';
          pill.textContent = `${m} (Imputed with Learned Token)`;
          missingPillsContainer.appendChild(pill);
        });
      } else {
        missingPillsContainer.innerHTML = '<span class="summary-pill active">None (Full Complete Data)</span>';
      }
    }

    // Cross-Attention Contributions
    const attnContainer = document.getElementById('studio-attention-pills');
    if (attnContainer) {
      attnContainer.innerHTML = '';
      if (data.modality_contributions) {
        Object.entries(data.modality_contributions).forEach(([m, val]) => {
          const pill = document.createElement('span');
          pill.className = `summary-pill ${val > 0 ? 'active' : 'missing'}`;
          pill.textContent = `${m}: ${val}%`;
          attnContainer.appendChild(pill);
        });
      }
    }

    // Biomarkers
    const bmContainer = document.getElementById('studio-biomarker-tags');
    if (bmContainer) {
      bmContainer.innerHTML = '';
      if (data.biomarkers && data.biomarkers.length > 0) {
        data.biomarkers.forEach((bm) => {
          const tag = document.createElement('span');
          tag.className = 'biomarker-tag';
          tag.textContent = bm.mod ? `${bm.feature} (${bm.mod})` : `${bm.feature}`;
          bmContainer.appendChild(tag);
        });
      } else {
        bmContainer.innerHTML = '<span class="biomarker-tag">General Pan-Cancer Genomic Profile</span>';
      }
    }

    // Show Panel
    studioResultsPanel.classList.add('visible');
    studioResultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Button Listeners
  if (btnStudioPredict) {
    btnStudioPredict.addEventListener('click', runStudioPrediction);
  }

  const btnStudioReset = document.getElementById('btn-studio-reset');
  if (btnStudioReset) {
    btnStudioReset.addEventListener('click', () => {
      // Re-check all checkboxes
      MODALITIES.forEach((m) => {
        const chk = document.getElementById(`check-mod-${m.toLowerCase()}`);
        if (chk && !chk.checked) {
          chk.checked = true;
          chk.dispatchEvent(new Event('change'));
        }
      });
      resetStudioFiles();
      // Clear manual textareas
      ['mrna', 'mirna', 'cnv', 'methy'].forEach((m) => {
        const txt = document.getElementById(`manual-${m}`);
        if (txt) txt.value = '';
      });
      clearStudioAlert();
      if (studioResultsPanel) studioResultsPanel.classList.remove('visible');
    });
  }

  const btnStudioAnother = document.getElementById('btn-studio-another');
  if (btnStudioAnother && studioPanel) {
    btnStudioAnother.addEventListener('click', () => {
      studioPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  // Initial Startup Sequence
  checkApiHealth().finally(() => {
    updateDiagnosticUI();
  });

})();

