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

    // Toggle Scrolled Class for Compact Navbar
    if (navbar) {
      navbar.classList.toggle('scrolled', scrollY > 30);
    }
  }

  window.addEventListener('scroll', () => {
    window.requestAnimationFrame(updateScrollState);
  }, { passive: true });
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

  // Smooth Section Jump for Navigation Links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });

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
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    revealElements.forEach((el, index) => {
      el.style.transitionDelay = `${(index % 4) * 75}ms`;
      revealObserver.observe(el);
    });
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
        setApiPillState(true, '🟢 Live Model Online');
        return true;
      }
    } catch (e) {
      // Offline fallback
    }
    isApiOnline = false;
    setApiPillState(false, '⚡ Offline Benchmark Mode');
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
          setApiPillState(true, `🟢 Live Model (${data.latency_ms}ms)`);
          return;
        }
      } catch (err) {
        console.warn('Live API request failed, switching to calibrated offline fallback.');
        isApiOnline = false;
        setApiPillState(false, '⚡ Offline Benchmark Mode');
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

  // Initial Startup Sequence
  checkApiHealth().finally(() => {
    updateDiagnosticUI();
  });

})();
