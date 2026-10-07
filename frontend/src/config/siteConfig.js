/**
 * AgriVision AI — Central Configuration & Content File
 * ====================================================
 * EDIT THIS FILE TO EASILY CHANGE:
 * - App title, logo, slogan, and branding
 * - Farmer profile and default farm settings
 * - Disease database (symptoms, remedies, severity levels)
 * - Sample plants & leaves
 * - Fertilizer crop guidelines
 * - Default stats and metric numbers
 */

export const siteConfig = {
  appName: "AgriVision AI",
  appSubtitle: "Precision AgriTech & Plant Pathology",
  tagline: "Diagnose your crop in seconds",
  heroScript: "Better Diagnosis,",
  heroScriptSub: "Healthier Harvests",
  heroDescription: "Upload a leaf image and let our AI detect plant diseases, analyze chlorophyll health, identify the problem, and provide instant treatment recommendations.",
  version: "2.4.0",
  activeStatusText: "AI MobileNetV2 Active",

  // Farmer & Farm Defaults
  defaultProfile: {
    farmerId: "FAR-2026-089",
    farmerName: "Krutin Patel",
    farmName: "Green Valley Farm",
    location: "Gujarat, India",
    totalArea: 14.5, // in hectares
    phone: "+91 98765 43210",
    email: "krutin.patel@agrivision.ai",
    farmingType: "Integrated Sustainable",
    farmingExperience: 8, // years
    primaryCrops: ["Tomato", "Cotton", "Potato", "Corn", "Wheat"],
    soilType: "Loamy Rich Alluvial",
    preferences: {
      language: "English",
      units: "Metric (ha / kg)",
      notifications: true,
      treatmentPreference: "integrated", // organic | chemical | integrated
    }
  },

  // Navigation Items
  navItems: [
    { id: "dashboard", label: "Dashboard", icon: "LayoutDashboard", desc: "Overview & quick diagnosis" },
    { id: "detection", label: "Disease Detection", icon: "ScanLine", desc: "AI leaf symptom scanner" },
    { id: "history", label: "Detection History", icon: "ClipboardList", desc: "Archived crop scans" },
    { id: "treatment", label: "Treatment History", icon: "Pill", desc: "Applied remedies & logs" },
    { id: "fertilizer", label: "Fertilizer Calculator", icon: "Sprout", desc: "NPK nutrition advisor" },
    { id: "analytics", label: "Farm Analytics", icon: "BarChart3", desc: "Disease & crop trends" },
    { id: "profile", label: "User Profile", icon: "UserCircle2", desc: "Farm & farmer settings" },
    { id: "export", label: "Export Reports", icon: "FileSpreadsheet", desc: "Download PDF & CSV reports" },
    { id: "data", label: "Data Management", icon: "Database", desc: "Backups & system health" },
  ],

  // Sample Plant Leaves for Quick Diagnostic Testing
  sampleLeaves: [
    {
      id: "sample-tomato-early-blight",
      name: "Tomato — Early Blight",
      crop: "Tomato",
      diseaseKey: "Early_Disease_Symptoms",
      badge: "Early Stage",
      description: "Concentric brown rings on lower leaves with yellow halo",
      accentColor: "#f59e0b",
      leafType: "tomato-blight"
    },
    {
      id: "sample-potato-late-blight",
      name: "Potato — Late Blight",
      crop: "Potato",
      diseaseKey: "Severe_Plant_Disease",
      badge: "Critical",
      description: "Dark water-soaked lesions spreading rapidly across foliage",
      accentColor: "#ef4444",
      leafType: "potato-blight"
    },
    {
      id: "sample-cotton-healthy",
      name: "Cotton — Healthy Foliage",
      crop: "Cotton",
      diseaseKey: "Plant_Healthy_Condition",
      badge: "Optimal",
      description: "Vibrant emerald leaf blade with balanced venation and vigor",
      accentColor: "#10b981",
      leafType: "cotton-healthy"
    },
    {
      id: "sample-maize-fungal",
      name: "Maize — Leaf Blight",
      crop: "Corn",
      diseaseKey: "Moderate_Fungal_Disease",
      badge: "Moderate",
      description: "Elongated grayish-tan lesions running parallel to veins",
      accentColor: "#f97316",
      leafType: "corn-fungal"
    },
    {
      id: "sample-severe-stress",
      name: "Eggplant — Severe Wilt Stress",
      crop: "Eggplant",
      diseaseKey: "Severe_Plant_Stress",
      badge: "Severe Stress",
      description: "Vascular collapse accompanied by chlorotic discoloration",
      accentColor: "#dc2626",
      leafType: "severe-stress"
    }
  ],

  // Diseases Knowledge Base (Exact Match & Expanded from Python Engine)
  diseases: {
    "Early_Disease_Symptoms": {
      name: "Early Disease Symptoms (Alternaria / Cercospora)",
      cropCommon: "Tomato / Pepper / Potato",
      severity: "Low",
      urgency: "Low",
      color: "#f59e0b",
      badgeClass: "sev-pill-low",
      symptoms: "Minor chlorotic spotting, faint concentric target rings on older leaves, early yellow margins, and localized tissue softening.",
      causes: "High relative humidity (>80%), splash dispersal from soil spore reservoir, moderate temperatures (24-29°C).",
      prevention: "Implement drip irrigation to avoid wet foliage, increase plant spacing for airflow, sterilize pruning shears.",
      organicTreatment: {
        products: ["Neem Oil (Cold-pressed 5ml/L)", "Bacillus subtilis Bio-Fungicide", "Copper Soap Spray"],
        instructions: "Foliar spray both upper and lower leaf surfaces during early dawn or post-sunset. Repeat every 7 days.",
        safety: "Zero pre-harvest interval; safe for beneficial pollinators when applied after sunset."
      },
      chemicalTreatment: {
        products: ["Mancozeb 75% WP (2.5g/L)", "Chlorothalonil 720 SC", "Azoxystrobin 23% SC"],
        instructions: "Apply contact protective spray prior to forecasted rainfall events. Ensure uniform leaf canopy coverage.",
        safety: "Wear nitrile gloves and respiratory mask. 7-day pre-harvest waiting period."
      }
    },
    "Severe_Plant_Disease": {
      name: "Severe Plant Blight (Phytophthora infestans)",
      cropCommon: "Potato / Tomato / Eggplant",
      severity: "Critical",
      urgency: "Immediate",
      color: "#ef4444",
      badgeClass: "sev-pill-high",
      symptoms: "Large irregular dark necrotic lesions, rapid leaf collapse, white moldy growth on leaf undersides in humid conditions.",
      causes: "Cool damp weather, waterlogged soil, windblown sporangia travelling from adjacent infected fields.",
      prevention: "Plant certified disease-free tubers, destroy volunteer host plants, maintain preventive copper spray schedules.",
      organicTreatment: {
        products: ["Bordeaux Mixture (1%)", "Copper Hydroxide Bio-Formulation", "Trichoderma harzianum soil drench"],
        instructions: "Prune and safely burn heavily blighted stems. Apply Bordeaux mixture immediately to arrest sporulation.",
        safety: "Wear eye protection; avoid inhalation of copper dust."
      },
      chemicalTreatment: {
        products: ["Metalaxyl-M + Mancozeb (Ridomil Gold 2.5g/L)", "Cymoxanil + Famoxadone", "Dimethomorph 50% WP"],
        instructions: "Dual-action systemic + contact fungicide rotation every 5 days until new foliage appears clear.",
        safety: "Strict 14-day pre-harvest interval. Keep livestock away from sprayed zones for 48 hours."
      }
    },
    "Plant_Healthy_Condition": {
      name: "Healthy Foliage Condition",
      cropCommon: "All Crops",
      severity: "None",
      urgency: "Normal Care",
      color: "#16a34a",
      badgeClass: "sev-pill-healthy",
      symptoms: "Crisp green leaf blade, normal chlorophyll density, unobstructed vascular veins, no signs of fungal or bacterial lesions.",
      causes: "Balanced nutrition, optimal soil moisture, healthy rhizosphere microbiome, proper aeration.",
      prevention: "Maintain balanced N-P-K fertigation, scout weekly for pest vectors, practice crop rotation.",
      organicTreatment: {
        products: ["Seaweed Extract Tonic", "Vermicompost Tea", "Panchagavya Foliar Spray"],
        instructions: "Apply fortnightly foliar bio-stimulant spray to boost natural plant immunity and photosystem efficiency.",
        safety: "100% Eco-friendly and biological."
      },
      chemicalTreatment: {
        products: ["Micronutrient Chelated Complex (Fe, Zn, Mn, B, Cu) 1g/L"],
        instructions: "Apply prophylactic micronutrient spray every 21 days during vegetative peak.",
        safety: "Standard agricultural handling."
      }
    },
    "Moderate_Fungal_Disease": {
      name: "Moderate Fungal Blight / Powdery Mildew",
      cropCommon: "Maize / Wheat / Cucurbits",
      severity: "Medium",
      urgency: "Medium",
      color: "#f97316",
      badgeClass: "sev-pill-medium",
      symptoms: "Powdery white or grayish patches, superficial mycelial growth across leaf surface, mild curling and premature senescence.",
      causes: "Warm dry days followed by cool humid nights, crowded plant canopy, nitrogen over-fertilization.",
      prevention: "Select resistant hybrids, reduce excessive nitrogen top-dressing, prune lower canopy leaves.",
      organicTreatment: {
        products: ["Potassium Bicarbonate (3g/L)", "Sulfur 80% WDG (2g/L)", "Milk-Water Emulsion (1:9)"],
        instructions: "Spray thoroughly at first sign of powdery patches. Potassium bicarbonate rapidly shifts surface pH to kill spores.",
        safety: "Do not spray sulfur when temperatures exceed 32°C to prevent phytotoxicity."
      },
      chemicalTreatment: {
        products: ["Propiconazole 25% EC (1ml/L)", "Tebuconazole 25.9% EC", "Difenoconazole 25% EC"],
        instructions: "Apply systemic triazole fungicide at 10-day intervals. Rotate FRAC groups to prevent resistance buildup.",
        safety: "Use chemical-resistant protective gear. 10-day PHI."
      }
    },
    "Severe_Plant_Stress": {
      name: "Severe Abiotic & Wilt Stress (Fusarium / Drought)",
      cropCommon: "Cotton / Legumes / Solanaceous",
      severity: "High",
      urgency: "High",
      color: "#b91c1c",
      badgeClass: "sev-pill-high",
      symptoms: "Vascular browning inside stem tissue, unilateral leaf drooping, marginal leaf scorch, stunted shoot growth.",
      causes: "Soil-borne fungal pathogen ingress or extreme moisture deficiency coupled with high transpiration stress.",
      prevention: "Solarize nursery beds, adjust soil pH to 6.5, apply organic mulch to conserve moisture and suppress pathogens.",
      organicTreatment: {
        products: ["Pseudomonas fluorescens 10g/L soil drench", "Humic & Fulvic Acid Root Stimulator", "Mycorrhiza inoculant"],
        instructions: "Drench root zone thoroughly. Restore soil microbial balance and provide shaded micro-climate if possible.",
        safety: "Safe and rejuvenating for soil microbiology."
      },
      chemicalTreatment: {
        products: ["Carbendazim 50% WP drench", "Fosetyl-Al 80% WP", "Potassium Phosphite"],
        instructions: "Apply root zone systemic drenching. Remove and incinerate completely collapsed specimens to prevent spread.",
        safety: "Follow registered chemical disposal rules. Protect waterways."
      }
    }
  },

  // Crop NPK Requirements for Fertilizer Calculator
  cropNutrients: {
    "Tomato": {
      name: "Tomato",
      icon: "🍅",
      baseNPK: { N: 120, P: 100, K: 150 }, // kg/ha
      micronutrients: ["Calcium", "Magnesium", "Boron"],
      phRange: [6.0, 7.0],
      stages: {
        seedling: { label: "Seedling Stage", N: 0.8, P: 1.2, K: 0.9 },
        vegetative: { label: "Vegetative Growth", N: 1.2, P: 1.0, K: 1.1 },
        flowering: { label: "Flowering Stage", N: 1.0, P: 1.5, K: 1.2 },
        fruiting: { label: "Fruiting & Ripening", N: 0.9, P: 1.1, K: 1.8 }
      }
    },
    "Potato": {
      name: "Potato",
      icon: "🥔",
      baseNPK: { N: 140, P: 60, K: 180 },
      micronutrients: ["Sulfur", "Calcium", "Magnesium"],
      phRange: [5.0, 6.5],
      stages: {
        planting: { label: "Sprouting / Planting", N: 0.5, P: 1.0, K: 0.6 },
        vegetative: { label: "Vegetative Growth", N: 1.1, P: 1.1, K: 1.0 },
        tuber_formation: { label: "Tuber Initiation", N: 1.2, P: 1.3, K: 1.5 },
        bulking: { label: "Tuber Bulking", N: 0.8, P: 1.0, K: 1.8 }
      }
    },
    "Corn": {
      name: "Corn / Maize",
      icon: "🌽",
      baseNPK: { N: 180, P: 80, K: 120 },
      micronutrients: ["Zinc", "Iron", "Manganese"],
      phRange: [6.0, 6.8],
      stages: {
        seedling: { label: "Emergence", N: 0.4, P: 1.0, K: 0.7 },
        vegetative: { label: "V6-V12 Vegetative", N: 1.5, P: 1.2, K: 1.0 },
        tasseling: { label: "Tasseling / Silking", N: 1.2, P: 1.0, K: 1.3 },
        grain_fill: { label: "Grain Filling", N: 0.7, P: 0.9, K: 1.1 }
      }
    },
    "Cotton": {
      name: "Cotton",
      icon: "🌿",
      baseNPK: { N: 150, P: 60, K: 120 },
      micronutrients: ["Boron", "Zinc", "Magnesium"],
      phRange: [6.0, 7.5],
      stages: {
        seedling: { label: "Squaring Stage", N: 0.8, P: 1.2, K: 0.9 },
        vegetative: { label: "Vegetative Peak", N: 1.3, P: 1.1, K: 1.1 },
        flowering: { label: "Peak Bloom", N: 1.2, P: 1.4, K: 1.4 },
        boll_fill: { label: "Boll Development", N: 0.8, P: 0.9, K: 1.6 }
      }
    },
    "Wheat": {
      name: "Wheat",
      icon: "🌾",
      baseNPK: { N: 130, P: 65, K: 60 },
      micronutrients: ["Zinc", "Copper", "Manganese"],
      phRange: [6.0, 7.0],
      stages: {
        tillering: { label: "Tillering Stage", N: 1.2, P: 1.1, K: 0.9 },
        jointing: { label: "Stem Elongation", N: 1.4, P: 1.0, K: 1.0 },
        booting: { label: "Booting / Heading", N: 1.0, P: 1.2, K: 1.1 },
        grain_fill: { label: "Milk & Dough Stage", N: 0.6, P: 0.8, K: 0.9 }
      }
    },
    "Apple": {
      name: "Apple Orchard",
      icon: "🍎",
      baseNPK: { N: 150, P: 75, K: 200 },
      micronutrients: ["Calcium", "Magnesium", "Boron", "Zinc"],
      phRange: [6.0, 7.0],
      stages: {
        dormant: { label: "Bud Break / Dormant", N: 0.6, P: 0.8, K: 0.7 },
        flowering: { label: "Pink Bud to Bloom", N: 1.2, P: 1.5, K: 1.0 },
        fruiting: { label: "Fruit Sizing & Maturation", N: 0.9, P: 1.2, K: 1.8 }
      }
    }
  },

  // Initial Detection Records (Matches Python workspace history)
  initialDetections: [
    {
      id: "scan-1",
      timestamp: "2026-10-06T10:15:00.000Z",
      crop: "Tomato",
      predictedDisease: "Early_Disease_Symptoms",
      confidence: 0.942,
      severity: "Low",
      imageName: "tomato_leaf_scan_101.jpg",
      fieldPlot: "North Plot B-4"
    },
    {
      id: "scan-2",
      timestamp: "2026-10-05T16:20:00.000Z",
      crop: "Cotton",
      predictedDisease: "Plant_Healthy_Condition",
      confidence: 0.985,
      severity: "None",
      imageName: "cotton_leaf_fresh.jpg",
      fieldPlot: "South Plot A-1"
    },
    {
      id: "scan-3",
      timestamp: "2026-10-04T09:40:00.000Z",
      crop: "Potato",
      predictedDisease: "Severe_Plant_Disease",
      confidence: 0.958,
      severity: "Critical",
      imageName: "potato_blight_sample.jpg",
      fieldPlot: "East Field Plot C-2"
    },
    {
      id: "scan-4",
      timestamp: "2026-10-02T14:10:00.000Z",
      crop: "Corn",
      predictedDisease: "Moderate_Fungal_Disease",
      confidence: 0.916,
      severity: "Medium",
      imageName: "maize_lesion_scan.jpg",
      fieldPlot: "Central Acre C"
    },
    {
      id: "scan-5",
      timestamp: "2026-09-29T11:35:00.000Z",
      crop: "Cotton",
      predictedDisease: "Severe_Plant_Stress",
      confidence: 0.893,
      severity: "High",
      imageName: "cotton_wilting_edge.jpg",
      fieldPlot: "West Plot D-1"
    },
    {
      id: "scan-6",
      timestamp: "2026-09-28T08:15:00.000Z",
      crop: "Tomato",
      predictedDisease: "Plant_Healthy_Condition",
      confidence: 0.978,
      severity: "None",
      imageName: "tomato_vigorous_foliage.jpg",
      fieldPlot: "North Plot B-2"
    }
  ],

  // Initial Treatment Logs (Matches Python workspace history)
  initialTreatments: [
    {
      id: "trt-1",
      timestamp: "2026-10-04T11:00:00.000Z",
      crop: "Potato",
      disease: "Severe_Plant_Disease",
      treatmentType: "chemical",
      productName: "Metalaxyl-M + Mancozeb (Ridomil Gold)",
      dosage: "2.5 g / Liter water",
      areaTreated: "3.2 ha",
      cost: 145.0,
      status: "in_progress",
      plannedDate: "2026-10-11T09:00:00.000Z",
      notes: "Systemic spray completed in East Plot C-2. Foliage necrosis arrested; second prophylactic check due next week."
    },
    {
      id: "trt-2",
      timestamp: "2026-10-02T15:30:00.000Z",
      crop: "Corn",
      disease: "Moderate_Fungal_Disease",
      treatmentType: "organic",
      productName: "Cold-Pressed Neem Oil (3000 PPM) + Potassium Bicarbonate",
      dosage: "5 ml / Liter water",
      areaTreated: "2.0 ha",
      cost: 65.0,
      status: "completed",
      plannedDate: "2026-10-07T08:00:00.000Z",
      notes: "Foliar mist applied at sunrise. Fungal mycelium dried up, healthy new leaves unfurling."
    },
    {
      id: "trt-3",
      timestamp: "2026-09-29T17:00:00.000Z",
      crop: "Cotton",
      disease: "Severe_Plant_Stress",
      treatmentType: "integrated",
      productName: "Pseudomonas fluorescens root drench + Humic Bio-Booster",
      dosage: "10 ml / Liter soil drench",
      areaTreated: "4.5 ha",
      cost: 110.0,
      status: "completed",
      plannedDate: "2026-10-03T10:00:00.000Z",
      notes: "Soil aerated and biological inoculant applied with drip irrigation. Plant turgor fully recovered."
    }
  ]
};
