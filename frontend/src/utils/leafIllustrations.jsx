import React from 'react';

/**
 * Botanical Leaf Illustrations Generator
 * Generates crisp botanical SVG leaf specimens for testing without external files!
 */
export const LeafSpecimen = ({ type, size = 180, className = "" }) => {
  switch (type) {
    case "tomato-blight":
      return (
        <svg width={size} height={size} viewBox="0 0 200 200" className={className}>
          <defs>
            <linearGradient id="tomatoLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#4ade80" />
              <stop offset="50%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#15803d" />
            </linearGradient>
            <radialGradient id="blightSpot1" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#451a03" />
              <stop offset="60%" stopColor="#78350f" />
              <stop offset="85%" stopColor="#d97706" />
              <stop offset="100%" stopColor="#fef08a" stopOpacity="0.8" />
            </radialGradient>
            <radialGradient id="blightSpot2" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#451a03" />
              <stop offset="70%" stopColor="#92400e" />
              <stop offset="90%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#fef08a" stopOpacity="0.7" />
            </radialGradient>
          </defs>
          {/* Main Leaf Blade */}
          <path d="M 100,20 C 145,50 170,110 125,165 C 100,185 85,185 75,165 C 30,110 55,50 100,20 Z" fill="url(#tomatoLeafGrad)" stroke="#166534" strokeWidth="2.5" />
          {/* Leaf Stem & Veins */}
          <path d="M 100,20 Q 98,100 95,190" stroke="#86efac" strokeWidth="3" fill="none" />
          <path d="M 98,60 Q 130,55 142,68" stroke="#86efac" strokeWidth="1.8" fill="none" opacity="0.85" />
          <path d="M 98,60 Q 70,55 58,68" stroke="#86efac" strokeWidth="1.8" fill="none" opacity="0.85" />
          <path d="M 97,95 Q 135,92 148,110" stroke="#86efac" strokeWidth="1.8" fill="none" opacity="0.85" />
          <path d="M 97,95 Q 65,92 52,110" stroke="#86efac" strokeWidth="1.8" fill="none" opacity="0.85" />
          <path d="M 96,130 Q 120,132 130,145" stroke="#86efac" strokeWidth="1.5" fill="none" opacity="0.85" />
          {/* Alternaria Concentric Blight Rings */}
          <circle cx="120" cy="85" r="18" fill="url(#blightSpot1)" stroke="#b45309" strokeWidth="1.2" />
          <circle cx="120" cy="85" r="10" fill="none" stroke="#fef08a" strokeWidth="1" strokeDasharray="2,2" />
          <circle cx="70" cy="115" r="14" fill="url(#blightSpot2)" stroke="#b45309" strokeWidth="1.2" />
          <circle cx="70" cy="115" r="7" fill="none" stroke="#fef08a" strokeWidth="0.9" strokeDasharray="2,2" />
          <circle cx="110" cy="140" r="10" fill="url(#blightSpot1)" />
        </svg>
      );

    case "potato-blight":
      return (
        <svg width={size} height={size} viewBox="0 0 200 200" className={className}>
          <defs>
            <linearGradient id="potatoLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#86efac" />
              <stop offset="40%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#14532d" />
            </linearGradient>
            <radialGradient id="lateBlightNecrosis" cx="45%" cy="45%" r="55%">
              <stop offset="0%" stopColor="#1c1917" />
              <stop offset="50%" stopColor="#451a03" />
              <stop offset="85%" stopColor="#78350f" />
              <stop offset="100%" stopColor="#a16207" stopOpacity="0.9" />
            </radialGradient>
          </defs>
          {/* Potato Leaf Shape with Serpentine Edges */}
          <path d="M 100,15 C 150,45 175,100 140,160 C 115,185 85,185 60,160 C 25,100 50,45 100,15 Z" fill="url(#potatoLeafGrad)" stroke="#166534" strokeWidth="2.5" />
          {/* Veins */}
          <path d="M 100,15 Q 98,100 96,192" stroke="#bbf7d0" strokeWidth="3.2" fill="none" />
          <path d="M 99,55 Q 135,50 150,65" stroke="#bbf7d0" strokeWidth="1.8" fill="none" opacity="0.8" />
          <path d="M 99,55 Q 65,50 50,65" stroke="#bbf7d0" strokeWidth="1.8" fill="none" opacity="0.8" />
          <path d="M 98,95 Q 140,90 155,115" stroke="#bbf7d0" strokeWidth="1.8" fill="none" opacity="0.8" />
          <path d="M 98,95 Q 60,90 45,115" stroke="#bbf7d0" strokeWidth="1.8" fill="none" opacity="0.8" />
          {/* Severe Phytophthora Necrotic Blotches */}
          <path d="M 105,45 C 135,45 155,75 145,95 C 130,110 100,90 105,45 Z" fill="url(#lateBlightNecrosis)" stroke="#78350f" strokeWidth="1.5" />
          <path d="M 45,85 C 75,80 85,110 70,135 C 50,145 35,115 45,85 Z" fill="url(#lateBlightNecrosis)" stroke="#78350f" strokeWidth="1.5" />
          <circle cx="120" cy="135" r="12" fill="url(#lateBlightNecrosis)" />
          {/* White Mold Mildew Margin Halo */}
          <path d="M 100,43 C 138,42 158,73 148,97" stroke="#f1f5f9" strokeWidth="1.2" fill="none" strokeDasharray="3,2" />
        </svg>
      );

    case "cotton-healthy":
      return (
        <svg width={size} height={size} viewBox="0 0 200 200" className={className}>
          <defs>
            <linearGradient id="cottonHealthyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#4ade80" />
              <stop offset="50%" stopColor="#16a34a" />
              <stop offset="100%" stopColor="#15803d" />
            </linearGradient>
            <radialGradient id="dewDrop" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
              <stop offset="60%" stopColor="#93c5fd" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.2" />
            </radialGradient>
          </defs>
          {/* 3-Lobed Cotton Leaf Blade */}
          <path d="M 100,20 C 120,45 135,50 165,65 C 145,85 135,100 155,145 C 130,135 115,145 100,175 C 85,145 70,135 45,145 C 65,100 55,85 35,65 C 65,50 80,45 100,20 Z" fill="url(#cottonHealthyGrad)" stroke="#166534" strokeWidth="2.5" />
          {/* Palmate Venation */}
          <path d="M 100,20 Q 98,100 96,190" stroke="#bbf7d0" strokeWidth="3" fill="none" />
          <path d="M 98,100 Q 130,80 165,65" stroke="#bbf7d0" strokeWidth="2.2" fill="none" />
          <path d="M 98,100 Q 66,80 35,65" stroke="#bbf7d0" strokeWidth="2.2" fill="none" />
          <path d="M 97,115 Q 125,125 155,145" stroke="#bbf7d0" strokeWidth="1.8" fill="none" />
          <path d="M 97,115 Q 70,125 45,145" stroke="#bbf7d0" strokeWidth="1.8" fill="none" />
          {/* Fresh Morning Dew Drops */}
          <circle cx="85" cy="70" r="5" fill="url(#dewDrop)" stroke="rgba(255,255,255,0.8)" strokeWidth="0.8" />
          <circle cx="125" cy="110" r="6" fill="url(#dewDrop)" stroke="rgba(255,255,255,0.8)" strokeWidth="0.8" />
          <circle cx="108" cy="55" r="3.5" fill="url(#dewDrop)" />
        </svg>
      );

    case "corn-fungal":
      return (
        <svg width={size} height={size} viewBox="0 0 200 200" className={className}>
          <defs>
            <linearGradient id="cornLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#86efac" />
              <stop offset="40%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#15803d" />
            </linearGradient>
            <linearGradient id="maizeLesionGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#fde68a" />
              <stop offset="40%" stopColor="#b45309" />
              <stop offset="100%" stopColor="#78350f" />
            </linearGradient>
          </defs>
          {/* Slender Elongated Corn Leaf Blade */}
          <path d="M 45,190 Q 75,100 130,40 Q 155,15 170,10 Q 160,35 130,85 Q 85,145 65,195 Z" fill="url(#cornLeafGrad)" stroke="#166534" strokeWidth="2.5" />
          {/* Parallel Midrib */}
          <path d="M 55,192 Q 95,95 150,25" stroke="#bbf7d0" strokeWidth="3" fill="none" />
          {/* Parallel Vein Lines */}
          <path d="M 62,180 Q 98,90 145,35" stroke="#86efac" strokeWidth="1" fill="none" opacity="0.6" />
          <path d="M 48,185 Q 88,105 138,50" stroke="#86efac" strokeWidth="1" fill="none" opacity="0.6" />
          {/* Elongated Fungal Lesions (Northern Corn Leaf Blight) */}
          <rect x="95" y="70" width="35" height="9" rx="4" transform="rotate(-38 95 70)" fill="url(#maizeLesionGrad)" stroke="#92400e" strokeWidth="1" />
          <rect x="80" y="110" width="42" height="11" rx="5" transform="rotate(-38 80 110)" fill="url(#maizeLesionGrad)" stroke="#92400e" strokeWidth="1" />
          <rect x="68" y="145" width="28" height="8" rx="4" transform="rotate(-38 68 145)" fill="url(#maizeLesionGrad)" stroke="#92400e" strokeWidth="1" />
        </svg>
      );

    case "severe-stress":
    default:
      return (
        <svg width={size} height={size} viewBox="0 0 200 200" className={className}>
          <defs>
            <linearGradient id="wiltLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#facc15" />
              <stop offset="40%" stopColor="#ca8a04" />
              <stop offset="100%" stopColor="#713f12" />
            </linearGradient>
            <linearGradient id="scorchEdge" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#b91c1c" />
              <stop offset="100%" stopColor="#451a03" />
            </linearGradient>
          </defs>
          {/* Drooping Wilted Leaf Blade */}
          <path d="M 100,25 C 135,45 160,85 145,145 C 130,175 110,185 90,180 C 60,175 45,120 55,70 C 65,40 85,25 100,25 Z" fill="url(#wiltLeafGrad)" stroke="#854d0e" strokeWidth="2.5" />
          {/* Severe Marginal Scorch */}
          <path d="M 100,25 C 135,45 160,85 145,145" stroke="url(#scorchEdge)" strokeWidth="6" fill="none" opacity="0.8" />
          <path d="M 55,70 C 45,120 60,175 90,180" stroke="url(#scorchEdge)" strokeWidth="6" fill="none" opacity="0.8" />
          {/* Weakened Midrib */}
          <path d="M 100,25 Q 98,100 90,188" stroke="#fef08a" strokeWidth="2.2" fill="none" />
          <path d="M 98,65 Q 125,75 135,95" stroke="#fef08a" strokeWidth="1.2" fill="none" opacity="0.7" />
          <path d="M 95,115 Q 120,125 130,140" stroke="#fef08a" strokeWidth="1.2" fill="none" opacity="0.7" />
          <path d="M 98,65 Q 75,75 62,95" stroke="#fef08a" strokeWidth="1.2" fill="none" opacity="0.7" />
        </svg>
      );
  }
};
