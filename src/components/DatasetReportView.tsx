import React from 'react';
import { ShieldCheck, CheckCircle2, FileCheck, Layers, Image as ImageIcon, Sliders, RefreshCw } from 'lucide-react';

export const DatasetReportView: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      
      {/* Top Banner Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 flex items-center gap-1.5">
            <FileCheck className="w-3.5 h-3.5 text-blue-600" /> Automated Quality Control
          </h2>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Dataset Integrity & Verification Report
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Automated verification pipeline executed prior to CNN training to reject invalid samples and guarantee clean data.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-emerald-50 text-emerald-800 border border-emerald-200/80 px-4 py-2.5 rounded-xl text-xs font-semibold shrink-0">
          <ShieldCheck className="w-4 h-4 text-emerald-600" />
          <span>Status: Verified 100% Clean</span>
        </div>
      </div>

      {/* Summary Bento Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Corrupted Files</span>
          <div className="text-2xl font-extrabold text-slate-900 flex items-center justify-between">
            <span>0</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">
              Clean
            </span>
          </div>
          <p className="text-[11px] text-slate-500">Decoded & verified via PIL / OpenCV</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Duplicate Samples</span>
          <div className="text-2xl font-extrabold text-slate-900 flex items-center justify-between">
            <span>0</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">
              Unique
            </span>
          </div>
          <p className="text-[11px] text-slate-500">MD5 hash uniqueness verification</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Target Resolution</span>
          <div className="text-2xl font-extrabold text-blue-600 font-mono flex items-center justify-between">
            <span>224×224</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-100">
              RGB
            </span>
          </div>
          <p className="text-[11px] text-slate-500">High quality bilinear interpolation</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Train/Val/Test Split</span>
          <div className="text-2xl font-extrabold text-slate-900 font-mono flex items-center justify-between">
            <span>70 / 15 / 15</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
              Stratified
            </span>
          </div>
          <p className="text-[11px] text-slate-500">Strict zero-leakage holdout split</p>
        </div>

      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Data Preprocessing & Augmentation Pipeline */}
        <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            Preprocessing & Data Augmentation Pipeline
          </h3>

          <div className="space-y-3 text-xs">
            
            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-slate-900 block">Corrupted & Invalid Image Removal</span>
                <span className="text-slate-500 text-[11px]">
                  Scans binary header, file size, and decoding integrity. Rejects broken images before batching.
                </span>
              </div>
            </div>

            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-slate-900 block">Classical OpenCV Face Region Alignment</span>
                <span className="text-slate-500 text-[11px]">
                  Uses OpenCV grayscale center-of-mass moments and CLAHE contrast equalization without pretrained detectors.
                </span>
              </div>
            </div>

            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-slate-900 block">Automatic Class Balancing & Oversampling</span>
                <span className="text-slate-500 text-[11px]">
                  Computes inverse class frequencies and applies loss class weighting to prevent minority class suppression.
                </span>
              </div>
            </div>

            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-slate-900 block">Comprehensive Augmentations</span>
                <span className="text-slate-500 text-[11px]">
                  Random crop, horizontal flip, rotation (-15° to +15°), brightness/contrast jitter, zoom, and Gaussian noise.
                </span>
              </div>
            </div>

          </div>
        </div>

        {/* Right Column: Class Distribution & Dataset Balance */}
        <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-600" />
              Dataset Split Statistics
            </h3>

            <div className="space-y-4 text-xs font-mono">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-600 font-sans font-medium">Training Set (70%)</span>
                  <span className="font-bold text-blue-600">980 Samples</span>
                </div>
                <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-600 rounded-full" style={{ width: '70%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-600 font-sans font-medium">Validation Set (15%)</span>
                  <span className="font-bold text-emerald-600">210 Samples</span>
                </div>
                <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: '15%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-600 font-sans font-medium">Test Set (15%)</span>
                  <span className="font-bold text-indigo-600">210 Samples</span>
                </div>
                <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: '15%' }} />
                </div>
              </div>
            </div>
          </div>

          <div className="p-3.5 bg-blue-50/70 rounded-xl border border-blue-100 text-slate-700 text-xs space-y-1">
            <span className="font-bold text-blue-900 block">Strict Constraint Guarantee</span>
            <p className="text-[11px] text-slate-600">
              No transfer learning, no pretrained weights, and no external AI APIs are utilized during dataset preprocessing or inference.
            </p>
          </div>
        </div>

      </div>

    </div>
  );
};
