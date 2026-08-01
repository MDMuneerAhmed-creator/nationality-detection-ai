import React, { useState } from 'react';
import { Layers, ShieldCheck, Box, Cpu, HardDrive } from 'lucide-react';
import { ModelArchSpec } from '../types';

interface ModelInspectorProps {
  archSpecs: Record<string, ModelArchSpec>;
}

export const ModelInspector: React.FC<ModelInspectorProps> = ({ archSpecs }) => {
  const [selectedModelKey, setSelectedModelKey] = useState<string>('Nationality_CNN');

  const modelKeys = Object.keys(archSpecs);
  const currentSpec = archSpecs[selectedModelKey];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      
      {/* Top Bento Info Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-600" /> Neural Architecture Inspector
            </h2>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              Layer-by-Layer Custom PyTorch CNN Specs
            </h1>
            <p className="text-xs text-slate-500 mt-1">
              Every CNN network is constructed manually from primitive tensor layers (`Conv2D`, `BatchNorm2D`, `ReLU`, `Dropout`).
            </p>
          </div>

          {/* Model Selection Tabs */}
          <div className="flex flex-wrap gap-2">
            {modelKeys.map((key) => (
              <button
                key={key}
                onClick={() => setSelectedModelKey(key)}
                className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all border cursor-pointer ${
                  selectedModelKey === key
                    ? 'bg-blue-600 text-white border-blue-600 shadow-md shadow-blue-500/20'
                    : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                }`}
              >
                {key.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>
      </div>

      {currentSpec ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left Summary Card (Bento Box - 4 cols) */}
          <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5 flex flex-col justify-between">
            <div className="space-y-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center shrink-0 border border-blue-100">
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">{selectedModelKey.replace('_', ' ')}</h3>
                  <p className="text-xs text-slate-400 font-mono">Custom PyTorch Model</p>
                </div>
              </div>

              <div className="space-y-3 pt-3 border-t border-slate-100 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-500 font-medium">Output Classes</span>
                  <span className="font-bold text-slate-900">{currentSpec.num_classes} Classes</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-500 font-medium">Trainable Params</span>
                  <span className="font-mono font-bold text-blue-600">
                    {currentSpec.total_params ? currentSpec.total_params.toLocaleString() : '8,910,119'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-500 font-medium">Input Tensor</span>
                  <span className="font-mono font-semibold text-slate-800">128 x 128 x 3 RGB</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500 font-medium">Initialization</span>
                  <span className="font-semibold text-emerald-700">He Normal (Random)</span>
                </div>
              </div>

              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Target Class Labels</p>
                <div className="flex flex-wrap gap-1.5">
                  {currentSpec.classes?.map((cls, i) => (
                    <span key={i} className="px-2.5 py-1 bg-slate-50 border border-slate-200 text-slate-700 rounded-lg text-[11px] font-mono font-semibold">
                      {cls}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="p-3.5 bg-emerald-50 rounded-xl border border-emerald-200/70 text-emerald-900 text-xs flex items-start gap-2.5 mt-4">
              <ShieldCheck className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold block">100% Pretrained-Free</span>
                <span className="text-[11px] text-emerald-800/80">
                  Zero transferred weights from ImageNet, ResNet, VGG, or MobileNet.
                </span>
              </div>
            </div>
          </div>

          {/* Right Layer Architecture Pipeline Table (Bento Box - 8 cols) */}
          <div className="lg:col-span-8 bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
              <Box className="w-4 h-4 text-blue-600" />
              Layer-by-Layer Forward Tensor Pipeline
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500 font-bold uppercase tracking-wider border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3">Layer Type</th>
                    <th className="py-2.5 px-3">Output Tensor Shape</th>
                    <th className="py-2.5 px-3">Parameters</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {(currentSpec.architecture || [
                    { layer: "Input", shape: "128x128x3", params: 0 },
                    { layer: "Conv2D(32, 3x3)", shape: "128x128x32", params: 896 },
                    { layer: "BatchNorm2d", shape: "128x128x32", params: 64 },
                    { layer: "ReLU", shape: "128x128x32", params: 0 },
                    { layer: "MaxPool2d(2x2)", shape: "64x64x32", params: 0 },
                    { layer: "Conv2D(64, 3x3)", shape: "64x64x64", params: 18496 },
                    { layer: "BatchNorm2d", shape: "64x64x64", params: 128 },
                    { layer: "ReLU", shape: "64x64x64", params: 0 },
                    { layer: "MaxPool2d(2x2)", shape: "32x32x64", params: 0 },
                    { layer: "Conv2D(128, 3x3)", shape: "32x32x128", params: 73856 },
                    { layer: "BatchNorm2d", shape: "32x32x128", params: 256 },
                    { layer: "ReLU", shape: "32x32x128", params: 0 },
                    { layer: "MaxPool2d(2x2)", shape: "16x16x128", params: 0 },
                    { layer: "Conv2D(256, 3x3)", shape: "16x16x256", params: 295168 },
                    { layer: "BatchNorm2d", shape: "16x16x256", params: 512 },
                    { layer: "ReLU", shape: "16x16x256", params: 0 },
                    { layer: "MaxPool2d(2x2)", shape: "8x8x256", params: 0 },
                    { layer: "Flatten", shape: "16384", params: 0 },
                    { layer: "Dense(512)", shape: "512", params: 8389120 },
                    { layer: "Dropout(0.5)", shape: "512", params: 0 },
                    { layer: "Dense(256)", shape: "256", params: 131328 },
                    { layer: "Dropout(0.3)", shape: "256", params: 0 },
                    { layer: "Dense(Softmax)", shape: `${currentSpec.num_classes}`, params: 256 * currentSpec.num_classes + currentSpec.num_classes }
                  ]).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="py-2.5 px-3 text-slate-400 font-sans">{idx + 1}</td>
                      <td className="py-2.5 px-3 font-semibold text-slate-800">{row.layer}</td>
                      <td className="py-2.5 px-3 text-blue-600">{row.shape}</td>
                      <td className="py-2.5 px-3 text-slate-600">{row.params.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

          </div>

        </div>
      ) : (
        <div className="p-8 text-center bg-white rounded-2xl border border-slate-200 text-slate-500">
          Loading architecture details...
        </div>
      )}

    </div>
  );
};

