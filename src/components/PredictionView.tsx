import React, { useState, useRef } from 'react';
import { Upload, Loader2, Image as ImageIcon } from 'lucide-react';
import { PredictionResult } from '../types';

interface PredictionViewProps {
  onPredict: (file: File) => Promise<PredictionResult>;
  loading: boolean;
  prediction: PredictionResult | null;
  error?: string | null;
}

export const PredictionView: React.FC<PredictionViewProps> = ({
  onPredict,
  loading,
  prediction,
  error,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handlePredictClick = () => {
    if (selectedFile) {
      onPredict(selectedFile).catch((err) => {
        // Error state is set and rendered in the UI by App state
        console.warn('Prediction request failed:', err);
      });
    }
  };

  // Extract prediction values
  const nationality = prediction?.predictions.nationality.value || '';
  const emotion = prediction?.predictions.emotion.value || '';
  const rawAge = prediction?.predictions.age?.value || null;
  const dressColor = prediction?.predictions.dress_color?.value || null;

  const nationalityLower = nationality.toLowerCase();
  const isIndian = nationalityLower === 'indian';
  const isAmerican = nationalityLower === 'american';
  const isAfrican = nationalityLower === 'african';

  // Display conditions:
  // Indian: Age, Dress Colour
  // American: Age
  // African: Dress Colour
  // Otherwise: Only Nationality, Emotion
  const showAge = isIndian || isAmerican || Boolean(rawAge);
  const showDressColor = isIndian || isAfrican || Boolean(dressColor);

  const formatAge = (ageVal: string | null) => {
    if (!ageVal) return '22 Years';
    if (ageVal.toLowerCase().includes('year')) return ageVal;
    return `${ageVal} Years`;
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-8">
      {/* Upload Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
        <h2 className="text-base font-bold text-slate-900">Upload an Image</h2>

        {/* Drag & Drop Preview Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative flex flex-col items-center justify-center w-full min-h-[260px] border-2 border-dashed rounded-2xl cursor-pointer transition-all overflow-hidden bg-slate-50/50 ${
            isDragging
              ? 'border-blue-600 bg-blue-50/50'
              : 'border-slate-300 hover:border-blue-500 hover:bg-blue-50/20'
          }`}
        >
          {previewUrl ? (
            <div className="w-full h-64 relative flex items-center justify-center p-2 bg-slate-900/5">
              <img
                src={previewUrl}
                alt="Image Preview"
                className="max-h-full max-w-full object-contain rounded-lg shadow-sm"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center p-6 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center text-blue-600">
                <ImageIcon className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Drag and drop your image here
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Supports JPG, PNG, or WEBP
                </p>
              </div>
            </div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInput}
            accept="image/*"
            className="hidden"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-5 py-3 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 font-semibold text-sm rounded-xl transition-all flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <Upload className="w-4 h-4 text-blue-600" />
            Upload Image
          </button>

          <button
            type="button"
            onClick={handlePredictClick}
            disabled={!selectedFile || loading}
            className="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white font-bold text-sm rounded-xl shadow-md shadow-blue-500/10 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Predicting...
              </>
            ) : (
              'Predict'
            )}
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl">
            {error}
          </div>
        )}
      </div>

      {/* Output Section */}
      {prediction && !loading && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-3">
            Prediction Result
          </h2>

          <div className="grid grid-cols-1 gap-4 text-sm font-medium">
            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-500">Nationality</span>
              <span className="font-bold text-slate-900 text-base">{nationality}</span>
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-500">Emotion</span>
              <span className="font-bold text-slate-900 text-base">{emotion}</span>
            </div>

            {showAge && rawAge && (
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-slate-500">Age</span>
                <span className="font-bold text-slate-900 text-base">{formatAge(rawAge)}</span>
              </div>
            )}

            {showDressColor && dressColor && (
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-slate-500">Dress Colour</span>
                <span className="font-bold text-slate-900 text-base">{dressColor}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
