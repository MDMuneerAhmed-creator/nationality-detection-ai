import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { PredictionView } from './components/PredictionView';
import { PredictionResult } from './types';

export default function App() {
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (file: File): Promise<PredictionResult> => {
    setLoading(true);
    setError(null);
    try {
      const createFormData = () => {
        const fd = new FormData();
        fd.append('file', file);
        return fd;
      };

      const endpoints = ['/api/predict', '/predict'];
      let res: Response | null = null;
      let lastErrMessage = '';

      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            body: createFormData(),
          });
          const contentType = response.headers.get('content-type') || '';
          if (response.ok && contentType.includes('application/json')) {
            res = response;
            break;
          } else {
            const json = await response.json().catch(() => null);
            if (json && json.error) {
              lastErrMessage = json.error;
            }
          }
        } catch (fetchErr: any) {
          if (fetchErr && fetchErr.message) {
            lastErrMessage = fetchErr.message;
          }
        }
      }

      if (!res) {
        throw new Error(
          lastErrMessage && !lastErrMessage.includes('Failed to fetch')
            ? lastErrMessage
            : 'Prediction service unavailable. Please check image format and server connection.'
        );
      }

      const result: PredictionResult = await res.json();
      setPrediction(result);
      return result;
    } catch (err: any) {
      console.error('Prediction error:', err);
      const userMessage = err?.message || 'Prediction failed. Please try again.';
      setError(userMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-600 selection:text-white pb-12">
      <Header />
      
      <main>
        <PredictionView
          onPredict={handlePredict}
          loading={loading}
          prediction={prediction}
          error={error}
        />
      </main>
    </div>
  );
}

