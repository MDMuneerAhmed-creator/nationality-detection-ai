import React from 'react';
import { Database, Calendar, User, Smile, Palette } from 'lucide-react';
import { HistoryRecord } from '../types';

interface HistoryViewProps {
  history: HistoryRecord[];
}

export const HistoryView: React.FC<HistoryViewProps> = ({ history }) => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      
      {/* Header Bento Box */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-blue-600" /> SQLite Persistent Prediction Log
          </h2>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Inference Records & DB Log
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Every upload & forward pass prediction is recorded in local SQLite database (`database.db`).
          </p>
        </div>
        <span className="text-xs font-mono font-bold bg-blue-50 text-blue-600 px-3 py-1.5 rounded-xl border border-blue-100 shrink-0">
          {history.length} Logs Stored
        </span>
      </div>

      {history.length === 0 ? (
        <div className="p-12 text-center bg-white rounded-2xl border border-slate-200 text-slate-400">
          No prediction logs recorded yet. Run a prediction in the Inference tab.
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden p-6">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">
            Recent Multi-CNN Inferences
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 font-bold uppercase tracking-wider border-b border-slate-200">
                <tr>
                  <th className="py-3 px-4"># ID</th>
                  <th className="py-3 px-4">Image</th>
                  <th className="py-3 px-4">Nationality</th>
                  <th className="py-3 px-4">Emotion</th>
                  <th className="py-3 px-4">Age (Conditional)</th>
                  <th className="py-3 px-4">Dress Colour (Conditional)</th>
                  <th className="py-3 px-4">Rule Applied</th>
                  <th className="py-3 px-4">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {history.map((record) => (
                  <tr key={record.id} className="hover:bg-slate-50 transition-colors">
                    <td className="py-3 px-4 font-bold text-slate-400">#{record.id}</td>
                    <td className="py-3 px-4">
                      <div className="w-10 h-10 rounded-lg bg-slate-100 overflow-hidden border border-slate-200">
                        {record.image_path ? (
                          <img src={record.image_path} alt="Thumb" className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                            IMG
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center gap-1 font-bold font-sans text-slate-900 bg-blue-50 px-2.5 py-1 rounded-lg text-blue-700 border border-blue-100">
                        <User className="w-3 h-3 text-blue-600" /> {record.nationality}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-semibold text-slate-800 font-sans">
                      <span className="flex items-center gap-1 text-slate-700">
                        <Smile className="w-3 h-3 text-blue-500" /> {record.emotion}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-sans">
                      {record.age ? (
                        <span className="font-semibold text-slate-800 flex items-center gap-1">
                          <Calendar className="w-3 h-3 text-blue-500" /> {record.age}
                        </span>
                      ) : (
                        <span className="text-slate-300 font-mono">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-sans">
                      {record.dress_color ? (
                        <span className="font-semibold text-slate-800 flex items-center gap-1">
                          <Palette className="w-3 h-3 text-blue-500" /> {record.dress_color}
                        </span>
                      ) : (
                        <span className="text-slate-300 font-mono">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-slate-500 max-w-xs truncate text-[11px] font-sans">
                      {record.conditional_rule || 'Default'}
                    </td>
                    <td className="py-3 px-4 text-slate-400 text-[11px]">
                      {record.created_at ? new Date(record.created_at).toLocaleString() : 'Just now'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};

