'use client';

import React, { useState, useEffect } from 'react';

interface ForecastPoint {
  month_index: number;
  target_date: string;
  predicted_price: number;
  lower_bound_95: number;
  upper_bound_95: number;
}

interface DashboardStats {
  latest_historical_price: number;
  latest_historical_date: string;
  active_model: string;
  active_model_mape: number;
  next_month_forecast: number;
  pending_recommended_actions: number;
}

interface CorporateAction {
  id: number;
  action_type: string;
  status: string;
  trigger_condition: string;
  action_details: string;
  expected_savings_inr: number;
}

export default function Way2AgriDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [actions, setActions] = useState<CorporateAction[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'forecast' | 'actions' | 'simulator'>('forecast');

  // Simulator state
  const [procurementVolume, setProcurementVolume] = useState<number>(500); // Quintals
  const [targetMonthIndex, setTargetMonthIndex] = useState<number>(5); // Feb 2026 dip

  const DJANGO_API_BASE = process.env.NEXT_PUBLIC_DJANGO_API || 'http://localhost:8000/api';
  const FASTAPI_BASE = process.env.NEXT_PUBLIC_FASTAPI || 'http://localhost:8001';

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        // 1. Fetch live 12-month forward curve from FastAPI microservice
        const forecastRes = await fetch(`${FASTAPI_BASE}/predict/forecast-12m`);
        const forecastJson = await forecastRes.json();
        setForecast(forecastJson.forecast_points || []);

        // 2. Fetch dashboard aggregated stats from Django REST API
        const statsRes = await fetch(`${DJANGO_API_BASE}/dashboard/stats/`);
        const statsJson = await statsRes.json();
        setStats(statsJson);

        // 3. Fetch AI-recommended corporate actions from Django REST API
        const actionsRes = await fetch(`${DJANGO_API_BASE}/actions/`);
        const actionsJson = await actionsRes.json();
        setActions(actionsJson.results || actionsJson);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const selectedMonth = forecast[targetMonthIndex - 1] || forecast[0];
  const baselineCost = (stats?.latest_historical_price || 15750) * procurementVolume;
  const forecastedCost = (selectedMonth ? selectedMonth.predicted_price : 15750) * procurementVolume;
  const costDifference = baselineCost - forecastedCost;

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-emerald-600 border-t-transparent mx-auto"></div>
          <p className="mt-4 text-emerald-800 font-medium">Loading Way2Agribusiness AI Engine...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-6 md:p-10 font-sans">
      {/* Header */}
      <header className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between border-b pb-6 border-slate-200">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded">
            Way2Agribusiness AI/ML Forecasting Platform
          </span>
          <h1 className="text-3xl font-extrabold text-slate-900 mt-2">
            12-Month Commodity Price Intelligence
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Station: Yeshwanthpur Hub | Serving Karnataka Agribusiness Ecosystem
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex gap-3">
          <button
            onClick={() => setActiveTab('forecast')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              activeTab === 'forecast'
                ? 'bg-emerald-600 text-white shadow'
                : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
          >
            Forward Curve
          </button>
          <button
            onClick={() => setActiveTab('simulator')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              activeTab === 'simulator'
                ? 'bg-emerald-600 text-white shadow'
                : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
          >
            Hedging Simulator
          </button>
          <button
            onClick={() => setActiveTab('actions')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              activeTab === 'actions'
                ? 'bg-emerald-600 text-white shadow'
                : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
          >
            Strategic Actions ({actions.length})
          </button>
        </div>
      </header>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Latest Spot Price</p>
          <p className="text-2xl font-bold text-slate-900 mt-2">
            ₹{stats?.latest_historical_price?.toLocaleString() || '15,750'}
          </p>
          <span className="text-xs text-slate-400 mt-1 block">
            As of {stats?.latest_historical_date || '2025-09-01'}
          </span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Next Month Projected</p>
          <p className="text-2xl font-bold text-emerald-600 mt-2">
            ₹{forecast[0]?.predicted_price?.toLocaleString() || '15,771.46'}
          </p>
          <span className="text-xs text-emerald-700 mt-1 block">
            CI: ₹{forecast[0]?.lower_bound_95.toLocaleString()} - ₹{forecast[0]?.upper_bound_95.toLocaleString()}
          </span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Active AI Model</p>
          <p className="text-lg font-bold text-slate-800 mt-2 truncate">
            {stats?.active_model || 'Hybrid Ensemble'}
          </p>
          <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded inline-block mt-1">
            Validation MAPE: {stats?.active_model_mape?.toFixed(2)}%
          </span>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Projected Net Cost Savings</p>
          <p className="text-2xl font-bold text-blue-600 mt-2">₹49.67 Lakhs</p>
          <span className="text-xs text-slate-400 mt-1 block">Across 3 Strategic Interventions</span>
        </div>
      </div>

      {/* Main Content Areas */}
      {activeTab === 'forecast' && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-lg font-bold text-slate-900">12-Month Forward Price Curve (Oct 2025 - Sep 2026)</h2>
              <p className="text-xs text-slate-500">FastAPI Real-Time Hybrid Ensemble Predictions</p>
            </div>
            <span className="text-xs bg-slate-100 text-slate-700 px-3 py-1 rounded font-medium">
              Confidence Interval: 95%
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Horizon</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Target Month</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">Predicted Price (₹/Qtl)</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">95% Lower Bound</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">95% Upper Bound</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-700">Seasonal Trend</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {forecast.map((pt) => {
                  const isDip = pt.predicted_price < 15000;
                  const isPeak = pt.predicted_price > 16100;
                  return (
                    <tr key={pt.month_index} className="hover:bg-slate-50/80 transition">
                      <td className="px-4 py-3 font-medium text-slate-500">M+{pt.month_index}</td>
                      <td className="px-4 py-3 font-semibold text-slate-800">{pt.target_date}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-900">
                        ₹{pt.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-500">
                        ₹{pt.lower_bound_95.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-500">
                        ₹{pt.upper_bound_95.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                      </td>
                      <td className="px-4 py-3 text-center">
                        {isDip && (
                          <span className="px-2.5 py-1 text-xs rounded-full font-bold bg-amber-100 text-amber-800">
                            Seasonal Trough (Buy Signal)
                          </span>
                        )}
                        {isPeak && (
                          <span className="px-2.5 py-1 text-xs rounded-full font-bold bg-rose-100 text-rose-800">
                            Seasonal Peak (Sell/Hedge)
                          </span>
                        )}
                        {!isDip && !isPeak && (
                          <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-slate-100 text-slate-600">
                            Stable
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'simulator' && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <h2 className="text-lg font-bold text-slate-900 mb-2">Hedging & Procurement Simulator</h2>
          <p className="text-xs text-slate-500 mb-6">
            Evaluate cost savings by forward purchasing during predicted seasonal troughs vs spot buying at peak.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-1">
                  Procurement Volume: {procurementVolume} Quintals
                </label>
                <input
                  type="range"
                  min="100"
                  max="5000"
                  step="100"
                  value={procurementVolume}
                  onChange={(e) => setProcurementVolume(Number(e.target.value))}
                  className="w-full accent-emerald-600"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-1">
                  Target Purchase Month: M+{targetMonthIndex} ({selectedMonth?.target_date})
                </label>
                <select
                  value={targetMonthIndex}
                  onChange={(e) => setTargetMonthIndex(Number(e.target.value))}
                  className="w-full border border-slate-300 rounded-lg p-2.5 text-sm"
                >
                  {forecast.map((pt) => (
                    <option key={pt.month_index} value={pt.month_index}>
                      M+{pt.month_index} ({pt.target_date}) - Projected ₹{pt.predicted_price.toFixed(2)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Procurement Impact</h3>
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Current Spot Outlay:</span>
                  <span className="font-semibold">₹{baselineCost.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Forward Contract Outlay:</span>
                  <span className="font-semibold text-emerald-700">₹{forecastedCost.toLocaleString()}</span>
                </div>
                <div className="border-t pt-2 flex justify-between font-bold text-base">
                  <span>Estimated Cost Variance:</span>
                  <span className={costDifference >= 0 ? 'text-emerald-600' : 'text-rose-600'}>
                    {costDifference >= 0 ? '+' : ''}₹{costDifference.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'actions' && (
        <div className="space-y-4">
          {actions.map((act) => (
            <div key={act.id} className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded">
                  {act.action_type.replace('_', ' ')}
                </span>
                <span className="text-xs font-semibold px-2 py-1 rounded bg-blue-50 text-blue-700">
                  {act.status}
                </span>
              </div>
              <h3 className="font-bold text-slate-900 mt-3">{act.trigger_condition}</h3>
              <p className="text-sm text-slate-600 mt-1">{act.action_details}</p>
              <div className="mt-4 pt-3 border-t border-slate-100 flex justify-between items-center text-xs">
                <span className="text-slate-500">Estimated Value Creation:</span>
                <span className="font-bold text-emerald-600 text-sm">
                  ₹{(act.expected_savings_inr / 100000).toFixed(2)} Lakhs
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
