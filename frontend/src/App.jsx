import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const API = "http://127.0.0.1:5000";

function kFormat(n) {
  const x = Number(n);
  if (Number.isNaN(x)) return n;
  if (Math.abs(x) >= 1_000_000) return `${(x / 1_000_000).toFixed(1)}M`;
  if (Math.abs(x) >= 1_000) return `${(x / 1_000).toFixed(1)}k`;
  return `${Math.round(x)}`;
}

function Card({ title, subtitle, right, children }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.78)",
        border: "1px solid rgba(255,255,255,0.55)",
        borderRadius: 18,
        padding: 18,
        boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700 }}>{title}</div>
          {subtitle && (
            <div style={{ fontSize: 12, opacity: 0.75, marginTop: 2 }}>{subtitle}</div>
          )}
        </div>
        {right}
      </div>
      <div style={{ marginTop: 12 }}>{children}</div>
    </div>
  );
}

function StatPill({ label, value }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 14,
        background: "rgba(255,255,255,0.65)",
        border: "1px solid rgba(255,255,255,0.5)",
        minWidth: 150,
      }}
    >
      <div style={{ fontSize: 12, opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800, marginTop: 4 }}>{value}</div>
    </div>
  );
}

export default function App() {
  const [titles, setTitles] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState("");
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [forecastModel, setForecastModel] = useState("");
  const [horizon, setHorizon] = useState(6);
  const [kpis, setKpis] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const tRes = await fetch(`${API}/api/titles`);
        const tData = await tRes.json();
        const list = tData.titles || [];
        setTitles(list);
        if (list.length) setSelectedTitle(list[0]);

        const kRes = await fetch(`${API}/api/kpis`);
        const kData = await kRes.json();
        setKpis(kData);
      } catch {
        setError("Failed to load initial data (titles/KPIs). Make sure Flask is running.");
      }
    })();
  }, []);

  const loadHistory = async () => {
    setError("");
    setForecast([]);
    setForecastModel("");
    try {
      const r = await fetch(`${API}/api/history?title=${encodeURIComponent(selectedTitle)}`);
      const d = await r.json();
      if (!r.ok) return setError(d.error || "Failed to load history");
      setHistory(d.history || []);
    } catch {
      setError("Network error while loading history");
    }
  };

  const loadForecast = async () => {
    setError("");
    setHistory([]);
    try {
      const r = await fetch(
        `${API}/api/forecast?title=${encodeURIComponent(selectedTitle)}&horizon=${horizon}`
      );
      const d = await r.json();
      if (!r.ok) return setError(d.error || "Failed to load forecast");
      setForecast(d.forecast || []);
      setForecastModel(d.model || "");
    } catch {
      setError("Network error while loading forecast");
    }
  };

  const topOpenings = useMemo(
    () => (kpis?.top_jobs_openings || []).slice(0, 7).map(r => ({ ...r, job_title: String(r.job_title) })),
    [kpis]
  );
  const topSalary = useMemo(
    () => (kpis?.top_jobs_salary || []).slice(0, 7).map(r => ({ ...r, job_title: String(r.job_title) })),
    [kpis]
  );
  const growthLeaders = useMemo(
    () => (kpis?.salary_growth_top10 || []).slice(0, 7).map(r => ({ ...r, job_title: String(r.job_title) })),
    [kpis]
  );

  const historyStats = useMemo(() => {
    if (!history.length) return null;
    const vals = history.map(r => Number(r.avg_salary)).filter(v => !Number.isNaN(v));
    if (!vals.length) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const last = vals[vals.length - 1];
    return { min, max, last };
  }, [history]);

  const forecastStats = useMemo(() => {
    if (!forecast.length) return null;
    const vals = forecast.map(r => Number(r.predicted_salary)).filter(v => !Number.isNaN(v));
    if (!vals.length) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const last = vals[vals.length - 1];
    return { min, max, last };
  }, [forecast]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 700px at 20% 10%, rgba(164, 165, 255, 0.55), transparent 60%)," +
          "radial-gradient(1200px 700px at 80% 20%, rgba(255, 170, 220, 0.55), transparent 60%)," +
          "linear-gradient(180deg, #f6f7ff 0%, #f6f7fb 40%, #f7f8fb 100%)",
      }}
    >
      {/* Top Bar */}
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          padding: "16px 18px",
          borderBottom: "1px solid rgba(0,0,0,0.06)",
          backdropFilter: "blur(10px)",
          background: "rgba(255,255,255,0.65)",
        }}
      >
        <div
          style={{
            maxWidth: 1400,
            width: "100%",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: 12,
                background: "rgba(79,70,229,0.15)",
                display: "grid",
                placeItems: "center",
                fontWeight: 900,
              }}
            >
              JM
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 900 }}>Job Market Dashboard</div>
              <div style={{ fontSize: 12, opacity: 0.7 }}>
                KPIs + Salary Forecast (Prophet/Linear)
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <select
              value={selectedTitle}
              onChange={(e) => setSelectedTitle(e.target.value)}
              style={{
                padding: "10px 12px",
                borderRadius: 12,
                border: "1px solid rgba(0,0,0,0.12)",
                background: "white",
                minWidth: 260,
                fontWeight: 600,
              }}
            >
              {titles.map((t, i) => (
                <option key={i} value={t}>
                  {t}
                </option>
              ))}
            </select>

            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{
                padding: "10px 12px",
                borderRadius: 12,
                border: "1px solid rgba(0,0,0,0.12)",
                background: "white",
                fontWeight: 700,
              }}
            >
              <option value={3}>3 mo</option>
              <option value={6}>6 mo</option>
              <option value={12}>12 mo</option>
            </select>

            <button
              onClick={loadHistory}
              style={{
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(0,0,0,0.12)",
                background: "white",
                fontWeight: 800,
                cursor: "pointer",
              }}
            >
              Load History
            </button>

            <button
              onClick={loadForecast}
              style={{
                padding: "10px 14px",
                borderRadius: 12,
                border: "0",
                background: "rgba(79,70,229,0.92)",
                color: "white",
                fontWeight: 900,
                cursor: "pointer",
              }}
            >
              Predict
            </button>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div style={{ padding: "22px 18px" }}>
        <div style={{ maxWidth: 1400, width:"100%", margin: "0 auto" }}>
          {error && (
            <div
              style={{
                padding: 14,
                borderRadius: 14,
                background: "rgba(220,38,38,0.10)",
                border: "1px solid rgba(220,38,38,0.25)",
                color: "#b91c1c",
                fontWeight: 700,
                marginBottom: 16,
              }}
            >
              {error}
            </div>
          )}

          {/* KPI Row */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 18 }}>
            <StatPill
              label="Current selected title"
              value={selectedTitle ? selectedTitle.slice(0, 20) + (selectedTitle.length > 20 ? "…" : "") : "—"}
            />
            <StatPill
              label="History (last avg)"
              value={historyStats ? `€${kFormat(historyStats.last)}` : "—"}
            />
            <StatPill
              label="Forecast (last predicted)"
              value={forecastStats ? `€${kFormat(forecastStats.last)}` : "—"}
            />
            <StatPill
              label="Forecast model"
              value={forecastModel || "—"}
            />
          </div>

          {/* Grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(12, 1fr)",
              gap: 16,
            }}
          >
            {/* KPIs charts */}
            <div style={{ gridColumn: "span 6" }}>
              <Card title="Top Jobs by Openings" subtitle="Total postings across months">
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={topOpenings} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" tickFormatter={kFormat} />
                      <YAxis
                        type="category"
                        dataKey="job_title"
                        width={160}
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip />
                      <Bar dataKey="job_count" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <div style={{ gridColumn: "span 6" }}>
              <Card title="Top Jobs by Salary" subtitle="Mean salary across months (avg_salary)">
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={topSalary} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" tickFormatter={kFormat} />
                      <YAxis
                        type="category"
                        dataKey="job_title"
                        width={160}
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip />
                      <Bar dataKey="avg_salary" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <div style={{ gridColumn: "span 5" }}>
              <Card title="Salary Growth Leaders" subtitle="First vs last month % change">
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={growthLeaders} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
                      <YAxis
                        type="category"
                        dataKey="job_title"
                        width={150}
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, "Growth"]} />
                      <Bar dataKey="growth_pct" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            {/* Trend chart */}
            <div style={{ gridColumn: "span 7" }}>
              <Card
                title="Salary Trend"
                subtitle={
                  history.length
                    ? "History: avg_salary by month"
                    : forecast.length
                    ? `Forecast: predicted_salary by month (${horizon} months)`
                    : "Click Load History or Predict"
                }
                right={
                  <div style={{ fontSize: 12, opacity: 0.75, textAlign: "right" }}>
                    {forecastModel ? `Model: ${forecastModel}` : ""}
                  </div>
                }
              >
                <div style={{ width: "100%", height: 320 }}>
                  <ResponsiveContainer>
                    <LineChart data={history.length ? history : forecast}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                      <YAxis tickFormatter={kFormat} tick={{ fontSize: 12 }} />
                      <Tooltip formatter={(v) => [`€${kFormat(v)}`, "Salary"]} />
                      {history.length > 0 && <Line type="monotone" dataKey="avg_salary" strokeWidth={2} dot={false} />}
                      {forecast.length > 0 && <Line type="monotone" dataKey="predicted_salary" strokeWidth={2} dot={false} />}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          </div>

          <div style={{ marginTop: 14, fontSize: 12, opacity: 0.7 }}>
            Tip: Use the dropdown and horizon selector in the top bar. “Predict” calls the API with title + horizon.
          </div>
        </div>
      </div>
    </div>
  );
}
