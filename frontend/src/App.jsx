import { useEffect, useMemo, useState } from "react";
import "./App.css";

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

function fmtMoney(v) {
  const n = Number(v);
  if (Number.isNaN(n)) return v;
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function fmtCompact(v) {
  const n = Number(v);
  if (Number.isNaN(n)) return v;
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${Math.round(n)}`;
}
function truncateLabel(s, max = 22) {
  const str = String(s ?? "");
  return str.length > max ? str.slice(0, max) + "…" : str;
}

const TOOLTIP_STYLE = {
  background: "rgba(17, 24, 39, 0.95)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 12,
  color: "#E5E7EB",
  fontWeight: 700,
  fontSize: 12,
  padding: "10px 12px",
};

const TOOLTIP_LABEL_STYLE = {
  color: "#E5E7EB",
  fontWeight: 900,
  marginBottom: 6,
};

const TOOLTIP_ITEM_STYLE = {
  color: "#E5E7EB",
};

function StatCard({ label, value, sub }) {
  return (
    <div className="statCard">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
      {sub ? <div className="statSub">{sub}</div> : null}
    </div>
  );
}

function Panel({ title, subtitle, children, right }) {
  return (
    <div className="panel">
      <div className="panelHeader">
        <div>
          <div className="panelTitle">{title}</div>
          {subtitle ? <div className="panelSub">{subtitle}</div> : null}
        </div>
        {right ? <div className="panelRight">{right}</div> : null}
      </div>
      <div className="panelBody">{children}</div>
    </div>
  );
}

export default function App() {
  // Page routing (simple state router)
  const [page, setPage] = useState("kpis"); // default KPIs

  // Data
  const [titles, setTitles] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState("");
  const [kpis, setKpis] = useState(null);

  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [forecastModel, setForecastModel] = useState("");
  const [horizon, setHorizon] = useState(6);

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
        setError(
          "Failed to load titles/KPIs. Check Flask is running on 127.0.0.1:5000"
        );
      }
    })();
  }, []);

  const loadHistory = async () => {
    if (!selectedTitle) return;
    setError("");
    try {
      const r = await fetch(
        `${API}/api/history?title=${encodeURIComponent(selectedTitle)}`
      );
      const d = await r.json();
      if (!r.ok) return setError(d.error || "Failed to load history");
      setHistory(d.history || []);
    } catch {
      setError("Network error while loading history");
    }
  };

  const loadForecast = async () => {
    if (!selectedTitle) return;
    setError("");
    try {
      const r = await fetch(
        `${API}/api/forecast?title=${encodeURIComponent(
          selectedTitle
        )}&horizon=${horizon}`
      );
      const d = await r.json();
      if (!r.ok) return setError(d.error || "Failed to load forecast");
      setForecast(d.forecast || []);
      setForecastModel(d.model || "");
    } catch {
      setError("Network error while loading forecast");
    }
  };

  // KPIs
  const topOpenings = useMemo(
    () => (kpis?.top_jobs_openings || []).slice(0, 8),
    [kpis]
  );
  const topSalary = useMemo(
    () => (kpis?.top_jobs_salary || []).slice(0, 8),
    [kpis]
  );
  const growthLeaders = useMemo(
    () => (kpis?.salary_growth_top10 || []).slice(0, 8),
    [kpis]
  );

  // Stats (for predict page)
  const historyStats = useMemo(() => {
    if (!history.length) return null;
    const vals = history
      .map((r) => Number(r.avg_salary))
      .filter((v) => !Number.isNaN(v));
    if (!vals.length) return null;
    return {
      last: vals[vals.length - 1],
      min: Math.min(...vals),
      max: Math.max(...vals),
    };
  }, [history]);

  const forecastStats = useMemo(() => {
    if (!forecast.length) return null;
    const vals = forecast
      .map((r) => Number(r.predicted_salary))
      .filter((v) => !Number.isNaN(v));
    if (!vals.length) return null;
    return {
      last: vals[vals.length - 1],
      min: Math.min(...vals),
      max: Math.max(...vals),
    };
  }, [forecast]);

  // Merge history + forecast into one series for a single chart with two lines
  const mergedSeries = useMemo(() => {
    const map = new Map();

    (history || []).forEach((r) => {
      map.set(r.month, {
        month: r.month,
        avg_salary: Number(r.avg_salary),
      });
    });

    (forecast || []).forEach((r) => {
      const prev = map.get(r.month) || { month: r.month };
      map.set(r.month, {
        ...prev,
        predicted_salary: Number(r.predicted_salary),
      });
    });

    return Array.from(map.values()).sort((a, b) =>
      String(a.month).localeCompare(String(b.month))
    );
  }, [history, forecast]);

  return (
    <div className="appShell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brandIcon">JM</div>
          <div>
            <div className="brandTitle">Job Market</div>
            <div className="brandSub">Analytics Dashboard</div>
          </div>
        </div>

        <nav className="nav">
          <button
            className={`navItem ${page === "kpis" ? "navItemActive" : ""}`}
            onClick={() => setPage("kpis")}
            type="button"
          >
            KPIs
          </button>

          <button
            className={`navItem ${page === "predict" ? "navItemActive" : ""}`}
            onClick={() => setPage("predict")}
            type="button"
          >
            Predict
          </button>
        </nav>

        <div className="sidebarFooter">
          <div className="hint">
            KPIs shows market summary. Predict gives history + forecast for a job.
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {/* Top bar (changes per page) */}
        <div className="topbar">
          <div className="topbarLeft">
            <div className="pageTitle">{page === "kpis" ? "KPIs" : "Predict"}</div>
            <div className="pageSub">
              {page === "kpis"
                ? "Openings, salary leaders, and salary growth insights"
                : "Select job title + horizon, then load history / forecast"}
            </div>
          </div>

          {/* Controls only on Predict page */}
          {page === "predict" ? (
            <div className="topbarRight">
              <select
                className="control"
                value={selectedTitle}
                onChange={(e) => setSelectedTitle(e.target.value)}
              >
                {titles.map((t, i) => (
                  <option key={i} value={t}>
                    {t}
                  </option>
                ))}
              </select>

              <select
                className="control controlSmall"
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
              >
                <option value={3}>3 mo</option>
                <option value={6}>6 mo</option>
                <option value={12}>12 mo</option>
              </select>

              <button className="btn" onClick={loadHistory}>
                Load History
              </button>
              <button className="btn btnPrimary" onClick={loadForecast}>
                Predict
              </button>
            </div>
          ) : (
            <div className="topbarRight">
              {/* for possible updatess */}
            </div>
          )}
        </div>

        {error ? <div className="alert">{error}</div> : null}

        {/* PAGE: KPIs */}
        {page === "kpis" && (
          <section className="grid">
            <Panel title="Top Jobs by Openings" subtitle="Most postings (aggregated monthly)">
              <div className="chartWrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={topOpenings}
                    layout="vertical"
                    margin={{ left: 10, right: 18, top: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.12)" />
                    <XAxis type="number" tickFormatter={fmtCompact} tick={{ fill: "#E5E7EB" }} />
                    <YAxis
                      type="category"
                      dataKey="job_title"
                      width={190}
                      tick={{ fontSize: 12, fill: "#E5E7EB" }}
                      tickFormatter={(v) => truncateLabel(v, 24)}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      labelStyle={TOOLTIP_LABEL_STYLE}
                      itemStyle={TOOLTIP_ITEM_STYLE}
                    />
                    <Bar dataKey="job_count" fill="rgba(99,102,241,0.95)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel title="Top Jobs by Salary" subtitle="Highest average salaries (monthly mean)">
              <div className="chartWrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={topSalary}
                    layout="vertical"
                    margin={{ left: 10, right: 18, top: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.12)" />
                    <XAxis type="number" tickFormatter={fmtCompact} tick={{ fill: "#E5E7EB" }} />
                    <YAxis
                      type="category"
                      dataKey="job_title"
                      width={190}
                      tick={{ fontSize: 12, fill: "#E5E7EB" }}
                      tickFormatter={(v) => truncateLabel(v, 24)}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      labelStyle={TOOLTIP_LABEL_STYLE}
                      itemStyle={TOOLTIP_ITEM_STYLE}
                      formatter={(v) => [`€${fmtMoney(v)}`, "Avg salary"]}
                    />
                    <Bar dataKey="avg_salary" fill="rgba(34,197,94,0.90)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel title="Salary Growth Leaders" subtitle="Top % growth from first to last month">
              <div className="chartWrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={growthLeaders}
                    layout="vertical"
                    margin={{ left: 10, right: 18, top: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.12)" />
                    <XAxis
                      type="number"
                      tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                      tick={{ fill: "#E5E7EB" }}
                    />
                    <YAxis
                      type="category"
                      dataKey="job_title"
                      width={190}
                      tick={{ fontSize: 12, fill: "#E5E7EB" }}
                      tickFormatter={(v) => truncateLabel(v, 24)}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      labelStyle={TOOLTIP_LABEL_STYLE}
                      itemStyle={TOOLTIP_ITEM_STYLE}
                      formatter={(v) => [`${Number(v).toFixed(1)}%`, "Growth"]}
                    />
                    <Bar dataKey="growth_pct" fill="rgba(245,158,11,0.92)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </section>
        )}

        {/* PAGE: Predict */}
        {page === "predict" && (
          <>
            <section className="statsRow">
              <StatCard
                label="Selected title"
                value={selectedTitle ? truncateLabel(selectedTitle, 28) : "—"}
                sub="Used for history + forecast"
              />
              <StatCard
                label="History (last avg)"
                value={historyStats ? `€${fmtCompact(historyStats.last)}` : "—"}
                sub={
                  historyStats
                    ? `min €${fmtCompact(historyStats.min)} • max €${fmtCompact(historyStats.max)}`
                    : ""
                }
              />
              <StatCard
                label="Forecast (last predicted)"
                value={forecastStats ? `€${fmtCompact(forecastStats.last)}` : "—"}
                sub={
                  forecastStats
                    ? `min €${fmtCompact(forecastStats.min)} • max €${fmtCompact(forecastStats.max)}`
                    : ""
                }
              />
              <StatCard
                label="Forecast model"
                value={forecastModel || "—"}
                sub="Chosen dynamically per job"
              />
            </section>

            <section className="grid">
              <Panel
                title="Salary Trend (History + Forecast)"
                subtitle="Solid line = history, dashed line = forecast"
                right={forecastModel ? <span className="badge">{forecastModel}</span> : null}
              >
                <div className="chartWrapTall">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={mergedSeries} margin={{ left: 10, right: 18, top: 8, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.12)" />
                      <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#E5E7EB" }} />
                      <YAxis tickFormatter={fmtCompact} tick={{ fontSize: 12, fill: "#E5E7EB" }} />
                      <Tooltip
                        contentStyle={TOOLTIP_STYLE}
                        labelStyle={TOOLTIP_LABEL_STYLE}
                        itemStyle={TOOLTIP_ITEM_STYLE}
                        formatter={(v) => [`€${fmtMoney(v)}`, "Salary"]}
                      />

                      {/* History */}
                      <Line
                        type="monotone"
                        dataKey="avg_salary"
                        strokeWidth={2}
                        dot={false}
                        name="History"
                        stroke="rgba(99,102,241,0.95)"
                      />

                      {/* Forecast */}
                      <Line
                        type="monotone"
                        dataKey="predicted_salary"
                        strokeWidth={2}
                        dot={false}
                        strokeDasharray="6 4"
                        name="Forecast"
                        stroke="rgba(34,197,94,0.90)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
