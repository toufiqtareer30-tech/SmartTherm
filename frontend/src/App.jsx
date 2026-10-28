import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bell,
  Check,
  Fan,
  History,
  LayoutDashboard,
  Radio,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Square,
  Thermometer,
  TriangleAlert,
  Users,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const REFRESH_MS = 5000;
const CALIBRATION_COOLDOWN_MS = 120_000;
const DEFAULT_CPU_LIMIT = 90;
const DEFAULT_GPU_LIMIT = 85;
const CALIBRATION_ACTIVITY_KEY = "smarttherm.calibrationActivity";
const SEEN_CALIBRATIONS_KEY = "smarttherm.seenCalibrationRuns";
const RESET_LIMIT_KEY = "smarttherm.resetCalibratedLimit";
const NOTIFIED_ALERTS_KEY = "smarttherm.notifiedAlerts";
const CLIENT_WARNINGS_KEY = "smarttherm.clientWarnings";
const HISTORY_RANGES = [
  { id: "recent", label: "Recent", durationMs: 20 * 60_000 },
  { id: "hour", label: "1h", durationMs: 60 * 60_000 },
];

async function getJson(path) {
  const response = await fetch(path);
  if (response.status === 204) return null;
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return response.json();
}

function readStorageArray(key) {
  try {
    const stored = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch (error) {
    console.error(`Could not read local SmartTherm data (${key}).`, error);
    return [];
  }
}

function readResetLimit() {
  try {
    return localStorage.getItem(RESET_LIMIT_KEY);
  } catch (error) {
    console.error("Could not read the calibrated-limit setting.", error);
    return null;
  }
}

function persistStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Could not save local SmartTherm data (${key}).`, error);
  }
}

function formatTemp(value, digits = 1) {
  return value == null || !Number.isFinite(Number(value))
    ? "Unavailable"
    : `${Number(value).toFixed(digits)}°C`;
}

function formatTime(value) {
  if (!value) return "Not available";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "Not available"
    : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function forecastHorizonMinutes(prediction, forecast) {
  const minutes = Number(forecast?.horizonMinutes ?? prediction?.horizonMinutes);
  return Number.isInteger(minutes) && minutes > 0 ? minutes : null;
}

function forecastTargetTimeMs(prediction, forecast) {
  return new Date(forecast?.targetTime || prediction?.targetTime).getTime();
}

function forecastIsUsable(prediction, forecast, now = Date.now()) {
  if (forecast?.status !== "READY" || forecast.predictedC == null ||
      !Number.isFinite(Number(forecast.predictedC)) ||
      !forecast.model || !forecast.modelVersion) return false;
  const horizon = forecastHorizonMinutes(prediction, forecast);
  const generatedAt = new Date(forecast.generatedAt || prediction?.generatedAt).getTime();
  const targetAt = forecastTargetTimeMs(prediction, forecast);
  return horizon != null && Number.isFinite(generatedAt) && Number.isFinite(targetAt) &&
    targetAt > now && Math.abs(targetAt - generatedAt - horizon * 60_000) <= 30_000;
}

function statusFor(value, limit) {
  if (value == null || !Number.isFinite(Number(value))) return "unavailable";
  if (Number(value) >= limit + 5) return "critical";
  if (Number(value) >= limit) return "warning";
  return "normal";
}

function forecastStateLabel(prediction, forecast) {
  if (!prediction) return "Loading forecast";
  if (forecast?.status === "READY") {
    return forecastIsUsable(prediction, forecast) ? "Ready" : "Unavailable";
  }
  if (["WAITING_FOR_DATA", "STALE_DATA", "INSUFFICIENT_HISTORY"].includes(forecast?.status || prediction.status)) {
    return "Warming up";
  }
  return "Unavailable";
}

function forecastReason(prediction, forecast) {
  if (!prediction) return "Loading the local trained-model forecast.";
  if (forecast?.status === "READY" && forecastIsUsable(prediction, forecast)) {
    return `${forecast.model || prediction.model || "Trained model"} forecast ready.`;
  }
  if (forecast?.status === "READY" && forecastHorizonMinutes(prediction, forecast) == null) {
    return "Forecast horizon unavailable; waiting for a compatible model response.";
  }
  if (forecast?.status === "READY" && !forecast.modelVersion) {
    return "Model version unavailable; waiting for a compatible model response.";
  }
  if (forecast?.status === "READY" &&
      forecastTargetTimeMs(prediction, forecast) <= Date.now()) {
    return "Forecast expired; waiting for a fresh model result.";
  }
  if (forecast?.status === "READY") {
    return "Forecast target does not match the reported horizon.";
  }
  if (forecast?.model) return forecast.model;
  if (prediction.status === "WAITING_FOR_DATA") return "Waiting for sensor data.";
  if (prediction.status === "STALE_DATA") return "Waiting for fresh sensor data.";
  return prediction.model || "Forecast unavailable.";
}

function calibrationRunning(calibration) {
  return calibration?.calibrationState === "running";
}

function calibrationTerminal(calibration) {
  return ["completed", "no_throttle_detected", "aborted_safety", "cancelled"]
    .includes(calibration?.calibrationState);
}

function isCalibrationResult(calibration) {
  return calibrationTerminal(calibration) && Boolean(calibration?.calibrationRunId);
}

function makeCalibrationActivity(calibration) {
  return {
    id: `calibration-${calibration.calibrationRunId}`,
    type: "calibration",
    runId: calibration.calibrationRunId,
    state: calibration.calibrationState,
    createdAt: calibration.calibrationEndedAtUtc || calibration.calibrationStartedAtUtc,
    message: calibration.message || "Calibration result recorded.",
    peakCpuTempC: calibration.calibrationPeakCpuTempC ?? null,
    peakGpuTempC: calibration.calibrationPeakGpuTempC ?? null,
    throttleCpuTempC: calibration.calibrationThrottleCpuTempC ?? null,
    abortReason: calibration.calibrationAbortReason ?? null,
    abortTempC: calibration.calibrationAbortTempC ?? null,
    startedAt: calibration.calibrationStartedAtUtc ?? null,
    endedAt: calibration.calibrationEndedAtUtc ?? null,
    plannedDurationSeconds: calibration.calibrationPlannedDurationSeconds ?? null,
  };
}

function statusText(state) {
  if (state === "aborted_safety") return "Stopped by safety limit";
  if (state === "no_throttle_detected") return "No reliable throttle event detected";
  if (state === "cancelled") return "Calibration was cancelled";
  return "Stress test complete";
}

function SensorRow({ label, value, available, accent = "green" }) {
  return (
    <div className={`sensor-row ${available ? "" : "sensor-unavailable"}`}>
      <span className={`sensor-dot ${available ? accent : ""}`} />
      <span className="sensor-label">{label}</span>
      <span className="sensor-value">{available ? value : "Unavailable"}</span>
    </div>
  );
}

function App() {
  const [page, setPage] = useState(() => window.location.hash.slice(1) || "overview");
  const [status, setStatus] = useState(null);
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [clientWarnings, setClientWarnings] = useState(() => readStorageArray(CLIENT_WARNINGS_KEY));
  const [settings, setSettings] = useState(null);
  const [processUsage, setProcessUsage] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [calibrationActivity, setCalibrationActivity] = useState(() => readStorageArray(CALIBRATION_ACTIVITY_KEY));
  const [previewWarnings, setPreviewWarnings] = useState(false);
  const [resetLimitForRun, setResetLimitForRun] = useState(readResetLimit);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [endingPid, setEndingPid] = useState(null);
  const [permission, setPermission] = useState(
    "Notification" in window ? Notification.permission : "unsupported",
  );
  const [clock, setClock] = useState(Date.now());
  const [historyRange, setHistoryRange] = useState("recent");
  const [completedRun, setCompletedRun] = useState(null);
  const [forecastPointFocused, setForecastPointFocused] = useState(false);
  const calibrateButtonRef = useRef(null);
  const modalRef = useRef(null);
  const running = calibrationRunning(calibration);
  const cpuForecast = prediction?.forecasts?.find((item) => item.component === "CPU");
  const gpuForecast = prediction?.forecasts?.find((item) => item.component === "GPU");
  const cpuModelReady = forecastIsUsable(prediction, cpuForecast, clock);
  const gpuModelReady = forecastIsUsable(prediction, gpuForecast, clock);
  const forecastMinutes = forecastHorizonMinutes(prediction, cpuForecast);
  const forecastHorizonText = forecastMinutes == null ? "Forecast horizon unavailable" : `${forecastMinutes} min`;
  const outlookTitle = forecastMinutes == null ? "Thermal outlook" : `${forecastMinutes}-minute thermal outlook`;
  const historyOutlookTitle = forecastMinutes == null
    ? "Predictive outlook"
    : `Predictive outlook - in ${forecastMinutes} minutes`;
  const cpuWarningLimit = Number(settings?.cpuWarningC ?? DEFAULT_CPU_LIMIT);
  const gpuWarningLimit = Number(settings?.gpuWarningC ?? DEFAULT_GPU_LIMIT);
  const calibratedCpuLimit = calibration?.calibrationThrottleCpuTempC ??
    calibrationActivity.find((item) => item.throttleCpuTempC != null)?.throttleCpuTempC;
  const resetLimit = resetLimitForRun === (calibration?.calibrationRunId || "none");
  const effectiveCpuLimit = !resetLimit && calibratedCpuLimit != null
    ? Number(calibratedCpuLimit)
    : cpuWarningLimit;
  const effectiveGpuLimit = gpuWarningLimit;
  const currentCpu = current?.cpuTemperatureC ?? null;
  const currentGpu = current?.gpuTemperatureC ?? null;
  const cpuState = running ? "calibrating" : statusFor(currentCpu, effectiveCpuLimit);
  const calibrationEndMs = calibration?.calibrationEndedAtUtc
    ? new Date(calibration.calibrationEndedAtUtc).getTime()
    : null;
  const cooldownActive = calibrationEndMs != null &&
    clock < calibrationEndMs + CALIBRATION_COOLDOWN_MS;
  const warningsSuppressed = running || cooldownActive;
  const activeAlerts = warningsSuppressed || previewWarnings
    ? []
    : clientWarnings.filter((alert) => alert.active);
  const pageTitles = {
    overview: "Thermal",
    processes: "Processes",
    history: "Temperature history",
    alerts: "Alert center",
  };
  const updatedTime = formatTime(current?.timestamp);

  const recordCalibration = useCallback((nextCalibration) => {
    if (!isCalibrationResult(nextCalibration)) return;
    const event = makeCalibrationActivity(nextCalibration);
    setCalibrationActivity((previous) => {
      if (previous.some((item) => item.runId === event.runId)) return previous;
      const next = [event, ...previous].slice(0, 100);
      persistStorage(CALIBRATION_ACTIVITY_KEY, next);
      return next;
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextCurrent, nextHistory, nextPrediction, nextAlerts, nextSettings, nextProcesses, nextCalibration] =
        await Promise.all([
          getJson("/api/monitoring/status"),
          getJson("/api/temperature/current"),
          getJson("/api/temperature/history?limit=720"),
          getJson("/api/predictions/latest"),
          getJson("/api/alerts"),
          getJson("/api/settings"),
          getJson("/api/processes"),
          getJson("/api/calibration/status"),
        ]);
      setStatus(nextStatus);
      setCurrent(nextCurrent);
      setHistory(Array.isArray(nextHistory) ? nextHistory : []);
      setPrediction(nextPrediction);
      setAlerts(Array.isArray(nextAlerts) ? nextAlerts : []);
      setSettings(nextSettings);
      setProcessUsage(nextProcesses);
      setCalibration((previous) =>
        previous?.calibrationRunId === nextCalibration?.calibrationRunId &&
        calibrationTerminal(previous) && calibrationRunning(nextCalibration)
          ? previous
          : nextCalibration);
      setError("");
      recordCalibration(nextCalibration);

    } catch (requestError) {
      setError(requestError.message || "Could not connect to the SmartTherm API.");
    } finally {
      setLoading(false);
    }
  }, [recordCalibration]);

  const lastWarningEvaluation = useRef("");
  useEffect(() => {
    const signature = [
      current?.timestamp || "",
      prediction?.generatedAt || "",
      effectiveCpuLimit,
      effectiveGpuLimit,
      cpuModelReady,
      gpuModelReady,
      running,
      cooldownActive,
    ].join(":");
    if (!current || !settings || signature === lastWarningEvaluation.current) return;
    lastWarningEvaluation.current = signature;
    if (warningsSuppressed) return;

    const evaluations = [
      { component: "CPU", currentC: current.cpuTemperatureC, forecast: cpuForecast, limit: effectiveCpuLimit },
      { component: "GPU", currentC: current.gpuTemperatureC, forecast: gpuForecast, limit: effectiveGpuLimit },
    ];
    let updatedWarnings = [...clientWarnings];
    let changed = false;
    const createdWarnings = [];
    for (const evaluation of evaluations) {
      const forecastReady = forecastIsUsable(prediction, evaluation.forecast);
      const predictionAbove = forecastReady &&
        Number(evaluation.forecast.predictedC) >= evaluation.limit;
      const activeForComponent = updatedWarnings.filter(
        (warning) => warning.component === evaluation.component && warning.active,
      );
      const legacyWarnings = activeForComponent.filter(
        (warning) => warning.trigger !== "forecast",
      );
      if (legacyWarnings.length) {
        const resolvedAt = new Date().toISOString();
        updatedWarnings = updatedWarnings.map((warning) =>
          warning.component === evaluation.component && warning.active &&
          warning.trigger !== "forecast"
            ? { ...warning, active: false, resolvedAt }
            : warning);
        changed = true;
      }
      const activeForecastWarnings = activeForComponent.filter(
        (warning) => warning.trigger === "forecast",
      );

      if (activeForecastWarnings.length && forecastReady &&
          Number(evaluation.forecast.predictedC) <= evaluation.limit - 2) {
        const resolvedAt = new Date().toISOString();
        updatedWarnings = updatedWarnings.map((warning) =>
          warning.component === evaluation.component && warning.active &&
          warning.trigger === "forecast"
            ? { ...warning, active: false, resolvedAt }
            : warning);
        changed = true;
      }
      if (!predictionAbove || activeForecastWarnings.length) continue;

      const value = Number(evaluation.forecast.predictedC);
      const severity = value >= evaluation.limit + 5 ? "CRITICAL" : "WARNING";
      const target = evaluation.forecast?.targetTime || prediction?.targetTime;
      const createdWarning = {
        id: `local-${evaluation.component}-${Date.now()}`,
        component: evaluation.component,
        trigger: "forecast",
        severity,
        active: true,
        createdAt: new Date().toISOString(),
        currentC: evaluation.currentC ?? evaluation.forecast?.currentC ?? null,
        predictedC: forecastReady ? evaluation.forecast.predictedC : null,
        thresholdC: evaluation.limit,
        message: `${evaluation.component} predicted to reach ${value.toFixed(1)}°C by ${formatTime(target)}, at or above the throttle limit of ${evaluation.limit.toFixed(1)}°C.`,
        suggestedAction: "Reduce demanding workloads and check that cooling and airflow are operating normally.",
      };
      updatedWarnings.unshift(createdWarning);
      createdWarnings.push(createdWarning);
      changed = true;
    }

    if (changed) {
      updatedWarnings = updatedWarnings.slice(0, 100);
      persistStorage(CLIENT_WARNINGS_KEY, updatedWarnings);
      setClientWarnings(updatedWarnings);
    }
    if (createdWarnings.length && settings.notificationsEnabled &&
        "Notification" in window && Notification.permission === "granted") {
      try {
        const notified = new Set(readStorageArray(NOTIFIED_ALERTS_KEY));
        createdWarnings.forEach((warning) => {
          if (notified.has(warning.id)) return;
          new Notification(`SmartTherm ${warning.severity.toLowerCase()}`, {
            body: warning.message,
            tag: warning.component,
          });
          notified.add(warning.id);
        });
        persistStorage(NOTIFIED_ALERTS_KEY, [...notified]);
      } catch (notificationError) {
        setError(`Could not show a browser notification: ${notificationError.message}`);
      }
    }
  }, [
    clientWarnings,
    current,
    cpuForecast,
    gpuForecast,
    prediction?.generatedAt,
    prediction?.targetTime,
    prediction?.horizonMinutes,
    settings,
    effectiveCpuLimit,
    effectiveGpuLimit,
    running,
    cooldownActive,
    warningsSuppressed,
    cpuModelReady,
    gpuModelReady,
  ]);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, REFRESH_MS);
    const clockTimer = window.setInterval(() => setClock(Date.now()), 1000);
    const onHashChange = () => setPage(window.location.hash.slice(1) || "overview");
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.clearInterval(timer);
      window.clearInterval(clockTimer);
      window.removeEventListener("hashchange", onHashChange);
    };
  }, [refresh]);

  useEffect(() => {
    if (!running) return undefined;
    const pollCalibration = async () => {
      try {
        const nextCalibration = await getJson("/api/calibration/status");
        setCalibration((previous) =>
          previous?.calibrationRunId === nextCalibration?.calibrationRunId &&
          calibrationTerminal(previous) && calibrationRunning(nextCalibration)
            ? previous
            : nextCalibration);
        recordCalibration(nextCalibration);
      } catch (requestError) {
        setError(requestError.message || "Could not read calibration status.");
      }
    };
    const timer = window.setInterval(pollCalibration, 1000);
    return () => window.clearInterval(timer);
  }, [running, recordCalibration]);

  useEffect(() => {
    if (!isCalibrationResult(calibration)) return;
    const seen = new Set(readStorageArray(SEEN_CALIBRATIONS_KEY));
    if (!seen.has(calibration.calibrationRunId)) setCompletedRun(calibration);
  }, [calibration]);

  useEffect(() => {
    if (completedRun) modalRef.current?.querySelector("button")?.focus();
  }, [completedRun]);

  useEffect(() => {
    if (!completedRun) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeCompletion();
      }
      if (event.key !== "Tab" || !modalRef.current) return;
      const focusable = modalRef.current.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [completedRun]);

  function closeCompletion() {
    if (!completedRun) return;
    const seen = new Set(readStorageArray(SEEN_CALIBRATIONS_KEY));
    seen.add(completedRun.calibrationRunId);
    persistStorage(SEEN_CALIBRATIONS_KEY, [...seen]);
    setCompletedRun(null);
    if (page !== "overview") {
      window.location.hash = "overview";
      setPage("overview");
    }
    window.setTimeout(() => calibrateButtonRef.current?.focus(), 0);
  }

  async function startCalibration() {
    setCalibrationBusy(true);
    setError("");
    try {
      const response = await fetch("/api/calibration/start", { method: "POST" });
      if (!response.ok) throw new Error(`Could not start calibration (${response.status}).`);
      await response.json();
      await refresh();
    } catch (requestError) {
      setError(requestError.message || "Could not contact SmartTherm.");
    } finally {
      setCalibrationBusy(false);
    }
  }

  async function stopCalibration() {
    setCalibrationBusy(true);
    setError("");
    try {
      const response = await fetch("/api/calibration/stop", { method: "POST" });
      if (!response.ok) throw new Error(`Could not stop calibration (${response.status}).`);
      await response.json();
      await refresh();
    } catch (requestError) {
      setError(requestError.message || "Could not contact SmartTherm.");
    } finally {
      setCalibrationBusy(false);
    }
  }

  async function updateSettings(changes) {
    if (!settings) return;
    setSaving(true);
    try {
      const response = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...settings, ...changes }),
      });
      if (!response.ok) throw new Error(`Could not save settings (${response.status}).`);
      setSettings(await response.json());
      setError("");
    } catch (requestError) {
      setError(requestError.message || "Could not save settings.");
    } finally {
      setSaving(false);
    }
  }

  async function requestNotificationPermission() {
    if (!("Notification" in window)) {
      setPermission("unsupported");
      return;
    }
    setPermission(await Notification.requestPermission());
  }

  async function endApp(process) {
    const accepted = window.confirm(
      `End ${process.processName} (PID ${process.processId})? It may lose unsaved work. SmartTherm will ask Windows to close it.`,
    );
    if (!accepted) return;
    setEndingPid(process.processId);
    try {
      const response = await fetch(`/api/processes/${process.processId}/terminate`, { method: "POST" });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.message || `Could not end app (${response.status}).`);
      }
      await refresh();
    } catch (requestError) {
      setError(requestError.message || "Could not end the selected app.");
    } finally {
      setEndingPid(null);
    }
  }

  function resetCalibratedLimit() {
    try {
      const runId = calibration?.calibrationRunId || "none";
      localStorage.setItem(RESET_LIMIT_KEY, runId);
      setResetLimitForRun(runId);
    } catch (storageError) {
      setError(`Could not reset the calibrated limit locally: ${storageError.message}`);
    }
  }

  const allChartData = useMemo(() => history.map((reading) => ({
    timestamp: new Date(reading.timestamp).getTime(),
    time: new Date(reading.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    cpu: reading.cpuTemperatureC ?? undefined,
    gpu: reading.gpuTemperatureC ?? undefined,
  })).filter((reading) => Number.isFinite(reading.timestamp)), [history]);
  const range = HISTORY_RANGES.find((item) => item.id === historyRange) || HISTORY_RANGES[0];
  const outlookChartData = allChartData.filter((point) => point.timestamp >= clock - range.durationMs);
  const forecastTargetMs = forecastTargetTimeMs(prediction, cpuForecast);
  const forecastTemperature = Number(cpuForecast?.predictedC);
  const latestHistoryCpu = outlookChartData.reduce((latest, point) =>
    Number.isFinite(Number(point.cpu)) && (!latest || point.timestamp > latest.timestamp)
      ? point
      : latest, null);
  const latestActualPoint = Number.isFinite(Number(currentCpu)) &&
    Number.isFinite(new Date(current?.timestamp).getTime())
    ? { timestamp: new Date(current.timestamp).getTime(), cpu: Number(currentCpu) }
    : latestHistoryCpu;
  const cpuForecastDrawable = cpuModelReady &&
    Number.isFinite(forecastTemperature) &&
    Number.isFinite(forecastTargetMs) &&
    forecastTargetMs > clock &&
    Number.isFinite(latestActualPoint?.timestamp) &&
    Number.isFinite(Number(latestActualPoint?.cpu));
  const forecastModel = cpuForecast?.model || "Trained model unavailable";
  const forecastModelVersion = cpuForecast?.modelVersion || "Model version unavailable";
  const forecastTargetLabel = Number.isFinite(forecastTargetMs)
    ? new Date(forecastTargetMs).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "time unavailable";
  const forecastPointLabel = `${formatTemp(forecastTemperature)} at ${forecastTargetLabel}`;
  const forecastAccessibleLabel = `Forecast, ${forecastPointLabel}, within ${forecastHorizonText}, model ${forecastModel}, version ${forecastModelVersion}`;
  const forecastPointColor = Number.isFinite(forecastTemperature) && forecastTemperature >= effectiveCpuLimit
    ? statusFor(forecastTemperature, effectiveCpuLimit) === "critical" ? "var(--danger)" : "var(--warm)"
    : "var(--warm)";
  const visibleCpuTemperatures = outlookChartData
    .map((point) => Number(point.cpu))
    .filter(Number.isFinite);
  if (cpuForecastDrawable) visibleCpuTemperatures.push(forecastTemperature);
  visibleCpuTemperatures.push(effectiveCpuLimit);
  const outlookYDomain = [
    Math.min(...visibleCpuTemperatures) - 5,
    Math.max(...visibleCpuTemperatures) + 5,
  ];
  const outlookStartMs = outlookChartData.length
    ? Math.min(...outlookChartData.map((point) => point.timestamp))
    : latestActualPoint?.timestamp;
  const outlookEndMs = cpuForecastDrawable
    ? forecastTargetMs
    : outlookChartData.length
      ? Math.max(...outlookChartData.map((point) => point.timestamp))
      : latestActualPoint?.timestamp;
  const outlookXDomain = Number.isFinite(outlookStartMs) && Number.isFinite(outlookEndMs) &&
    outlookStartMs === outlookEndMs
    ? [outlookStartMs - 1, outlookEndMs + 1]
    : [outlookStartMs, outlookEndMs];

  useEffect(() => {
    if (!cpuForecastDrawable) setForecastPointFocused(false);
  }, [cpuForecastDrawable]);

  const rangeReadings = history.filter((reading) => {
    const time = new Date(reading.timestamp).getTime();
    return Number.isFinite(time) && time >= clock - range.durationMs;
  });
  const cpuReadings = rangeReadings.map((item) => item.cpuTemperatureC).filter(Number.isFinite);
  const latestCpuReading = [...rangeReadings].reverse().find((item) => item.cpuTemperatureC != null);
  const latestGpuReading = [...rangeReadings].reverse().find((item) => item.gpuTemperatureC != null);
  const historyChartData = rangeReadings.map((reading) => ({
    timestamp: new Date(reading.timestamp).getTime(),
    time: new Date(reading.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    cpu: reading.cpuTemperatureC ?? undefined,
    gpu: reading.gpuTemperatureC ?? undefined,
  }));
  const cpuPeak = cpuReadings.length ? Math.max(...cpuReadings) : null;
  const cpuLow = cpuReadings.length ? Math.min(...cpuReadings) : null;
  const processList = Array.isArray(processUsage?.processes) ? processUsage.processes : [];
  const calibrationStart = calibration?.calibrationStartedAtUtc
    ? new Date(calibration.calibrationStartedAtUtc).getTime()
    : null;
  const plannedSeconds = Number(calibration?.calibrationPlannedDurationSeconds);
  const elapsedSeconds = calibrationStart == null
    ? null
    : Math.max(0, Math.floor((clock - calibrationStart) / 1000));
  const remainingSeconds = calibrationStart == null || !Number.isFinite(plannedSeconds)
    ? null
    : Math.max(0, plannedSeconds - elapsedSeconds);
  const formattedRemaining = remainingSeconds == null
    ? "--:--"
    : `${String(Math.floor(remainingSeconds / 60)).padStart(2, "0")}:${String(remainingSeconds % 60).padStart(2, "0")}`;
  const formattedElapsed = elapsedSeconds == null || !Number.isFinite(plannedSeconds)
    ? "Elapsed time unavailable"
    : `Elapsed ${String(Math.floor(elapsedSeconds / 60)).padStart(2, "0")}:${String(elapsedSeconds % 60).padStart(2, "0")} of ${String(Math.floor(plannedSeconds / 60)).padStart(2, "0")}:${String(plannedSeconds % 60).padStart(2, "0")}`;

  const showCalibratedTag = !resetLimit && calibratedCpuLimit != null;
  const calibrationWindow = calibrationStart == null
    ? null
    : { start: calibrationStart, end: calibration?.calibrationEndedAtUtc ? new Date(calibration.calibrationEndedAtUtc).getTime() : null };
  const serverActivity = alerts.filter((alert) => {
    if (warningsSuppressed) return false;
    if (!calibrationWindow) return true;
    const created = new Date(alert.createdAt).getTime();
    const duringCalibration = created >= calibrationWindow.start &&
      (calibrationWindow.end == null || created <= calibrationWindow.end);
    const duplicate = clientWarnings.some((warning) =>
      warning.component === alert.component &&
      Math.abs(new Date(warning.createdAt).getTime() - created) < 60_000);
    return !duringCalibration && !duplicate;
  }).map((alert) => ({ ...alert, type: "alert", id: alert.id }));
  const clientActivity = warningsSuppressed ? [] : clientWarnings.map((alert) => ({ ...alert, type: "alert" }));
  const recentActivity = [...calibrationActivity, ...serverActivity, ...clientActivity]
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, 30);

  const gaugeState = running ? "calibrating" : statusFor(currentCpu, effectiveCpuLimit);
  const cpuOutlookValue = Math.max(
    currentCpu != null && Number.isFinite(Number(currentCpu)) ? Number(currentCpu) : Number.NEGATIVE_INFINITY,
    cpuModelReady ? Number(cpuForecast.predictedC) : Number.NEGATIVE_INFINITY,
  );
  const gpuOutlookValue = Math.max(
    currentGpu != null && Number.isFinite(Number(currentGpu)) ? Number(currentGpu) : Number.NEGATIVE_INFINITY,
    gpuModelReady ? Number(gpuForecast.predictedC) : Number.NEGATIVE_INFINITY,
  );
  const cpuPredictionState = running ? "calibrating" : statusFor(
    Number.isFinite(cpuOutlookValue) ? cpuOutlookValue : null,
    effectiveCpuLimit,
  );
  const gpuPredictionState = running ? "calibrating" : statusFor(
    Number.isFinite(gpuOutlookValue) ? gpuOutlookValue : null,
    effectiveGpuLimit,
  );
  const systemRiskState = running ? "calibrating"
    : [cpuPredictionState, gpuPredictionState].includes("critical") ? "critical"
      : [cpuPredictionState, gpuPredictionState].includes("warning") ? "warning"
        : [cpuPredictionState, gpuPredictionState].every((item) => item === "unavailable")
          ? "unavailable"
          : "normal";
  const readyHistory = historyChartData.length > 1;
  const calibrateLabel = running ? "Stop test" : "Start CPU throttle calibration";
  const currentPageTitle = pageTitles[page] || pageTitles.overview;
  const cpuPercent = currentCpu == null ? null : Math.round((Number(currentCpu) / effectiveCpuLimit) * 100);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="SmartTherm overview">
          <span className="brand-mark"><Thermometer size={19} /></span>
          <span>smarttherm</span>
        </a>
        <nav aria-label="Main navigation">
          <div className="nav-label">MONITORING</div>
          <a className={`nav-item ${page === "overview" ? "active" : ""}`} href="#overview" aria-current={page === "overview" ? "page" : undefined}>
            <LayoutDashboard size={15} />Overview
          </a>
          <a className={`nav-item ${page === "processes" ? "active" : ""}`} href="#processes" aria-current={page === "processes" ? "page" : undefined}>
            <Users size={15} />Processes
          </a>
          <div className="nav-label">ANALYTICS</div>
          <a className={`nav-item ${page === "history" ? "active" : ""}`} href="#history" aria-current={page === "history" ? "page" : undefined}>
            <History size={15} />Temperature history
          </a>
          <a className={`nav-item ${page === "alerts" ? "active" : ""}`} href="#alerts" aria-current={page === "alerts" ? "page" : undefined}>
            <Bell size={15} />Alert center
            {activeAlerts.length > 0 && <span className="nav-count">{activeAlerts.length}</span>}
          </a>
        </nav>
        <div className="sidebar-bottom">
          <ShieldCheck size={15} />
          <span>Private by design - data stays on this PC</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">SmartTherm <span>/</span> <b>{currentPageTitle}</b></div>
          <div className="topbar-right">
            <span className={`connection ${status?.active ? "connected" : ""}`}>
              <i />{status?.active ? "Monitoring active" : "Collector unavailable"}
            </span>
            <button className="icon-button" onClick={refresh} aria-label="Refresh data">
              <RefreshCw size={16} />
            </button>
          </div>
        </header>

        <div className="dashboard">
          <header className="page-heading">
            <div>
              <h1>{currentPageTitle}</h1>
              <p>{page === "overview" ? `Updated ${updatedTime}` : "Local thermal telemetry and model output"}</p>
            </div>
            <span className={`monitoring-badge ${status?.active ? "online" : "offline"}`}>
              <i />{status?.active ? "Monitoring active" : "Monitoring inactive"}
            </span>
          </header>

          {error && <div className="error-banner" role="alert">
            <TriangleAlert size={17} /><span>{error}</span><button onClick={refresh}>Retry</button>
          </div>}

          {page === "overview" && <>
            <div className="dashboard-actions">
              <button ref={calibrateButtonRef}
                className={`button primary ${running ? "neutral" : ""}`}
                onClick={running ? stopCalibration : startCalibration}
                disabled={calibrationBusy}>
                {running ? <Square size={14} /> : <Activity size={15} />}
                {calibrationBusy ? "Updating calibration…" : calibrateLabel}
              </button>
              <button className="button secondary" onClick={() => setPreviewWarnings((value) => !value)}
                disabled={running} aria-pressed={previewWarnings}>
                <TriangleAlert size={15} />{previewWarnings ? "Close warning preview" : "Preview warnings"}
              </button>
            </div>
            {calibration && calibration.calibrationState && calibration.calibrationState !== "idle" &&
              <div className={`notice-strip ${running ? "calibrating" : ""}`} role="status">
                <Thermometer size={15} />
                <span>{running ? "Calibration is running. Thermal warnings and notifications are paused." : calibration.message}</span>
                {running && <b className="neutral-pill">Calibrating</b>}
              </div>}
            {running && <section className="panel calibration-run-panel">
              <div className="run-timer">
                <span>TIME REMAINING</span>
                <strong>{formattedRemaining}</strong>
                <small>{formattedElapsed}</small>
              </div>
              <div className="run-progress">
                <div className="run-progress-head">
                  <b>CPU throttle calibration</b>
                  <span>{remainingSeconds == null || !plannedSeconds ? "Timer unavailable" : `${Math.min(100, Math.round((elapsedSeconds / plannedSeconds) * 100))}%`}</span>
                </div>
                <progress max={plannedSeconds || 1} value={Math.min(elapsedSeconds || 0, plannedSeconds || 1)} />
                <div className="run-peaks">
                  <span>CPU now <b>{formatTemp(currentCpu)}</b></span>
                  <span>GPU now <b>{formatTemp(currentGpu)}</b></span>
                  <span>Peak CPU <b>{formatTemp(calibration?.calibrationPeakCpuTempC)}</b></span>
                  <span>Peak GPU <b>{formatTemp(calibration?.calibrationPeakGpuTempC)}</b></span>
                </div>
              </div>
            </section>}

            {previewWarnings && <section className="preview-banner" role="status">
              <AlertTriangle size={17} />
              <div><b>Warning preview — sample values only</b>
                <span>CPU predicted to reach {formatTemp(effectiveCpuLimit + 1)} within {forecastHorizonText} (by {formatTime(cpuForecast?.targetTime || prediction?.targetTime)}), at or above the throttle limit of {effectiveCpuLimit}°C.</span>
              </div>
              <button className="icon-button" aria-label="Close warning preview" onClick={() => setPreviewWarnings(false)}>×</button>
            </section>}
            {!warningsSuppressed && !previewWarnings && activeAlerts.length > 0 &&
              <section className={`alert-banner ${activeAlerts[0].severity.toLowerCase()}`} aria-live="polite">
                <AlertTriangle size={18} />
                <div><b>{activeAlerts[0].message}</b><span>{activeAlerts[0].suggestedAction}</span></div>
              </section>}
            {warningsSuppressed && cooldownActive && !running &&
              <div className="cooldown-note" role="status">Warnings resume after a short cooldown.</div>}

            <div className="overview-grid">
              <section className={`panel gauge-card ${running ? "calibrating" : ""}`}>
                <div className="panel-heading">
                  <div><h2>Thermal status</h2><p>Current temperature against effective CPU limit</p></div>
                  <span className={`status-pill ${systemRiskState}`}>{running ? "Calibrating" : systemRiskState}</span>
                </div>
                <div className="gauge-layout">
                  <svg className="temperature-gauge" viewBox="0 0 150 150" role="img"
                    aria-label={`CPU gauge ${cpuPercent == null ? "unavailable" : `${cpuPercent} percent of effective limit`}`}>
                    <defs><linearGradient id="gaugeGradient" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="var(--cpu-deep)" /><stop offset="100%" stopColor="var(--warm)" />
                    </linearGradient></defs>
                    <circle className="gauge-track" cx="75" cy="75" r="58" />
                    <circle className="gauge-progress" cx="75" cy="75" r="58"
                      strokeDasharray={`${2 * Math.PI * 58 * (cpuPercent == null ? 0 : Math.min(1, Math.max(0, cpuPercent / 100)))} ${2 * Math.PI * 58}`}
                      transform="rotate(-90 75 75)" />
                    <text className="gauge-percent" x="75" y="71" textAnchor="middle">{cpuPercent == null ? "--" : `${cpuPercent}%`}</text>
                    <text className="gauge-caption" x="75" y="91" textAnchor="middle">{running ? "Calibrating" : gaugeState}</text>
                  </svg>
                  <div className="gauge-readings">
                    <div className="gauge-reading"><span><i className="dot cpu-dot" />CPU temperature</span><b>{formatTemp(currentCpu)}</b></div>
                    <div className="gauge-reading"><span><i className={`dot ${running ? "cpu-dot" : "gpu-dot"}`} />GPU temperature</span><b className={running ? "forecast-temp" : "gpu-text"}>{formatTemp(currentGpu)}</b></div>
                    <div className="gauge-reading"><span><i className="dot forecast-dot" />CPU forecast · {forecastHorizonText}</span><b className={running ? "neutral-temp" : "forecast-temp"}>{cpuModelReady ? formatTemp(cpuForecast.predictedC) : "Unavailable"}</b></div>
                    {!cpuModelReady && <small className="forecast-detail">{forecastReason(prediction, cpuForecast)}</small>}
                    <div className="gauge-limits">
                      <span>CPU limit <b>{effectiveCpuLimit.toFixed(1)}°C</b>{showCalibratedTag && <em>Calibrated</em>}</span>
                      <span>GPU limit <b>{effectiveGpuLimit.toFixed(1)}°C</b></span>
                    </div>
                  </div>
                </div>
                <div className="gauge-status-line"><i className={`sensor-dot ${running ? "orange" : gaugeState}`} />
                  {running ? "Thermal warnings paused during calibration" : `CPU status: ${gaugeState}`}
                </div>
              </section>

              <section className={`panel outlook-card ${previewWarnings ? "warning" : warningsSuppressed ? "calibrating" : cpuPredictionState}`}>
                <div className="panel-heading">
                  <div><h2>{outlookTitle}</h2><p>{cpuModelReady ? `${cpuForecast.model} · ${cpuForecast.modelVersion}` : prediction?.model || "Forecast model unavailable"}</p></div>
                  <div className="outlook-controls">
                    <span className="history-range">{forecastStateLabel(prediction, cpuForecast)}</span>
                    <div className="segmented-control" role="group" aria-label="Outlook history range">
                      {HISTORY_RANGES.map((item) => <button key={item.id} type="button"
                        className={historyRange === item.id ? "selected" : ""}
                        aria-pressed={historyRange === item.id}
                        onClick={() => setHistoryRange(item.id)}>{item.label}</button>)}
                    </div>
                  </div>
                </div>
                {allChartData.length > 1 || cpuForecastDrawable ? <div className="outlook-chart" role="group"
                  aria-label={cpuForecastDrawable
                    ? `CPU temperature history. ${forecastAccessibleLabel}.`
                    : "CPU temperature history"}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={outlookChartData} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                      <defs><linearGradient id="cpuAreaFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--cpu)" stopOpacity={0.28} />
                        <stop offset="100%" stopColor="var(--cpu)" stopOpacity={0} />
                      </linearGradient></defs>
                      <CartesianGrid stroke="var(--border-card)" vertical={false} />
                      <XAxis dataKey="timestamp" type="number" scale="time" domain={outlookXDomain}
                        tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={30} />
                      <YAxis domain={outlookYDomain} tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                        axisLine={false} tickLine={false} tickFormatter={(value) => `${value}°`} />
                      <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border-card)", borderRadius: "var(--radius-input)" }}
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value) => [`${Number(value).toFixed(1)}°C`, "CPU"]} />
                      <ReferenceLine y={effectiveCpuLimit} stroke="var(--warm)" strokeDasharray="5 5" />
                      <Area type="monotone" dataKey="cpu" name="CPU actual" stroke="var(--cpu)" strokeWidth={2}
                        fill="url(#cpuAreaFill)" connectNulls />
                      {cpuForecastDrawable && <>
                        <ReferenceLine segment={[
                          { x: latestActualPoint.timestamp, y: Number(latestActualPoint.cpu) },
                          { x: forecastTargetMs, y: forecastTemperature },
                        ]} stroke="var(--warm)" strokeWidth={2} strokeDasharray="6 5" ifOverflow="extendDomain" />
                        <ReferenceDot x={forecastTargetMs} y={forecastTemperature} r={5} fill={forecastPointColor}
                          stroke="var(--card)" strokeWidth={2} ifOverflow="extendDomain"
                          shape={({ cx, cy }) => <g className="forecast-chart-point" tabIndex={0} role="img"
                            aria-label={forecastAccessibleLabel}
                            onFocus={() => setForecastPointFocused(true)}
                            onBlur={() => setForecastPointFocused(false)}>
                            <title>{forecastAccessibleLabel}</title>
                            <circle cx={cx} cy={cy} r={5} fill={forecastPointColor} stroke="var(--card)" strokeWidth={2} />
                            <text x={cx - 9} y={cy - 10} textAnchor="end" fill={forecastPointColor}
                              fontSize={10} fontWeight={600}>{forecastPointLabel}</text>
                          </g>} />
                      </>}
                    </AreaChart>
                  </ResponsiveContainer>
                  {cpuForecastDrawable && forecastPointFocused && <div className="forecast-point-tooltip" role="tooltip">
                    <b>Forecast · {forecastTargetLabel}</b>
                    <span>{formatTemp(forecastTemperature)} · {forecastModel} · {forecastModelVersion}</span>
                  </div>}
                </div> : <div className="chart-empty">
                  <Radio size={18} /><span>{cpuModelReady ? "Sensor history unavailable" : "Forecast unavailable"}</span>
                  <small>{cpuForecast?.status || prediction?.status || "Waiting for model status"}</small>
                </div>}
                <div className="chart-legend"><span><i className="cpu-dot" />CPU actual</span>
                  {cpuForecastDrawable && <span><i className="forecast-dot" />CPU forecast</span>}
                  <span className="legend-limit"><i />Effective CPU limit</span></div>
                {!warningsSuppressed && cpuModelReady && Number(cpuForecast.predictedC) >= effectiveCpuLimit &&
                  <p className="forecast-warning">CPU forecast is at or above the effective limit.</p>}
                {!warningsSuppressed && !cpuModelReady && currentCpu != null && Number(currentCpu) >= effectiveCpuLimit &&
                  <p className="forecast-warning">Current CPU temperature is at or above the effective limit.</p>}
                {!cpuModelReady && <p className="forecast-detail">{forecastReason(prediction, cpuForecast)}</p>}
                {running && <p className="neutral-copy">Warnings are paused while the stress test runs.</p>}
              </section>
            </div>

            <div className="overview-grid lower-overview-grid">
              <section className="panel sensors-card">
                <div className="panel-heading"><div><h2>Sensors</h2><p>Live hardware readings</p></div>
                  <span className="history-range">{status?.active ? "Live" : "Unavailable"}</span></div>
                <div className="sensor-list">
                  <SensorRow label="CPU temperature" value={formatTemp(currentCpu)} available={currentCpu != null}
                    accent={running ? "orange" : cpuState} />
                  <SensorRow label="GPU temperature" value={formatTemp(currentGpu)} available={currentGpu != null}
                    accent={running ? "orange" : "purple"} />
                  <SensorRow label="CPU utilization" value={current?.cpuUtilizationPercent == null ? null : `${Number(current.cpuUtilizationPercent).toFixed(0)}%`} available={current?.cpuUtilizationPercent != null} />
                  <SensorRow label="CPU clock speed" value={current?.cpuClockMhz == null ? null : `${(Number(current.cpuClockMhz) / 1000).toFixed(2)} GHz`} available={current?.cpuClockMhz != null} accent="blue" />
                  <SensorRow label="CPU power" value={current?.cpuPowerW == null ? null : `${Number(current.cpuPowerW).toFixed(1)} W`} available={current?.cpuPowerW != null} accent="orange" />
                  <SensorRow label="CPU fan speed" value={current?.cpuFanSpeedRpm == null ? null : `${Math.round(current.cpuFanSpeedRpm)} RPM`} available={current?.cpuFanSpeedRpm != null} accent="purple" />
                  <SensorRow label="RAM usage" value={current?.memoryUtilizationPercent == null ? null : `${Number(current.memoryUtilizationPercent).toFixed(0)}%`} available={current?.memoryUtilizationPercent != null} accent="blue" />
                </div>
              </section>
              <section className="panel consumers-card">
                <div className="panel-heading"><div><h2>Top CPU consumers</h2><p>Highest process CPU usage</p></div>
                  <Users size={16} /></div>
                {processList.length ? <div className="consumer-list">
                  {processList.slice(0, 5).map((process) => {
                    const value = Number(process.cpuUsagePercent);
                    return <div className="consumer-row" key={`${process.processId}-${process.processName}`}>
                      <span title={process.processName}>{process.processName}</span>
                      <i><b style={{ width: `${Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0}%` }} /></i>
                      <strong>{Number.isFinite(value) ? `${value.toFixed(1)}%` : "Unavailable"}</strong>
                    </div>;
                  })}
                </div> : <div className="inline-empty">{processUsage?.message || "No process data available."}</div>}
              </section>
            </div>
          </>}

          {page === "processes" && <section className="panel process-panel">
            <div className="panel-heading"><div><h2>Running apps and processes</h2>
              <p>CPU share, memory use, and foreground status</p></div>
              <span className="history-range">{processUsage?.available ? "Refreshes every 5 seconds" : "Waiting for collector"}</span></div>
            <p className="process-note">The list refreshes about every five seconds. Ending a process may discard unsaved work.</p>
            {processList.length ? <div className="process-list">
              {processList.map((process) => {
                const cpu = Number(process.cpuUsagePercent);
                return <div className="process-row" key={process.processId}>
                  <div className="process-title"><b>{process.processName}</b><span>PID {process.processId}{process.foreground ? " · In use" : ""}</span></div>
                  <div className="process-meter" aria-label={`CPU usage ${cpu.toFixed(1)} percent`}><i style={{ width: `${Math.min(100, cpu)}%` }} /></div>
                  <b className="process-cpu">{cpu.toFixed(1)}%</b>
                  <span className="process-memory">{Number(process.memoryMb).toFixed(0)} MB RAM</span>
                  <button className="button secondary end-app-button" disabled={endingPid === process.processId}
                    onClick={() => endApp(process)}>{endingPid === process.processId ? "Closing…" : "End app"}</button>
                </div>;
              })}
            </div> : <div className="inline-empty"><Activity size={17} />{processUsage?.message || "Waiting for process CPU samples."}</div>}
          </section>}

          {page === "history" && <>
            <div className="history-controls">
              <div><h2>Recent readings</h2><p>Available history is limited to the collector’s latest 720 samples.</p></div>
              <div className="segmented-control" role="group" aria-label="Temperature history range">
                {HISTORY_RANGES.map((item) => <button key={item.id} type="button"
                  className={historyRange === item.id ? "selected" : ""}
                  aria-pressed={historyRange === item.id}
                  onClick={() => setHistoryRange(item.id)}>{item.label}</button>)}
              </div>
            </div>
            <section className="panel history-panel">
              <div className="history-stats">
                <div><span>CPU peak</span><b>{cpuPeak == null ? "Unavailable" : formatTemp(cpuPeak)}</b></div>
                <div><span>CPU low</span><b>{cpuLow == null ? "Unavailable" : formatTemp(cpuLow)}</b></div>
                <div><span>Latest CPU</span><b>{formatTemp(latestCpuReading?.cpuTemperatureC)}</b></div>
                <div><span>Latest GPU</span><b>{formatTemp(latestGpuReading?.gpuTemperatureC)}</b></div>
              </div>
              {readyHistory ? <div className="history-chart" role="img" aria-label="CPU and GPU temperatures over the selected recent history range">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={historyChartData} margin={{ top: 15, right: 18, left: -15, bottom: 5 }}>
                    <defs><linearGradient id="historyCpuFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--cpu)" stopOpacity={0.22} /><stop offset="100%" stopColor="var(--cpu)" stopOpacity={0} />
                    </linearGradient></defs>
                    <CartesianGrid stroke="var(--border-card)" vertical={false} />
                    <XAxis dataKey="timestamp" type="number" scale="time" domain={["dataMin", "dataMax"]}
                      tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={35} />
                    <YAxis domain={["dataMin - 5", "dataMax + 5"]} tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                      axisLine={false} tickLine={false} tickFormatter={(value) => `${value}°`} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border-card)", borderRadius: "var(--radius-input)" }}
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                      formatter={(value, name) => [`${Number(value).toFixed(1)}°C`, name]} />
                    <ReferenceLine y={effectiveCpuLimit} stroke="var(--warm)" strokeDasharray="5 5" />
                    <Area type="monotone" dataKey="cpu" name="CPU" stroke="var(--cpu)" strokeWidth={2}
                      fill="url(#historyCpuFill)" connectNulls />
                    <Line type="monotone" dataKey="gpu" name="GPU" stroke="var(--gpu)" strokeWidth={2} connectNulls dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div> : <div className="chart-empty"><History size={18} /><span>Not enough temperature history for this range.</span></div>}
              <div className="chart-legend"><span><i className="cpu-dot" />CPU</span><span><i className="gpu-dot" />GPU</span><span className="legend-limit"><i />Effective CPU limit</span></div>
            </section>
            <div className="history-bottom-grid">
              <section className="panel">
                <div className="panel-heading"><div><h2>{historyOutlookTitle}</h2><p>Latest current and predicted temperatures</p></div></div>
                <div className="history-forecast-row"><span>CPU</span><b>{formatTemp(cpuForecast?.currentC)}</b><i>→</i><b>{cpuModelReady ? formatTemp(cpuForecast.predictedC) : "Unavailable"}</b></div>
                <div className="history-forecast-row"><span>GPU</span><b>{formatTemp(gpuForecast?.currentC)}</b><i>→</i><b>{gpuModelReady ? formatTemp(gpuForecast.predictedC) : "Unavailable"}</b></div>
                {!cpuModelReady && <p className="forecast-detail">CPU: {forecastReason(prediction, cpuForecast)}</p>}
                {!gpuModelReady && <p className="forecast-detail">GPU: {forecastReason(prediction, gpuForecast)}</p>}
              </section>
              <section className="panel model-card">
                <div className="panel-heading"><div><h2>Model</h2><p>{cpuModelReady ? `${cpuForecast.model} · ${cpuForecast.modelVersion}` : prediction?.model || "Model unavailable"}</p></div></div>
                {(cpuModelReady || gpuModelReady) && <>
                  <div className="target-time"><span>Forecast horizon</span><b>{forecastMinutes} minutes</b></div>
                  <div className="target-time"><span>Prediction generated</span><b>{formatTime(cpuForecast?.generatedAt || prediction?.generatedAt)}</b></div>
                  <div className="target-time"><span>Forecast target</span><b>{formatTime(cpuForecast?.targetTime || prediction?.targetTime)}</b></div>
                </>}
                <div className="model-state">
                  CPU {forecastStateLabel(prediction, cpuForecast).toLowerCase()} · GPU {forecastStateLabel(prediction, gpuForecast).toLowerCase()}
                </div>
                {(!cpuModelReady || !gpuModelReady) && <p className="forecast-detail">
                  {!cpuModelReady && `CPU: ${forecastReason(prediction, cpuForecast)}`}
                  {!cpuModelReady && !gpuModelReady && " · "}
                  {!gpuModelReady && `GPU: ${forecastReason(prediction, gpuForecast)}`}
                </p>}
              </section>
            </div>
          </>}

          {page === "alerts" && <>
            <section className="panel settings-panel">
              <div className="panel-heading"><div><h2>Warning settings</h2><p>Fallback thresholds and browser notification preferences</p></div>
                <Settings2 size={17} /></div>
              {settings && <div className="settings-layout">
                <label className="threshold-field"><span>CPU threshold</span><div className="number-input">
                  <input type="number" min="50" max="110" step="1" value={settings.cpuWarningC}
                    onChange={(event) => setSettings({ ...settings, cpuWarningC: Number(event.target.value) })}
                    onBlur={() => updateSettings({ cpuWarningC: settings.cpuWarningC })} /><small>°C</small>
                </div></label>
                <label className="threshold-field"><span>GPU threshold</span><div className="number-input">
                  <input type="number" min="50" max="110" step="1" value={settings.gpuWarningC}
                    onChange={(event) => setSettings({ ...settings, gpuWarningC: Number(event.target.value) })}
                    onBlur={() => updateSettings({ gpuWarningC: settings.gpuWarningC })} /><small>°C</small>
                </div></label>
                <div className="calibrated-limit">
                  <div><span>Effective CPU limit</span><b>{effectiveCpuLimit.toFixed(1)}°C {showCalibratedTag && <em>Calibrated</em>}</b></div>
                  {showCalibratedTag && <button className="text-button" onClick={resetCalibratedLimit}>Reset to default</button>}
                </div>
                <div className="notification-setting">
                  <div><b>Browser notifications</b><span>In-app alerts always remain available</span></div>
                  <button className={`toggle ${settings.notificationsEnabled ? "on" : ""}`}
                    role="switch" aria-checked={settings.notificationsEnabled}
                    aria-label="Toggle browser notifications"
                    onClick={() => updateSettings({ notificationsEnabled: !settings.notificationsEnabled })}><i /></button>
                </div>
                {settings.notificationsEnabled && permission !== "granted" && <button className="button secondary permission-button"
                  onClick={requestNotificationPermission}><Bell size={14} />
                  {permission === "denied" ? "Enable in browser settings" : "Allow desktop notifications"}
                </button>}
                <div className={saving ? "saving-note" : "saved-note"}>{saving ? <RefreshCw size={12} /> : <Check size={12} />}
                  {saving ? "Saving settings…" : error ? "Settings need attention" : "Settings saved locally"}</div>
              </div>}
            </section>
            <section className="panel recent-activity-panel">
              <div className="panel-heading"><div><h2>Recent activity</h2><p>Warnings and calibration results</p></div>
                <span className="history-range">{recentActivity.length} events</span></div>
              {!warningsSuppressed && !previewWarnings && activeAlerts.length > 0 &&
                <div className={`alert-banner ${activeAlerts[0].severity.toLowerCase()}`}>
                  <AlertTriangle size={17} /><div><b>{activeAlerts[0].message}</b><span>{activeAlerts[0].suggestedAction}</span></div>
                </div>}
              {recentActivity.length ? <div className="activity-list">
                {recentActivity.map((item) => <article className={`activity-row ${item.type === "calibration" ? "calibration-activity" : ""}`} key={item.id}>
                  <span className={`activity-icon ${item.type === "calibration" ? "orange" : (item.severity || "warning").toLowerCase()}`}>
                    {item.type === "calibration" ? <Activity size={15} /> : <AlertTriangle size={15} />}
                  </span>
                  <div className="activity-copy">
                    <b>{item.type === "calibration" ? statusText(item.state) : item.message}</b>
                    <span>{item.type === "calibration"
                      ? `Peak CPU ${formatTemp(item.peakCpuTempC)} · throttle ${formatTemp(item.throttleCpuTempC)}${item.abortReason ? ` · ${item.abortReason}` : ""}`
                      : item.suggestedAction}</span>
                  </div>
                  <time>{formatTime(item.createdAt)}</time>
                  {item.type !== "calibration" && <span className={`activity-state ${item.active ? "active" : ""}`}>{item.active ? "Active" : "Resolved"}</span>}
                </article>)}
              </div> : <div className="inline-empty"><ShieldCheck size={17} />No recent activity.</div>}
              <div className="current-limit-summary">
                <span>Current CPU <b>{formatTemp(currentCpu)}</b></span>
                <span>CPU limit <b>{effectiveCpuLimit.toFixed(1)}°C</b></span>
                <span>Current GPU <b>{formatTemp(currentGpu)}</b></span>
                <span>GPU limit <b>{effectiveGpuLimit.toFixed(1)}°C</b></span>
              </div>
            </section>
          </>}

          <footer className="footer">
            <span><Fan size={13} /> SmartTherm only reads sensor and process state.</span>
            <span>{loading ? "Connecting…" : `Sensor update ${updatedTime} · refresh every 5 seconds`}</span>
          </footer>
        </div>
      </main>

      {completedRun && <div className="modal-backdrop" onMouseDown={(event) => {
        if (event.target === event.currentTarget) closeCompletion();
      }}>
        <section className="completion-modal" role="dialog" aria-modal="true"
          aria-labelledby="completion-title" tabIndex="-1" ref={modalRef}>
          <button className="modal-close icon-button" aria-label="Close completion dialog" onClick={closeCompletion}>×</button>
          <span className="modal-icon"><Check size={19} /></span>
          <h2 id="completion-title">{completedRun.calibrationState === "aborted_safety"
            ? `Stopped by safety limit${completedRun.calibrationAbortTempC == null ? "" : ` at ${formatTemp(completedRun.calibrationAbortTempC)}`}`
            : completedRun.calibrationState === "no_throttle_detected"
              ? "Test complete - no reliable throttle event detected"
              : completedRun.calibrationState === "cancelled"
                ? "Calibration was cancelled before the 300-second run completed."
                : "Stress test complete"}</h2>
          <div className="completion-temperature">
            {completedRun.calibrationThrottleCpuTempC == null
              ? "Throttle temperature unavailable"
              : formatTemp(completedRun.calibrationThrottleCpuTempC)}
          </div>
          <dl className="completion-details">
            <div><dt>Peak CPU temperature</dt><dd>{formatTemp(completedRun.calibrationPeakCpuTempC)}</dd></div>
            <div><dt>Peak GPU temperature</dt><dd>{formatTemp(completedRun.calibrationPeakGpuTempC)}</dd></div>
            <div><dt>Test duration</dt><dd>{completedRun.calibrationStartedAtUtc && completedRun.calibrationEndedAtUtc
              ? `${Math.max(0, (new Date(completedRun.calibrationEndedAtUtc) - new Date(completedRun.calibrationStartedAtUtc)) / 1000).toFixed(1)} seconds`
              : "Duration unavailable"}</dd></div>
            <div><dt>Warning limit change</dt><dd>{completedRun.calibrationThrottleCpuTempC == null
              ? "Unchanged"
              : `${cpuWarningLimit.toFixed(1)}°C to ${Number(completedRun.calibrationThrottleCpuTempC).toFixed(1)}°C`}</dd></div>
            {completedRun.calibrationAbortReason && <div><dt>Abort reason</dt><dd>{completedRun.calibrationAbortReason}</dd></div>}
          </dl>
          <p>Warnings resume after a short cooldown.</p>
          <button className="button primary modal-action" onClick={closeCompletion}>Close</button>
        </section>
      </div>}
    </div>
  );
}

export default App;
