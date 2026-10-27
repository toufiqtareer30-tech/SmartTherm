package local.smarttherm.api;

import local.smarttherm.config.WebConfig;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.MonitoringStatus;
import local.smarttherm.service.AlertService;
import local.smarttherm.service.CollectorProcessManager;
import local.smarttherm.service.ProcessTelemetryService;
import local.smarttherm.service.PythonPredictionClient;
import local.smarttherm.service.PredictionService;
import local.smarttherm.service.SettingsService;
import local.smarttherm.service.TelemetryService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.options;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.nio.file.Files;
import java.nio.file.Path;

@WebMvcTest(ApiController.class)
@Import(WebConfig.class)
class ApiControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private TelemetryService telemetryService;

    @MockitoBean
    private PredictionService predictionService;

    @MockitoBean
    private SettingsService settingsService;

    @MockitoBean
    private AlertService alertService;

    @MockitoBean
    private ProcessTelemetryService processTelemetryService;

    @MockitoBean
    private CollectorProcessManager collectorProcessManager;

    @MockitoBean
    private PythonPredictionClient pythonPredictionClient;

    @TempDir
    private Path tempDir;

    @Test
    void returnsHealthAndDisconnectedMonitoringState() throws Exception {
        when(telemetryService.status()).thenReturn(
                new MonitoringStatus(false, false, null, "Start the collector."));

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"));
        mockMvc.perform(get("/api/monitoring/status"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(false))
                .andExpect(jsonPath("$.message").value("Start the collector."));
    }

    @Test
    void validatesSettingsRequests() throws Exception {
        when(settingsService.update(new AppSettings(90.0, 85.0, true)))
                .thenReturn(new AppSettings(90.0, 85.0, true));

        mockMvc.perform(put("/api/settings")
                        .contentType("application/json")
                        .content("""
                                {"cpuWarningC":90,"gpuWarningC":85,"notificationsEnabled":true}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.cpuWarningC").value(90.0));

        mockMvc.perform(put("/api/settings")
                        .contentType("application/json")
                        .content("""
                                {"cpuWarningC":120,"gpuWarningC":85,"notificationsEnabled":true}
                                """))
                .andExpect(status().isBadRequest());
    }

    @Test
    void startsCollectionAndRequestsImmediateTrainingCheck() throws Exception {
        when(collectorProcessManager.startThrottleCalibration()).thenReturn(true);
        when(pythonPredictionClient.requestTrainingCheck()).thenReturn(true);

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post("/api/calibration/start"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.collectorStarted").value(true))
                .andExpect(jsonPath("$.trainingCheckRequested").value(true))
                .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("throttle calibration requested")));

        verify(collectorProcessManager).startThrottleCalibration();
        verify(pythonPredictionClient).requestTrainingCheck();
    }

    @Test
    void allowsBrowserPreflightForCalibrationPost() throws Exception {
        mockMvc.perform(options("/api/calibration/start")
                        .header("Origin", "http://127.0.0.1:5173")
                        .header("Access-Control-Request-Method", "POST"))
                .andExpect(status().isOk())
                .andExpect(header().string("Access-Control-Allow-Methods",
                        org.hamcrest.Matchers.containsString("POST")));
    }

    @Test
    void stopsCalibrationFromDashboard() throws Exception {
        when(collectorProcessManager.stopCollection()).thenReturn(true);

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post("/api/calibration/stop"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.collectorStopped").value(true))
                .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("Calibration stopped")));

        verify(collectorProcessManager).stopCollection();
    }

    @Test
    void exposesIdleCalibrationFieldsWhenNoStatusFileExists() throws Exception {
        when(collectorProcessManager.isCollectionRunning()).thenReturn(false);
        when(collectorProcessManager.throttleCalibrationStatusFile())
                .thenReturn(tempDir.resolve("missing-status.json"));

        mockMvc.perform(get("/api/calibration/status"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.calibrationState").value("idle"))
                .andExpect(jsonPath("$.calibrationPlannedDurationSeconds").value(300))
                .andExpect(jsonPath("$.calibrationRunId").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationStartedAtUtc").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationEndedAtUtc").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationPeakCpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationPeakGpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationThrottleCpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationThrottleGpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationAbortReason").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationAbortTempC").value(org.hamcrest.Matchers.nullValue()));
    }

    @Test
    void returnsPersistedCalibrationRunFields() throws Exception {
        when(collectorProcessManager.isCollectionRunning()).thenReturn(true);
        Path statusFile = tempDir.resolve("throttle-calibration-status.json");
        Files.writeString(statusFile, """
                {
                  "state":"running",
                  "message":"CPU stress and sensor collection are running.",
                  "calibrationRunId":"run-123",
                  "calibrationState":"running",
                  "calibrationStartedAtUtc":"2026-10-27T06:30:00Z",
                  "calibrationEndedAtUtc":null,
                  "calibrationPlannedDurationSeconds":300,
                  "calibrationPeakCpuTempC":91.5,
                  "calibrationPeakGpuTempC":74.2,
                  "calibrationThrottleCpuTempC":null,
                  "calibrationThrottleGpuTempC":null,
                  "calibrationAbortReason":null,
                  "calibrationAbortTempC":null
                }
                """);
        when(collectorProcessManager.throttleCalibrationStatusFile()).thenReturn(statusFile);

        mockMvc.perform(get("/api/calibration/status"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.calibrationRunId").value("run-123"))
                .andExpect(jsonPath("$.calibrationState").value("running"))
                .andExpect(jsonPath("$.calibrationPeakCpuTempC").value(91.5))
                .andExpect(jsonPath("$.calibrationPeakGpuTempC").value(74.2))
                .andExpect(jsonPath("$.calibrationEndedAtUtc").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationThrottleCpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationThrottleGpuTempC").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.calibrationPlannedDurationSeconds").value(300));
    }
}
